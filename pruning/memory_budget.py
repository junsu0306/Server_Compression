"""Measured memory excess added as a scalar loss term (no weight gradient)."""

from __future__ import annotations

import json
import hashlib
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

PROFILER = Path(__file__).resolve().parents[1] / "scripts/profiling/profile_reduced_memory.py"


def memory_penalty(memory_mib, max_mib, weight):
    if not all(math.isfinite(x) for x in (memory_mib, max_mib, weight)):
        raise ValueError("Memory values and penalty weight must be finite")
    if memory_mib < 0 or max_mib <= 0 or weight < 0:
        raise ValueError("Memory/weight must be nonnegative and budget must be positive")
    return weight * max(0.0, memory_mib - max_mib)


class MemoryBudget:
    def __init__(self, args, device):
        self.model_name = args.model
        self.device = str(device)
        self.weight = args.memory_penalty_weight
        budget_path = Path(args.memory_budget_json).resolve()
        budget = json.loads(budget_path.read_text())
        if (budget["model"] != self.model_name or budget["trained_steps"] != 0 or
                budget["metric"] != "cuda_peak_allocated_bytes" or
                budget["target_compression"] != args.target_compression):
            raise ValueError("Memory budget provenance does not match this experiment")
        measured = budget["target_profile"]
        if getattr(args, "initial_checkpoint", None):
            if Path(args.initial_checkpoint).resolve() != Path(budget["baseline_checkpoint"]).resolve():
                raise ValueError("Training must start from the budget baseline checkpoint")
        expected = {"dtype": "float32", "warmups": args.memory_warmups, "runs": args.memory_runs,
                    "input_shape": [args.memory_batch_size, 3, args.input_size, args.input_size]}
        if any(measured[k] != v for k, v in expected.items()):
            raise ValueError("Budget measurement settings differ from candidate settings")
        for name in ("baseline", "target"):
            if hashlib.sha256(Path(budget[f"{name}_checkpoint"]).read_bytes()).hexdigest() != budget[f"{name}_sha256"]:
                raise ValueError(f"Budget {name} checkpoint has changed")
        self.max_mib = budget["max_mib"]
        if self.max_mib != measured["peak_allocated_bytes"] / 2**20:
            raise ValueError("Budget peak does not match recorded measurement")
        memory_penalty(0, self.max_mib, self.weight)
        self.interval = args.memory_profile_interval
        self.profile_args = ["--device", self.device, "--batch-size", str(args.memory_batch_size),
                             "--input-size", str(args.input_size), "--warmups", str(args.memory_warmups),
                             "--runs", str(args.memory_runs), "--expected-model", self.model_name]
        self.directory = Path(args.output_dir) / "memory_budget"
        self.memory_mib = None
        self.last_epoch = None
        self.config = {"model": self.model_name, "budget_json": str(budget_path),
                       "budget_sha256": hashlib.sha256(budget_path.read_bytes()).hexdigest(),
                       "max_mib": self.max_mib, "weight": self.weight,
                       "interval": self.interval, "batch_size": args.memory_batch_size,
                       "input_size": args.input_size, "warmups": args.memory_warmups,
                       "runs": args.memory_runs}

    @property
    def penalty(self):
        if self.memory_mib is None:
            return 0.0
        return memory_penalty(self.memory_mib, self.max_mib, self.weight)

    def _profile(self, checkpoint, output):
        command = [sys.executable, str(PROFILER), "--checkpoint", str(checkpoint),
                   "--output", str(output), *self.profile_args]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"Memory profiler failed:\n{completed.stdout}\n{completed.stderr}")
        return json.loads(output.read_text())["peak_allocated_bytes"] / 2**20

    def maybe_refresh(self, model, model_ema, epoch):
        if self.last_epoch is not None and epoch < self.last_epoch + self.interval:
            return
        distributed = dist.is_available() and dist.is_initialized()
        result = [None]
        if not distributed or dist.get_rank() == 0:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                raw = model.module if hasattr(model, "module") else model
                # CPU snapshots do not move/resize the training model or touch its RNG.
                snapshot = {"args": {"model": self.model_name},
                            "model": {k: v.detach().to("cpu", copy=True) for k, v in raw.state_dict().items()}}
                if model_ema is not None:
                    snapshot["model_ema"] = {
                        k: v.detach().to("cpu", copy=True) for k, v in model_ema.module.state_dict().items()}
                with tempfile.TemporaryDirectory(prefix="snapshot_", dir=self.directory) as temp:
                    checkpoint = Path(temp) / "candidate.pt"
                    torch.save(snapshot, checkpoint)
                    self.memory_mib = self._profile(checkpoint, self.directory / f"epoch_{epoch:04d}.json")
                self.last_epoch = epoch
                # Validate before sharing with other ranks.
                penalty = self.penalty
                result[0] = {"state": self.state_dict()}
                print(f"[Memory] reduced={self.memory_mib:.4f} MiB  "
                      f"M_max={self.max_mib:.4f} MiB  penalty={penalty:.6f}")
            except Exception as exc:
                result[0] = {"error": str(exc)}
        if distributed:
            dist.broadcast_object_list(result, src=0)
        if "error" in result[0]:
            raise RuntimeError(result[0]["error"])
        self.load_state_dict(result[0]["state"])

    def metrics(self):
        if self.memory_mib is None:
            return {}
        return {"memory/reduced_mib": self.memory_mib, "memory/max_mib": self.max_mib,
                "memory/excess_mib": max(0.0, self.memory_mib - self.max_mib),
                "memory/penalty": self.penalty, "memory/profile_epoch": self.last_epoch}

    def state_dict(self):
        return {"config": self.config, "max_mib": self.max_mib,
                "memory_mib": self.memory_mib, "last_epoch": self.last_epoch}

    def load_state_dict(self, state):
        if state["config"] != self.config:
            raise ValueError("Memory budget settings differ from checkpoint; use matching settings to resume")
        self.max_mib = state["max_mib"]
        self.memory_mib = state["memory_mib"]
        self.last_epoch = state["last_epoch"]
