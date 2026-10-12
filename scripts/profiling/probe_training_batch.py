#!/usr/bin/env python3
"""Short, isolated synthetic training probes; never save/update production weights."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "patch_slimming")]

import torch
import timm
import yaml
from timm.utils import ModelEmaV2
from pruning.engine import train_one_epoch
from pruning.memory_budget import MemoryBudget
from pruning.vit_pruning import ViTPruner


def worker(args, matrix):
    entry = matrix["models"][args.key]
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.manual_seed(2022)
    free_before, total = torch.cuda.mem_get_info()
    teacher = ema = pruner = budget = None
    if args.phase == "structural":
        initial = ROOT / matrix["output_root"] / args.key / "00_budget/baseline.pt"
        model = timm.create_model(entry["model"], pretrained=False).cuda()
        model.load_state_dict(torch.load(initial, map_location="cpu", weights_only=False)["state_dict"])
        ema = ModelEmaV2(model, device="cuda:0")
        teacher = copy.deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        pruner = ViTPruner(model, target_compression=entry["target_compression"],
                           index_refresh_steps=100)
        settings = dict(matrix["structural"])
        settings.update({k: v for k, v in entry.items() if k.startswith("memory_")})
        settings["memory_budget_json"] = str(initial.with_name("budget.json"))
        if settings.get("memory_penalty_weight", 0) > 0:
            budget = MemoryBudget(SimpleNamespace(
                **{k: v for k, v in settings.items() if k.startswith("memory_")},
                model=entry["model"], target_compression=entry["target_compression"], input_size=224,
                output_dir=str(Path(args.output).with_suffix(""))), "cuda:0")
    else:
        from psvit.architecture import ArchitectureSpec
        from psvit.compact import CompactPSViT
        from psvit.model_utils import apply_checkpoint_mlp_dims
        folder = ROOT / matrix["output_root"] / args.key / "03_patch_slimming_paper"
        spec = ArchitectureSpec.from_json(str(folder / "architecture.json"))
        checkpoint = torch.load(folder / "searched_student.pt", map_location="cpu", weights_only=False)
        base = timm.create_model(entry["model"], pretrained=False)
        apply_checkpoint_mlp_dims(base, checkpoint)
        base.load_state_dict(checkpoint["model"], strict=True)
        model = CompactPSViT(base, spec.keep_ids_list(), select_mode="index_select").cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda")
    samples = torch.randn(args.batch_size, 3, 224, 224, device="cuda")
    targets = torch.randint(1000, (args.batch_size,), device="cuda")
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    # Includes EMA, KD, pruning, optimizer state, and the isolated memory probe.
    train_one_epoch(model, criterion, [(samples, targets)] * 3, optimizer, scaler,
                    torch.device("cuda:0"), 0, model_ema=ema, pruner=pruner,
                    teacher=teacher, memory_budget=budget, clip_grad=1.0)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    stats = train_one_epoch(model, criterion, [(samples, targets)] * args.steps,
                           optimizer, scaler, torch.device("cuda:0"), 0,
                           model_ema=ema, pruner=pruner, teacher=teacher,
                           memory_budget=budget, clip_grad=1.0)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    free_after, _ = torch.cuda.mem_get_info()
    result = {
        "key": args.key, "phase": args.phase, "batch_size": args.batch_size,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "steps": args.steps, "synthetic_input": True,
        "images_per_second": args.batch_size * args.steps / seconds,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "free_before_mib": free_before / 2**20, "free_after_mib": free_after / 2**20,
        "total_mib": total / 2**20, "stats": stats,
        "memory": budget.metrics() if budget else {},
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--phase", choices=["structural", "patch"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--candidates", type=int, nargs="+")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--headroom-mib", type=float, default=1536)
    args = parser.parse_args()
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Probe output exists; inspect/reuse it or choose a new path: {target}")
    if args.batch_size:
        worker(args, matrix)
        return
    if not args.candidates or min(args.candidates) < 1:
        parser.error("--batch-size or positive --candidates required")
    rows = []
    for batch in sorted(set(args.candidates)):
        output = target.with_name(f"{target.stem}_b{batch}.json")
        command = [sys.executable, str(Path(__file__).resolve()), "--matrix", args.matrix,
                   "--key", args.key, "--phase", args.phase, "--batch-size", str(batch),
                   "--steps", str(args.steps), "--output", str(output)]
        # A previous parent may have stopped after a completed child. Reuse that
        # completed trial, rather than failing on the worker's no-overwrite guard.
        result_code = 0
        if not output.exists():
            with output.with_suffix(".log").open("w") as log:
                result_code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        if result_code:
            detail = output.with_suffix(".log").read_text()
            if "out of memory" not in detail.lower():
                raise RuntimeError(f"Probe failed (not OOM): {output.with_suffix('.log')}\n{detail[-4000:]}")
            rows.append({"batch_size": batch, "status": "oom"})
            break
        row = json.loads(output.read_text())
        if (row["key"], row["phase"], row["batch_size"]) != (args.key, args.phase, batch):
            raise ValueError(f"Cached trial belongs to another probe: {output}")
        row["status"] = "ok" if row["free_after_mib"] >= args.headroom_mib else "insufficient_headroom"
        rows.append(row)
        print(f"[probe] {args.key}/{args.phase} batch={batch}: {row['status']} "
              f"{row['images_per_second']:.1f} img/s, free={row['free_after_mib']:.0f} MiB", flush=True)
        if row["status"] != "ok":
            break
    accepted = [row for row in rows if row["status"] == "ok"]
    if not accepted:
        raise RuntimeError(f"No batch passed with {args.headroom_mib} MiB headroom; inspect {target.parent}")
    selected = max(accepted, key=lambda row: row["images_per_second"])
    target.write_text(json.dumps({"key": args.key, "phase": args.phase,
                                 "selected_batch": selected["batch_size"],
                                 "headroom_mib": args.headroom_mib, "trials": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
