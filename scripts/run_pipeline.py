#!/usr/bin/env python3
"""Resumable final pipeline: global-L2 KD -> reducing -> paper Patch Slimming.

One process owns one physical GPU queue. Generated configs and every command are
stored below the experiment root. Heavy phases are intentionally separable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "pruning/train.py"
REDUCE = ROOT / "pruning/reduce.py"
EVALUATE = ROOT / "scripts/evaluation/evaluate.py"
PS_SEARCH = ROOT / "patch_slimming/scripts/search.py"
PS_FINETUNE = ROOT / "patch_slimming/scripts/finetune.py"
EXPORT_ONNX = ROOT / "scripts/export_final_onnx.py"
PROFILE_MEMORY = ROOT / "scripts/profiling/profile_reduced_memory.py"
PREPARE_BUDGET = ROOT / "scripts/profiling/prepare_budget.py"
BATCH_PROBE = ROOT / "scripts/profiling/probe_training_batch.py"


def run_id(matrix, key, stage):
    return f"{matrix['run_prefix']}-{key}-{stage}"


def write_json(path, data):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def paths(matrix: dict, key: str) -> dict[str, Path]:
    root = ROOT / matrix["output_root"] / key
    return {
        "root": root,
        "budget": root / "00_budget",
        "structural": root / "01_structural",
        "reduced": root / "02_reduced",
        "patch": root / "03_patch_slimming_paper",
        "onnx": root / "04_onnx",
        "memory": root / "05_memory",
        "evaluation": root / "06_evaluation",
    }


def structural_config(matrix: dict, key: str) -> tuple[dict, Path]:
    model = matrix["models"][key]
    common = matrix["structural"]
    p = paths(matrix, key)
    cfg = {
        "model": model["model"], "data_path": matrix["data_path"],
        "epochs": common["epochs"], "batch_size": model["structural_batch"],
        "lr_schedule_epochs": common["lr_schedule_epochs"],
        "lr": common["lr"], "weight_decay": common["weight_decay"],
        "clip_grad": common["clip_grad"], "warmup_epochs": common["warmup_epochs"],
        "min_lr": common["min_lr"], "smoothing": common["smoothing"],
        "amp": common["amp"], "num_workers": common["num_workers"],
        "model_ema": common["model_ema"], "model_ema_decay": common["model_ema_decay"],
        "target_compression": model["target_compression"],
        "pruning_max_sparsity": common["pruning_max_sparsity"],
        "prune_refresh_steps": common["prune_refresh_steps"],
        "seed": matrix["seed"],
        "initial_checkpoint": str(p["budget"] / "baseline.pt"),
        "memory_budget_json": str(p["budget"] / "budget.json"),
        "kd_alpha": common["kd_alpha"], "kd_temperature": common["kd_temperature"],
        "kd_teacher": model["teacher"], "output_dir": str(p["structural"]),
        "wandb": True, "wandb_project": matrix["project"],
        "wandb_run_name": f"{key}__global_l2_kd{matrix.get('run_label', '')}__structural{common['epochs']}",
        "wandb_run_id": run_id(matrix, key, "structural"),
    }
    # Shared measurement protocol; the budget always comes from the preparation stage.
    for settings in (common, model):
        cfg.update({k: v for k, v in settings.items() if k.startswith("memory_")})
    return cfg, p["root"] / "configs/structural.yaml"


def patch_config(matrix: dict, key: str) -> tuple[dict, Path]:
    model = matrix["models"][key]
    ps = matrix["patch_slimming"]
    p = paths(matrix, key)
    reduced = p["reduced"] / "reduced.pt"
    cfg = {
        "method": "patch_slimming_static",
        "seed": matrix["seed"],
        "model": {"name": model["model"], "num_classes": 1000,
                  "reduced_checkpoint": str(reduced)},
        "data_path": matrix["data_path"], "output_dir": str(p["patch"]),
        "calibration": {"num_samples": ps["score_samples"],
                        "batch_size": model["search_batch"], "seed": 2022,
                        "split": "train", "num_workers": 8},
        "score_data": {"num_samples": ps["score_samples"],
                       "batch_size": model["search_batch"], "seed": 2022,
                       "split": "train", "num_workers": 8},
        "reconstruction_train": {
            "num_samples": ps["reconstruction_train_samples"],
            "batch_size": model["search_batch"], "seed": 2023,
            "split": "train", "num_workers": 8},
        "reconstruction_eval": {
            "num_samples": ps["reconstruction_eval_samples"],
            "batch_size": model["search_batch"], "seed": 2024,
            "split": "train", "num_workers": 8},
        "score": {"mode": "paper_path_energy_dp"},
        "search": {"epsilon": ps["epsilon"], "error_metric": ps["error_metric"],
                   "search_step": ps["search_step"], "max_search_iters": 10000,
                   "score_max_batches": None},
        "block_finetune": {"optimizer": "adamw", "learning_rate": 1e-5,
                           "weight_decay": 0.0, "epochs": ps["block_finetune_epochs"],
                           "grad_clip_norm": 1.0},
        "final_finetune": {"epochs": ps["final_finetune_epochs"], "lr": 2e-5,
                           "weight_decay": 0.05,
                           "batch_size": model["patch_finetune_batch"],
                           "warmup_epochs": 3, "num_workers": 8, "wandb": True,
                           "wandb_project": matrix["project"],
                           "wandb_run_name": f"{key}{matrix.get('run_label', '')}__paper_rel002__finetune{ps['final_finetune_epochs']}",
                           "wandb_run_id": run_id(matrix, key, "patch-finetune")},
        "wandb": {"enabled": True, "project": matrix["project"],
                  "run_name": f"{key}{matrix.get('run_label', '')}__paper_rel002__search",
                  "run_id": run_id(matrix, key, "patch-search")},
    }
    return cfg, p["root"] / "configs/patch_slimming_paper.yaml"


def materialize(matrix: dict) -> list[dict]:
    if matrix["method"] != "baseline_target_peak_global_l2_kd_static_patch":
        raise ValueError("Unsupported experiment methodology")
    manifest = []
    for key, model in matrix["models"].items():
        p = paths(matrix, key)
        for folder in p.values():
            folder.mkdir(parents=True, exist_ok=True)
        scfg, spath = structural_config(matrix, key)
        pcfg, ppath = patch_config(matrix, key)
        spath.parent.mkdir(parents=True, exist_ok=True)
        ppath.parent.mkdir(parents=True, exist_ok=True)
        for path, cfg in ((spath, scfg), (ppath, pcfg)):
            if path.exists():
                if read_yaml(path) != cfg:
                    raise ValueError(f"Existing experiment config differs; use a new output_root: {path}")
            else:
                temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
                temporary.write_text(yaml.safe_dump(cfg, sort_keys=False))
                temporary.replace(path)
        manifest.append({"key": key, "gpu": model["gpu"],
                         "paths": {k: str(v) for k, v in p.items()},
                         "structural_config": str(spath), "patch_config": str(ppath)})
    experiment_root = ROOT / matrix["output_root"]
    write_json(experiment_root / "experiment_manifest.json", manifest)
    return manifest


def run(cmd: list[str], gpu: int, log: Path) -> None:
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_DIR"] = str(log.parent)
    env["TMPDIR"] = str(ROOT / ".cache/tmp")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    deps = str(ROOT / ".deps")
    env["PYTHONPATH"] = deps + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as stream:
        stream.write("\n$ " + " ".join(cmd) + "\n"); stream.flush()
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def training_batch(matrix, job, phase):
    entry = matrix["models"][job["key"]]
    fallback = entry["structural_batch" if phase == "structural" else "patch_finetune_batch"]
    candidates = entry.get(f"{phase}_batch_candidates")
    if not candidates:
        return fallback
    output = Path(job["paths"]["root"]) / "batch_probe" / f"{phase}.json"
    if not output.exists():
        run([sys.executable, str(BATCH_PROBE), "--matrix", matrix["_matrix_path"],
             "--key", job["key"], "--phase", phase, "--output", str(output),
             "--headroom-mib", str(matrix.get("batch_headroom_mib", 1536)),
             "--candidates", *map(str, candidates)], int(job["gpu"]), output.with_suffix(".log"))
    return int(json.loads(output.read_text())["selected_batch"])


def training_complete(path: Path, epochs: int) -> bool:
    if not path.exists():
        return False
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", -1)) + 1 >= epochs


def pin_terminal_checkpoint(directory: Path, epochs: int) -> Path:
    """Pin the exact terminal epoch; never substitute a later or best epoch."""
    source = directory / "checkpoint_last.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    completed = int(checkpoint.get("epoch", -1)) + 1
    if completed != epochs:
        raise ValueError(f"Need exact epoch {epochs}, got {completed}: {source}")
    del checkpoint
    target = directory / "checkpoint_final.pt"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError("Pinned terminal checkpoint differs from checkpoint_last")
    else:
        temporary = target.with_suffix(".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)
    write_json(directory / "terminal_checkpoint.json", {
        "completed_epochs": completed, "selection": "terminal_epoch",
        "source": str(source), "checkpoint": str(target), "sha256": digest})
    return target


def budget_phase(matrix: dict, job: dict) -> None:
    p = {k: Path(v) for k, v in job["paths"].items()}
    if not (p["budget"] / "budget.json").exists():
        run([sys.executable, str(PREPARE_BUDGET), "--matrix", matrix["_matrix_path"],
             "--key", job["key"]], int(job["gpu"]), p["budget"] / "prepare.log")


def structural_phase(matrix: dict, job: dict) -> None:
    key, gpu = job["key"], int(job["gpu"]); p = {k: Path(v) for k, v in job["paths"].items()}
    cfg = read_yaml(Path(job["structural_config"])); epochs = int(cfg["epochs"])
    last = p["structural"] / "checkpoint_last.pt"
    if matrix["structural"]["checkpoint_selection"] != "terminal_epoch":
        raise ValueError("Structural compression uses the terminal epoch checkpoint")
    if not training_complete(last, epochs):
        batch = training_batch(matrix, job, "structural")
        cmd = [sys.executable, str(TRAIN), "--config", job["structural_config"], "--batch-size", str(batch)]
        if last.exists(): cmd += ["--resume", str(last)]
        run(cmd, gpu, p["structural"] / "train.log")
    terminal = pin_terminal_checkpoint(p["structural"], epochs)
    reduced = p["reduced"] / "reduced.pt"
    if not reduced.exists():
        run([sys.executable, str(REDUCE), "--model", cfg["model"],
             "--checkpoint", str(terminal), "--output", str(reduced)],
            gpu, p["reduced"] / "reduce.log")


def patch_phase(matrix: dict, job: dict) -> None:
    gpu = int(job["gpu"]); p = {k: Path(v) for k, v in job["paths"].items()}
    reduced = p["reduced"] / "reduced.pt"
    if not reduced.exists(): raise FileNotFoundError(reduced)
    if not all((p["patch"] / name).exists() for name in ("architecture.json", "searched_student.pt")):
        run([sys.executable, str(PS_SEARCH), "--config", job["patch_config"], "--gpu", "0"],
            gpu, p["patch"] / "search.log")
    last = p["patch"] / "compact_finetuned_last.pt"
    epochs = int(matrix["patch_slimming"]["final_finetune_epochs"])
    if not training_complete(last, epochs):
        batch = training_batch(matrix, job, "patch")
        cmd = [sys.executable, str(PS_FINETUNE), "--config", job["patch_config"], "--batch-size", str(batch)]
        if last.exists(): cmd.append("--resume")
        run(cmd, gpu, p["patch"] / "finetune.log")


def export_phase(matrix: dict, job: dict) -> None:
    key = job["key"]; gpu = int(job["gpu"])
    p = {k: Path(v) for k, v in job["paths"].items()}
    model = matrix["models"][key]["model"]
    commands = [
        ("baseline", ["--kind", "baseline", "--model", model,
                      "--baseline-checkpoint", str(p["budget"] / "baseline.pt")]),
        ("reduced", ["--kind", "reduced", "--reduced", str(p["reduced"] / "reduced.pt")]),
        ("patch_slimming", ["--kind", "patch",
          "--arch", str(p["patch"] / "architecture.json"),
          "--weights", str(p["patch"] / "compact_finetuned_best.pt")]),
    ]
    for name, arguments in commands:
        suffix = "_npusafe" if matrix.get("npu_safe_export") else ""
        output = p["onnx"] / f"{key}__{name}__nhwc_b1{suffix}.onnx"
        marker = output.with_suffix(".verified.json")
        if output.exists() and marker.exists():
            print(f"[skip] ONNX exists: {output}"); continue
        flags = ["--npu-safe"] if matrix.get("npu_safe_export") else []
        run([sys.executable, str(EXPORT_ONNX), *arguments, "--output", str(output), *flags],
            gpu, p["onnx"] / f"export_{name}.log")


def memory_phase(matrix: dict, job: dict) -> None:
    key = job["key"]; gpu = int(job["gpu"])
    p = {k: Path(v) for k, v in job["paths"].items()}
    model = matrix["models"][key]["model"]
    commands = [
        ("baseline", ["--checkpoint", str(p["budget"] / "baseline.pt")]),
        ("reduced", ["--checkpoint", str(p["reduced"] / "reduced.pt")]),
        ("patch_slimming", ["--arch", str(p["patch"] / "architecture.json"),
          "--weights", str(p["patch"] / "compact_finetuned_best.pt")]),
    ]
    for name, arguments in commands:
        output = p["memory"] / f"{key}__{name}__b1_fp32.json"
        if output.exists():
            print(f"[skip] memory profile exists: {output}"); continue
        run([sys.executable, str(PROFILE_MEMORY), *arguments,
             "--device", "cuda:0", "--batch-size", "1", "--details",
             "--warmups", "10", "--runs", "50", "--output", str(output)],
            gpu, p["memory"] / f"profile_{name}.log")


def evaluate_phase(matrix, job):
    root = Path(job["paths"]["evaluation"])
    for kind in ("baseline", "pruned", "reduced", "patch_slimming"):
        output = root / f"{kind}.json"
        if not output.exists():
            run([sys.executable, str(EVALUATE), "--matrix", matrix["_matrix_path"],
                 "--key", job["key"], "--kind", kind, "--output", str(output)],
                int(job["gpu"]), output.with_suffix(".log"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/final.yaml")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--phase", required=True,
                        choices=["materialize", "budget", "structural", "patch", "export", "memory", "evaluate", "all"])
    args = parser.parse_args()
    matrix_path = (ROOT / args.matrix).resolve()
    matrix = read_yaml(matrix_path)
    matrix["_matrix_path"] = str(matrix_path)
    jobs = materialize(matrix)
    if args.phase == "materialize":
        print(f"materialized {len(jobs)} jobs under {matrix['output_root']}"); return
    if args.gpu is None: parser.error("--gpu is required for heavy phases")
    jobs = [job for job in jobs if int(job["gpu"]) == args.gpu]
    if not jobs:
        parser.error(f"No models assigned to GPU {args.gpu} in this matrix")
    root = ROOT / matrix["output_root"]
    with (root / f"gpu{args.gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = root / f"gpu{args.gpu}_status.json"
        status = {"gpu": args.gpu, "pid": os.getpid(), "jobs": [j["key"] for j in jobs],
                  "completed": [], "state": "running"}
        try:
            for job in jobs:
                for phase, function in (("budget", budget_phase), ("structural", structural_phase),
                                        ("patch", patch_phase), ("export", export_phase),
                                        ("memory", memory_phase), ("evaluate", evaluate_phase)):
                    if args.phase not in (phase, "all"):
                        continue
                    status.update(key=job["key"], phase=phase, updated_utc=datetime.now(timezone.utc).isoformat())
                    write_json(status_path, status)
                    print(f"[queue] gpu={args.gpu} job={job['key']} phase={phase}", flush=True)
                    function(matrix, job)
                    status["completed"].append(f"{job['key']}/{phase}")
                    write_json(status_path, status)
            status["state"] = "complete"
        except BaseException as exc:
            status.update(state="failed", error=str(exc))
            raise
        finally:
            status["updated_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(status_path, status)
        if args.phase == "all":
            # Serialize shared report writes across independent GPU workers.
            with (root / "report.lock").open("a") as report_lock:
                fcntl.flock(report_lock, fcntl.LOCK_EX)
                run([sys.executable, str(ROOT / "scripts/build_report.py"),
                     "--matrix", str(matrix_path)], args.gpu, root / "report.log")


if __name__ == "__main__":
    main()
