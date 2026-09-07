#!/usr/bin/env python3
"""Resumable final pipeline: global-L2 KD -> reducing -> paper Patch Slimming.

One process owns one physical GPU queue. Generated configs and every command are
stored below the experiment root. Heavy phases are intentionally separable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "pruning/train.py"
REDUCE = ROOT / "pruning/reduce.py"
EVAL_REDUCED = ROOT / "scripts/evaluation/eval_reduced.py"
PS_SEARCH = ROOT / "patch_slimming/scripts/search.py"
PS_FINETUNE = ROOT / "patch_slimming/scripts/finetune.py"
EXPORT_ONNX = ROOT / "scripts/export_final_onnx.py"
PROFILE_MEMORY = ROOT / "patch_slimming/scripts/profile_memory.py"


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def paths(matrix: dict, key: str) -> dict[str, Path]:
    root = ROOT / matrix["output_root"] / key
    return {
        "root": root,
        "structural": root / "01_structural",
        "reduced": root / "02_reduced",
        "patch": root / "03_patch_slimming_paper",
        "onnx": root / "04_onnx",
        "memory": root / "05_memory",
    }


def structural_config(matrix: dict, key: str) -> tuple[dict, Path]:
    model = matrix["models"][key]
    common = matrix["structural"]
    p = paths(matrix, key)
    cfg = {
        "model": model["model"], "data_path": matrix["data_path"],
        "epochs": common["epochs"], "batch_size": model["structural_batch"],
        "lr": common["lr"], "weight_decay": common["weight_decay"],
        "clip_grad": common["clip_grad"], "warmup_epochs": common["warmup_epochs"],
        "min_lr": common["min_lr"], "smoothing": common["smoothing"],
        "amp": common["amp"], "num_workers": common["num_workers"],
        "model_ema": common["model_ema"], "model_ema_decay": common["model_ema_decay"],
        "target_compression": model["target_compression"],
        "pruning_max_sparsity": common["pruning_max_sparsity"],
        "prune_refresh_steps": common["prune_refresh_steps"],
        "pruning_mode": common["pruning_mode"],
        "pruning_importance": common["pruning_importance"],
        "prune_warmup_epochs": common["prune_warmup_epochs"],
        "prune_ramp_epochs": common["prune_ramp_epochs"],
        "kd_alpha": common["kd_alpha"], "kd_temperature": common["kd_temperature"],
        "kd_teacher": model["teacher"], "output_dir": str(p["structural"]),
        "wandb": True, "wandb_project": matrix["project"],
        "wandb_run_name": f"{key}__global_l2_kd__structural50",
        "wandb_run_id": f"final-gkd-l2-{key}-struct-corrected-20260814",
    }
    return cfg, p["root"] / "configs/structural.yaml"


def patch_config(matrix: dict, key: str) -> tuple[dict, Path]:
    model = matrix["models"][key]
    ps = matrix["patch_slimming"]
    p = paths(matrix, key)
    reduced = p["reduced"] / "reduced.pt"
    cfg = {
        "method": "patch_slimming_static",
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
                   "score_max_batches": None, "min_growth_per_shallower_layer": 0},
        "block_finetune": {"optimizer": "adamw", "learning_rate": 1e-5,
                           "weight_decay": 0.0, "epochs": ps["block_finetune_epochs"],
                           "grad_clip_norm": 1.0},
        "final_finetune": {"epochs": ps["final_finetune_epochs"], "lr": 2e-5,
                           "weight_decay": 0.05,
                           "batch_size": model["patch_finetune_batch"],
                           "warmup_epochs": 3, "num_workers": 8, "wandb": True,
                           "wandb_project": matrix["project"],
                           "wandb_run_name": f"{key}__paper_rel002__finetune30",
                           "wandb_run_id": f"final-gkd-l2-{key}-ps-ft-20260814"},
        "wandb": {"enabled": True, "project": matrix["project"],
                  "run_name": f"{key}__paper_rel002__search",
                  "run_id": f"final-gkd-l2-{key}-ps-search-20260814"},
    }
    return cfg, p["root"] / "configs/patch_slimming_paper.yaml"


def materialize(matrix: dict) -> list[dict]:
    manifest = []
    for key, model in matrix["models"].items():
        p = paths(matrix, key)
        for folder in p.values():
            folder.mkdir(parents=True, exist_ok=True)
        scfg, spath = structural_config(matrix, key)
        pcfg, ppath = patch_config(matrix, key)
        spath.parent.mkdir(parents=True, exist_ok=True)
        ppath.parent.mkdir(parents=True, exist_ok=True)
        spath.write_text(yaml.safe_dump(scfg, sort_keys=False))
        ppath.write_text(yaml.safe_dump(pcfg, sort_keys=False))
        manifest.append({"key": key, "gpu": model["gpu"],
                         "paths": {k: str(v) for k, v in p.items()},
                         "structural_config": str(spath), "patch_config": str(ppath)})
    experiment_root = ROOT / matrix["output_root"]
    (experiment_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run(cmd: list[str], gpu: int, log: Path) -> None:
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    deps = str(ROOT / ".deps")
    env["PYTHONPATH"] = deps + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as stream:
        stream.write("\n$ " + " ".join(cmd) + "\n"); stream.flush()
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def training_complete(path: Path, epochs: int) -> bool:
    if not path.exists():
        return False
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("epoch", -1)) + 1 >= epochs


def structural_phase(matrix: dict, job: dict) -> None:
    key, gpu = job["key"], int(job["gpu"]); p = {k: Path(v) for k, v in job["paths"].items()}
    cfg = read_yaml(Path(job["structural_config"])); epochs = int(cfg["epochs"])
    last, best = p["structural"] / "checkpoint_last.pt", p["structural"] / "checkpoint_best.pt"
    if not training_complete(last, epochs):
        cmd = [sys.executable, str(TRAIN), "--config", job["structural_config"]]
        if last.exists(): cmd += ["--resume", str(last)]
        run(cmd, gpu, p["structural"] / "train.log")
    if not best.exists():
        raise FileNotFoundError(f"missing best checkpoint: {best}")
    reduced = p["reduced"] / "reduced.pt"
    if not reduced.exists():
        run([sys.executable, str(REDUCE), "--model", cfg["model"],
             "--checkpoint", str(best), "--output", str(reduced)],
            gpu, p["reduced"] / "reduce.log")
    marker = p["reduced"] / "eval_complete.json"
    if not marker.exists():
        run([sys.executable, str(EVAL_REDUCED), "--reduced", str(reduced),
             "--data-path", matrix["data_path"], "--gpu", "0", "--wandb",
             "--wandb-project", matrix["project"],
             "--wandb-run-name", f"{key}__global_l2_kd__reduced_eval"],
            gpu, p["reduced"] / "eval.log")
        marker.write_text(json.dumps({"completed": True}))


def patch_phase(matrix: dict, job: dict) -> None:
    gpu = int(job["gpu"]); p = {k: Path(v) for k, v in job["paths"].items()}
    reduced = p["reduced"] / "reduced.pt"
    if not reduced.exists(): raise FileNotFoundError(reduced)
    if not (p["patch"] / "architecture.json").exists():
        run([sys.executable, str(PS_SEARCH), "--config", job["patch_config"], "--gpu", "0"],
            gpu, p["patch"] / "search.log")
    last = p["patch"] / "compact_finetuned_last.pt"
    epochs = int(matrix["patch_slimming"]["final_finetune_epochs"])
    if not training_complete(last, epochs):
        cmd = [sys.executable, str(PS_FINETUNE), "--config", job["patch_config"]]
        if last.exists(): cmd.append("--resume")
        run(cmd, gpu, p["patch"] / "finetune.log")


def export_phase(matrix: dict, job: dict) -> None:
    key = job["key"]; gpu = int(job["gpu"])
    p = {k: Path(v) for k, v in job["paths"].items()}
    model = matrix["models"][key]["model"]
    commands = [
        ("baseline", ["--kind", "baseline", "--model", model]),
        ("reduced", ["--kind", "reduced", "--reduced", str(p["reduced"] / "reduced.pt")]),
        ("patch_slimming", ["--kind", "patch",
          "--arch", str(p["patch"] / "architecture.json"),
          "--weights", str(p["patch"] / "compact_finetuned_best.pt")]),
    ]
    for name, arguments in commands:
        output = p["onnx"] / f"{key}__{name}__nhwc_b1.onnx"
        if output.exists():
            print(f"[skip] ONNX exists: {output}"); continue
        run([sys.executable, str(EXPORT_ONNX), *arguments, "--output", str(output)],
            gpu, p["onnx"] / f"export_{name}.log")


def memory_phase(matrix: dict, job: dict) -> None:
    key = job["key"]; gpu = int(job["gpu"])
    p = {k: Path(v) for k, v in job["paths"].items()}
    model = matrix["models"][key]["model"]
    commands = [
        ("baseline", ["--model", model, "--pretrained"]),
        ("reduced", ["--reduced", str(p["reduced"] / "reduced.pt")]),
        ("patch_slimming", ["--arch", str(p["patch"] / "architecture.json"),
          "--weights", str(p["patch"] / "compact_finetuned_best.pt"),
          "--select-mode", "matmul"]),
    ]
    for name, arguments in commands:
        output = p["memory"] / f"{key}__{name}__b1_fp32.json"
        if output.exists():
            print(f"[skip] memory profile exists: {output}"); continue
        run([sys.executable, str(PROFILE_MEMORY), *arguments,
             "--device", "cuda:0", "--batch-size", "1", "--dtype", "float32",
             "--warmups", "10", "--runs", "50", "--output", str(output)],
            gpu, p["memory"] / f"profile_{name}.log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/final_global_kd_l2_20260814.yaml")
    parser.add_argument("--gpu", type=int, choices=[6, 7])
    parser.add_argument("--phase", required=True,
                        choices=["materialize", "structural", "patch", "export", "memory", "all"])
    args = parser.parse_args()
    matrix = read_yaml(ROOT / args.matrix)
    jobs = materialize(matrix)
    if args.phase == "materialize":
        print(f"materialized {len(jobs)} jobs under {matrix['output_root']}"); return
    if args.gpu is None: parser.error("--gpu is required for heavy phases")
    jobs = [job for job in jobs if int(job["gpu"]) == args.gpu]
    for job in jobs:
        print(f"[queue] gpu={args.gpu} job={job['key']} phase={args.phase}")
        if args.phase in ("structural", "all"): structural_phase(matrix, job)
        if args.phase in ("patch", "all"): patch_phase(matrix, job)
        if args.phase in ("export", "all"): export_phase(matrix, job)
        if args.phase in ("memory", "all"): memory_phase(matrix, job)


if __name__ == "__main__":
    main()
