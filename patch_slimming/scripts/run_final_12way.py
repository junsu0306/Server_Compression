"""Materialize and execute the final 4-model x 3-policy experiment matrix.

One process owns one physical GPU queue. Search and fine-tune phases are separate
so all architectures can be reviewed before the expensive 30-epoch recovery.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SEARCH = ROOT / "patch_slimming/scripts/search.py"
FINETUNE = ROOT / "patch_slimming/scripts/finetune.py"


def resolved_config(matrix: dict, model_key: str, policy_key: str) -> tuple[dict, Path]:
    model = matrix["models"][model_key]
    policy = matrix["policies"][policy_key]
    run_id = f"{model_key}__{policy_key}__rel002"
    out = ROOT / matrix["output_root"] / run_id
    batch = int(model["search_batch"])
    cfg = {
        "method": "patch_slimming_static",
        "model": {"name": model["name"], "num_classes": 1000,
                  "reduced_checkpoint": model["checkpoint"]},
        "data_path": matrix["data_path"], "output_dir": str(out),
        "calibration": {"num_samples": matrix["score_samples"], "batch_size": batch,
                        "seed": 2022, "split": "train", "num_workers": 8},
        "score_data": {"num_samples": matrix["score_samples"], "batch_size": batch,
                       "seed": 2022, "split": "train", "num_workers": 8},
        "reconstruction_train": {
            "num_samples": matrix["reconstruction_train_samples"], "batch_size": batch,
            "seed": 2023, "split": "train", "num_workers": 8},
        "reconstruction_eval": {
            "num_samples": matrix["reconstruction_eval_samples"], "batch_size": batch,
            "seed": 2024, "split": "train", "num_workers": 8},
        "score": {"mode": "paper_path_energy_dp"},
        "search": {"epsilon": matrix["epsilon"], "error_metric": matrix["error_metric"],
                   "search_step": matrix["search_step"], "max_search_iters": 10_000,
                   "score_max_batches": None,
                   "min_growth_per_shallower_layer": policy["min_growth_per_shallower_layer"]},
        "block_finetune": {"optimizer": "adamw", "learning_rate": 1e-5,
                           "weight_decay": 0.0, "epochs": matrix["block_finetune_epochs"],
                           "grad_clip_norm": 1.0},
        "final_finetune": {"epochs": matrix["final_finetune_epochs"], "lr": 2e-5,
                           "weight_decay": 0.05, "batch_size": model["final_batch"],
                           "warmup_epochs": 3, "num_workers": 8, "wandb": True,
                           "wandb_project": matrix["project"],
                           "wandb_run_name": f"{run_id}__finetune30",
                           "wandb_run_id": f"{model_key}-{policy_key}-ft30-20260806"},
        "wandb": {"enabled": True, "project": matrix["project"],
                  "run_name": f"{run_id}__search__{run_id}.yaml",
                  "run_id": f"{model_key}-{policy_key}-search-20260806"},
    }
    return cfg, out / f"{run_id}.yaml"


def materialize(matrix: dict) -> list[dict]:
    manifest = []
    for model_key, model in matrix["models"].items():
        for policy_key in matrix["policies"]:
            cfg, path = resolved_config(matrix, model_key, policy_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            manifest.append({"model": model_key, "policy": policy_key,
                             "gpu": int(model["gpu"]), "config": str(path),
                             "output_dir": cfg["output_dir"]})
    root = ROOT / matrix["output_root"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def run(cmd: list[str], gpu: int, log_path: Path) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", buffering=1) as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                       check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--phase", choices=("materialize", "search", "finetune"), required=True)
    args = ap.parse_args()
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    jobs = materialize(matrix)
    if args.phase == "materialize":
        print(f"materialized {len(jobs)} jobs")
        return
    jobs = [j for j in jobs if j["gpu"] == args.gpu]
    for job in jobs:
        out = Path(job["output_dir"])
        log = out / f"{args.phase}.log"
        if args.phase == "search":
            if (out / "architecture.json").exists():
                print(f"[skip] search complete: {out}")
                continue
            cmd = [sys.executable, str(SEARCH), "--config", job["config"], "--gpu", "0"]
        else:
            if not (out / "architecture.json").exists():
                raise FileNotFoundError(f"Search incomplete: {out}")
            last = out / "compact_finetuned_last.pt"
            if last.exists():
                ckpt = __import__("torch").load(last, map_location="cpu", weights_only=False)
                if int(ckpt.get("epoch", -1)) + 1 >= int(matrix["final_finetune_epochs"]):
                    print(f"[skip] finetune complete: {out}")
                    continue
            cmd = [sys.executable, str(FINETUNE), "--config", job["config"], "--resume"]
        print(f"[run] gpu={args.gpu} {job['model']} {job['policy']} {args.phase}")
        run(cmd, args.gpu, log)


if __name__ == "__main__":
    main()
