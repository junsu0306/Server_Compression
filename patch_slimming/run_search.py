"""
Phase 5 — Top-down mask search 진입점 (SPEC §11, §17 Phase 5).

Pretrained ViT/DeiT → calibration subset 기반 top-down search → layer별 고정 keep mask →
architecture.json + searched student checkpoint 저장.

사용법 (repo 루트에서):
    python patch_slimming/run_search.py --config patch_slimming/configs/deit_tiny_ps.yaml

재개: 같은 output_dir로 다시 실행하면 search_last.pt에서 layer 단위 재개 (SPEC §16.8).
"""

from __future__ import annotations

import os
import sys
import copy
import json
import argparse

import yaml
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _REPO_ROOT):                       # psvit(sibling) + repo 루트(engine 등)
    if p not in sys.path:
        sys.path.insert(0, p)

import timm
from psvit.model_utils import get_shape, validate_vit
from psvit.data import build_calibration_loader
from psvit.search import run_patch_slimming_search
from psvit.architecture import build_spec


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser("PS-ViT top-down mask search")
    ap.add_argument("--config", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    mcfg = cfg["model"]
    model_name = mcfg["name"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # ── student(탐색 대상) + teacher(frozen 원본) ────────────────────────────────
    student = timm.create_model(model_name, pretrained=True,
                                num_classes=mcfg.get("num_classes", 1000)).to(device)
    validate_vit(student)
    shp = get_shape(student)
    print(f"[model] {model_name}  L={shp.num_blocks}  D={shp.embed_dim}  H={shp.num_heads}  "
          f"N={shp.num_global_tokens}  classes={shp.num_classes}")

    teacher = copy.deepcopy(student).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ── calibration subset (deterministic) ─────────────────────────────────────
    ccfg = cfg["calibration"]
    calib_loader, data_cfg, indices = build_calibration_loader(
        model_name=model_name, data_path=cfg["data_path"],
        num_samples=ccfg["num_samples"], batch_size=ccfg["batch_size"],
        seed=ccfg.get("seed", 2022), split=ccfg.get("split", "train"),
        num_workers=ccfg.get("num_workers", 8),
        sample_id_file=os.path.join(out_dir, "calibration_ids.txt"),
    )
    print(f"[calib] {len(indices)} samples  batch={ccfg['batch_size']}  "
          f"mean/std={data_cfg['mean']}/{data_cfg['std']}")

    # ── search config ──────────────────────────────────────────────────────────
    scfg = cfg["search"]
    search_cfg = {
        "epsilon": scfg["epsilon"],
        "error_metric": scfg.get("error_metric", "mse"),
        "search_step": scfg["search_step"],
        "max_search_iters": scfg.get("max_search_iters", 10_000),
        "score_max_batches": scfg.get("score_max_batches"),
        "block_finetune": cfg.get("block_finetune", {}),
    }

    keep_ids, accepted_errors, records = run_patch_slimming_search(
        student, teacher, calib_loader, device, search_cfg,
        checkpoint_dir=out_dir, resume=not args.no_resume,
    )

    # ── 저장: architecture.json + searched student + records ────────────────────
    spec = build_spec(
        model_name, shp, keep_ids, accepted_errors,
        epsilon=scfg["epsilon"], error_metric=scfg.get("error_metric", "mse"),
        search_step=scfg["search_step"], score_mode=cfg.get("score", {}).get("mode", "paper_path_energy_dp"),
    )
    spec.to_json(os.path.join(out_dir, "architecture.json"))
    torch.save({"model": student.state_dict(), "model_name": model_name},
               os.path.join(out_dir, "searched_student.pt"))
    with open(os.path.join(out_dir, "search_records.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\n[done] architecture.json / searched_student.pt / search_records.jsonl → {out_dir}")
    print(f"       token schedule: {spec.token_schedule()}")


if __name__ == "__main__":
    main()
