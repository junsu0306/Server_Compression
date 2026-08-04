"""
Phase 6a — Searched student + architecture → CompactPSViT 평가 (SPEC §12, §16.9, §17 Phase 6).

실제 token tensor 축소가 반영된 compact 모델을 만들어 ImageNet val로 평가한다
(최종 fine-tuning 전 baseline). token schedule / 파라미터 / FLOPs(추정) / top1 보고.

사용법:
    python patch_slimming/build_compact.py \
        --arch   ./output/ps_deit_tiny/architecture.json \
        --weights ./output/ps_deit_tiny/searched_student.pt \
        --data-path /workspace/.../imagenet --gpu 4
"""

from __future__ import annotations

import os
import sys
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import timm
import timm.data
from torchvision import datasets

from psvit.architecture import ArchitectureSpec
from psvit.compact import CompactPSViT
from psvit.model_utils import get_shape
from engine import evaluate


def build_val_loader(model_name, data_path, batch_size, num_workers=8):
    ref = timm.create_model(model_name, pretrained=False)
    cfg = timm.data.resolve_model_data_config(ref)
    del ref
    tf = timm.data.create_transform(**cfg, is_training=False)
    ds = datasets.ImageFolder(os.path.join(data_path, "val"), transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=True), cfg


def load_compact(arch_path, weights_path, device, select_mode="index_select"):
    spec = ArchitectureSpec.from_json(arch_path)
    spec.validate()
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model = timm.create_model(spec.model_name, pretrained=False,
                              num_classes=state["head.weight"].shape[0])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    keep_ids = spec.keep_ids_list(device=device)
    return CompactPSViT(model, keep_ids, select_mode=select_mode).to(device).eval(), spec


def main():
    ap = argparse.ArgumentParser("Build & evaluate CompactPSViT")
    ap.add_argument("--arch", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--save", default="", help="compact.pt 저장 경로(선택)")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    compact, spec = load_compact(args.arch, args.weights, device)
    shp = get_shape(compact.model)

    n_params = sum(p.numel() for p in compact.model.parameters())
    print(f"[compact] {spec.model_name}  params={n_params:,}  (Patch Slimming은 weight 유지)")
    print(f"          token schedule: {compact.token_schedule()}")

    val_loader, dcfg = build_val_loader(spec.model_name, args.data_path, args.batch_size)
    print(f"[eval] ImageNet val  mean/std={dcfg['mean']}/{dcfg['std']}")
    metrics = evaluate(val_loader, compact, device, amp=True)
    print(f"[eval] top1={metrics['acc1']:.2f}%  top5={metrics['acc5']:.2f}%  loss={metrics['loss']:.4f}")

    if args.save:
        torch.save({"model": compact.model.state_dict(), "model_name": spec.model_name,
                    "architecture": args.arch}, args.save)
        print(f"[save] {args.save}")


if __name__ == "__main__":
    main()
