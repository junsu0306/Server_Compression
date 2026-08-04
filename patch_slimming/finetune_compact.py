"""
Phase 6b — Compact PS-ViT 전체 classification fine-tuning (SPEC §13, §17 Phase 6).

모든 keep_ids 고정, compact 모델 전체 parameter 학습, CE loss. baseline fine-tuning
recipe 재사용. distillation은 논문 핵심 알고리즘이 아니므로 자동 추가 안 함(§13.2).

사용법:
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 patch_slimming/finetune_compact.py \
        --config patch_slimming/configs/deit_tiny_ps.yaml
"""

from __future__ import annotations

import os
import sys
import argparse

import yaml
import torch
import torch.nn as nn
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import timm
import timm.data
from timm.data import create_transform
from torchvision import datasets

from psvit.architecture import ArchitectureSpec
from psvit.compact import CompactPSViT
from engine import train_one_epoch, evaluate


def setup_ddp(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"]); args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl"); torch.cuda.set_device(args.gpu); args.distributed = True
    else:
        args.rank, args.gpu, args.distributed = 0, 0, False
    return args.rank == 0


def build_loaders(model_name, data_path, batch_size, num_workers, distributed):
    ref = timm.create_model(model_name, pretrained=False)
    dc = timm.data.resolve_model_data_config(ref); del ref
    train_tf = create_transform(input_size=dc["input_size"], is_training=True,
                                color_jitter=0.4, auto_augment="rand-m9-mstd0.5-inc1",
                                interpolation=dc["interpolation"], re_prob=0.25,
                                re_mode="pixel", re_count=1, mean=dc["mean"], std=dc["std"])
    val_tf = timm.data.create_transform(**dc, is_training=False)
    tr = datasets.ImageFolder(os.path.join(data_path, "train"), transform=train_tf)
    va = datasets.ImageFolder(os.path.join(data_path, "val"), transform=val_tf)
    tr_samp = torch.utils.data.DistributedSampler(tr) if distributed else torch.utils.data.RandomSampler(tr)
    tl = torch.utils.data.DataLoader(tr, batch_size=batch_size, sampler=tr_samp,
                                     num_workers=num_workers, pin_memory=True, drop_last=True)
    vl = torch.utils.data.DataLoader(va, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, pin_memory=True)
    return tl, vl, tr_samp


def main():
    ap = argparse.ArgumentParser("Final fine-tune CompactPSViT")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    is_main = setup_ddp(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    out_dir = cfg["output_dir"]
    arch_path = os.path.join(out_dir, "architecture.json")
    weights_path = os.path.join(out_dir, "searched_student.pt")
    spec = ArchitectureSpec.from_json(arch_path); spec.validate()
    state = torch.load(weights_path, map_location="cpu", weights_only=False)["model"]

    base = timm.create_model(spec.model_name, pretrained=False,
                             num_classes=state["head.weight"].shape[0])
    base.load_state_dict(state, strict=True)
    keep_ids = spec.keep_ids_list(device=device)
    model = CompactPSViT(base, keep_ids, select_mode="index_select").to(device)

    ft = cfg.get("final_finetune", {})
    epochs = int(ft.get("epochs") or 30)
    lr = float(ft.get("lr", 2e-5)); wd = float(ft.get("weight_decay", 0.05))
    batch_size = int(ft.get("batch_size", 256)); warmup = int(ft.get("warmup_epochs", 3))

    tl, vl, tr_samp = build_loaders(spec.model_name, cfg["data_path"], batch_size,
                                    ft.get("num_workers", 8), args.distributed)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, 1.0, total_iters=warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - warmup, 1), eta_min=1e-6)
    sched = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[warmup])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best = 0.0
    for epoch in range(epochs):
        if args.distributed:
            tr_samp.set_epoch(epoch)
        if is_main:
            print(f"\n── Epoch {epoch}/{epochs-1}  lr={optimizer.param_groups[0]['lr']:.2e} ──")
        train_one_epoch(model, criterion, tl, optimizer, scaler, device, epoch,
                        amp=True, clip_grad=1.0)
        sched.step()
        eval_model = model.module if hasattr(model, "module") else model
        stats = evaluate(vl, eval_model, device, amp=True)
        if is_main:
            best = max(best, stats["acc1"])
            print(f"  val_top1={stats['acc1']:.2f}%  best={best:.2f}%")
            torch.save({"model": eval_model.model.state_dict(), "model_name": spec.model_name,
                        "architecture": arch_path, "epoch": epoch, "best_acc1": best},
                       os.path.join(out_dir, "compact_finetuned_last.pt"))
            if stats["acc1"] >= best:
                torch.save({"model": eval_model.model.state_dict(), "model_name": spec.model_name,
                            "architecture": arch_path, "epoch": epoch, "best_acc1": best},
                           os.path.join(out_dir, "compact_finetuned_best.pt"))

    if is_main:
        print(f"\n[done] best val top1={best:.2f}%  → {out_dir}/compact_finetuned_best.pt")
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
