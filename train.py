"""
timm ViT / DeiT Soft Pruning Fine-tuning

단일 GPU:
    python train.py \\
        --model vit_small_patch16_224 \\
        --data-path /data/imagenet \\
        --epochs 30 \\
        --batch-size 256 \\
        --target-compression 0.20 \\
        --output-dir ./output/vit_small_prune20 \\
        --wandb

멀티 GPU (DDP, torchrun):
    torchrun --nproc_per_node=4 train.py \\
        --model vit_base_patch16_224 \\
        --data-path /data/imagenet \\
        --epochs 30 \\
        --batch-size 128 \\
        --target-compression 0.30 \\
        --output-dir ./output/vit_base_prune30

체크포인트 재개:
    python train.py ... --resume ./output/vit_small_prune20/checkpoint_last.pt
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import timm
from timm.utils import ModelEmaV2
from timm.data import create_transform
import timm.data
from torchvision import datasets

from pruning.vit_pruning import ViTPruner
from engine import train_one_epoch, evaluate


# ── argparse ───────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ViT Soft Pruning Fine-tuning")

    # 데이터 — 서버 경로는 실행 시 --data-path 로 지정
    p.add_argument("--data-path", required=True,
                   help="ImageNet 루트 디렉터리 (하위에 train/ val/ 존재)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--pin-mem",     action="store_true", default=True)

    # 모델
    p.add_argument("--model",       default="vit_tiny_patch16_224",
                   help="timm 모델 이름 (예: vit_base_patch16_224)")
    p.add_argument("--input-size",  type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000)

    # 학습 하이퍼파라미터
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--batch-size",     type=int,   default=256,
                   help="GPU 당 배치 크기")
    p.add_argument("--lr",             type=float, default=5e-5,
                   help="초기 learning rate (ViT fine-tuning 권장: 5e-5 ~ 1e-4)")
    p.add_argument("--weight-decay",   type=float, default=0.05)
    p.add_argument("--clip-grad",      type=float, default=1.0)
    p.add_argument("--warmup-epochs",  type=int,   default=5)
    p.add_argument("--min-lr",         type=float, default=1e-6,
                   help="Cosine LR 최솟값")
    p.add_argument("--smoothing",      type=float, default=0.1,
                   help="Label smoothing epsilon")
    p.add_argument("--amp",            action="store_true", default=True,
                   help="자동 혼합 정밀도 (AMP) 사용")

    # EMA
    p.add_argument("--model-ema",       action="store_true", default=True)
    p.add_argument("--model-ema-decay", type=float,          default=0.9998)

    # Soft Pruning
    p.add_argument("--target-compression",  type=float, default=0.0,
                   help="목표 파라미터 압축률 0.0~1.0 (0=비활성)")
    p.add_argument("--pruning-max-sparsity", type=float, default=0.95)
    p.add_argument("--prune-refresh-steps",  type=int,   default=100,
                   help="마스크 재계산 주기 (step 단위). 0=매 step.")

    # 출력 / 체크포인트
    p.add_argument("--output-dir",   default="./output")
    p.add_argument("--resume",       default="",
                   help="재개할 체크포인트 경로")
    p.add_argument("--log-interval", type=int, default=50,
                   help="배치 로그 출력 주기")

    # WandB
    p.add_argument("--wandb",          action="store_true")
    p.add_argument("--wandb-project",  default="vit-pruning")
    p.add_argument("--wandb-run-name", default="",
                   help="비어있으면 {model}_prune{compression%} 로 자동 설정")
    p.add_argument("--wandb-run-id",   default="",
                   help="Resume 시 기존 run id 입력")

    # DDP
    p.add_argument("--dist-url", default="env://")

    return p.parse_args()


# ── 분산 학습 설정 ──────────────────────────────────────────────────────────────

def setup_distributed(args: argparse.Namespace) -> bool:
    """DDP 초기화. is_main_process 반환."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank       = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu        = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl", init_method=args.dist_url)
        torch.cuda.set_device(args.gpu)
        args.distributed = True
    else:
        args.distributed = False
        args.rank        = 0
        args.gpu         = 0
    return args.rank == 0


# ── 데이터셋 ───────────────────────────────────────────────────────────────────

def build_loaders(args: argparse.Namespace):
    """ImageNet DataLoader 생성. 모델별 권장 mean/std/crop_pct 적용."""
    # 가중치 없이 config만 읽음 (모델마다 mean/std/crop_pct 가 다를 수 있음)
    _cfg_model = timm.create_model(args.model, pretrained=False)
    data_config = timm.data.resolve_model_data_config(_cfg_model)
    del _cfg_model

    train_transform = create_transform(
        input_size=data_config["input_size"],
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation=data_config["interpolation"],
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=data_config["mean"],
        std=data_config["std"],
    )
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    train_ds = datasets.ImageFolder(
        os.path.join(args.data_path, "train"), transform=train_transform
    )
    val_ds = datasets.ImageFolder(
        os.path.join(args.data_path, "val"), transform=val_transform
    )

    if args.distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_ds)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_ds)
    val_sampler = torch.utils.data.SequentialSampler(val_ds)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    return train_loader, val_loader, train_sampler


# ── LR 스케줄러 ────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, args: argparse.Namespace):
    """Warmup(Linear) + Cosine 조합 스케줄러."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=args.min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[args.warmup_epochs],
    )


# ── 체크포인트 ──────────────────────────────────────────────────────────────────

def save_checkpoint(
    path:         str,
    model:        nn.Module,
    model_ema,
    optimizer,
    lr_scheduler,
    scaler,
    pruner,
    epoch:        int,
    best_acc1:    float,
    args:         argparse.Namespace,
) -> None:
    raw = model.module if hasattr(model, "module") else model
    ckpt = {
        "model":        raw.state_dict(),
        "model_ema":    model_ema.module.state_dict() if model_ema else None,
        "optimizer":    optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler":       scaler.state_dict(),
        "pruner":       pruner.state_dict() if pruner else None,
        "epoch":        epoch,
        "best_acc1":    best_acc1,
        "args":         vars(args),
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, model, model_ema, optimizer, lr_scheduler, scaler, pruner):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model"])

    if model_ema is not None and ckpt.get("model_ema") is not None:
        model_ema.module.load_state_dict(ckpt["model_ema"])

    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    scaler.load_state_dict(ckpt["scaler"])

    if pruner is not None and ckpt.get("pruner") is not None:
        pruner.load_state_dict(ckpt["pruner"])

    return ckpt["epoch"] + 1, ckpt.get("best_acc1", 0.0)


# ── 메인 ───────────────────────────────────────────────────────────────────────

def main():
    args     = get_args()
    is_main  = setup_distributed(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )

    # ── WandB ──────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb and is_main:
        import wandb
        run_name = args.wandb_run_name or (
            f"{args.model}_prune{int(args.target_compression * 100)}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            id=args.wandb_run_id or None,
            resume="allow" if args.wandb_run_id else None,
            config=vars(args),
        )

    # ── 데이터 ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, train_sampler = build_loaders(args)

    # ── 모델 (pretrained ImageNet-1k) ──────────────────────────────────────────
    model = timm.create_model(
        args.model, pretrained=True, num_classes=args.num_classes
    )
    model = model.to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[Model] {args.model}  params={n_params:,}")

    # ── EMA ────────────────────────────────────────────────────────────────────
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay, device=device)

    # ── Pruner ─────────────────────────────────────────────────────────────────
    pruner = None
    if args.target_compression > 0:
        pruner = ViTPruner(
            model,
            target_compression=args.target_compression,
            max_sparsity=args.pruning_max_sparsity,
            index_refresh_steps=args.prune_refresh_steps,
        )

    # ── Optimizer & Scheduler ──────────────────────────────────────────────────
    optimizer    = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = build_scheduler(optimizer, args)
    criterion    = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    scaler       = torch.amp.GradScaler("cuda", enabled=args.amp)

    # ── DDP 래핑 ───────────────────────────────────────────────────────────────
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_acc1   = 0.0
    if args.resume:
        start_epoch, best_acc1 = load_checkpoint(
            args.resume, model, model_ema, optimizer, lr_scheduler, scaler, pruner
        )
        if is_main:
            print(f"[Resume] epoch={start_epoch}  best_acc1={best_acc1:.2f}%")

    # ── 학습 루프 ──────────────────────────────────────────────────────────────
    if is_main:
        print(
            f"\n=== Training: {args.model} ===\n"
            f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}\n"
            f"  compression={args.target_compression}  amp={args.amp}  "
            f"ema={args.model_ema}\n"
        )

    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if is_main:
            print(f"\n── Epoch {epoch}/{args.epochs - 1}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} ──")

        # 학습
        train_stats = train_one_epoch(
            model, criterion, train_loader,
            optimizer, scaler, device, epoch,
            model_ema=model_ema,
            pruner=pruner,
            amp=args.amp,
            clip_grad=args.clip_grad,
            log_interval=args.log_interval,
        )
        lr_scheduler.step()

        # 검증 — EMA 모델 우선
        eval_model = (
            model_ema.module if model_ema is not None
            else (model.module if hasattr(model, "module") else model)
        )
        val_stats = evaluate(val_loader, eval_model, device, amp=args.amp)

        acc1    = val_stats["acc1"]
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1

        # Pruner sparsity 지표
        sparsity_stats: dict = {}
        if pruner is not None and is_main:
            actual        = model.module if hasattr(model, "module") else model
            sparsity_stats = pruner.log_sparsity(actual)

        # WandB 로깅
        if wandb_run is not None and is_main:
            log_dict = {
                "epoch":         epoch,
                "train/loss":    train_stats["loss"],
                "train/top1":    train_stats["top1"],
                "train/lr":      optimizer.param_groups[0]["lr"],
                "val/loss":      val_stats["loss"],
                "val/top1":      acc1,
                "val/top5":      val_stats["acc5"],
                "val/top1_best": best_acc1,
            }
            log_dict.update(sparsity_stats)
            wandb_run.log(log_dict)

        if is_main:
            sp_str = (
                f"  sparsity={sparsity_stats.get('pruning/actual_sparsity', 0):.4f}"
                if sparsity_stats else ""
            )
            print(
                f"  val_top1={acc1:.2f}%  best={best_acc1:.2f}%"
                + sp_str
                + (" ← BEST" if is_best else "")
            )

        # 체크포인트 저장
        if is_main:
            save_checkpoint(
                os.path.join(args.output_dir, "checkpoint_last.pt"),
                model, model_ema, optimizer, lr_scheduler, scaler,
                pruner, epoch, best_acc1, args,
            )
            if is_best:
                save_checkpoint(
                    os.path.join(args.output_dir, "checkpoint_best.pt"),
                    model, model_ema, optimizer, lr_scheduler, scaler,
                    pruner, epoch, best_acc1, args,
                )

    if is_main:
        print(f"\n=== Done. Best val top-1: {best_acc1:.2f}% ===")

    if wandb_run is not None:
        wandb_run.finish()

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
