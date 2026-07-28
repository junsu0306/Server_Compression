"""
Cropr Token Pruning Fine-tuning

FFN pruning → reduce 이후의 reduced.pt 위에 Cropr 토큰 pruning을 추가 학습.

파이프라인:
    [Phase 1] train.py         → checkpoint_best.pt  (FFN channel pruning)
    [Phase 2] reduce.py        → reduced.pt           (Dense 변환)
    [Phase 3] train_cropr.py   → cropr_best.pt        (Token pruning 추가)

단일 GPU:
    python train_cropr.py \\
        --reduced ./output/vit_tiny_prune50_progressive_taylor/reduced.pt \\
        --data-path /data/imagenet \\
        --epochs 30

멀티 GPU (DDP):
    CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 train_cropr.py \\
        --config configs/vit_tiny_prune50_cropr.yaml
"""

from __future__ import annotations

import os
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import timm
from timm.utils import ModelEmaV2
import timm.data
from torchvision import datasets

from pruning.vit_reducing import apply_reduced_config
from cropr import CroprWrapper


# ── argparse ───────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Cropr Token Pruning Fine-tuning")

    p.add_argument("--config", default="")

    # 입력 모델
    p.add_argument("--reduced", default="",
                   help="reduced.pt 경로 (FFN pruning + reduce 완료 모델)")

    # 데이터
    p.add_argument("--data-path",   default="")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--pin-mem",     action="store_true", default=True)
    p.add_argument("--num-classes", type=int, default=1000)

    # 학습 하이퍼파라미터
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--clip-grad",    type=float, default=1.0)
    p.add_argument("--warmup-epochs",type=int,   default=3)
    p.add_argument("--min-lr",       type=float, default=1e-6)
    p.add_argument("--smoothing",    type=float, default=0.1)
    p.add_argument("--amp",          action="store_true", default=True)

    # Backbone freeze
    p.add_argument("--freeze-backbone", action="store_true", default=False,
                   help="Cropr 모듈만 학습 (backbone frozen). 초반 안정화에 유용.")
    p.add_argument("--freeze-epochs", type=int, default=0,
                   help="이 epoch까지 backbone frozen, 이후 전체 학습 (0=처음부터 전체 학습)")

    # EMA
    p.add_argument("--model-ema",       action="store_true", default=True)
    p.add_argument("--model-ema-decay", type=float,          default=0.9998)

    # Cropr 설정
    p.add_argument("--cropr-locs",      type=int, nargs="+", default=[3, 6, 9],
                   help="Cropr 모듈을 삽입할 블록 인덱스 (기본: 3 6 9)")
    p.add_argument("--cropr-rate",      type=int, default=32,
                   help="각 Cropr 모듈에서 제거할 토큰 수 (기본: 32)")
    p.add_argument("--cropr-heads",     type=int, default=None,
                   help="Cropr cross-attention 헤드 수 (None=모델과 동일)")
    p.add_argument("--cropr-llf",       action="store_true", default=False,
                   help="Last Layer Fusion: 마지막 블록 전 제거 토큰 재결합")

    # 출력 / 체크포인트
    p.add_argument("--output-dir",   default="./output/cropr")
    p.add_argument("--resume",       default="")
    p.add_argument("--log-interval", type=int, default=50)

    # WandB
    p.add_argument("--wandb",          action="store_true")
    p.add_argument("--wandb-project",  default="vit-pruning")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--wandb-run-id",   default="")

    # DDP
    p.add_argument("--dist-url", default="env://")

    # YAML override
    pre, _ = p.parse_known_args()
    if pre.config:
        with open(pre.config) as f:
            yaml_cfg = yaml.safe_load(f)
        p.set_defaults(**{k.replace("-", "_"): v for k, v in yaml_cfg.items()})

    args = p.parse_args()
    if not args.data_path:
        p.error("--data-path 를 지정해야 합니다.")
    if not args.reduced:
        p.error("--reduced (reduced.pt 경로) 를 지정해야 합니다.")
    return args


# ── 분산 학습 ───────────────────────────────────────────────────────────────────

def setup_distributed(args) -> bool:
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

def build_loaders(args, model_name: str):
    _cfg_model  = timm.create_model(model_name, pretrained=False)
    data_config = timm.data.resolve_model_data_config(_cfg_model)
    del _cfg_model

    train_transform = timm.data.create_transform(
        **data_config, is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        re_prob=0.25, re_mode="pixel", re_count=1,
    )
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    train_ds = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_path, "val"),   transform=val_transform)

    train_sampler = (
        torch.utils.data.DistributedSampler(train_ds)
        if args.distributed else
        torch.utils.data.RandomSampler(train_ds)
    )
    val_sampler = torch.utils.data.SequentialSampler(val_ds)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_mem,
    )
    return train_loader, val_loader, train_sampler


# ── LR 스케줄러 ────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, args):
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=args.min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs],
    )


# ── 체크포인트 ──────────────────────────────────────────────────────────────────

def save_checkpoint(path, model, model_ema, optimizer, lr_scheduler, scaler, epoch, best_acc1, args):
    raw = model.module if hasattr(model, "module") else model
    torch.save({
        "model":        raw.state_dict(),
        "model_ema":    model_ema.module.state_dict() if model_ema else None,
        "optimizer":    optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler":       scaler.state_dict(),
        "epoch":        epoch,
        "best_acc1":    best_acc1,
        "args":         vars(args),
    }, path)


def load_checkpoint(path, model, model_ema, optimizer, lr_scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw  = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model"])
    if model_ema and ckpt.get("model_ema"):
        model_ema.module.load_state_dict(ckpt["model_ema"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"] + 1, ckpt.get("best_acc1", 0.0)


# ── Accuracy ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    bs   = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum() * 100.0 / bs for k in topk]


# ── 학습 루프 ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, loader, optimizer, scaler, device, epoch, args,
                    model_ema=None):
    model.train()

    loss_sum, top1_sum, top5_sum, n = 0.0, 0.0, 0.0, 0
    import time; t0 = time.time()

    for step, (samples, targets) in enumerate(loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.amp):
            output = model(samples)

            # multi-head loss: main + auxiliary heads
            if isinstance(output, list):
                losses = [criterion(o, targets) for o in output]
                loss   = sum(losses)
            else:
                loss = criterion(output, targets)
                output = [output]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.clip_grad:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        scaler.step(optimizer)
        scaler.update()

        if model_ema is not None:
            actual = model.module if hasattr(model, "module") else model
            model_ema.update(actual)

        with torch.no_grad():
            acc1, acc5 = accuracy(output[0].detach().float(), targets)
        bs = samples.size(0)
        loss_sum += loss.item() * bs
        top1_sum += acc1.item() * bs
        top5_sum += acc5.item() * bs
        n        += bs

        if step % args.log_interval == 0:
            print(
                f"  Epoch[{epoch}] [{step:>4d}/{len(loader)}]  "
                f"loss={loss_sum/n:.4f}  top1={top1_sum/n:.2f}%  "
                f"t={time.time()-t0:.0f}s"
            )

    return {"loss": loss_sum / n, "top1": top1_sum / n}


@torch.no_grad()
def evaluate(loader, model, device, amp=True):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum, top1_sum, top5_sum, n = 0.0, 0.0, 0.0, 0

    for samples, targets in loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp):
            output = model(samples)
            if isinstance(output, list):
                output = output[0]
            loss = criterion(output, targets)

        acc1, acc5 = accuracy(output.float(), targets)
        bs = samples.size(0)
        loss_sum += loss.item() * bs
        top1_sum += acc1.item() * bs
        top5_sum += acc5.item() * bs
        n        += bs

    print(f"  [Val] loss={loss_sum/n:.4f}  top1={top1_sum/n:.2f}%  top5={top5_sum/n:.2f}%")
    return {"loss": loss_sum / n, "acc1": top1_sum / n, "acc5": top5_sum / n}


# ── 메인 ───────────────────────────────────────────────────────────────────────

def main():
    args    = get_args()
    is_main = setup_distributed(args)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ── WandB ──────────────────────────────────────────────────────────────────
    wandb_run = None
    if args.wandb and is_main:
        import wandb
        run_name = args.wandb_run_name or "cropr"
        wandb_run = wandb.init(
            project=args.wandb_project, name=run_name,
            id=args.wandb_run_id or None,
            resume="allow" if args.wandb_run_id else None,
            config=vars(args),
        )

    # ── Reduced 모델 로드 ───────────────────────────────────────────────────────
    ckpt       = torch.load(args.reduced, map_location="cpu", weights_only=False)
    model_name = ckpt["model_name"]

    base_model = timm.create_model(model_name, pretrained=False, num_classes=args.num_classes)
    apply_reduced_config(base_model, ckpt["mlp_dims"])
    base_model.load_state_dict(ckpt["state_dict"])
    base_model = base_model.to(device)

    if is_main:
        n_base = sum(p.numel() for p in base_model.parameters())
        print(f"\n[Base] {model_name}  params={n_base:,}  (reduced FFN)")

    # ── CroprWrapper ────────────────────────────────────────────────────────────
    model = CroprWrapper(
        model       = base_model,
        pruning_locs= args.cropr_locs,
        pruning_rate= args.cropr_rate,
        num_heads   = args.cropr_heads,
        num_classes = args.num_classes,
        llf         = args.cropr_llf,
    ).to(device)

    if is_main:
        n_total  = sum(p.numel() for p in model.parameters())
        n_cropr  = sum(p.numel() for p in model.cropr.parameters())
        print(f"[CroprWrapper] total={n_total:,}  cropr_modules={n_cropr:,} (+{n_cropr/n_base*100:.1f}%)")

    # ── 데이터 ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, train_sampler = build_loaders(args, model_name)

    # ── EMA ────────────────────────────────────────────────────────────────────
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay, device=device)

    # ── Optimizer ──────────────────────────────────────────────────────────────
    # Backbone frozen 여부에 따라 파라미터 그룹 분리
    if args.freeze_backbone:
        for p in model.base_model.parameters():
            p.requires_grad_(False)
        optim_params = model.cropr.parameters()
        if is_main:
            print("[CroprWrapper] backbone frozen — Cropr 모듈만 학습")
    else:
        optim_params = model.parameters()

    optimizer    = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = build_scheduler(optimizer, args)
    criterion    = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    scaler       = torch.amp.GradScaler("cuda", enabled=args.amp)

    # ── DDP ────────────────────────────────────────────────────────────────────
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch, best_acc1 = 0, 0.0
    if args.resume:
        start_epoch, best_acc1 = load_checkpoint(
            args.resume, model, model_ema, optimizer, lr_scheduler, scaler
        )
        if is_main:
            print(f"[Resume] epoch={start_epoch}  best_acc1={best_acc1:.2f}%")

    if is_main:
        print(
            f"\n=== Cropr Training: {model_name} ===\n"
            f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}\n"
            f"  cropr_locs={args.cropr_locs}  cropr_rate={args.cropr_rate}  llf={args.cropr_llf}\n"
            f"  freeze_backbone={args.freeze_backbone}  freeze_epochs={args.freeze_epochs}\n"
        )

    # ── 학습 루프 ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # freeze_epochs 이후 backbone unfreeze
        actual = model.module if hasattr(model, "module") else model
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs and is_main:
            for p in actual.base_model.parameters():
                p.requires_grad_(True)
            # optimizer 재설정 (새 파라미터 그룹 추가)
            optimizer = torch.optim.AdamW(
                actual.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay
            )
            lr_scheduler = build_scheduler(optimizer, args)
            print(f"[CroprWrapper] epoch={epoch}: backbone unfrozen, lr={args.lr * 0.1:.2e}")

        if is_main:
            print(f"\n── Epoch {epoch}/{args.epochs - 1}  lr={optimizer.param_groups[0]['lr']:.2e} ──")

        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, device, epoch, args, model_ema
        )
        lr_scheduler.step()

        eval_model = (
            model_ema.module if model_ema is not None
            else (model.module if hasattr(model, "module") else model)
        )
        val_stats = evaluate(val_loader, eval_model, device, amp=args.amp)

        acc1    = val_stats["acc1"]
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1

        if wandb_run is not None and is_main:
            wandb_run.log({
                "epoch":         epoch,
                "train/loss":    train_stats["loss"],
                "train/top1":    train_stats["top1"],
                "train/lr":      optimizer.param_groups[0]["lr"],
                "val/loss":      val_stats["loss"],
                "val/top1":      acc1,
                "val/top5":      val_stats["acc5"],
                "val/top1_best": best_acc1,
            })

        if is_main:
            print(f"  val_top1={acc1:.2f}%  best={best_acc1:.2f}%" + (" ← BEST" if is_best else ""))
            save_checkpoint(
                os.path.join(args.output_dir, "cropr_last.pt"),
                model, model_ema, optimizer, lr_scheduler, scaler, epoch, best_acc1, args,
            )
            if is_best:
                save_checkpoint(
                    os.path.join(args.output_dir, "cropr_best.pt"),
                    model, model_ema, optimizer, lr_scheduler, scaler, epoch, best_acc1, args,
                )

    if is_main:
        print(f"\n=== Done. Best val top-1: {best_acc1:.2f}% ===")

    if wandb_run:
        wandb_run.finish()
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
