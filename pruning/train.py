"""Global L2 FFN pruning with logit KD and an isolated target-peak memory budget.

Use scripts/run_pipeline.py with configs/final.yaml to prepare the budget first.
The measured memory excess is a scalar monitoring term, not a differentiable
memory controller. The student starts from the saved pretrained baseline.
"""

from __future__ import annotations

import os
import argparse
import math
import random
import json
import numpy as np
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.distributed as dist
import timm
from timm.utils import ModelEmaV2
from timm.data import create_transform
import timm.data
from torchvision import datasets

from pruning.vit_pruning import ViTPruner
from pruning.engine import train_one_epoch, evaluate
from pruning.memory_budget import MemoryBudget


# ── argparse ───────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ViT Soft Pruning Fine-tuning")

    # Config 파일 (YAML) — 지정 시 파일 값이 기본값으로 적용되고, CLI 인자로 override 가능
    p.add_argument("--config", default="",
                   help="run_pipeline.py가 생성한 configs/structural.yaml")

    # 데이터
    p.add_argument("--data-path", default="")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--pin-mem",     action="store_true", default=True)

    # 모델
    p.add_argument("--model",       default="vit_tiny_patch16_224")
    p.add_argument("--input-size",  type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000)

    # 학습 하이퍼파라미터
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--lr-schedule-epochs", type=int, default=50,
                   help="LR horizon, independent of the terminal training epoch")
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=5e-5)
    p.add_argument("--weight-decay",   type=float, default=0.05)
    p.add_argument("--clip-grad",      type=float, default=1.0)
    p.add_argument("--warmup-epochs",  type=int,   default=5)
    p.add_argument("--min-lr",         type=float, default=1e-6)
    p.add_argument("--smoothing",      type=float, default=0.1)
    p.add_argument("--amp",            action="store_true", default=True)

    # EMA
    p.add_argument("--model-ema",       action="store_true", default=True)
    p.add_argument("--model-ema-decay", type=float,          default=0.9998)

    # Soft Pruning
    p.add_argument("--target-compression",   type=float, default=0.0)
    p.add_argument("--pruning-max-sparsity", type=float, default=0.95)
    p.add_argument("--prune-refresh-steps",  type=int,   default=100)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--initial-checkpoint", required=False, default="",
                   help="baseline.pt saved before target budget preparation")

    # Knowledge Distillation
    p.add_argument("--kd-alpha",       type=float, default=0.0,
                   help="KD loss 가중치 (0=비활성, 0.5 권장). loss=(1-α)·CE + α·KL")
    p.add_argument("--kd-temperature", type=float, default=4.0,
                   help="KD softening temperature (권장: 3~5)")
    p.add_argument("--kd-teacher",     default="",
                   help="Teacher 모델 이름. 비어있으면 student와 동일 아키텍처 사용")

    # Measured memory excess (scalar penalty, MiB).
    p.add_argument("--memory-penalty-weight", type=float, default=0.0,
                   help="lambda_mem; 0 disables memory profiling/penalty")
    p.add_argument("--memory-budget-json", default="",
                   help="immutable budget from an un-fine-tuned target structure")
    p.add_argument("--memory-profile-interval", type=int, default=1, help="profiling interval in epochs")
    p.add_argument("--memory-batch-size", type=int, default=1)
    p.add_argument("--memory-warmups", type=int, default=10)
    p.add_argument("--memory-runs", type=int, default=50)

    # 출력 / 체크포인트
    p.add_argument("--output-dir",   default="./output")
    p.add_argument("--resume",       default="")
    p.add_argument("--log-interval", type=int, default=50)

    # WandB
    p.add_argument("--wandb",          action="store_true")
    p.add_argument("--wandb-project",  default="vit-pruning")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--wandb-run-id",   default="")

    # DDP
    p.add_argument("--dist-url", default="env://")

    # ── YAML config 로드 후 기본값 덮어쓰기 ──────────────────────────────────
    # 1차 파싱으로 --config 경로만 추출
    pre, _ = p.parse_known_args()
    if pre.config:
        with open(pre.config) as f:
            yaml_cfg = yaml.safe_load(f)
        # YAML 키(snake_case) → argparse dest(snake_case) 변환 후 set_defaults
        p.set_defaults(**{k.replace("-", "_"): v for k, v in yaml_cfg.items()})

    args = p.parse_args()

    if not args.data_path:
        p.error("--data-path 또는 config 내 data_path 를 지정해야 합니다.")

    if not math.isfinite(args.memory_penalty_weight) or args.memory_penalty_weight < 0:
        p.error("--memory-penalty-weight must be finite and nonnegative")
    if args.memory_penalty_weight > 0:
        if args.target_compression <= 0:
            p.error("memory penalty requires --target-compression > 0")
        if not Path(args.memory_budget_json).is_file():
            p.error("--memory-budget-json is required; run the budget preparation stage first")
        if not Path(args.initial_checkpoint).is_file():
            p.error("--initial-checkpoint baseline.pt is required")
        if min(args.memory_profile_interval, args.memory_batch_size, args.memory_runs) < 1 or args.memory_warmups < 0:
            p.error("memory interval/batch/runs must be positive; warmups must be nonnegative")

    return args


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
        T_max=max(args.lr_schedule_epochs - args.warmup_epochs, 1),
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
    memory_budget=None,
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
        "memory_budget": memory_budget.state_dict() if memory_budget is not None else None,
    }
    temporary = Path(path).with_suffix(".tmp")
    torch.save(ckpt, temporary)
    temporary.replace(path)


def load_checkpoint(path: str, model, model_ema, optimizer, lr_scheduler, scaler, pruner, memory_budget=None):
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
    if memory_budget is not None and ckpt.get("memory_budget") is not None:
        memory_budget.load_state_dict(ckpt["memory_budget"])

    return ckpt["epoch"] + 1, ckpt.get("best_acc1", 0.0)


def validate_resume_schedule(checkpoint, args):
    """Never silently replace an already-trained learning-rate trajectory."""
    previous = checkpoint["args"]
    horizon = previous.get("lr_schedule_epochs", previous["epochs"])
    if horizon != args.lr_schedule_epochs:
        raise ValueError(f"Stored LR horizon {horizon} differs from requested {args.lr_schedule_epochs}")
    if checkpoint["epoch"] + 1 > args.epochs:
        raise ValueError("Checkpoint is beyond the requested terminal epoch; an exact earlier checkpoint is required")


# ── 메인 ───────────────────────────────────────────────────────────────────────

def main():
    args     = get_args()
    is_main  = setup_distributed(args)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    if args.memory_penalty_weight > 0 and device.type != "cuda":
        raise ValueError("Memory penalty profiling requires a CUDA device")
    memory_budget = MemoryBudget(args, device) if args.memory_penalty_weight > 0 else None

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
            config=None if args.resume else vars(args),
        )
        wandb_run.name = run_name
        wandb_run.config.update(vars(args), allow_val_change=True)

    # ── 데이터 ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, train_sampler = build_loaders(args)
    if is_main:
        data_info = {"train_samples": len(train_loader.dataset), "val_samples": len(val_loader.dataset),
                     "batch_size": args.batch_size, "seed": args.seed,
                     "source_baseline": args.initial_checkpoint, "memory_budget_json": args.memory_budget_json}
        (Path(args.output_dir) / "run_metadata.json").write_text(json.dumps(data_info, indent=2) + "\n")
        print(f"[Data] {data_info}")

    # ── 모델 (pretrained ImageNet-1k) ──────────────────────────────────────────
    model = timm.create_model(
        args.model, pretrained=not bool(args.initial_checkpoint), num_classes=args.num_classes
    )
    initial_state = None
    if args.initial_checkpoint:
        initial = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        if initial["model_name"] != args.model or initial["trained_steps"] != 0:
            raise ValueError("Initial checkpoint must be the budget baseline")
        initial_state = initial["state_dict"]
        model.load_state_dict(initial_state, strict=True)
    model = model.to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[Model] {args.model}  params={n_params:,}")

    # ── EMA ────────────────────────────────────────────────────────────────────
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay, device=device)

    # ── Teacher (Knowledge Distillation) ──────────────────────────────────────
    teacher = None
    if args.kd_alpha > 0:
        teacher_name = args.kd_teacher if args.kd_teacher else args.model
        teacher = timm.create_model(
            teacher_name, pretrained=not bool(initial_state), num_classes=args.num_classes
        ).to(device)
        if initial_state is not None:
            if teacher_name != args.model:
                raise ValueError("The final KD teacher is the same baseline architecture")
            teacher.load_state_dict(initial_state, strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # normalization 불일치 경고
        if args.kd_teacher and args.kd_teacher != args.model and is_main:
            student_cfg = timm.data.resolve_model_data_config(
                timm.create_model(args.model, pretrained=False)
            )
            teacher_cfg = timm.data.resolve_model_data_config(teacher)
            if student_cfg["mean"] != teacher_cfg["mean"]:
                print(
                    f"[KD] ⚠ normalization 불일치!\n"
                    f"     student mean={student_cfg['mean']}  std={student_cfg['std']}\n"
                    f"     teacher mean={teacher_cfg['mean']}  std={teacher_cfg['std']}\n"
                    f"     DataLoader는 student 기준으로 생성됩니다. "
                    f"teacher 성능이 저하될 수 있습니다."
                )
            else:
                print(f"[KD] normalization 일치 ✓  mean={student_cfg['mean']}")

        if is_main:
            n_teacher = sum(p.numel() for p in teacher.parameters())
            print(
                f"[KD] teacher={teacher_name}  params={n_teacher:,}  "
                f"alpha={args.kd_alpha}  T={args.kd_temperature}"
            )

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
        resume_meta = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_resume_schedule(resume_meta, args)
        del resume_meta
        start_epoch, best_acc1 = load_checkpoint(
            args.resume, model, model_ema, optimizer, lr_scheduler, scaler, pruner, memory_budget
        )
        if is_main:
            print(f"[Resume] epoch={start_epoch}  best_acc1={best_acc1:.2f}%")
    elif pruner is not None:
        raw = model.module if hasattr(model, "module") else model
        pruner.apply(raw)
        if model_ema is not None:
            from pruning.vit_reducing import transfer_pruning_mask
            transfer_pruning_mask(raw, model_ema.module)
        if memory_budget is not None:
            memory_budget.maybe_refresh(model, model_ema, 0)

    # ── 학습 루프 ──────────────────────────────────────────────────────────────
    if is_main:
        print(
            f"\n=== Training: {args.model} ===\n"
            f"  terminal_epochs={args.epochs}  lr_schedule_epochs={args.lr_schedule_epochs}  "
            f"batch={args.batch_size}  lr={args.lr}\n"
            f"  compression={args.target_compression}  amp={args.amp}  "
            f"ema={args.model_ema}\n"
            f"  kd={'ON (α=' + str(args.kd_alpha) + ', T=' + str(args.kd_temperature) + ')' if args.kd_alpha > 0 else 'OFF'}\n"
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
            teacher=teacher,
            kd_alpha=args.kd_alpha,
            kd_temperature=args.kd_temperature,
            memory_budget=memory_budget,
        )
        lr_scheduler.step()

        # 검증 — EMA 모델 우선
        eval_model = (
            model_ema.module if model_ema is not None
            else (model.module if hasattr(model, "module") else model)
        )
        val_stats = evaluate(val_loader, eval_model, device, amp=args.amp)

        acc1    = val_stats["acc1"]
        if is_main:
            with (Path(args.output_dir) / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"epoch": epoch, "train": train_stats, "val": val_stats,
                                         "memory": memory_budget.metrics() if memory_budget else {}}) + "\n")
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
            import wandb as _wandb
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
            if memory_budget is not None:
                log_dict.update(memory_budget.metrics())
                log_dict["train/task_kd_loss"] = train_stats["task_kd_loss"]
                log_dict["train/memory_penalty"] = train_stats["memory_loss"]

            # 블록별 sparsity 한 번에 보기 (bar chart)
            # pruning/layer/blocks/N/mlp 키만 사용 (ratio 0~1)
            # pruning/survived/blocks/N/mlp (절대값) 는 제외
            if sparsity_stats:
                layer_rows = sorted(
                    ([k.split("/blocks/")[1].split("/")[0], v]
                     for k, v in sparsity_stats.items()
                     if "/layer/blocks/" in k),
                    key=lambda x: int(x[0])
                )
                if layer_rows:
                    tbl = _wandb.Table(
                        data=[[f"block {r[0]}", r[1]] for r in layer_rows],
                        columns=["block", "sparsity"],
                    )
                    log_dict["pruning/layer_sparsity"] = _wandb.plot.bar(
                        tbl, "block", "sparsity",
                        title="FFN Sparsity per Block"
                    )

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
                memory_budget=memory_budget,
            )
            if is_best:
                save_checkpoint(
                    os.path.join(args.output_dir, "checkpoint_best.pt"),
                    model, model_ema, optimizer, lr_scheduler, scaler,
                    pruner, epoch, best_acc1, args,
                    memory_budget=memory_budget,
                )

    if is_main:
        print(f"\n=== Done. Best val top-1: {best_acc1:.2f}% ===")

    if wandb_run is not None:
        wandb_run.finish()

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
