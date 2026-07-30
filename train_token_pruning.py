"""
Stage 2 — reduce.py로 물리적으로 축소된 Dense 모델 위에 EViT Token Pruning fine-tuning.

전제: Stage 1(soft channel pruning) → reduce.py 완료 후 생성된 reduced.pt가 있어야 한다.
      이 스크립트는 그 reduced.pt를 시작점으로 로드하고, 일부 block에 EViT token
      pruning을 얹어 fine-tuning한다. FFN 채널 구조(mlp_dims)는 이 단계에서
      변하지 않는다 — 시퀀스 차원(토큰 개수)만 forward 중 동적으로 줄어든다.

단일 GPU:
    python train_token_pruning.py \\
        --reduced ./output/vit_tiny_prune50_progressive_taylor/reduced.pt \\
        --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \\
        --epochs 30 \\
        --base-keep-rate 0.7 \\
        --output-dir ./output/vit_tiny_prune50_token70 \\
        --wandb

멀티 GPU (DDP):
    torchrun --nproc_per_node=2 train_token_pruning.py \\
        --config configs/vit_tiny_token_prune70.yaml

체크포인트 재개:
    python train_token_pruning.py --config ... \\
        --resume ./output/vit_tiny_prune50_token70/checkpoint_last.pt

결과물:
    checkpoint_last.pt / checkpoint_best.pt — optimizer state 포함, 재개용
    token_pruned_best.pt                    — eval_token_pruned.py / export_onnx.py 입력
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
from timm.data import create_transform
import timm.data
from torchvision import datasets

from pruning.vit_reducing import apply_reduced_config
from pruning.token_pruning import EvitTokenPruner
from engine import train_one_epoch, evaluate


# ── argparse ───────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("EViT Token Pruning Fine-tuning (Stage 2)")

    p.add_argument("--config", default="",
                   help="YAML config 파일 경로 (예: configs/vit_tiny_token_prune70.yaml)")

    # 입력 — Stage 1 산출물. --resume 시에도 항상 필요하다 (model_name/mlp_dims
    # 구조 재현용 — train.py가 resume 시에도 --model을 항상 요구하는 것과 동일한 이유).
    p.add_argument("--reduced", default="", help="reduce.py 출력 (reduced.pt) 경로")

    # 데이터
    p.add_argument("--data-path", default="")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--pin-mem",     action="store_true", default=True)

    p.add_argument("--input-size",  type=int, default=224)
    p.add_argument("--num-classes", type=int, default=1000)

    # 학습 하이퍼파라미터 — Stage 1(train.py)보다 짧게, 더 작은 lr 권장
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=2e-5)
    p.add_argument("--weight-decay",   type=float, default=0.05)
    p.add_argument("--clip-grad",      type=float, default=1.0)
    p.add_argument("--warmup-epochs",  type=int,   default=3)
    p.add_argument("--min-lr",         type=float, default=1e-6)
    p.add_argument("--smoothing",      type=float, default=0.1)
    p.add_argument("--amp",            action="store_true", default=True)

    # EMA
    p.add_argument("--model-ema",       action="store_true", default=True)
    p.add_argument("--model-ema-decay", type=float,          default=0.9998)

    # EViT Token Pruning
    p.add_argument("--base-keep-rate",       type=float, default=0.7,
                   help="목표 keep_rate. patch 토큰 중 유지할 비율 (0.5~0.9 권장)")
    p.add_argument("--prune-layers",         default="",
                   help="쉼표구분 block 인덱스 (예: '3,6,9'). 비어있으면 자동(depth//4 등분)")
    p.add_argument("--fuse-token",           action="store_true", default=True)
    p.add_argument("--no-fuse-token",        dest="fuse_token", action="store_false",
                   help="버려지는 토큰을 fused token으로 합치지 않고 그냥 버림")
    p.add_argument("--keep-rate-warmup-epochs", type=int, default=0,
                   help="token pruning 시작 전 유예 epoch 수 (0=즉시 적용)")
    p.add_argument("--keep-rate-ramp-epochs",   type=int, default=0,
                   help="keep_rate 1.0→target 점진 감소 epoch 수 (0=즉시 target)")

    # Knowledge Distillation — teacher 선택
    p.add_argument("--kd-alpha",       type=float, default=0.5,
                   help="KD loss 가중치 (0=비활성)")
    p.add_argument("--kd-temperature", type=float, default=4.0)
    p.add_argument("--kd-teacher-mode", default="reduced",
                   choices=["reduced", "original", "none"],
                   help="reduced=token pruning 적용 전 reduced 모델 자체(self-distillation, 권장) | "
                        "original=원본 pretrained dense 모델 | none=KD 비활성")
    p.add_argument("--kd-teacher-model", default="",
                   help="kd_teacher_mode=original일 때 사용할 timm 모델명. "
                        "비어있으면 reduced.pt에 저장된 model_name 사용")

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

    pre, _ = p.parse_known_args()
    if pre.config:
        with open(pre.config) as f:
            yaml_cfg = yaml.safe_load(f)
        p.set_defaults(**{k.replace("-", "_"): v for k, v in yaml_cfg.items()})

    args = p.parse_args()

    if not args.data_path:
        p.error("--data-path 또는 config 내 data_path 를 지정해야 합니다.")
    if not args.reduced:
        p.error(
            "--reduced (Stage 1 reduce.py 출력) 를 지정해야 합니다. "
            "--resume 시에도 모델 구조(model_name/mlp_dims) 재현을 위해 항상 필요합니다."
        )

    return args


# ── 분산 학습 설정 ──────────────────────────────────────────────────────────────

def setup_distributed(args: argparse.Namespace) -> bool:
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


# ── 모델 로드 ──────────────────────────────────────────────────────────────────

def load_reduced_base(reduced_path: str) -> tuple[nn.Module, dict]:
    """reduce.py 출력을 token pruning 없는 순수 Dense 모델로 로드."""
    ckpt = torch.load(reduced_path, map_location="cpu", weights_only=False)
    model = timm.create_model(ckpt["model_name"], pretrained=False)
    apply_reduced_config(model, ckpt["mlp_dims"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model, ckpt


# ── 데이터셋 ───────────────────────────────────────────────────────────────────

def build_loaders(args: argparse.Namespace, model_name: str):
    _cfg_model = timm.create_model(model_name, pretrained=False)
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

    train_ds = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(args.data_path, "val"),   transform=val_transform)

    if args.distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_ds)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_ds)
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

def build_scheduler(optimizer, args: argparse.Namespace):
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

def save_checkpoint(path, model, model_ema, optimizer, lr_scheduler, scaler,
                     token_pruner, epoch, best_acc1, args) -> None:
    raw = model.module if hasattr(model, "module") else model
    ckpt = {
        "model":         raw.state_dict(),
        "model_ema":     model_ema.module.state_dict() if model_ema else None,
        "optimizer":     optimizer.state_dict(),
        "lr_scheduler":  lr_scheduler.state_dict(),
        "scaler":        scaler.state_dict(),
        "token_pruner":  token_pruner.state_dict(),
        "epoch":         epoch,
        "best_acc1":     best_acc1,
        "args":          vars(args),
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, model_ema, optimizer, lr_scheduler, scaler, token_pruner):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(ckpt["model"])

    if model_ema is not None and ckpt.get("model_ema") is not None:
        model_ema.module.load_state_dict(ckpt["model_ema"])

    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    scaler.load_state_dict(ckpt["scaler"])

    if ckpt.get("token_pruner") is not None:
        token_pruner.load_state_dict(ckpt["token_pruner"])

    return ckpt["epoch"] + 1, ckpt.get("best_acc1", 0.0)


def save_token_pruned_artifact(path, model, model_name, mlp_dims, token_pruner,
                                n_params_before, n_params_after) -> None:
    """eval_token_pruned.py / export_onnx.py가 바로 로드할 수 있는 배포용 아티팩트.

    reduce.py의 reduced.pt와 같은 성격 — 여기서는 물리적 구조 변화가 없으므로
    (mlp_dims 그대로) state_dict + token pruning 설정만 추가로 저장한다.
    """
    torch.save(
        {
            "state_dict":        model.state_dict(),
            "model_name":        model_name,
            "mlp_dims":          mlp_dims,
            "token_pruning":     token_pruner.config(),
            "n_params_before":   n_params_before,
            "n_params_after":    n_params_after,
        },
        path,
    )


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
        run_name = args.wandb_run_name or "token_pruning_stage2"
        wandb_run = wandb.init(
            project=args.wandb_project, name=run_name,
            id=args.wandb_run_id or None,
            resume="allow" if args.wandb_run_id else None,
            config=vars(args),
        )

    # ── Stage 1 reduced 모델 로드 ─────────────────────────────────────────────
    reduced_path = args.reduced
    model, reduced_ckpt = load_reduced_base(reduced_path)
    model_name = reduced_ckpt["model_name"]
    mlp_dims   = reduced_ckpt["mlp_dims"]
    n_params_before_total = reduced_ckpt.get("n_params_before", 0)
    model = model.to(device)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[Model] base={model_name}  (reduced from {reduced_path})  params={n_params:,}")

    # ── 데이터 ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, train_sampler = build_loaders(args, model_name)

    # ── EMA ────────────────────────────────────────────────────────────────────
    # 주의: 이 시점의 model은 아직 token pruning이 patch되기 전이다.
    # ModelEmaV2는 deepcopy로 shadow module을 만드는데, block.forward patch나
    # _evit_keep_rate 속성은 deepcopy 신뢰성이 보장되지 않으므로, token_pruner
    # 생성 후 token_pruner.attach_mirror(model_ema.module)로 명시적으로
    # 재적용하고 동기화 대상으로 등록한다 (아래 참고).
    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay, device=device)

    # ── Teacher (Knowledge Distillation) ──────────────────────────────────────
    teacher = None
    if args.kd_alpha > 0 and args.kd_teacher_mode != "none":
        if args.kd_teacher_mode == "reduced":
            # self-distillation: token pruning 적용 전의 동일 reduced 모델(전체 토큰 사용)
            teacher, _ = load_reduced_base(reduced_path)
            teacher = teacher.to(device)
            teacher_desc = f"reduced base ({reduced_path}, full tokens)"
        else:  # "original"
            teacher_name = args.kd_teacher_model or model_name
            teacher = timm.create_model(teacher_name, pretrained=True, num_classes=args.num_classes).to(device)
            teacher_desc = f"original pretrained ({teacher_name})"
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        if is_main:
            n_teacher = sum(p.numel() for p in teacher.parameters())
            print(
                f"[KD] teacher={teacher_desc}  params={n_teacher:,}  "
                f"alpha={args.kd_alpha}  T={args.kd_temperature}"
            )

    # ── Token Pruner ───────────────────────────────────────────────────────────
    prune_layers = None
    if args.prune_layers:
        prune_layers = [int(x) for x in args.prune_layers.split(",") if x.strip() != ""]

    token_pruner = EvitTokenPruner(
        model,
        base_keep_rate=args.base_keep_rate,
        fuse_token=args.fuse_token,
        prune_layers=prune_layers,
        warmup_epochs=args.keep_rate_warmup_epochs,
        ramp_epochs=args.keep_rate_ramp_epochs,
    )
    if model_ema is not None:
        token_pruner.attach_mirror(model_ema.module)

    # ── Optimizer & Scheduler ──────────────────────────────────────────────────
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
            args.resume, model, model_ema, optimizer, lr_scheduler, scaler, token_pruner
        )
        if is_main:
            print(f"[Resume] epoch={start_epoch}  best_acc1={best_acc1:.2f}%")

    if is_main:
        print(
            f"\n=== Stage 2: Token Pruning Fine-tuning ({model_name}) ===\n"
            f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}\n"
            f"  base_keep_rate={args.base_keep_rate}  prune_layers={token_pruner.prune_layers}  "
            f"fuse_token={args.fuse_token}\n"
            f"  kd={'ON (' + args.kd_teacher_mode + ', α=' + str(args.kd_alpha) + ')' if teacher else 'OFF'}\n"
        )

    # ── 학습 루프 ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        token_pruner.set_epoch(epoch)

        if is_main:
            print(f"\n── Epoch {epoch}/{args.epochs - 1}  lr={optimizer.param_groups[0]['lr']:.2e} ──")

        train_stats = train_one_epoch(
            model, criterion, train_loader,
            optimizer, scaler, device, epoch,
            model_ema=model_ema,
            pruner=None,  # Stage 2에는 channel pruning 없음 — weight 마스킹 불필요
            amp=args.amp,
            clip_grad=args.clip_grad,
            log_interval=args.log_interval,
            teacher=teacher,
            kd_alpha=args.kd_alpha if teacher is not None else 0.0,
            kd_temperature=args.kd_temperature,
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

        token_stats = token_pruner.log_info(eval_model) if is_main else {}

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
            log_dict.update({k: v for k, v in token_stats.items() if not isinstance(v, list)})
            wandb_run.log(log_dict)

        if is_main:
            print(
                f"  val_top1={acc1:.2f}%  best={best_acc1:.2f}%  "
                f"keep_rate={token_stats.get('token_pruning/keep_rate', '?')}"
                + (" ← BEST" if is_best else "")
            )

        if is_main:
            save_checkpoint(
                os.path.join(args.output_dir, "checkpoint_last.pt"),
                model, model_ema, optimizer, lr_scheduler, scaler,
                token_pruner, epoch, best_acc1, args,
            )
            if is_best:
                save_checkpoint(
                    os.path.join(args.output_dir, "checkpoint_best.pt"),
                    model, model_ema, optimizer, lr_scheduler, scaler,
                    token_pruner, epoch, best_acc1, args,
                )
                n_after = sum(p.numel() for p in eval_model.parameters())
                save_token_pruned_artifact(
                    os.path.join(args.output_dir, "token_pruned_best.pt"),
                    eval_model, model_name, mlp_dims, token_pruner,
                    n_params_before_total, n_after,
                )

    if is_main:
        print(f"\n=== Done. Best val top-1: {best_acc1:.2f}% ===")
        print(f"배포용 아티팩트: {os.path.join(args.output_dir, 'token_pruned_best.pt')}")

    if wandb_run is not None:
        wandb_run.finish()

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
