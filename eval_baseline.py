"""
Pruning 전 사전학습 모델 baseline evaluation → WandB 기록

단일 GPU:
    python eval_baseline.py \\
        --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \\
        --wandb

멀티 GPU (DDP):
    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 eval_baseline.py \\
        --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \\
        --wandb

특정 모델만:
    python eval_baseline.py \\
        --model vit_small_patch16_224 \\
        --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \\
        --wandb
"""

from __future__ import annotations

import os
import argparse

import torch
import torch.distributed as dist
import timm
import timm.data
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets

from engine import evaluate

# 평가 대상 모델 (--model 미지정 시 순서대로 실행)
BASELINE_MODELS = [
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
]


# ── argparse ───────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ViT Baseline Evaluation")
    p.add_argument("--data-path",    required=True,
                   help="ImageNet 루트 (하위에 val/ 폴더 존재)")
    p.add_argument("--model",        default="",
                   help="단일 모델 지정. 비어있으면 BASELINE_MODELS 전체 평가")
    p.add_argument("--input-size",   type=int, default=224)
    p.add_argument("--batch-size",   type=int, default=256,
                   help="GPU 당 배치 크기")
    p.add_argument("--num-workers",  type=int, default=8)
    p.add_argument("--amp",          action="store_true", default=True)
    p.add_argument("--wandb",        action="store_true")
    p.add_argument("--wandb-project", default="vit-pruning")
    p.add_argument("--dist-url",     default="env://")
    return p.parse_args()


# ── 분산 설정 ──────────────────────────────────────────────────────────────────

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


# ── val DataLoader ─────────────────────────────────────────────────────────────

def build_val_loader(args: argparse.Namespace, model: torch.nn.Module) -> DataLoader:
    # 모델의 실제 권장 data config 사용 (mean/std/crop_pct 등이 모델마다 다름)
    data_config = timm.data.resolve_model_data_config(model)
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    val_ds = datasets.ImageFolder(
        os.path.join(args.data_path, "val"), transform=val_transform
    )
    sampler = DistributedSampler(val_ds, shuffle=False) if args.distributed else None
    return DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ── 모델 평가 ──────────────────────────────────────────────────────────────────

def eval_one_model(
    model_name: str,
    device: torch.device,
    args: argparse.Namespace,
    is_main: bool,
) -> dict[str, float]:
    if is_main:
        print(f"\n{'='*55}")
        print(f"  Evaluating: {model_name}")
        print(f"{'='*55}")

    model = timm.create_model(model_name, pretrained=True)
    model = model.to(device)

    # 모델별 권장 data config 출력 (디버깅용)
    if is_main:
        cfg = timm.data.resolve_model_data_config(model)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params:    {n_params:,}")
        print(f"  mean/std:  {cfg['mean']} / {cfg['std']}")
        print(f"  crop_pct:  {cfg.get('crop_pct', 'default')}")

    val_loader = build_val_loader(args, model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    metrics = evaluate(val_loader, model, device, amp=args.amp)
    return metrics


# ── WandB 기록 ─────────────────────────────────────────────────────────────────

def log_to_wandb(model_name: str, metrics: dict[str, float], project: str) -> None:
    import wandb
    run = wandb.init(
        project=project,
        name=f"{model_name}_baseline",
        config={"model": model_name, "type": "baseline"},
        reinit=True,
    )
    run.log({
        "baseline/top1": metrics["acc1"],
        "baseline/top5": metrics["acc5"],
        "baseline/loss": metrics["loss"],
    })
    run.summary.update({
        "top1": metrics["acc1"],
        "top5": metrics["acc5"],
    })
    run.finish()


# ── 메인 ───────────────────────────────────────────────────────────────────────

def main():
    args    = get_args()
    is_main = setup_distributed(args)
    device  = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    models_to_eval = [args.model] if args.model else BASELINE_MODELS

    results: dict[str, dict] = {}
    for model_name in models_to_eval:
        metrics = eval_one_model(model_name, device, args, is_main)
        results[model_name] = metrics

        if is_main:
            print(f"\n  [{model_name}] top1={metrics['acc1']:.2f}%  "
                  f"top5={metrics['acc5']:.2f}%  loss={metrics['loss']:.4f}")

            if args.wandb:
                log_to_wandb(model_name, metrics, args.wandb_project)
                print(f"  → WandB 기록 완료: {model_name}_baseline")

    # ── 최종 요약 ──────────────────────────────────────────────────────────────
    if is_main and len(results) > 1:
        print(f"\n{'='*55}")
        print(f"  Baseline Summary")
        print(f"{'='*55}")
        print(f"  {'model':<30} {'top1':>8} {'top5':>8}")
        print(f"  {'-'*50}")
        for name, m in results.items():
            print(f"  {name:<30} {m['acc1']:>7.2f}% {m['acc5']:>7.2f}%")

    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
