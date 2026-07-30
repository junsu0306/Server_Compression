"""
Reduced 모델 (reduced.pt) → ImageNet val 평가 → WandB 기록

사용법:
    python eval_reduced.py \\
        --reduced  ./output/vit_tiny_prune50_global/reduced.pt \\
        --data-path /workspace/etri_iitp/JS/Server_Compression/data/imagenet \\
        --wandb
"""

from __future__ import annotations

import argparse
import os

import torch
import timm
import timm.data
from torchvision import datasets

from pruning.vit_reducing import apply_reduced_config
from pruning.vit_flops import analyze_vit_compute, compute_reduction, format_flops, format_bytes
from engine import evaluate


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Reduced ViT Evaluation")
    p.add_argument("--reduced",        required=True,
                   help="reduced.pt 경로 (reduce.py 출력)")
    p.add_argument("--data-path",      required=True,
                   help="ImageNet 루트 (하위에 val/ 폴더 존재)")
    p.add_argument("--batch-size",     type=int, default=256)
    p.add_argument("--num-workers",    type=int, default=8)
    p.add_argument("--amp",            action="store_true", default=True)
    p.add_argument("--gpu",            type=int, default=0)
    p.add_argument("--wandb",          action="store_true")
    p.add_argument("--wandb-project",  default="vit-pruning")
    p.add_argument("--wandb-run-name", default="",
                   help="WandB run 이름. 비어있으면 자동 생성")
    return p.parse_args()


def load_reduced_model(path: str) -> tuple[torch.nn.Module, dict]:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    model = timm.create_model(ckpt["model_name"], pretrained=False)
    apply_reduced_config(model, ckpt["mlp_dims"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def build_val_loader(args: argparse.Namespace, model_name: str):
    # 원본 모델의 권장 mean/std/crop_pct 사용 (모델마다 다름)
    ref = timm.create_model(model_name, pretrained=False)
    data_config = timm.data.resolve_model_data_config(ref)
    del ref

    val_transform = timm.data.create_transform(**data_config, is_training=False)
    val_ds = datasets.ImageFolder(
        os.path.join(args.data_path, "val"), transform=val_transform
    )
    loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, data_config


def main():
    args   = get_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    model, ckpt = load_reduced_model(args.reduced)
    model = model.to(device)

    model_name       = ckpt["model_name"]
    n_before         = ckpt.get("n_params_before", 0)
    n_after          = ckpt.get("n_params_after",
                                sum(p.numel() for p in model.parameters()))
    compression_rate = ckpt.get("compression_rate", 0.0)

    print(f"\n{'='*55}")
    print(f"  Evaluating Reduced Model")
    print(f"{'='*55}")
    print(f"  file:        {args.reduced}")
    print(f"  base model:  {model_name}")
    print(f"  params:      {n_after:,}  (before: {n_before:,}, -{compression_rate:.2f}%)")

    # 블록별 survived mlp_dim 출력
    print(f"\n  블록별 mlp_dim: {ckpt['mlp_dims']}")

    # ── FLOPs / Activation Footprint 분석 (baseline vs reduced) ─────────────────
    # baseline: 압축 전 원본 구조 (mlp_dim이 block마다 동일)
    baseline_ref = timm.create_model(model_name, pretrained=False)
    baseline_mlp_dims = [b.mlp.fc1.out_features for b in baseline_ref.blocks]
    embed_dim = model.embed_dim
    num_heads = model.blocks[0].attn.num_heads
    n_patches = model.patch_embed.num_patches
    del baseline_ref

    baseline_compute = analyze_vit_compute(embed_dim, num_heads, baseline_mlp_dims, n_patches)
    reduced_compute  = analyze_vit_compute(embed_dim, num_heads, ckpt["mlp_dims"], n_patches)
    reduction = compute_reduction(baseline_compute, reduced_compute)

    print(f"\n  FLOPs:       {format_flops(baseline_compute['flops_total'])} → "
          f"{format_flops(reduced_compute['flops_total'])}  "
          f"(-{reduction['flops_reduction_pct']:.2f}%)")
    print(f"  Activation:  {format_bytes(baseline_compute['peak_activation_bytes'])} → "
          f"{format_bytes(reduced_compute['peak_activation_bytes'])}  "
          f"(-{reduction['activation_reduction_pct']:.2f}%, block별 peak 기준 추정치)")

    # ── 데이터 ─────────────────────────────────────────────────────────────────
    val_loader, data_config = build_val_loader(args, model_name)
    print(f"\n  mean/std:  {data_config['mean']} / {data_config['std']}")
    print(f"  crop_pct:  {data_config.get('crop_pct', 'default')}")

    # ── 평가 ───────────────────────────────────────────────────────────────────
    print(f"\n  Evaluating on ImageNet val (50,000 images)...")
    metrics = evaluate(val_loader, model, device, amp=args.amp)

    print(f"\n  top1 = {metrics['acc1']:.2f}%")
    print(f"  top5 = {metrics['acc5']:.2f}%")
    print(f"  loss = {metrics['loss']:.4f}")

    # ── WandB ──────────────────────────────────────────────────────────────────
    if args.wandb:
        import wandb
        run_name = args.wandb_run_name or (
            f"{model_name}_reduced_{compression_rate:.0f}pct"
        )
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model":            model_name,
                "type":             "reduced",
                "reduced_path":     args.reduced,
                "n_params_before":  n_before,
                "n_params_after":   n_after,
                "compression_rate": compression_rate,
                "mlp_dims":         ckpt["mlp_dims"],
            },
        )
        run.log({
            "test/top1":            metrics["acc1"],
            "test/top5":            metrics["acc5"],
            "test/loss":            metrics["loss"],
            "test/n_params":        n_after,
            "test/compression_pct": compression_rate,
            "test/flops_baseline":            baseline_compute["flops_total"],
            "test/flops_reduced":             reduced_compute["flops_total"],
            "test/flops_reduction_pct":       reduction["flops_reduction_pct"],
            "test/activation_baseline_bytes": baseline_compute["peak_activation_bytes"],
            "test/activation_reduced_bytes":  reduced_compute["peak_activation_bytes"],
            "test/activation_reduction_pct":  reduction["activation_reduction_pct"],
        })
        run.summary.update({
            "top1": metrics["acc1"],
            "top5": metrics["acc5"],
        })

        # 압축 요약 bar chart — params/FLOPs/activation 절감률 한눈에 비교
        summary_tbl = wandb.Table(
            data=[
                ["params",     compression_rate],
                ["flops",      reduction["flops_reduction_pct"]],
                ["activation", reduction["activation_reduction_pct"]],
            ],
            columns=["metric", "reduction_pct"],
        )
        run.log({
            "compression_summary": wandb.plot.bar(
                summary_tbl, "metric", "reduction_pct",
                title="Reduction % — Params vs FLOPs vs Activation Footprint",
            )
        })

        # block별 FLOPs / activation footprint 절감률 — FFN만 줄고 attention(QKV/proj)은
        # 그대로라 block마다 절감률이 다르게 나오는 걸 시각적으로 확인할 수 있다.
        flops_tbl = wandb.Table(
            data=[[f"block {i}", v] for i, v in enumerate(reduction["flops_per_block_pct"])],
            columns=["block", "flops_reduction_pct"],
        )
        run.log({
            "flops_per_block": wandb.plot.bar(
                flops_tbl, "block", "flops_reduction_pct",
                title="FLOPs Reduction % per Block",
            )
        })

        act_tbl = wandb.Table(
            data=[[f"block {i}", v] for i, v in enumerate(reduction["activation_per_block_pct"])],
            columns=["block", "activation_reduction_pct"],
        )
        run.log({
            "activation_per_block": wandb.plot.bar(
                act_tbl, "block", "activation_reduction_pct",
                title="Activation Footprint Reduction % per Block",
            )
        })

        run.finish()
        print(f"\n  WandB 기록 완료: {run_name}")


if __name__ == "__main__":
    main()
