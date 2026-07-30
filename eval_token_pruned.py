"""
Token Pruned 모델 (token_pruned_best.pt) → ImageNet val 평가 → WandB 기록

사용법:
    python eval_token_pruned.py \\
        --token-pruned ./output/vit_tiny_prune50_token70/token_pruned_best.pt \\
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
from pruning.token_pruning import apply_token_pruning
from engine import evaluate


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Token Pruned ViT Evaluation")
    p.add_argument("--token-pruned",   required=True,
                   help="token_pruned_best.pt 경로 (train_token_pruning.py 출력)")
    p.add_argument("--data-path",      required=True)
    p.add_argument("--batch-size",     type=int, default=256)
    p.add_argument("--num-workers",    type=int, default=8)
    p.add_argument("--amp",            action="store_true", default=True)
    p.add_argument("--gpu",            type=int, default=0)
    p.add_argument("--wandb",          action="store_true")
    p.add_argument("--wandb-project",  default="vit-pruning")
    p.add_argument("--wandb-run-name", default="")
    return p.parse_args()


def load_token_pruned_model(path: str) -> tuple[torch.nn.Module, dict]:
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    model = timm.create_model(ckpt["model_name"], pretrained=False)
    apply_reduced_config(model, ckpt["mlp_dims"])

    tp_cfg = ckpt["token_pruning"]
    apply_token_pruning(
        model,
        prune_layers=tp_cfg["prune_layers"],
        base_keep_rate=tp_cfg["base_keep_rate"],
        fuse_token=tp_cfg["fuse_token"],
    )

    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def build_val_loader(args: argparse.Namespace, model_name: str):
    ref = timm.create_model(model_name, pretrained=False)
    data_config = timm.data.resolve_model_data_config(ref)
    del ref

    val_transform = timm.data.create_transform(**data_config, is_training=False)
    val_ds = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    return loader, data_config


def main():
    args   = get_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_token_pruned_model(args.token_pruned)
    model = model.to(device)

    model_name = ckpt["model_name"]
    tp_cfg     = ckpt["token_pruning"]
    n_before   = ckpt.get("n_params_before", 0)
    n_after    = ckpt.get("n_params_after", sum(p.numel() for p in model.parameters()))

    print(f"\n{'='*55}")
    print(f"  Evaluating Token Pruned Model")
    print(f"{'='*55}")
    print(f"  file:            {args.token_pruned}")
    print(f"  base model:      {model_name}")
    print(f"  params:          {n_after:,}  (channel pruning 전: {n_before:,})")
    print(f"  prune_layers:    {tp_cfg['prune_layers']}")
    print(f"  base_keep_rate:  {tp_cfg['base_keep_rate']}")
    print(f"  fuse_token:      {tp_cfg['fuse_token']}")
    print(f"  mlp_dims:        {ckpt['mlp_dims']}")

    val_loader, data_config = build_val_loader(args, model_name)
    print(f"\n  mean/std:  {data_config['mean']} / {data_config['std']}")

    print(f"\n  Evaluating on ImageNet val (50,000 images)...")
    metrics = evaluate(val_loader, model, device, amp=args.amp)

    print(f"\n  top1 = {metrics['acc1']:.2f}%")
    print(f"  top5 = {metrics['acc5']:.2f}%")
    print(f"  loss = {metrics['loss']:.4f}")

    if args.wandb:
        import wandb
        run_name = args.wandb_run_name or (
            f"{model_name}_token_kr{int(tp_cfg['base_keep_rate']*100)}"
        )
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model":           model_name,
                "type":            "token_pruned",
                "token_pruned_path": args.token_pruned,
                "n_params":        n_after,
                "prune_layers":    tp_cfg["prune_layers"],
                "base_keep_rate":  tp_cfg["base_keep_rate"],
                "fuse_token":      tp_cfg["fuse_token"],
                "mlp_dims":        ckpt["mlp_dims"],
            },
        )
        run.log({
            "test/top1": metrics["acc1"],
            "test/top5": metrics["acc5"],
            "test/loss": metrics["loss"],
        })
        run.summary.update({"top1": metrics["acc1"], "top5": metrics["acc5"]})
        run.finish()
        print(f"\n  WandB 기록 완료: {run_name}")


if __name__ == "__main__":
    main()
