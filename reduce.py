"""
Soft Pruning 완료 체크포인트 → Dense 모델 생성 CLI

사용법:
    python reduce.py \\
        --model vit_small_patch16_224 \\
        --checkpoint ./output/vit_small_prune20/checkpoint_best.pt \\
        --output     ./output/vit_small_prune20/reduced.pt

reduced.pt 로드 방법:
    import timm, torch
    from pruning.vit_reducing import apply_reduced_config

    ckpt  = torch.load("reduced.pt", map_location="cpu")
    model = timm.create_model(ckpt["model_name"], pretrained=False)
    apply_reduced_config(model, ckpt["mlp_dims"])   # 구조 축소
    model.load_state_dict(ckpt["state_dict"])        # 가중치 복원
    model.eval()
"""

from __future__ import annotations

import argparse
import torch
import timm
from pruning.vit_reducing import (
    reduce_vit_model, get_reduced_config, transfer_pruning_mask
)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("ViT Soft Pruning Reducer")
    p.add_argument("--model",       required=True, help="timm 모델 이름")
    p.add_argument("--checkpoint",  required=True, help="학습 완료 체크포인트 경로")
    p.add_argument("--output",      required=True, help="저장할 reduced 모델 경로")
    p.add_argument("--input-size",  type=int, default=224)
    p.add_argument("--no-ema",      action="store_true",
                   help="EMA weights 대신 raw model weights 사용")
    return p.parse_args()


def main():
    args = get_args()

    # ── 1. 빈 timm 모델 생성 ───────────────────────────────────────────────────
    model = timm.create_model(args.model, pretrained=False)

    # ── 2. 체크포인트 로드 ─────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        raw_sd = ckpt.get("model")
        ema_sd = ckpt.get("model_ema")
        use_ema = (not args.no_ema) and ema_sd is not None
    else:
        raw_sd, ema_sd, use_ema = ckpt, None, False

    def strip_ddp(sd: dict) -> dict:
        return {k.removeprefix("module."): v for k, v in sd.items()}

    if use_ema:
        # EMA weights로 alive 채널의 학습된 값을 복원하되,
        # dead 채널 판별은 raw model의 정확한 0을 기준으로 한다.
        # (EMA decay^N 이 초기값에 곱해진 상태라 dead 채널이 정확히 0이 아님)
        print("[Reducer] EMA weights 사용  (dead 채널 마스크는 raw model 기준)")
        raw_model = timm.create_model(args.model, pretrained=False)
        raw_model.load_state_dict(strip_ddp(raw_sd), strict=True)

        model.load_state_dict(strip_ddp(ema_sd), strict=True)
        transfer_pruning_mask(raw_model, model)  # raw의 zero 패턴 → EMA model
        del raw_model
    elif raw_sd is not None:
        print("[Reducer] raw model weights 사용")
        model.load_state_dict(strip_ddp(raw_sd), strict=True)
    else:
        print("[Reducer] state_dict 직접 사용")
        model.load_state_dict(strip_ddp(ckpt), strict=True)

    # ── 3. Reducing ────────────────────────────────────────────────────────────
    n_before = sum(p.numel() for p in model.parameters())
    print(f"\nBEFORE reduce: {n_before:,} params")

    reduce_vit_model(model)

    n_after = sum(p.numel() for p in model.parameters())
    rate    = 100.0 * (n_before - n_after) / n_before
    print(f"AFTER  reduce: {n_after:,} params  ({rate:.2f}% removed)")

    # ── 4. Forward 검증 ────────────────────────────────────────────────────────
    model.eval()
    dummy = torch.zeros(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        out = model(dummy)
    print(f"Forward OK: output shape = {tuple(out.shape)}")

    # ── 5. 블록별 new mlp_dim 출력 ─────────────────────────────────────────────
    mlp_dims = get_reduced_config(model)
    orig_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    orig_dim  = orig_args.get("model", args.model)

    print(f"\n블록별 reduced mlp_dim:")
    for i, d in enumerate(mlp_dims):
        print(f"  block {i:>2d}: {d}")

    # 압축률 검증
    assert rate >= 0, "Reducing 후 파라미터가 늘어남 — 체크포인트 확인 필요"
    print(f"\n압축률: {rate:.2f}%")

    # ── 6. 저장 ────────────────────────────────────────────────────────────────
    torch.save(
        {
            "state_dict":       model.state_dict(),
            "model_name":       args.model,
            "mlp_dims":         mlp_dims,
            "n_params_before":  n_before,
            "n_params_after":   n_after,
            "compression_rate": rate,
        },
        args.output,
    )
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
