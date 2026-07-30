"""
ViT FLOPs / Activation Footprint 분석적(analytical) 추정.

목적: reduced 모델(§8, channel pruning으로 block별 mlp_dim이 non-uniform하게
줄어든 Dense 모델)이 baseline(원본 pretrained) 대비 연산량/활성화 메모리가
얼마나 줄었는지 eval_reduced.py에서 WandB로 시각화하기 위함.

측정이 아니라 "구조로부터 계산"이다 — hook으로 실제 forward를 프로파일링하는
대신, ViT block 구조(embed_dim, num_heads, mlp_dim, 토큰 개수)로부터 표준
공식을 이용해 계산한다. 이렇게 한 이유:
  - timm 1.0.x의 Attention은 fused_attn=True 시 F.scaled_dot_product_attention
    하나로 attention 전체를 처리해서, nn.Module 단위 forward hook으로는
    Q@K^T / softmax / attn@V FLOPs를 분해해서 잡을 수 없다.
  - channel pruning 결과는 block마다 mlp_dim이 다르므로(§4), 이미 알고 있는
    구조 정보(mlp_dims)로 계산하는 쪽이 hook 기반 프로파일링보다 오히려
    간단하고 정확하다.

주의 — activation footprint는 "추정치"다:
    실제 peak memory profiler가 아니라, block 하나의 forward 동안 존재하는
    주요 중간 텐서(qkv, attention matrix, mlp hidden 등) 크기의 합을 "그 block의
    activation footprint"로 정의한 것이다. 실제 framework의 메모리 재사용/해제
    시점에 따라 진짜 peak memory와는 차이가 있을 수 있다. 다만 baseline과
    reduced에 동일한 정의를 적용하므로 "상대 비교" 용도로는 유효하다.

FLOPs 정의: 곱-덧셈 1쌍(MAC) = 2 FLOPs.
"""

from __future__ import annotations

from typing import List


def _block_compute(embed_dim: int, num_heads: int, mlp_dim: int, n_tokens: int) -> dict:
    """block 하나의 MACs/FLOPs/activation footprint(elements)."""
    C, H, M, N = embed_dim, num_heads, mlp_dim, n_tokens

    qkv_macs  = N * C * (3 * C)
    proj_macs = N * C * C
    attn_macs = 2 * H * N * N * (C // H)   # QK^T + attn@V
    mlp_macs  = 2 * N * C * M              # fc1 + fc2

    attn_block_macs = qkv_macs + proj_macs + attn_macs
    total_macs = attn_block_macs + mlp_macs

    act_elems = (
        N * C          # 입력/잔차
        + N * 3 * C    # qkv
        + H * N * N    # attention matrix
        + N * C        # attn 출력
        + N * M        # mlp hidden
        + N * C        # mlp 출력
    )

    return {
        "macs":        total_macs,
        "flops":       2 * total_macs,
        "attn_flops":  2 * attn_block_macs,
        "mlp_flops":   2 * mlp_macs,
        "act_elems":   act_elems,
    }


def analyze_vit_compute(
    embed_dim: int,
    num_heads: int,
    mlp_dims: List[int],
    n_patches: int,
    num_classes: int = 1000,
    patch_size: int = 16,
    in_chans: int = 3,
    dtype_bytes: int = 4,
) -> dict:
    """전체 ViT의 FLOPs/activation footprint 추정.

    Args:
        embed_dim, num_heads: 모델 전역 설정 (channel pruning으로 변하지 않음)
        mlp_dims:    block별 mlp_dim 리스트 (reduced 모델은 non-uniform)
        n_patches:   patch_embed.num_patches (예: 224/16 → 196)
        dtype_bytes: activation footprint를 바이트로 환산할 dtype 크기 (fp32=4)

    반환:
        flops_total, macs_total, patch_embed_flops, head_flops,
        blocks: [{flops, attn_flops, mlp_flops, macs, act_bytes}, ...],
        peak_activation_bytes: block별 activation footprint 중 최댓값
        (실제 배포 시 peak memory에 가장 가까운 프록시)
    """
    n_tokens = n_patches + 1  # + CLS

    patch_embed_macs = n_patches * (patch_size * patch_size * in_chans) * embed_dim
    patch_embed_flops = 2 * patch_embed_macs

    head_macs  = embed_dim * num_classes
    head_flops = 2 * head_macs

    blocks = []
    for mlp_dim in mlp_dims:
        b = _block_compute(embed_dim, num_heads, mlp_dim, n_tokens)
        b["act_bytes"] = b.pop("act_elems") * dtype_bytes
        blocks.append(b)

    flops_total = patch_embed_flops + head_flops + sum(b["flops"] for b in blocks)
    macs_total  = patch_embed_macs + head_macs + sum(b["macs"] for b in blocks)
    peak_activation_bytes = max((b["act_bytes"] for b in blocks), default=0)

    return {
        "n_tokens":              n_tokens,
        "patch_embed_flops":     patch_embed_flops,
        "head_flops":            head_flops,
        "flops_total":           flops_total,
        "macs_total":            macs_total,
        "blocks":                blocks,
        "peak_activation_bytes": peak_activation_bytes,
    }


def compute_reduction(baseline: dict, reduced: dict) -> dict:
    """baseline 대비 reduced의 절감률(%). 값이 클수록 더 절감됨."""
    def pct(base, red):
        return 100.0 * (base - red) / base if base > 0 else 0.0

    return {
        "flops_reduction_pct":      pct(baseline["flops_total"], reduced["flops_total"]),
        "activation_reduction_pct": pct(
            baseline["peak_activation_bytes"], reduced["peak_activation_bytes"]
        ),
        "flops_per_block_pct": [
            pct(b["flops"], r["flops"])
            for b, r in zip(baseline["blocks"], reduced["blocks"])
        ],
        "activation_per_block_pct": [
            pct(b["act_bytes"], r["act_bytes"])
            for b, r in zip(baseline["blocks"], reduced["blocks"])
        ],
    }


# ── 출력 포매팅 ────────────────────────────────────────────────────────────────

def format_flops(n: float) -> str:
    for unit, div in (("GFLOPs", 1e9), ("MFLOPs", 1e6), ("KFLOPs", 1e3)):
        if n >= div:
            return f"{n / div:.3f} {unit}"
    return f"{n:.0f} FLOPs"


def format_bytes(n: float) -> str:
    for unit, div in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:.0f} B"
