"""
timm ViT/DeiT adapter — 모델 검증, block/attention 속성 접근, 형상 정보.

SPEC §1.2/§1.3: 표준 단일 CLS 토큰 ViT/DeiT만 지원. distilled(dist_token) 또는
prefix token이 CLS 하나가 아닌 모델은 조기 실패(§1.3의 명시적 adapter 정책).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ViTShape:
    num_blocks: int          # L
    embed_dim: int           # D
    num_heads: int           # H
    head_dim: int
    num_patches: int         # N_patch
    num_prefix_tokens: int   # 1 (CLS)
    num_global_tokens: int   # N = N_patch + 1
    num_classes: int


def validate_vit(model: nn.Module) -> None:
    """지원 범위 검증 (SPEC §1.3). 조용히 우회하지 않고 명시적으로 실패한다."""
    if getattr(model, "dist_token", None) is not None:
        raise NotImplementedError(
            "distilled 모델(dist_token 존재) 미지원 — 논문 재현은 CLS 하나인 모델로 제한(SPEC §1.3)."
        )
    if not hasattr(model, "cls_token") or model.cls_token is None:
        raise NotImplementedError("cls_token이 없는 모델 미지원 (no_embed_class 등).")
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    if n_prefix != 1:
        raise NotImplementedError(
            f"num_prefix_tokens={n_prefix} 미지원. 첫 구현은 CLS 하나만 지원(SPEC §1.3)."
        )
    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        raise RuntimeError("model.blocks 접근 불가 — 표준 timm ViT 구조가 아님.")
    blk0 = model.blocks[0]
    for attr in ("norm1", "attn", "norm2", "mlp"):
        if not hasattr(blk0, attr):
            raise RuntimeError(f"block에 {attr} 없음 — timm 버전/구조 확인 필요.")
    for attr in ("qkv", "proj", "num_heads"):
        if not hasattr(blk0.attn, attr):
            raise RuntimeError(f"attn에 {attr} 없음 — timm Attention 구조 확인 필요.")


def get_shape(model: nn.Module) -> ViTShape:
    validate_vit(model)
    L = len(model.blocks)
    attn0 = model.blocks[0].attn
    H = int(attn0.num_heads)
    # embed_dim: cls_token feature 크기
    D = int(model.cls_token.shape[-1])
    head_dim = int(getattr(attn0, "head_dim", D // H))
    n_patch = int(model.patch_embed.num_patches)
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    num_classes = int(getattr(model, "num_classes", 0)) or int(model.head.out_features)
    return ViTShape(
        num_blocks=L, embed_dim=D, num_heads=H, head_dim=head_dim,
        num_patches=n_patch, num_prefix_tokens=n_prefix,
        num_global_tokens=n_patch + n_prefix, num_classes=num_classes,
    )


def embed_tokens(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """이미지 → block 입력 토큰 시퀀스 [B, N, D] (patch_embed + cls + pos_embed).

    timm의 forward_features 앞부분과 동일. block들은 태우지 않는다.
    """
    x = model.patch_embed(images)
    x = model._pos_embed(x)              # cls prepend + pos_embed (+ patch_drop 포함될 수 있음)
    if hasattr(model, "norm_pre"):
        x = model.norm_pre(x)
    return x


def head_from_tokens(model: nn.Module, tokens: torch.Tensor, cls_local_index: int = 0) -> torch.Tensor:
    """최종 block 출력 토큰 → logits (norm + head). CLS 토큰으로 분류.

    tokens: [B, N_out, D] (마지막 block 출력). cls_local_index: CLS의 local 위치.
    """
    x = model.norm(tokens)
    if getattr(model, "global_pool", "token") == "avg":
        feat = x.mean(dim=1)
    else:
        feat = x[:, cls_local_index]
    if hasattr(model, "fc_norm"):
        feat = model.fc_norm(feat)
    if hasattr(model, "head_drop"):
        feat = model.head_drop(feat)
    return model.head(feat)


def attn_qkv(attn: nn.Module, x_norm: torch.Tensor):
    """norm1(x) → (q, k, v) 각 [B, H, N, head_dim], q_norm/k_norm 적용 후.

    timm Attention 내부와 동일하게 fused qkv를 분해한다 (SPEC §5.3). attn.forward는
    호출하지 않으므로 fused_attn 경로/정확도에 영향 없다.
    """
    B, N, C = x_norm.shape
    H = attn.num_heads
    dh = getattr(attn, "head_dim", C // H)
    qkv = attn.qkv(x_norm).reshape(B, N, 3, H, dh).permute(2, 0, 3, 1, 4)  # [3,B,H,N,dh]
    q, k, v = qkv.unbind(0)
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)
    if q_norm is not None and not isinstance(q_norm, nn.Identity):
        q = q_norm(q)
    if k_norm is not None and not isinstance(k_norm, nn.Identity):
        k = k_norm(k)
    return q, k, v


def attn_scale(attn: nn.Module) -> float:
    C_per_head = getattr(attn, "head_dim", None)
    return float(getattr(attn, "scale", (C_per_head ** -0.5) if C_per_head else 1.0))
