"""
§6 — Attention 및 feature 계측.

계측을 켜도 baseline logits가 변하면 안 된다 (SPEC §6.2). 그래서 원본 block.forward는
그대로 실행해 x를 진행시키고, attention probability는 norm1(x)로부터 별도로 재계산한다
(x에 영향 없음). softmax **이후** probability를 저장한다 (SPEC §6.1).

메모리: attention [B,H,N,N]는 크므로 target layer별 streaming(scoring.py에서 처리)을
권장. 이 모듈의 capture_full은 Phase 1 검증/소규모용이다 (SPEC §6.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor

from .model_utils import embed_tokens, head_from_tokens, attn_qkv, attn_scale


@dataclass
class CaptureResult:
    z_in: List[Tensor]                       # z_in[l] = block l 입력 [B,N,D], l=0..L-1
    z_final: Tensor                          # 마지막 block 출력(= block L-1 출력) [B,N,D]
    logits: Tensor                           # [B, num_classes]
    attn_probs: Optional[List[Tensor]] = None  # attn_probs[l] = [B,H,N,N] post-softmax, or None

    def block_input(self, l: int) -> Tensor:
        return self.z_in[l]

    def block_output(self, l: int) -> Tensor:
        """block l의 출력 = block l+1의 입력. 마지막 block은 z_final."""
        L = len(self.z_in)
        return self.z_final if l == L - 1 else self.z_in[l + 1]


@torch.no_grad()
def _full_attention(block, x: Tensor) -> Tensor:
    """block의 full attention probability [B,H,N,N] (모든 토큰). x는 안 바꾼다."""
    q, k, v = attn_qkv(block.attn, block.norm1(x))
    scale = attn_scale(block.attn)
    attn = (q @ k.transpose(-2, -1)) * scale        # [B,H,N,N]
    return attn.softmax(dim=-1)


@torch.no_grad()
def capture_full(model, images: Tensor, capture_attention: bool = True) -> CaptureResult:
    """full ViT를 block별로 돌리며 z_in / (선택) attention / logits 캡처.

    logits는 timm forward와 동일해야 한다 (§6.2 test). model은 eval 상태로 호출.
    """
    x = embed_tokens(model, images)
    z_in: List[Tensor] = []
    attn_probs: List[Tensor] = [] if capture_attention else None

    for block in model.blocks:
        z_in.append(x)
        if capture_attention:
            attn_probs.append(_full_attention(block, x))
        x = block(x)                                # 원본 forward → logits 정확도 보존

    logits = head_from_tokens(model, x)
    return CaptureResult(z_in=z_in, z_final=x, logits=logits, attn_probs=attn_probs)


@torch.no_grad()
def logits_match(model, images: Tensor, atol: float = 1e-4) -> tuple[bool, float]:
    """§6.2 — 계측 forward가 timm 표준 forward와 logit이 일치하는지 검증."""
    ref = model(images)
    got = capture_full(model, images, capture_attention=False).logits
    max_diff = (ref - got).abs().max().item()
    return (max_diff <= atol), max_diff
