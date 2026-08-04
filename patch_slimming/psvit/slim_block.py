"""
§5 — SlimTransformerBlock (rectangular attention).

기존 timm Block의 weight를 그대로 재사용하되, 토큰 흐름만 바꾼다:
    Q      : 출력으로 유지할 N_out개 토큰에서만 계산
    K/V    : 입력 N_in개 토큰 전체에서 계산
    attention map : N_out × N_in  (rectangular)
    residual: 입력에서 output 토큰 row만 선택
    MLP    : N_out개 output 토큰에서만

토큰 선택 방식 (SPEC §14.0):
    "index_select" — 학습/검색(GPU)용. differentiable, 정확.
    "matmul"       — NPU export용. 상수 선택행렬 P(N_out×N_in) @ x. Gather 없음.
    두 방식은 수치적으로 동일하다.

수치 등가성 요구 (SPEC §5.5, test로 게이트):
    output_ids == active_ids(전체)  → slim_block(x) ≈ original_block(x)
    output_ids == subset            → slim_block(x) ≈ original_block(x)[subset]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ids import global_to_local
from .model_utils import attn_scale


@dataclass
class SlimBlockOutput:
    x: Tensor                          # [B, N_out, D]
    output_global_ids: Tensor          # [N_out]
    attention_probs: Optional[Tensor]  # [B, H, N_out, N_in]  post-softmax, or None


def _split_qkv(attn: nn.Module):
    """fused qkv Linear를 Wq/Wk/Wv (+bias)로 논리 분할 (SPEC §5.3). weight 복제 아님(view)."""
    W = attn.qkv.weight                       # [3D, D]
    D = W.shape[1]
    b = attn.qkv.bias                         # [3D] or None
    Wq, Wk, Wv = W[0:D], W[D:2 * D], W[2 * D:3 * D]
    if b is not None:
        bq, bk, bv = b[0:D], b[D:2 * D], b[2 * D:3 * D]
    else:
        bq = bk = bv = None
    return (Wq, bq), (Wk, bk), (Wv, bv)


def selection_matrix(out_local: Tensor, n_in: int, device, dtype) -> Tensor:
    """P[i, out_local[i]] = 1 인 상수 선택행렬 [N_out, N_in] (matmul 선택용)."""
    n_out = out_local.numel()
    P = torch.zeros(n_out, n_in, device=device, dtype=dtype)
    P[torch.arange(n_out, device=device), out_local.to(device)] = 1.0
    return P


def select_rows(x: Tensor, out_local: Tensor, mode: str, sel_mat: Optional[Tensor] = None) -> Tensor:
    """[B, N_in, D] → [B, N_out, D]. index_select / 상수행렬 matmul 둘 다 결과 동일."""
    if mode == "index_select":
        return x.index_select(1, out_local.to(x.device))
    if mode == "matmul":
        P = sel_mat if sel_mat is not None else selection_matrix(out_local, x.shape[1], x.device, x.dtype)
        return torch.matmul(P, x)          # [N_out,N_in] @ [B,N_in,D] → [B,N_out,D]
    raise ValueError(f"select mode: {mode!r}")


class SlimBlock(nn.Module):
    """timm Block 하나를 감싸 rectangular attention으로 forward. weight 공유(복제 X)."""

    def __init__(self, block: nn.Module, select_mode: str = "index_select"):
        super().__init__()
        self.block = block                 # 원본 block (norm1, attn, ls1, drop_path1, ...)
        self.attn = block.attn
        self.num_heads = int(block.attn.num_heads)
        self.head_dim = int(getattr(block.attn, "head_dim",
                                    block.attn.qkv.weight.shape[1] // self.num_heads))
        self.scale = attn_scale(block.attn)
        self.select_mode = select_mode
        # matmul 모드에서 고정 출력을 쓸 때 precompute할 선택행렬 (compact export용)
        self._sel_mat_q: Optional[Tensor] = None
        self._sel_mat_r: Optional[Tensor] = None

    # ── 내부 ────────────────────────────────────────────────────────────────────
    def _heads(self, t: Tensor) -> Tensor:
        B, N, D = t.shape
        return t.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B,H,N,dh]

    def _apply_norm(self, name: str, t: Tensor) -> Tensor:
        m = getattr(self.attn, name, None)
        if m is not None and not isinstance(m, nn.Identity):
            return m(t)
        return t

    # ── forward ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: Tensor,                     # [B, N_in, D]
        active_global_ids: Tensor,     # [N_in]
        output_global_ids: Tensor,     # [N_out] ⊆ active
        return_attention: bool = False,
    ) -> SlimBlockOutput:
        blk = self.block
        B, N_in, D = x.shape
        out_local = global_to_local(active_global_ids, output_global_ids)

        (Wq, bq), (Wk, bk), (Wv, bv) = _split_qkv(self.attn)
        x_norm = blk.norm1(x)                                       # [B,N_in,D]

        # K/V: 전체 입력 토큰
        k = self._apply_norm("k_norm", self._heads(F.linear(x_norm, Wk, bk)))   # [B,H,N_in,dh]
        v = self._heads(F.linear(x_norm, Wv, bv))                               # [B,H,N_in,dh]

        # Q: 출력 토큰만 (선택 후 projection — SPEC §5.3, Q 연산 절감)
        x_norm_sel = select_rows(x_norm, out_local, self.select_mode, self._sel_mat_q)  # [B,N_out,D]
        q = self._apply_norm("q_norm", self._heads(F.linear(x_norm_sel, Wq, bq)))       # [B,H,N_out,dh]

        # rectangular attention
        attn = (q @ k.transpose(-2, -1)) * self.scale              # [B,H,N_out,N_in]
        attn = attn.softmax(dim=-1)
        attn = self._apply_norm_drop("attn_drop", attn)
        ctx = attn @ v                                             # [B,H,N_out,dh]
        N_out = ctx.shape[2]
        ctx = ctx.transpose(1, 2).reshape(B, N_out, D)
        ctx = self.attn.proj(ctx)
        ctx = self._apply_norm_drop("proj_drop", ctx)

        # residual: 입력에서 output 토큰 선택
        x_sel = select_rows(x, out_local, self.select_mode, self._sel_mat_r)   # [B,N_out,D]
        x_out = x_sel + blk.drop_path1(blk.ls1(ctx))
        x_out = x_out + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x_out))))

        return SlimBlockOutput(
            x=x_out,
            output_global_ids=output_global_ids,
            attention_probs=attn if return_attention else None,
        )

    def _apply_norm_drop(self, name: str, t: Tensor) -> Tensor:
        m = getattr(self.attn, name, None)
        if m is not None and not isinstance(m, nn.Identity):
            return m(t)
        return t

    # ── compact export: 고정 출력에 대한 선택행렬 미리 굽기 ─────────────────────
    @torch.no_grad()
    def bake_selection(self, active_global_ids: Tensor, output_global_ids: Tensor,
                       device, dtype) -> None:
        """matmul 모드용 상수 선택행렬 precompute (compact 모델 export 시 1회)."""
        out_local = global_to_local(active_global_ids, output_global_ids)
        n_in = int(active_global_ids.numel())
        P = selection_matrix(out_local, n_in, device, dtype)
        self._sel_mat_q = P
        self._sel_mat_r = P
