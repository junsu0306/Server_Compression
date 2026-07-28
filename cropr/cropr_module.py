"""
Cropr: token pruning module.

Reference: Benbergner et al. "Token Cropr" (https://github.com/benbergner/cropr)
Adapted for integration with timm ViT / reduced models.

학습 시: CrossAttention(full) + ClassificationHead → auxiliary loss 신호 제공
추론 시: CrossAttention.forward_scorer() 만 실행 → 분류 헤드 연산 생략
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """Learnable query로 모든 토큰을 cross-attend → 토큰 중요도 점수 계산.

    forward_scorer(): 추론 전용. attention score만 계산 (값 projection 없음).
    forward():        학습 전용. 완전한 cross-attention + aggregated representation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_queries: int = 1,
        q_proj: bool = True,
        k_proj: bool = True,
        v_proj: bool = True,
        pre_attn_norm: bool = False,
        mlp: bool = False,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.queries = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.q_proj = nn.Linear(embed_dim, embed_dim) if q_proj else nn.Identity()
        self.k_proj = nn.Linear(embed_dim, embed_dim) if k_proj else nn.Identity()
        self.v_proj = nn.Linear(embed_dim, embed_dim) if v_proj else nn.Identity()
        self.norm   = nn.LayerNorm(embed_dim) if pre_attn_norm else nn.Identity()

        self.mlp_block = None
        if mlp:
            mlp_dim = int(embed_dim * mlp_ratio)
            self.mlp_block = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, embed_dim),
            )

    def forward_scorer(self, x: torch.Tensor) -> torch.Tensor:
        """추론 전용: attention weight 합산 → 토큰 중요도 점수 (B, N)."""
        B, N, _ = x.shape
        x_norm = self.norm(x)

        q = self.queries.expand(B, -1, -1)
        q = self.q_proj(q).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return attn.sum(dim=1).sum(dim=1)   # (B, N)

    def forward(self, x: torch.Tensor):
        """학습 전용: cross-attention + aggregated representation."""
        B, N, C = x.shape
        x_norm = self.norm(x)

        q = self.queries.expand(B, -1, -1)
        q = self.q_proj(q).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x_aggr = (attn @ v).transpose(1, 2).reshape(B, -1, C).mean(dim=1)  # (B, C)
        if self.mlp_block is not None:
            x_aggr = x_aggr + self.mlp_block(x_aggr.unsqueeze(1)).squeeze(1)

        scores = attn.sum(dim=1).sum(dim=1)   # (B, N)
        return x_aggr, scores


class ClassificationHead(nn.Module):
    """Auxiliary 분류 헤드. 학습 중에만 사용, 추론 시 연산 생략."""

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(x))


class CroprModule(nn.Module):
    """단일 Cropr 모듈: 토큰 중요도 점수 계산 → 하위 pruning_rate개 제거.

    Args:
        pruning_rate:   제거할 토큰 수 (정수)
        embed_dim:      토큰 임베딩 차원
        num_heads:      cross-attention 헤드 수
        num_queries:    learnable query 수 (기본 1)
        num_classes:    ImageNet 클래스 수 (auxiliary head용)
    """

    def __init__(
        self,
        pruning_rate: int,
        embed_dim: int,
        num_heads: int,
        num_queries: int = 1,
        num_classes: int = 1000,
        q_proj: bool = True,
        k_proj: bool = True,
        v_proj: bool = True,
        pre_attn_norm: bool = False,
        mlp: bool = False,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.pruning_rate = pruning_rate
        self.cross_attn   = CrossAttention(
            embed_dim=embed_dim, num_heads=num_heads, num_queries=num_queries,
            q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
            pre_attn_norm=pre_attn_norm, mlp=mlp, mlp_ratio=mlp_ratio,
        )
        self.cls_head = ClassificationHead(embed_dim, num_classes)

    def _prune(self, x: torch.Tensor, scores: torch.Tensor):
        """중요도 낮은 pruning_rate개 토큰 분리. CLS 토큰(idx 0)은 항상 보존."""
        B, N, C = x.shape
        scores = scores.clone()
        scores[:, 0] = math.inf   # CLS 항상 보존

        n_keep = N - self.pruning_rate
        idx = torch.argsort(scores, dim=1, descending=True, stable=False)

        idx_keep  = idx[:, :n_keep].sort(dim=1).values
        idx_prune = idx[:, n_keep:].sort(dim=1).values

        x_keep  = torch.gather(x, 1, idx_keep.unsqueeze(-1).expand(-1, -1, C))
        x_prune = torch.gather(x, 1, idx_prune.unsqueeze(-1).expand(-1, -1, C))
        return x_keep, x_prune

    def forward(self, x: torch.Tensor, inference: bool = False):
        """
        Returns:
            x_keep: 살아남은 토큰  (B, N-pruning_rate, C)
            x_prune: 제거된 토큰   (B, pruning_rate, C)
            pred:    auxiliary logit (B, num_classes) or None (inference 시)
        """
        if inference:
            scores = self.cross_attn.forward_scorer(x)
            x_keep, x_prune = self._prune(x, scores)
            return x_keep, x_prune, None
        else:
            x_aggr, scores = self.cross_attn(x)
            pred = self.cls_head(x_aggr)
            x_keep, x_prune = self._prune(x, scores)
            return x_keep, x_prune, pred
