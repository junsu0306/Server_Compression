"""
CroprWrapper: 어떤 timm ViT (reduced 포함)에도 Cropr 토큰 pruning을 붙이는 래퍼.

- reduced.pt (가변 mlp_dim) 모델에서 동작
- checkpoint_best.pt (soft-pruned, 표준 dim) 모델에서도 동작
- 학습: 보조 분류 헤드가 auxiliary loss 신호 제공
- 추론: model.eval() 시 보조 헤드 연산 생략, scorer만 실행

사용:
    ckpt  = torch.load("reduced.pt")
    base  = timm.create_model(ckpt["model_name"], pretrained=False)
    apply_reduced_config(base, ckpt["mlp_dims"])
    base.load_state_dict(ckpt["state_dict"])

    model = CroprWrapper(base, pruning_locs=[3, 6, 9], pruning_rate=32)
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .cropr_module import CroprModule


class CroprWrapper(nn.Module):
    """timm ViT에 Cropr 토큰 pruning 모듈을 추가하는 래퍼.

    Args:
        model:          timm ViT 기반 모델 (reduced.pt 로드 후 포함)
        pruning_locs:   Cropr를 삽입할 블록 인덱스 (예: [3, 6, 9])
        pruning_rate:   각 Cropr 모듈에서 제거할 토큰 수 (첫 번째는 +1)
        num_queries:    learnable query 수 (기본 1)
        num_heads:      Cropr cross-attention 헤드 수 (None → 모델과 동일)
        num_classes:    분류 클래스 수 (auxiliary head용)
        llf:            Last Layer Fusion — 마지막 블록 전에 제거된 토큰 재결합
    """

    def __init__(
        self,
        model: nn.Module,
        pruning_locs: list[int],
        pruning_rate: int,
        num_queries: int = 1,
        num_heads: int | None = None,
        num_classes: int = 1000,
        llf: bool = False,
        q_proj: bool = True,
        k_proj: bool = True,
        v_proj: bool = True,
        pre_attn_norm: bool = False,
        mlp: bool = False,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.base_model   = model
        self.pruning_locs = sorted(pruning_locs)
        self.llf          = llf

        embed_dim = model.embed_dim
        if num_heads is None:
            num_heads = model.blocks[0].attn.num_heads

        # 첫 번째 모듈은 토큰 1개 추가 제거 (원본 Cropr 스케줄 방식)
        n_cropr  = len(pruning_locs)
        schedule = [pruning_rate] * n_cropr
        schedule[0] += 1

        self.cropr = nn.ModuleList([
            CroprModule(
                pruning_rate=schedule[i],
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_queries=num_queries,
                num_classes=num_classes,
                q_proj=q_proj, k_proj=k_proj, v_proj=v_proj,
                pre_attn_norm=pre_attn_norm,
                mlp=mlp, mlp_ratio=mlp_ratio,
            )
            for i in range(n_cropr)
        ])

        num_tokens = model.patch_embed.num_patches + 1  # +1 for CLS
        remaining  = [num_tokens - sum(schedule[:i + 1]) for i in range(n_cropr)]
        print(
            f"[CroprWrapper] pruning_locs={self.pruning_locs}  "
            f"tokens per stage: {num_tokens} → {' → '.join(map(str, remaining))}  "
            f"llf={llf}"
        )

    # ── Forward ─────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        inference = not self.training

        # ── Patch embedding ────────────────────────────────────────────────────
        x = self.base_model.patch_embed(x)
        x = self.base_model._pos_embed(x)
        if hasattr(self.base_model, "patch_drop"):
            x = self.base_model.patch_drop(x)
        if hasattr(self.base_model, "norm_pre"):
            x = self.base_model.norm_pre(x)

        preds   = []
        prnd    = []   # LLF용: 제거된 토큰 보관
        c_idx   = 0   # Cropr 모듈 인덱스
        n_blks  = len(self.base_model.blocks)

        for i, blk in enumerate(self.base_model.blocks):
            is_last        = (i == n_blks - 1)
            is_penultimate = (i == n_blks - 2)

            if self.llf and is_penultimate:
                # penultimate 블록: 처리 후 제거 토큰 재결합 (LLF)
                x = blk(x)
                if prnd:
                    x = torch.cat([x] + prnd, dim=1)
                continue

            if self.llf and is_last:
                x = blk(x)
                continue

            x = blk(x)

            if i in self.pruning_locs:
                x, x_p, pred = self.cropr[c_idx](x, inference=inference)
                preds.append(pred)
                if self.llf:
                    prnd.append(x_p)
                c_idx += 1

        # ── Classification head ────────────────────────────────────────────────
        x = self.base_model.norm(x)
        x = self.base_model.forward_head(x)

        # 학습 중: auxiliary logit 포함해서 반환 → multi-head loss 계산
        preds_valid = [p for p in preds if p is not None]
        if preds_valid:
            return [x] + preds_valid
        return x

    # ── 파라미터 분리: backbone freeze 지원 ─────────────────────────────────────

    def backbone_parameters(self):
        return self.base_model.parameters()

    def cropr_parameters(self):
        return self.cropr.parameters()
