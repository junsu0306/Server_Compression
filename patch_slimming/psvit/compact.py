"""
Compact PS-ViT 모델.

search로 얻은 layer별 고정 keep_ids로 각 block에서 실제 token tensor 길이를 줄인다.

토큰 선택:
    select_mode="index_select" — 학습/평가(GPU). searched student와 수치 동일.
    select_mode="matmul"       — NPU export. 상수 선택행렬 P@x (Gather 없음).
NHWC 입력 옵션 — Mobilint calibration 파이프라인이 [B,224,224,3]을 넣기 때문.

CLS는 항상 local index 0: keep_ids가 정렬되어 있고 CLS_ID=0이 모든 keep에 포함되므로
가장 작은 값 → 첫 번째 위치. classifier는 x[:,0]을 계속 쓴다.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from .ids import as_long_ids, CLS_ID
from .model_utils import embed_tokens, head_from_tokens, get_shape, validate_vit
from .slim_block import SlimBlock


class CompactPSViT(nn.Module):
    """timm ViT weight를 공유하며 layer별 고정 mask로 토큰을 실제 축소하는 compact 모델."""

    def __init__(self, model: nn.Module, keep_ids: List[Tensor],
                 select_mode: str = "index_select", nhwc_input: bool = False,
                 npu_static_batch_one: bool = False):
        super().__init__()
        validate_vit(model)
        self.model = model                       # patch_embed / _pos_embed / norm / head 재사용
        self.shp = get_shape(model)
        L = self.shp.num_blocks
        assert len(keep_ids) == L, "keep_ids 길이 = block 수"
        self.nhwc_input = nhwc_input
        self.npu_static_batch_one = npu_static_batch_one
        self.select_mode = select_mode

        self.slim = nn.ModuleList(SlimBlock(model.blocks[l], select_mode) for l in range(L))

        # 각 block의 active(입력) / output(출력) global id
        all_ids = torch.arange(self.shp.num_global_tokens)
        self._active_ids: List[Tensor] = []
        for l in range(L):
            active = all_ids if l == 0 else as_long_ids(keep_ids[l - 1])
            self.register_buffer(f"active_{l}", active, persistent=False)
            self.register_buffer(f"keep_{l}", as_long_ids(keep_ids[l]), persistent=False)
            self._active_ids.append(active)

        # CLS가 최종 출력 local 0인지 확인 (keep_ids[L-1] 정렬 → CLS=0이 맨 앞)
        last_keep = as_long_ids(keep_ids[L - 1]).tolist()
        assert last_keep[0] == CLS_ID, "최종 block 출력의 첫 토큰이 CLS가 아님 (정렬/CLS 포함 확인)"

    def _active(self, l: int) -> Tensor:
        return getattr(self, f"active_{l}")

    def _keep(self, l: int) -> Tensor:
        return getattr(self, f"keep_{l}")

    def forward(self, images: Tensor) -> Tensor:
        if self.nhwc_input:
            images = images.permute(0, 3, 1, 2)          # [B,H,W,3] → [B,3,H,W]
        if self.npu_static_batch_one:
            # NPU export is fixed to B=1. Avoid timm _pos_embed's dynamic CLS
            # expansion (Equal/Where/Expand in ONNX) by concatenating [1,1,D]
            # cls_token directly with the fixed-batch patch sequence.
            x = self.model.patch_embed(images)
            x = torch.cat((self.model.cls_token, x), dim=1)
            x = self.model.pos_drop(x + self.model.pos_embed)
            if hasattr(self.model, "patch_drop"):
                x = self.model.patch_drop(x)
            if hasattr(self.model, "norm_pre"):
                x = self.model.norm_pre(x)
        else:
            x = embed_tokens(self.model, images)         # [B, N, D]
        for l, sb in enumerate(self.slim):
            out = sb(x, self._active(l).to(x.device), self._keep(l).to(x.device))
            x = out.x
        if self.npu_static_batch_one:
            # Final architecture output is CLS-only [1,1,D]. squeeze(1) exports
            # as constant-axis Squeeze, unlike x[:, 0] which exports as Gather.
            feat = self.model.norm(x).squeeze(1)
            if hasattr(self.model, "fc_norm"):
                feat = self.model.fc_norm(feat)
            if hasattr(self.model, "head_drop"):
                feat = self.model.head_drop(feat)
            return self.model.head(feat)
        return head_from_tokens(self.model, x, cls_local_index=0)

    @torch.no_grad()
    def bake_for_export(self, device, dtype=torch.float32) -> "CompactPSViT":
        """모든 block을 matmul 선택 모드로 전환하고 상수 선택행렬 precompute."""
        for l, sb in enumerate(self.slim):
            sb.select_mode = "matmul"
            sb.bake_selection(self._active(l).to(device), self._keep(l).to(device), device, dtype)
        self.select_mode = "matmul"
        return self

    def token_schedule(self) -> List[int]:
        return [self.shp.num_global_tokens] + [int(self._keep(l).numel())
                                               for l in range(self.shp.num_blocks)]
