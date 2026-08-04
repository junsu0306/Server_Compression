"""
§12 / §14 — Compact PS-ViT 모델.

search로 얻은 layer별 고정 keep_ids로 각 block에서 실제 token tensor 길이를 줄인다
(§12.1: zero-mask만 적용하는 게 아니라 [B,N_in,D]→[B,N_out,D]로 실제 축소).

토큰 선택 (SPEC §14.0 실측 반영):
    select_mode="index_select" — 학습/평가(GPU). searched student와 수치 동일(§16.9).
    select_mode="matmul"       — NPU export. 상수 선택행렬 P@x (Gather 없음).
NHWC 입력 옵션 — Mobilint calibration 파이프라인이 [B,224,224,3]을 넣기 때문(§14.0).

CLS는 항상 local index 0: keep_ids가 정렬되어 있고 CLS_ID=0이 모든 keep에 포함되므로
가장 작은 값 → 첫 번째 위치. classifier는 x[:,0]을 계속 쓴다 (§12.3).
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
                 select_mode: str = "index_select", nhwc_input: bool = False):
        super().__init__()
        validate_vit(model)
        self.model = model                       # patch_embed / _pos_embed / norm / head 재사용
        self.shp = get_shape(model)
        L = self.shp.num_blocks
        assert len(keep_ids) == L, "keep_ids 길이 = block 수"
        self.nhwc_input = nhwc_input
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
        x = embed_tokens(self.model, images)             # [B, N, D]
        for l, sb in enumerate(self.slim):
            out = sb(x, self._active(l).to(x.device), self._keep(l).to(x.device))
            x = out.x
        return head_from_tokens(self.model, x, cls_local_index=0)

    @torch.no_grad()
    def bake_for_export(self, device, dtype=torch.float32) -> "CompactPSViT":
        """모든 block을 matmul 선택 모드로 전환하고 상수 선택행렬 precompute (§14.0)."""
        for l, sb in enumerate(self.slim):
            sb.select_mode = "matmul"
            sb.bake_selection(self._active(l).to(device), self._keep(l).to(device), device, dtype)
        self.select_mode = "matmul"
        return self

    def token_schedule(self) -> List[int]:
        return [self.shp.num_global_tokens] + [int(self._keep(l).numel())
                                               for l in range(self.shp.num_blocks)]


def build_compact_from_spec(arch_spec, state_dict: dict, device,
                            select_mode: str = "index_select", nhwc_input: bool = False):
    """architecture spec + full-model state_dict → CompactPSViT.

    Patch Slimming은 weight shape을 안 바꾸므로 compact 모델 = 원본 timm 모델 weight +
    keep mask. timm 모델을 만들어 state_dict를 로드한 뒤 mask로 감싼다.
    """
    import timm
    num_classes = int(state_dict["head.weight"].shape[0])
    model = timm.create_model(arch_spec.model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    keep_ids = arch_spec.keep_ids_list(device=device)
    return CompactPSViT(model, keep_ids, select_mode=select_mode, nhwc_input=nhwc_input).to(device)
