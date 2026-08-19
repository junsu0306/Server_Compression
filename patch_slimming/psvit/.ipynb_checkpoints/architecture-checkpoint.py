"""
§4 — Architecture specification 직렬화.

search 결과(layer별 고정 keep global IDs)를 weight와 별도 JSON으로 저장/복원한다.
compact 모델은 이 spec으로 재현한다 (SPEC §4.1, §12).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import List

import torch
from torch import Tensor

from .ids import as_long_ids, validate_nested_masks


@dataclass
class LayerSpec:
    block_index: int
    keep_global_ids: List[int]
    num_output_tokens: int
    accepted_error: float


@dataclass
class ArchitectureSpec:
    method: str
    model_name: str
    num_blocks: int
    num_global_tokens: int
    num_prefix_tokens: int
    classifier_token_ids: List[int]
    epsilon: float
    error_metric: str
    search_step: int
    score_mode: str
    layers: List[LayerSpec] = field(default_factory=list)

    # ── 직렬화 ─────────────────────────────────────────────────────────────────
    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def from_json(path: str) -> "ArchitectureSpec":
        with open(path) as f:
            d = json.load(f)
        layers = [LayerSpec(**l) for l in d.pop("layers")]
        return ArchitectureSpec(layers=layers, **d)

    # ── 편의 ───────────────────────────────────────────────────────────────────
    def keep_ids_list(self, device=None) -> List[Tensor]:
        """block_index 순서로 정렬된 keep_ids 텐서 리스트 [L]."""
        by_idx = {l.block_index: l for l in self.layers}
        out = []
        for i in range(self.num_blocks):
            ids = as_long_ids(by_idx[i].keep_global_ids)
            out.append(ids.to(device) if device is not None else ids)
        return out

    def validate(self) -> None:
        keep = self.keep_ids_list()
        validate_nested_masks(keep)

    def token_schedule(self) -> List[int]:
        """[N, out0, out1, ..., out_{L-1}] — 입력 N에서 layer별 출력 토큰 수."""
        by_idx = {l.block_index: l.num_output_tokens for l in self.layers}
        return [self.num_global_tokens] + [by_idx[i] for i in range(self.num_blocks)]


def build_spec(model_name: str, shp, keep_ids: List[Tensor], accepted_errors: List[float],
               epsilon: float, error_metric: str, search_step: int, score_mode: str,
               method: str = "patch_slimming_static") -> ArchitectureSpec:
    layers = []
    for i in range(shp.num_blocks):
        ids = as_long_ids(keep_ids[i]).tolist()
        layers.append(LayerSpec(
            block_index=i, keep_global_ids=ids, num_output_tokens=len(ids),
            accepted_error=float(accepted_errors[i]),
        ))
    spec = ArchitectureSpec(
        method=method, model_name=model_name, num_blocks=shp.num_blocks,
        num_global_tokens=shp.num_global_tokens, num_prefix_tokens=shp.num_prefix_tokens,
        classifier_token_ids=[0], epsilon=epsilon, error_metric=error_metric,
        search_step=search_step, score_mode=score_mode, layers=layers,
    )
    spec.validate()
    return spec
