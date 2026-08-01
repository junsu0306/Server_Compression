"""
EViT Token Pruning — Aries2 NPU 컴파일 호환 변형.

pruning/token_pruning.py의 기본 구현이 Aries2(qbcompiler) 컴파일에서 실패한
3가지 문제에 대한 우회책:

  1. ScatterElements 미지원
     원인: _complement_idx()가 torch.scatter로 "top-k에 안 뽑힌 나머지 인덱스"를
     구하는 트릭을 씀.
     해결: 어차피 cls_attn 전체가 정렬 가능한 점수라, "안 뽑힌 나머지"는 그냥
     largest=False인 두 번째 TopK로 바로 나온다. scatter가 아예 필요 없다.
     (학습 시에도 항상 이걸 쓴다 — 결과는 동일하고 그래프만 더 단순해짐)

  2. GatherElements 미지원
     원인: torch.gather(dim=1, index=(B,k,C))가 배치마다 다른 인덱스를 한 번에
     처리하려고 element-wise gather(GatherElements)로 export됨.
     해결: NPU 추론은 어차피 거의 항상 batch=1이므로, 배치 축을 떼어내고
     torch.index_select(dim=0, index=1D)를 쓰면 plain Gather로 export된다.
     단, 이건 batch=1을 강제로 가정하므로 학습(batch>1) 중에는 쓸 수 없다 —
     set_npu_export_mode()로 export 직전에만 활성화한다.

  3. ONNX export 시 그래프 output이 여러 개로 노출되는 문제
     원인 추정: torch.onnx.export의 dynamo 기반 exporter가 TopK처럼 데이터
     의존적 shape을 가진 중간 연산을 안전하게 추적하려고 부수적으로 output에
     노출시키는 것으로 보임 — 확실하지 않아 export_onnx.py 쪽에서
     dynamo=False(레거시 TorchScript 기반 exporter)로 우회를 시도한다.
     (이 파일과는 무관 — export_onnx.py --npu-safe 참고)

주의: 이 파일은 검증되지 않았다. 서버에서 실제로 학습 → export → qbcompiler까지
돌려봐야 1, 2번이 실제로 문제를 해결하는지 확인 가능하다. 기존
pruning/token_pruning.py, train_token_pruning.py의 기본 동작(forward_fn
기본값)은 전혀 건드리지 않으므로 이미 진행 중인 다른 3개 run에는 영향 없다.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

from pruning.token_pruning import _cls_attention_scores, apply_token_pruning


# ── 패치된 Block.forward (NPU 호환 변형) ───────────────────────────────────────

def _evit_block_forward_npu(
    self,
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """token_pruning._evit_block_forward와 동일한 알고리즘, 다른 연산으로 구현.

    - complement(안 뽑힌 나머지 인덱스): scatter 대신 largest=False TopK (항상)
    - gather: self._evit_npu_export=True일 때만 batch=1 index_select로 전환
      (기본 False → 학습 중엔 기존과 동일한 batched gather 사용)
    """
    if attn_mask is not None or is_causal:
        raise NotImplementedError(
            "EvitTokenPruner는 attn_mask/is_causal을 지원하지 않는다 — "
            "이 repo의 ViT 분류 학습에서는 사용되지 않는 경로다."
        )

    keep_rate = getattr(self, "_evit_keep_rate", 1.0)
    fuse_token = getattr(self, "_evit_fuse_token", True)
    npu_export = getattr(self, "_evit_npu_export", False)

    x_norm = self.norm1(x)
    attn_out = self.attn(x_norm)
    x = x + self.drop_path1(self.ls1(attn_out))

    if keep_rate < 1.0:
        B, N, C = x.shape
        n_patch = N - 1
        n_keep = max(1, math.ceil(n_patch * keep_rate))

        if n_keep < n_patch:
            cls_attn = _cls_attention_scores(self.attn, x_norm)   # (B, n_patch)

            _, idx = torch.topk(cls_attn, n_keep, dim=1, largest=True, sorted=True)
            non_cls = x[:, 1:]

            if npu_export:
                # ── batch=1 전용: index_select → ONNX Gather (GatherElements 아님) ──
                assert B == 1, "NPU export mode는 batch_size=1을 가정한다"
                x_kept = non_cls[0].index_select(0, idx[0]).unsqueeze(0)
            else:
                index = idx.unsqueeze(-1).expand(-1, -1, C)
                x_kept = torch.gather(non_cls, dim=1, index=index)

            if fuse_token:
                # complement: scatter 대신 largest=False TopK (ScatterElements 회피)
                _, compl = torch.topk(cls_attn, n_patch - n_keep, dim=1, largest=False, sorted=False)

                if npu_export:
                    assert B == 1, "NPU export mode는 batch_size=1을 가정한다"
                    non_topk = non_cls[0].index_select(0, compl[0]).unsqueeze(0)
                    non_topk_attn = cls_attn[0].index_select(0, compl[0]).unsqueeze(0)
                else:
                    compl_idx = compl.unsqueeze(-1).expand(-1, -1, C)
                    non_topk = torch.gather(non_cls, dim=1, index=compl_idx)
                    non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)

                extra_token = (non_topk * non_topk_attn.unsqueeze(-1)).sum(dim=1, keepdim=True)
                x = torch.cat([x[:, 0:1], x_kept, extra_token], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_kept], dim=1)

    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


# ── 공개 API ───────────────────────────────────────────────────────────────────

def apply_token_pruning_npu(
    model: nn.Module,
    prune_layers: List[int],
    base_keep_rate: float = 0.7,
    fuse_token: bool = True,
) -> nn.Module:
    """pruning.token_pruning.apply_token_pruning()과 동일하지만 NPU 호환 forward를 바인딩."""
    return apply_token_pruning(
        model, prune_layers, base_keep_rate=base_keep_rate, fuse_token=fuse_token,
        forward_fn=_evit_block_forward_npu,
    )


def set_npu_export_mode(model: nn.Module, enabled: bool = True) -> None:
    """ONNX export 직전에 호출 — batch=1 index_select 경로로 전환.

    학습 중에는 절대 켜면 안 된다 (batch>1이면 index_select 경로가 잘못된 shape을
    만든다 — assert로 걸리긴 하지만 애초에 켤 이유가 없다).
    """
    for block in model.blocks:
        if getattr(block, "_evit_pruned", False):
            block._evit_npu_export = enabled
