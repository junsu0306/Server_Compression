"""
EViT Token Pruning — Aries2 NPU 컴파일 호환 변형.

pruning/token_pruning.py의 기본 구현이 Aries2(qbcompiler) 컴파일에서 실패한
문제에 대한 우회책. 실제 컴파일 로그로 확인된 사실 관계(§14.6-1 참고):

  1. ScatterElements 미지원 → 해결됨, 검증 완료
     원인: _complement_idx()가 torch.scatter로 "top-k에 안 뽑힌 나머지 인덱스"를
     구하는 트릭을 씀.
     해결: 어차피 cls_attn 전체가 정렬 가능한 점수라, "안 뽑힌 나머지"는 그냥
     largest=False인 두 번째 TopK로 바로 나온다. scatter가 아예 필요 없다.
     (학습 시에도 항상 이걸 쓴다 — 결과는 동일하고 그래프만 더 단순해짐)
     실제 컴파일 로그에 ScatterElements가 더 이상 안 나타남 — 확인됨.

  2. GatherElements 미지원 → "plain Gather로 바꾸면 될까" 시도, 실패로 확인됨
     원인: torch.gather(dim=1, index=(B,k,C))가 배치마다 다른 인덱스를 한 번에
     처리하려고 element-wise gather(GatherElements)로 export됨.
     시도: batch=1 가정하고 torch.index_select(dim=0, index=1D)로 바꾸면 ONNX
     Gather(GatherElements 아님)로 export된다 — mode="index_select".
     결과: optypes에서 GatherElements는 사라졌지만, 실제 컴파일 로그에서
     plain Gather도 9개 전부 Unsupported(0%)로 나왔다. 즉 "GatherElements vs
     Gather" op 이름 문제가 아니라, **런타임에 계산된 인덱스로 하는 gather 자체를
     Aries2가 지원하지 않는 것으로 보인다.**

  3. ONNX 그래프 output이 여러 개로 노출되는 문제 → 애초에 문제가 아니었음
     원인 추정이 있었지만(dynamo exporter), 실제로는 qbcompiler의 HL 컴파일
     단계가 안 쓰이는 TopK 중간 출력을 자체적으로 dead-code로 정리해서 최종
     컴파일 그래프는 이미 output 1개였다 — dynamo=False는 불필요했던 것으로 보임.

**mode="onehot_matmul"** (신규, 미검증): 2번이 진짜 하드 블로커일 가능성이 높아,
gather 자체를 없애는 시도. `gather(src, idx)`는 수학적으로 one-hot 행렬곱과
동일하다:

    onehot[i,j] = 1 if j==idx[i] else 0   (Equal + Cast, N은 상수라 arange 고정)
    out = onehot @ src                     (MatMul — 실제 컴파일 로그에서 27개 전부
                                             100% Supported로 확인됨)

이미지마다 다른 토큰을 고르는 성질(idx가 런타임 값)은 그대로 유지된다 — gather를
없애는 게 아니라 "gather를 표현하는 방식"만 바꾸는 것. 리스크: 현재 그래프의
Equal/Cast는 전부 상수 입력이라 컴파일 시 const-fold되어 사라지므로, **런타임
값이 들어간 Equal/Cast가 실제로 컴파일되는지는 이 파일만으로는 검증되지 않는다**
— 최소 2-input(더미 idx를 그래프 input으로) toy 그래프로 별도 확인 필요.

전제: 이 파일의 모든 우회는 export 시점(batch=1)에만 켜진다
(set_npu_export_mode). 학습(batch>1) 중에는 기존 pruning/token_pruning.py와
수학적으로 동일한 batched torch.gather를 그대로 쓰므로, 이미 학습된 checkpoint를
재사용해서 export만 다시 하면 된다 — 재학습 불필요.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn

from pruning.token_pruning import _cls_attention_scores, apply_token_pruning


# ── batch=1 전용 gather 대체 구현 ───────────────────────────────────────────────

def _onehot_gather(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """dim=0 gather(src, idx)와 동등한 연산을 GatherElements/Gather 없이 구현.

    gather(src, idx)[i] == src[idx[i]]
    onehot[i,j] = 1 if j==idx[i] else 0  →  (onehot @ src)[i] == src[idx[i]]

    src: (N,) 또는 (N, C) — 배치 축 없는 단일 샘플
    idx: (k,) — 런타임 값(top-k 결과 등). idx가 그래프에서 상수로 접히지 않아야
         (즉 실제로 export 시 데이터 의존적이어야) 이 우회가 의미가 있다.
    반환: (k,) 또는 (k, C)
    """
    N = src.shape[0]
    arange_n = torch.arange(N, device=src.device)             # 상수
    onehot = (idx.unsqueeze(-1) == arange_n).to(src.dtype)    # (k, N) — Equal + Cast
    return onehot @ src                                        # (k,) 또는 (k, C) — MatMul


def _npu_select(src: torch.Tensor, idx: torch.Tensor, mode: str) -> torch.Tensor:
    """batch=1 gather(dim=0)의 NPU 호환 구현. mode로 전략 선택."""
    if mode == "onehot_matmul":
        return _onehot_gather(src, idx)
    if mode == "index_select":
        return src.index_select(0, idx)
    raise ValueError(f"알 수 없는 npu_mode: {mode!r} (index_select | onehot_matmul)")


# ── 패치된 Block.forward (NPU 호환 변형) ───────────────────────────────────────

def _evit_block_forward_npu(
    self,
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """token_pruning._evit_block_forward와 동일한 알고리즘, 다른 연산으로 구현.

    - complement(안 뽑힌 나머지 인덱스): scatter 대신 largest=False TopK (항상)
    - gather: self._evit_npu_export=True일 때만 batch=1 전용 경로(_npu_select,
      self._evit_npu_mode로 index_select/onehot_matmul 선택)로 전환.
      기본 False → 학습 중엔 기존과 동일한 batched torch.gather 사용.
    """
    if attn_mask is not None or is_causal:
        raise NotImplementedError(
            "EvitTokenPruner는 attn_mask/is_causal을 지원하지 않는다 — "
            "이 repo의 ViT 분류 학습에서는 사용되지 않는 경로다."
        )

    keep_rate = getattr(self, "_evit_keep_rate", 1.0)
    fuse_token = getattr(self, "_evit_fuse_token", True)
    npu_export = getattr(self, "_evit_npu_export", False)
    npu_mode = getattr(self, "_evit_npu_mode", "index_select")

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
                assert B == 1, "NPU export mode는 batch_size=1을 가정한다"
                x_kept = _npu_select(non_cls[0], idx[0], npu_mode).unsqueeze(0)
            else:
                index = idx.unsqueeze(-1).expand(-1, -1, C)
                x_kept = torch.gather(non_cls, dim=1, index=index)

            if fuse_token:
                # complement: scatter 대신 largest=False TopK (ScatterElements 회피)
                _, compl = torch.topk(cls_attn, n_patch - n_keep, dim=1, largest=False, sorted=False)

                if npu_export:
                    non_topk = _npu_select(non_cls[0], compl[0], npu_mode).unsqueeze(0)
                    non_topk_attn = _npu_select(cls_attn[0], compl[0], npu_mode).unsqueeze(0)
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


def set_npu_export_mode(model: nn.Module, enabled: bool = True, mode: str = "index_select") -> None:
    """ONNX export 직전에 호출 — batch=1 전용 NPU 우회 경로로 전환.

    mode:
        "index_select"  — 실측 결과 여전히 Gather 9개 전부 Unsupported로 확인됨
                           (§14.6-1). 하위호환 기본값으로만 남겨둠.
        "onehot_matmul" — gather를 Equal+Cast+MatMul로 대체. MatMul은 실측
                           100% Supported. Equal/Cast의 동적 입력 지원 여부는
                           미검증 (§14.6-1).

    학습 중에는 절대 켜면 안 된다 (batch>1이면 이 경로들이 잘못된 shape을
    만든다 — assert로 걸리긴 하지만 애초에 켤 이유가 없다).
    """
    for block in model.blocks:
        if getattr(block, "_evit_pruned", False):
            block._evit_npu_export = enabled
            block._evit_npu_mode = mode
