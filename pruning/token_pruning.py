"""
EViT 스타일 Token Pruning
레퍼런스: Liang et al., "Not All Patches are What You Need: Expediting
          Vision Transformers via Token Reorganizations", ICLR 2022
          https://github.com/youweiliang/evit

channel pruning(FFN width, pruning/vit_pruning.py)과 직교하는 축 —
이쪽은 sequence 차원(패치 토큰 개수)을 줄인다. reduce.py로 이미 물리적으로
축소된 Dense 모델 위에 이어서 적용하는 2단계(Stage 2) 구성을 전제로 한다.

핵심 아이디어
    선택된 일부 block에서: CLS 토큰이 각 patch 토큰에 준 attention score를
    기준으로 상위 keep_rate만 남기고, 버려지는 토큰들은 attention 가중합으로
    하나의 "fused token"에 합쳐서 함께 유지한다 (완전 폐기 대비 정보 손실 최소화).

    keep_rate가 고정 비율이고 패치 개수 N도 고정(입력 해상도 고정)이므로,
    남는 토큰 개수 k는 모든 입력에 대해 동일한 "컴파일타임 상수"다.
    즉 그래프의 텐서 shape은 완전히 정적이고, 어떤 토큰이 선택되는지(index 값)만
    입력에 따라 달라진다 — ONNX/NPU 컴파일러 입장에서 DynamicViT류(샘플마다
    남는 토큰 "개수" 자체가 다름)보다 훨씬 우호적인 조건이다.

왜 timm Attention.forward를 직접 건드리지 않는가
    timm 1.0.x의 Attention은 기본적으로 fused_attn=True 로
    F.scaled_dot_product_attention 을 사용하며 attention 행렬을 material화하지
    않는다. fused_attn을 강제로 끄고 원본 forward를 재구현하면, 구현 실수로
    실제 attention 출력(x)이 원본과 미묘하게 어긋날 위험이 있다 — 이건 그냥
    사고다.
    대신 CLS→patch attention score만 별도로 계산한다: block에 이미 존재하는
    attn.qkv Linear를 한 번 더 통과시켜 q, k만 뽑고 CLS row만 계산한다
    (O(N) 크기의 작은 matmul 하나 추가, 전체 attention O(N^2) 대비 무시할
    수준의 오버헤드). 이렇게 하면 attn.forward는 원본 그대로 유지되어
    fused_attn 경로/정확도가 전혀 손상되지 않는다.

전제 조건
    - 표준 단일 CLS 토큰 ViT (timm vit_*_patch16_224 등). distilled(dist_token)
      모델이나 no_embed_class 모델은 지원하지 않는다 (assert로 조기 실패).
    - model.blocks[i].attn 에 qkv / q_norm / k_norm / num_heads 속성이 있어야
      한다 (timm 표준 Attention 구조).

사용법
    token_pruner = EvitTokenPruner(
        model,                      # reduce.py로 이미 축소된 Dense 모델
        base_keep_rate=0.7,
        warmup_epochs=5,
        ramp_epochs=15,
    )

    for epoch in range(epochs):
        token_pruner.set_epoch(epoch)   # progressive keep_rate 스케줄
        for samples, targets in loader:
            output = model(samples)     # 내부적으로 알아서 토큰이 줄어듦
            ...

    metrics = token_pruner.log_info(model)  # WandB 로깅용
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn


# ── complement_idx ────────────────────────────────────────────────────────────

def _complement_idx(idx: torch.Tensor, dim: int) -> torch.Tensor:
    """idx(선택된 인덱스, shape=(B, k))의 여집합 인덱스를 (B, dim-k)로 반환.

    scatter로 선택된 위치를 0으로 지운 뒤 오름차순 정렬 → 마지막 (dim-k)개를
    취하면 선택되지 않은 인덱스만 남는다 (선택된 위치가 우연히 값 0을 가지고
    있었어도 정렬 후 앞쪽에 몰리므로 결과에 영향 없음).
    """
    B = idx.shape[0]
    a = torch.arange(dim, device=idx.device).unsqueeze(0).expand(B, -1)
    masked = torch.scatter(a, 1, idx, 0)
    compl, _ = torch.sort(masked, dim=1, descending=False)
    return compl[:, idx.shape[1]:]


# ── CLS attention score (attn.forward를 건드리지 않고 별도 계산) ──────────────

def _cls_attention_scores(attn: nn.Module, x_norm: torch.Tensor) -> torch.Tensor:
    """block.norm1(x) 입력에 대해 CLS→patch attention score를 계산.

    attn.qkv Linear를 한 번 더 통과시켜 q, k만 뽑아 CLS row만 계산한다.
    attn.forward 본체(및 fused_attn 경로)는 전혀 건드리지 않는다.

    반환: (B, N-1)  — N-1 = patch 토큰 개수 (CLS 제외), head 평균 softmax score.
    """
    B, N, C = x_norm.shape
    num_heads = attn.num_heads
    head_dim = getattr(attn, "head_dim", C // num_heads)
    scale = getattr(attn, "scale", head_dim ** -0.5)

    qkv = attn.qkv(x_norm).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)                      # each (B, H, N, head_dim)
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)
    if q_norm is not None:
        q = q_norm(q)
    if k_norm is not None:
        k = k_norm(k)

    q_cls = q[:, :, 0:1, :]                                    # (B, H, 1, head_dim)
    cls_attn = (q_cls @ k.transpose(-2, -1)) * scale           # (B, H, 1, N)
    cls_attn = cls_attn.softmax(dim=-1)[:, :, 0, 1:]           # (B, H, N-1) — CLS 자신 제외
    return cls_attn.mean(dim=1)                                # (B, N-1) — head 평균


# ── 패치된 Block.forward ───────────────────────────────────────────────────────

def _evit_block_forward(
    self,
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """timm Block.forward를 대체하는 bound method.

    표준 forward와 동일하게 attn → (선택적 token pruning) → mlp 순서로 진행한다.
    self.attn(...) 호출 자체는 원본 그대로이므로 attention 출력은 정확히 동일.
    keep_rate=1.0(또는 미설정)이면 원본 forward와 완전히 동일하게 동작한다.
    """
    keep_rate = getattr(self, "_evit_keep_rate", 1.0)
    fuse_token = getattr(self, "_evit_fuse_token", True)

    x_norm = self.norm1(x)
    attn_out = self.attn(x_norm, attn_mask=attn_mask, is_causal=is_causal)
    x = x + self.drop_path1(self.ls1(attn_out))

    if keep_rate < 1.0:
        B, N, C = x.shape
        n_patch = N - 1
        n_keep = max(1, math.ceil(n_patch * keep_rate))

        if n_keep < n_patch:
            cls_attn = _cls_attention_scores(self.attn, x_norm)   # (B, n_patch)
            _, idx = torch.topk(cls_attn, n_keep, dim=1, largest=True, sorted=True)
            index = idx.unsqueeze(-1).expand(-1, -1, C)

            non_cls = x[:, 1:]
            x_kept = torch.gather(non_cls, dim=1, index=index)

            if fuse_token:
                compl = _complement_idx(idx, n_patch)                       # (B, n_patch-n_keep)
                compl_idx = compl.unsqueeze(-1).expand(-1, -1, C)
                non_topk = torch.gather(non_cls, dim=1, index=compl_idx)
                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # (B, n_patch-n_keep)
                extra_token = (non_topk * non_topk_attn.unsqueeze(-1)).sum(dim=1, keepdim=True)
                x = torch.cat([x[:, 0:1], x_kept, extra_token], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_kept], dim=1)

    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x


# ── 모델 검증 & 패치 적용 (stateless) ──────────────────────────────────────────

def _validate_model(model: nn.Module) -> None:
    if getattr(model, "dist_token", None) is not None:
        raise NotImplementedError(
            "distilled ViT(dist_token 존재)는 token pruning 미지원. "
            "표준 단일 CLS 토큰 ViT만 지원한다."
        )
    if not hasattr(model, "cls_token"):
        raise NotImplementedError(
            "model.cls_token이 없다 — no_embed_class 변형이거나 표준 timm ViT가 아님."
        )
    for i, block in enumerate(model.blocks):
        for attr in ("qkv", "num_heads"):
            if not hasattr(block.attn, attr):
                raise RuntimeError(
                    f"block {i}: attn.{attr} 속성이 없다 — timm 버전이 달라 "
                    f"Attention 내부 구조가 예상과 다를 수 있다. "
                    f"pruning/token_pruning.py의 _cls_attention_scores()를 "
                    f"현재 timm 버전에 맞게 확인해야 한다."
                )


def default_prune_layers(depth: int, n_layers: int = 3) -> List[int]:
    """EViT 논문 기본값과 동일한 방식: depth를 (n_layers+1) 등분한 지점.

    depth=12, n_layers=3 → [3, 6, 9]  (0-indexed block 인덱스)
    """
    return sorted({round(depth * i / (n_layers + 1)) for i in range(1, n_layers + 1)})


def apply_token_pruning(
    model: nn.Module,
    prune_layers: List[int],
    base_keep_rate: float = 0.7,
    fuse_token: bool = True,
) -> nn.Module:
    """model.blocks[i]의 forward를 EViT token pruning 버전으로 교체 (in-place, stateless).

    학습 중 progressive 스케줄이 필요하면 EvitTokenPruner를 쓰고,
    eval/export 시 고정된 keep_rate로 그래프를 재현할 때는 이 함수를 직접 호출한다.

    checkpoint에서 재현할 때:
        model = timm.create_model(model_name, pretrained=False)
        apply_reduced_config(model, mlp_dims)              # Stage 1 축소 구조
        apply_token_pruning(model, prune_layers, base_keep_rate, fuse_token)
        model.load_state_dict(state_dict)
    """
    _validate_model(model)
    depth = len(model.blocks)
    for i in prune_layers:
        if not (0 <= i < depth):
            raise ValueError(f"prune_layers 인덱스 {i}가 block 개수({depth}) 범위를 벗어남")

    for i, block in enumerate(model.blocks):
        if i in prune_layers:
            block.forward = types.MethodType(_evit_block_forward, block)
            block._evit_keep_rate = base_keep_rate
            block._evit_fuse_token = fuse_token
            block._evit_pruned = True
        else:
            block._evit_pruned = False

    return model


# ── EvitTokenPruner (progressive 스케줄 컨트롤러) ──────────────────────────────

@dataclass
class EvitTokenPruner:
    """EViT token pruning 학습 컨트롤러.

    ViTPruner(channel pruning)와 달리 weight를 마스킹하지 않는다 — 순수하게
    forward 시점의 시퀀스 길이만 바꾸므로 optimizer.step() 이후 호출할 apply()가
    없다. 매 epoch 시작 시 set_epoch()만 호출하면 된다.

    Args:
        model:           reduce.py로 축소된 Dense 모델 (Stage 2 입력)
        prune_layers:    token pruning을 적용할 block 인덱스 목록.
                         None이면 default_prune_layers(depth)로 자동 설정.
        base_keep_rate:  목표 keep_rate (예: 0.7 = patch 토큰의 70% 유지)
        fuse_token:      버려지는 토큰을 attention 가중합으로 fused token에 합칠지 여부
        warmup_epochs:   pruning 없이 정상 학습할 epoch 수 (keep_rate=1.0 유지)
        ramp_epochs:     keep_rate가 1.0 → base_keep_rate로 점진 감소하는 epoch 수
    """

    model: nn.Module
    base_keep_rate: float = 0.7
    fuse_token: bool = True
    prune_layers: Optional[List[int]] = None
    warmup_epochs: int = 0
    ramp_epochs: int = 0
    _current_epoch: int = field(default=0, init=False)
    _keep_rate: float = field(default=1.0, init=False)
    _mirrors: List[nn.Module] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        depth = len(self.model.blocks)
        if self.prune_layers is None:
            self.prune_layers = default_prune_layers(depth)

        apply_token_pruning(
            self.model, self.prune_layers, base_keep_rate=1.0, fuse_token=self.fuse_token
        )
        progressive = self.warmup_epochs > 0 or self.ramp_epochs > 0
        self._keep_rate = 1.0 if progressive else self.base_keep_rate
        self._set_block_keep_rate(self._keep_rate)

        print(
            f"[EvitTokenPruner] prune_layers={self.prune_layers}  "
            f"base_keep_rate={self.base_keep_rate}  fuse_token={self.fuse_token}"
        )
        if progressive:
            print(
                f"[EvitTokenPruner] progressive=ON  "
                f"warmup={self.warmup_epochs}  ramp={self.ramp_epochs}  "
                f"(epoch {self.warmup_epochs}~{self.warmup_epochs + self.ramp_epochs}"
                f"에서 keep_rate 1.0 → {self.base_keep_rate} 점진 감소)"
            )

    # ── 내부 ────────────────────────────────────────────────────────────────────

    def _set_block_keep_rate(self, keep_rate: float) -> None:
        for m in (self.model, *self._mirrors):
            for block in m.blocks:
                if getattr(block, "_evit_pruned", False):
                    block._evit_keep_rate = keep_rate

    def _scheduled_keep_rate(self, epoch: int) -> float:
        """Zhu & Gupta cubic ease-out — ViTPruner._scheduled_sparsity와 동일 형태.

        1.0(pruning 없음)에서 시작해 base_keep_rate까지 점진적으로 내려간다.
        """
        if self.ramp_epochs == 0 and self.warmup_epochs == 0:
            return self.base_keep_rate
        if epoch < self.warmup_epochs:
            return 1.0
        ramp_end = self.warmup_epochs + self.ramp_epochs
        if epoch >= ramp_end:
            return self.base_keep_rate
        progress = (epoch - self.warmup_epochs) / max(self.ramp_epochs, 1)
        drop = 1.0 - self.base_keep_rate
        return 1.0 - drop * (1.0 - (1.0 - progress) ** 3)

    # ── 공개 API ────────────────────────────────────────────────────────────────

    def attach_mirror(self, mirror_model: nn.Module) -> None:
        """model_ema.module 등 별도 인스턴스에도 동일한 patch를 적용하고 동기화 대상으로 등록.

        ModelEmaV2는 파라미터 텐서만 running-average로 갱신할 뿐, block.forward
        patch나 _evit_keep_rate 같은 (state_dict에 잡히지 않는) 인스턴스 속성은
        전파하지 않는다. 그래서 model_ema.module은 self.model과 별개로 patch를
        적용받아야 하고, 이후 set_epoch()가 호출될 때마다 keep_rate도 함께
        갱신되어야 한다 — 안 하면 EMA 모델은 evaluate() 때 token pruning이
        전혀 적용되지 않은 채로 평가되어 val/top1이 실제와 다르게 나온다.
        """
        apply_token_pruning(
            mirror_model, self.prune_layers,
            base_keep_rate=self._keep_rate, fuse_token=self.fuse_token,
        )
        self._mirrors.append(mirror_model)

    def set_epoch(self, epoch: int) -> None:
        """에포크 시작 전에 학습 스크립트에서 호출."""
        self._current_epoch = epoch
        new_kr = self._scheduled_keep_rate(epoch)
        if abs(new_kr - self._keep_rate) > 1e-7:
            old_kr = self._keep_rate
            self._keep_rate = new_kr
            self._set_block_keep_rate(new_kr)
            print(
                f"[EvitTokenPruner] epoch={epoch}  "
                f"keep_rate: {old_kr:.4f} → {new_kr:.4f}"
            )

    @torch.no_grad()
    def log_info(self, model: Optional[nn.Module] = None) -> dict:
        """WandB 로깅용 — block별 살아남는 토큰 개수(추정)와 현재 keep_rate."""
        m = model if model is not None else self.model
        depth = len(m.blocks)
        n_patch = None
        # patch_embed의 grid_size가 있으면 정확한 N을 알 수 있다.
        pe = getattr(m, "patch_embed", None)
        if pe is not None and hasattr(pe, "num_patches"):
            n_patch = pe.num_patches

        result: dict = {
            "token_pruning/keep_rate": self._keep_rate,
            "token_pruning/prune_layers": self.prune_layers,
        }
        if n_patch is not None:
            cur = n_patch
            for i in range(depth):
                block = m.blocks[i]
                if getattr(block, "_evit_pruned", False) and block._evit_keep_rate < 1.0:
                    n_patch_cur = cur - 1  # CLS(+fused, 있었다면) 제외한 patch 수
                    n_keep = max(1, math.ceil(n_patch_cur * block._evit_keep_rate))
                    cur = 1 + n_keep + (1 if self.fuse_token else 0)  # CLS + kept (+ fused)
                result[f"token_pruning/tokens_after_block/{i}"] = cur
            result["token_pruning/tokens_final"] = cur
            result["token_pruning/tokens_original"] = n_patch + 1
        return result

    def config(self) -> dict:
        """checkpoint에 저장할 구조 정보. eval/export 시 apply_token_pruning()에 그대로 전달."""
        return {
            "prune_layers": list(self.prune_layers),
            "base_keep_rate": self.base_keep_rate,
            "fuse_token": self.fuse_token,
        }

    def state_dict(self) -> dict:
        return {
            "current_epoch": self._current_epoch,
            "keep_rate": self._keep_rate,
            "base_keep_rate": self.base_keep_rate,
            "fuse_token": self.fuse_token,
            "prune_layers": list(self.prune_layers),
            "warmup_epochs": self.warmup_epochs,
            "ramp_epochs": self.ramp_epochs,
        }

    def load_state_dict(self, state: dict) -> None:
        self._current_epoch = state.get("current_epoch", 0)
        self._keep_rate = state["keep_rate"]
        self.base_keep_rate = state.get("base_keep_rate", self.base_keep_rate)
        self.fuse_token = state.get("fuse_token", self.fuse_token)
        self.prune_layers = state.get("prune_layers", self.prune_layers)
        self.warmup_epochs = state.get("warmup_epochs", self.warmup_epochs)
        self.ramp_epochs = state.get("ramp_epochs", self.ramp_epochs)
        self._set_block_keep_rate(self._keep_rate)
