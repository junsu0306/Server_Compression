"""
§7 — Significance score (path-energy dynamic programming).

논문 Theorem 1 (SPEC §7.1):
    s_{t,i} = Σ_h || A_t^h[:,i] U_t^h[i,:] ||_F^2
    A_t^h = Π_{l=t+1..L} diag(m_l) P_l^h      (downstream influence)
    U_t^h = P_t^h |Z_{t-1}|                    (target layer feature energy)

논문은 head-path 집계 구현을 안 준다 (SPEC §7.2). [DECISION]: head-path energy를
동적 계획법으로 계산하고, 작은 case에서 brute-force head-path 열거와 일치함을
검증한다 (SPEC §7.6, §16.5). 이 test 통과 전에는 전체 search를 시작하지 않는다.

핵심 DP (SPEC §7.5):
    G = I_N
    for P in reversed(downstream P_k [B,H,N,N]):
        G = (1/H) Σ_h P_h^T G P_h
    downstream_energy_i = G_ii
    U = target_attn_global @ |Z_{t-1}|
    current_energy_i = (1/H) Σ_h Σ_d U^h[i,d]^2
    s_i = clamp(downstream_energy_i, ≥0) * current_energy_i
"""

from __future__ import annotations

import itertools
from typing import List, Optional

import torch
from torch import Tensor

from .ids import CLS_ID, as_long_ids
from .model_utils import embed_tokens, get_shape
from .slim_block import SlimBlock


# ── §7.3 local attention → global NxN scatter ──────────────────────────────────

def scatter_attention_to_global(
    attn_local: Tensor,        # [B,H,N_out,N_in]
    output_global_ids: Tensor,  # [N_out]
    input_global_ids: Tensor,   # [N_in]
    num_global_tokens: int,     # N
) -> Tensor:                    # [B,H,N,N]
    """선택되지 않은 row/column은 0 (mask 효과가 global matrix에 이미 반영)."""
    B, H, N_out, N_in = attn_local.shape
    out_ids = as_long_ids(output_global_ids).to(attn_local.device)
    in_ids = as_long_ids(input_global_ids).to(attn_local.device)
    Pg = attn_local.new_zeros(B, H, num_global_tokens, num_global_tokens)
    Pg[:, :, out_ids.unsqueeze(1), in_ids.unsqueeze(0)] = attn_local
    return Pg


# ── §7.4 hybrid network forward (blocks<t full, block t full-out, blocks>t slim) ─

@torch.no_grad()
def hybrid_forward_for_target(model, images: Tensor, t: int, keep_ids: List[Tensor]):
    """target layer t의 score 계산에 필요한 값 산출.

    반환:
        z_prev              : block t 입력 [B,N,D] (전체 토큰, = Z_{t-1})
        target_attn_global  : block t의 full attention [B,H,N,N]
        downstream_globals  : [P_{t+1}, ..., P_{L-1}] 각 [B,H,N,N] (global scatter)
    """
    shp = get_shape(model)
    N = shp.num_global_tokens
    L = shp.num_blocks
    device = images.device
    all_ids = torch.arange(N, device=device)

    x = embed_tokens(model, images)
    for l in range(t):                                   # blocks < t: 원본 full forward
        x = model.blocks[l](x)
    z_prev = x                                           # [B,N,D]

    # block t: 전체 출력(= all tokens), full attention 캡처
    out_t = SlimBlock(model.blocks[t])(x, all_ids, all_ids, return_attention=True)
    target_attn_global = out_t.attention_probs           # [B,H,N,N]
    x = out_t.x                                          # [B,N,D]
    active = all_ids

    downstream: List[Tensor] = []
    for k in range(t + 1, L):
        keep_k = as_long_ids(keep_ids[k]).to(device)
        out_k = SlimBlock(model.blocks[k])(x, active, keep_k, return_attention=True)
        Pg = scatter_attention_to_global(out_k.attention_probs, keep_k, active, N)
        downstream.append(Pg)
        x = out_k.x
        active = keep_k
    return z_prev, target_attn_global, downstream


# ── §7.5 DP score ──────────────────────────────────────────────────────────────

def _downstream_energy_dp(downstream: List[Tensor]) -> Tensor:
    """DP로 downstream influence의 대각(G_ii) 계산. 반환 [B,N]."""
    if len(downstream) == 0:
        # target이 마지막 layer 직전인 특수 case: downstream 없음 → G=I → 대각=1
        B, H, N, _ = 1, 1, 1, 1  # placeholder; 실제로는 아래 shape에서 결정
        raise ValueError("downstream이 비었다 — hybrid_forward가 target=L-1을 넘겨줬는지 확인")
    B, H, N, _ = downstream[0].shape
    G = torch.eye(N, device=downstream[0].device, dtype=downstream[0].dtype)
    G = G.unsqueeze(0).expand(B, -1, -1).clone()         # [B,N,N]
    for P in reversed(downstream):                       # P: [B,H,N,N]
        GP = torch.matmul(G.unsqueeze(1), P)             # [B,H,N,N]
        PtGP = torch.matmul(P.transpose(-2, -1), GP)     # [B,H,N,N]
        G = PtGP.mean(dim=1)                             # head 평균 → [B,N,N]
    return torch.diagonal(G, dim1=-2, dim2=-1)           # [B,N]


def compute_sample_scores(
    z_prev: Tensor,                  # [B,N,D]
    target_attn_global: Tensor,      # [B,H,N,N]
    downstream_globals: List[Tensor],  # list of [B,H,N,N]
) -> Tensor:                         # [B,N]
    """SPEC §7.5 그대로."""
    downstream_energy = _downstream_energy_dp(downstream_globals)     # [B,N]
    U = torch.matmul(target_attn_global, z_prev.abs().unsqueeze(1))   # [B,H,N,D]
    current_energy = U.square().sum(dim=-1).mean(dim=1)               # [B,N]
    return downstream_energy.clamp_min(0) * current_energy            # [B,N]


# ── §7.7 dataset 평균 score ─────────────────────────────────────────────────────

@torch.no_grad()
def estimate_mean_scores(model, target_layer: int, keep_ids: List[Tensor],
                         calib_loader, device, max_batches: Optional[int] = None,
                         log_every: int = 0) -> Tensor:
    """calibration subset 전체에 대한 patch significance 평균 score [N] (float64 누적).

    SPEC §7.7: NaN/Inf 즉시 오류, eval mode. CLS 제외/keep 제외는 candidate ranking에서.
    """
    model.eval()
    shp = get_shape(model)
    N = shp.num_global_tokens
    score_sum = torch.zeros(N, dtype=torch.float64)
    n_samples = 0
    for bi, batch in enumerate(calib_loader):
        images = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device, non_blocking=True)
        z_prev, tgt_attn, downstream = hybrid_forward_for_target(model, images, target_layer, keep_ids)
        s = compute_sample_scores(z_prev, tgt_attn, downstream)       # [B,N]
        if not torch.isfinite(s).all():
            raise FloatingPointError(f"score에 NaN/Inf 발생 (layer={target_layer}, batch={bi})")
        score_sum += s.double().sum(dim=0).cpu()
        n_samples += s.shape[0]
        if log_every and bi % log_every == 0:
            print(f"    [score] layer={target_layer} batch={bi} samples={n_samples}")
        if max_batches is not None and bi + 1 >= max_batches:
            break
    if n_samples == 0:
        raise RuntimeError("calibration batch가 없음")
    return (score_sum / n_samples)                                    # [N] float64


# ── §7.6 brute-force 대조 (test 전용) ──────────────────────────────────────────

def bruteforce_downstream_energy(downstream: List[Tensor]) -> Tensor:
    """head-path를 명시적으로 열거해 G_ii 계산 (DP 검증용, SPEC §7.6, §16.5).

    DP: G_k-1 = (1/H) Σ_h P_k^h,T G_k P_k^h.  이를 head 인덱스로 완전 전개.
    m개 downstream layer, H heads → H^m 개 head-sequence. 작은 case에서만 쓴다.
    """
    B, H, N, _ = downstream[0].shape
    rev = list(reversed(downstream))                     # [P_{L-1}, ..., P_{t+1}]
    m = len(rev)
    device, dtype = downstream[0].device, downstream[0].dtype
    G_total = torch.zeros(B, N, N, device=device, dtype=dtype)
    for seq in itertools.product(range(H), repeat=m):    # 각 layer의 head 선택
        G = torch.eye(N, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()
        for step, h in enumerate(seq):
            Ph = rev[step][:, h]                          # [B,N,N]
            G = torch.matmul(Ph.transpose(-2, -1), torch.matmul(G, Ph))
        G_total += G
    G_total /= (H ** m)
    return torch.diagonal(G_total, dim1=-2, dim2=-1)      # [B,N]
