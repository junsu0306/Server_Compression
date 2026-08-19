"""
§8 / §11 — Algorithm 1 top-down mask search.

마지막 CLS 출력에서 시작(keep_ids[L-1]=[CLS]), t=L-2..0 방향으로 이동하며 각 layer의
고정 keep mask를 찾는다. 각 target layer t:
    1. hybrid network로 patch significance 평균 score 계산 (scoring.estimate_mean_scores)
    2. score 내림차순으로 candidate 정렬 (base=keep_ids[t+1], prefix 제외)
    3. r을 search_step 단위로 늘리며 block t reconstruction fine-tuning → 다음 layer 복원
       오차가 epsilon 이하가 되면 mask 확정 (§8.5)
    4. nested invariant 검증 + layer checkpoint 저장 (resume 가능, §8.7/§16.8)

논문 미기재 값은 SPEC의 [DECISION]: cumulative fine-tuning(§8.6), MSE metric(§9.4),
path-energy DP(§7.5).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import torch
from torch import Tensor

from .ids import CLS_ID, as_long_ids, sorted_union, validate_nested_masks
from .model_utils import get_shape
from .scoring import estimate_mean_scores
from .reconstruct import finetune_block_for_candidate, evaluate_reconstruction


def rank_candidate_patch_ids(mean_scores: Tensor, base_ids: Tensor,
                             num_global: int, prefix_ids=(CLS_ID,)) -> Tensor:
    """score 내림차순으로 candidate global patch ID 정렬. base/prefix는 제외 (SPEC §8.4)."""
    exclude = set(as_long_ids(base_ids).tolist()) | set(int(p) for p in prefix_ids)
    cand = torch.tensor(
        [g for g in range(num_global) if g not in exclude],
        dtype=torch.long,
        device=mean_scores.device,
    )
    if cand.numel() == 0:
        return cand
    order = torch.argsort(mean_scores[cand], descending=True)
    return cand[order]


def initial_growth_count(t: int, last_searchable_layer: int,
                         candidate_count: int, cfg: dict) -> int:
    """Return the optional minimum growth beyond the next layer's base mask."""
    if t >= last_searchable_layer:
        return 0
    growth = max(0, int(cfg.get("min_growth_per_shallower_layer", 0)))
    return min(growth, candidate_count)


def search_layer(
    model, teacher, t: int, keep_ids: List[Optional[Tensor]],
    score_loader, reconstruction_train_loader, reconstruction_eval_loader,
    device, cfg: dict, records: List[dict],
) -> float:
    """target layer t의 keep mask를 찾아 keep_ids[t]에 채운다. 반환: accepted error."""
    shp = get_shape(model)
    N = shp.num_global_tokens
    all_ids = torch.arange(N, device=device)
    epsilon = float(cfg["epsilon"])
    error_metric = cfg.get("error_metric", "mse")
    search_step = int(cfg["search_step"])
    max_iters = int(cfg.get("max_search_iters", 10_000))

    print(f"[search] layer {t}: mean score 추정 중 ...")
    mean_scores = estimate_mean_scores(
        model, target_layer=t, keep_ids=keep_ids, calib_loader=score_loader,
        device=device, max_batches=cfg.get("score_max_batches"),
    ).to(device)

    base_ids = as_long_ids(keep_ids[t + 1]).to(device)
    candidate_ids = rank_candidate_patch_ids(mean_scores, base_ids, N).to(device)
    patch_scores = mean_scores[1:]
    score_stats = {
        "min": float(patch_scores.min().item()),
        "max": float(patch_scores.max().item()),
        "mean": float(patch_scores.mean().item()),
    }

    r = initial_growth_count(t, shp.num_blocks - 2, int(candidate_ids.numel()), cfg)
    it, accepted, accepted_err = 0, False, float("nan")
    while not accepted:
        it += 1
        if it > max_iters:
            raise RuntimeError(f"layer {t}: max_search_iters 초과")
        current_ids = sorted_union(base_ids, candidate_ids[:r])
        candidate_started = time.perf_counter()
        ft = finetune_block_for_candidate(
            model, teacher, t, current_ids, base_ids, reconstruction_train_loader, device,
            cfg.get("block_finetune", {}),
        )
        metrics = evaluate_reconstruction(model, teacher, t, current_ids, base_ids,
                                          reconstruction_eval_loader, device)
        candidate_seconds = time.perf_counter() - candidate_started
        err = metrics[error_metric]
        records.append({
            "layer": t, "r": r, "n_keep": int(current_ids.numel()),
            "minimum_growth_constraint": int(cfg.get("min_growth_per_shallower_layer", 0)),
            "candidate_keep_ids": current_ids.detach().cpu().tolist(),
            "train_loss": ft.get("loss"),
            "train_loss_by_epoch": ft.get("loss_by_epoch", []),
            "score_stats": score_stats,
            "candidate_seconds": candidate_seconds,
            **{f"val_{k}": v for k, v in metrics.items()},
            "accepted": None,
        })
        callback = cfg.get("record_callback")
        if callback is not None:
            callback(records[-1])
        print(f"    layer {t}  r={r}  n_keep={current_ids.numel()}  "
              f"{error_metric}={err:.5f}  (ε={epsilon})")

        if err <= epsilon:
            keep_ids[t] = current_ids
            accepted, accepted_err = True, err
            records[-1]["accepted"] = True
        elif r >= candidate_ids.numel():
            keep_ids[t] = all_ids.clone()               # 전부 유지 (복원 실패 fallback)
            accepted, accepted_err = True, err
            records[-1]["accepted"] = "all_tokens"
        else:
            r = min(r + search_step, int(candidate_ids.numel()))

    validate_nested_masks(keep_ids, upto=t)
    print(f"[search] layer {t} 확정: {int(keep_ids[t].numel())} tokens  (err={accepted_err:.5f})")
    return accepted_err


def _save_checkpoint(path, model, keep_ids, accepted_errors, records, next_layer):
    torch.save({
        "model": model.state_dict(),
        "keep_ids": [None if k is None else as_long_ids(k).cpu().tolist() for k in keep_ids],
        "accepted_errors": accepted_errors,
        "records": records,
        "next_layer": next_layer,          # 다음에 탐색할 t (이보다 큰 t는 이미 완료)
    }, path)


def _load_checkpoint(path, model, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    keep_ids = [None if k is None else torch.tensor(k, dtype=torch.long, device=device)
                for k in ckpt["keep_ids"]]
    return keep_ids, ckpt["accepted_errors"], ckpt["records"], ckpt["next_layer"]


def run_patch_slimming_search(
    model, teacher, score_loader, device, cfg: dict,
    checkpoint_dir: str, resume: bool = True,
    reconstruction_train_loader=None, reconstruction_eval_loader=None,
):
    """SPEC §11 — 전체 top-down search. layer checkpoint로 재개 가능.

    model: student (탐색·fine-tuning 대상, in-place로 blocks[t] weight 갱신)
    teacher: frozen 원본 (deepcopy). 둘 다 이미 device 위, eval 준비.
    반환: (keep_ids[L], accepted_errors[L], records)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    reconstruction_train_loader = reconstruction_train_loader or score_loader
    reconstruction_eval_loader = reconstruction_eval_loader or score_loader
    shp = get_shape(model)
    L = shp.num_blocks

    keep_ids: List[Optional[Tensor]] = [None] * L
    keep_ids[L - 1] = torch.tensor([CLS_ID], dtype=torch.long, device=device)
    accepted_errors: List[float] = [float("nan")] * L
    accepted_errors[L - 1] = 0.0
    records: List[dict] = []
    start_t = L - 2

    last_ckpt = os.path.join(checkpoint_dir, "search_last.pt")
    if resume and os.path.exists(last_ckpt):
        keep_ids, accepted_errors, records, start_t = _load_checkpoint(last_ckpt, model, device)
        print(f"[resume] search_last.pt 로드 — layer {start_t}부터 재개")

    for t in range(start_t, -1, -1):
        err = search_layer(
            model, teacher, t, keep_ids, score_loader,
            reconstruction_train_loader, reconstruction_eval_loader,
            device, cfg, records,
        )
        accepted_errors[t] = err
        _save_checkpoint(os.path.join(checkpoint_dir, f"search_layer_{t}.pt"),
                         model, keep_ids, accepted_errors, records, next_layer=t - 1)
        _save_checkpoint(last_ckpt, model, keep_ids, accepted_errors, records, next_layer=t - 1)

    validate_nested_masks(keep_ids)
    sched = [shp.num_global_tokens] + [int(keep_ids[i].numel()) for i in range(L)]
    print(f"[search] 완료. token schedule = {sched}")
    return keep_ids, accepted_errors, records
