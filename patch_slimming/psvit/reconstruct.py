"""
§9 — Block reconstruction fine-tuning.

현재 탐색 중인 student.blocks[t]의 parameter만 학습한다. 다음 block student.blocks[t+1]은
parameter는 고정하되 gradient는 통과시킨다 (torch.no_grad로 감싸지 않음, SPEC §9.1).

목표(SPEC §9.2): teacher(frozen 원본)의 block t 입력 Z_{t-1}을 주면, slim block t
(current_keep_ids) → frozen slim block t+1 (next_keep_ids) 의 출력이 teacher의 block t+1
출력(next_keep_ids row)을 복원하도록 한다.

loss: element-wise MSE (SPEC §9.4 [DECISION]). raw/relative frobenius도 로깅.
error_metric 기본값 mse로 acceptance 판정 (SPEC §9.4).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ids import as_long_ids
from .model_utils import embed_tokens, get_shape
from .slim_block import SlimBlock


@torch.no_grad()
def teacher_features(teacher: nn.Module, images: Tensor, t: int, next_keep_ids: Tensor):
    """teacher(frozen 원본)에서 block t 입력과 block t+1 출력(next_keep_ids) 산출.

    blocks 0..t+1 까지만 부분 forward (전체 forward 불필요).
    반환: z_prev_full [B,N,D],  target [B, |next_keep|, D]
    """
    L = len(teacher.blocks)
    x = embed_tokens(teacher, images)
    for l in range(t):
        x = teacher.blocks[l](x)
    z_prev = x                                  # [B,N,D]  Z_{t-1}
    for l in range(t, min(t + 2, L)):           # block t, t+1 실행
        x = teacher.blocks[l](x)
    next_ids = as_long_ids(next_keep_ids).to(x.device)
    target = x.index_select(1, next_ids)        # teacher block t+1 출력의 next_keep row
    return z_prev, target


def _student_prediction(student: nn.Module, t: int, z_prev: Tensor,
                        current_keep_ids: Tensor, next_keep_ids: Tensor,
                        all_ids: Tensor) -> Tensor:
    """slim block t (current) → slim block t+1 (next) → [B, |next|, D]. block t에 grad 흐름."""
    sb_t = SlimBlock(student.blocks[t])
    sb_next = SlimBlock(student.blocks[t + 1])
    out_t = sb_t(z_prev, all_ids, current_keep_ids)
    out_next = sb_next(out_t.x, current_keep_ids, next_keep_ids)
    return out_next.x


def _set_trainable_only_block(student: nn.Module, t: int) -> None:
    for p in student.parameters():
        p.requires_grad_(False)
    for p in student.blocks[t].parameters():
        p.requires_grad_(True)


def finetune_block_for_candidate(
    student: nn.Module, teacher: nn.Module, t: int,
    current_keep_ids: Tensor, next_keep_ids: Tensor,
    calib_loader, device, cfg: dict,
) -> Dict[str, float]:
    """candidate mask에 대해 block t를 cfg.epochs 만큼 fine-tuning (cumulative, SPEC §8.6).

    cfg: {optimizer, learning_rate, weight_decay, epochs, grad_clip_norm} (SPEC §9.5).
    """
    shp = get_shape(student)
    all_ids = torch.arange(shp.num_global_tokens, device=device)

    _set_trainable_only_block(student, t)
    params = [p for p in student.blocks[t].parameters() if p.requires_grad]
    opt_name = cfg.get("optimizer", "adamw").lower()
    lr = float(cfg.get("learning_rate", 1e-5))
    wd = float(cfg.get("weight_decay", 0.0))
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"optimizer: {opt_name}")
    clip = float(cfg.get("grad_clip_norm", 0.0))
    epochs = int(cfg.get("epochs", 3))

    student.train()
    teacher.eval()
    last = {"loss": float("nan")}
    for ep in range(epochs):
        loss_sum, n = 0.0, 0
        for batch in calib_loader:
            images = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device, non_blocking=True)
            z_prev, target = teacher_features(teacher, images, t, next_keep_ids)
            pred = _student_prediction(student, t, z_prev, current_keep_ids, next_keep_ids, all_ids)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(params, clip)
            optimizer.step()
            bs = images.shape[0]
            loss_sum += loss.item() * bs
            n += bs
        last = {"loss": loss_sum / max(n, 1), "epoch": ep}
    student.eval()
    return last


@torch.no_grad()
def evaluate_reconstruction(
    student: nn.Module, teacher: nn.Module, t: int,
    current_keep_ids: Tensor, next_keep_ids: Tensor,
    calib_loader, device,
) -> Dict[str, float]:
    """다음 layer feature 복원 오차 (SPEC §9.4). mse / raw_frobenius_sq / relative_frobenius_sq."""
    shp = get_shape(student)
    all_ids = torch.arange(shp.num_global_tokens, device=device)
    student.eval()
    teacher.eval()

    se_sum, sq_sum, tgt_sq_sum, elem = 0.0, 0.0, 0.0, 0
    for batch in calib_loader:
        images = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device, non_blocking=True)
        z_prev, target = teacher_features(teacher, images, t, next_keep_ids)
        pred = _student_prediction(student, t, z_prev, current_keep_ids, next_keep_ids, all_ids)
        diff = pred - target
        se_sum += diff.pow(2).sum().item()
        sq_sum += diff.pow(2).sum().item()
        tgt_sq_sum += target.pow(2).sum().item()
        elem += diff.numel()
    mse = se_sum / max(elem, 1)
    return {
        "mse": mse,
        "raw_frobenius_sq": sq_sum,
        "relative_frobenius_sq": sq_sum / (tgt_sq_sum + 1e-12),
    }
