"""
한 epoch 단위 학습/검증 루프.

train.py 에서 import 해서 사용한다.
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn


# ── 유틸 ───────────────────────────────────────────────────────────────────────

class AverageMeter:
    """배치 평균을 누적 관리."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    """Top-k accuracy (%) 반환."""
    maxk      = max(topk)
    batch_size = target.size(0)
    _, pred   = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred      = pred.t()                                    # (maxk, B)
    correct   = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum() * 100.0 / batch_size for k in topk]


# ── 학습 ───────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:        nn.Module,
    criterion:    nn.Module,
    data_loader,
    optimizer:    torch.optim.Optimizer,
    scaler:       torch.cuda.amp.GradScaler,
    device:       torch.device,
    epoch:        int,
    model_ema=None,
    pruner=None,
    amp:          bool = True,
    clip_grad:    float | None = None,
    log_interval: int = 50,
) -> dict[str, float]:
    """
    한 epoch 학습.

    Soft Pruning 삽입 위치:
        optimizer.step() → pruner.apply() → model_ema.update()

    Args:
        pruner:    ViTPruner 인스턴스 (None이면 일반 학습)
        model_ema: timm ModelEmaV2 인스턴스 (None이면 EMA 미사용)
    """
    model.train()

    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    start  = time.time()

    for batch_idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # ── Forward ────────────────────────────────────────────────────────────
        with torch.amp.autocast("cuda", enabled=amp):
            output = model(samples)
            loss   = criterion(output, targets)

        # ── Backward + optimizer.step ──────────────────────────────────────────
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        if clip_grad is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        scaler.step(optimizer)
        scaler.update()

        # ── ★ Soft Pruning Hook ★ ──────────────────────────────────────────────
        # optimizer.step() 직후, model_ema.update() 이전에 호출해야 한다.
        # DDP인 경우 .module로 unwrap해서 전달.
        if pruner is not None:
            actual = model.module if hasattr(model, "module") else model
            pruner.apply(actual)

        # ── EMA 업데이트 (pruning 후 weight 기준) ─────────────────────────────
        if model_ema is not None:
            actual = model.module if hasattr(model, "module") else model
            model_ema.update(actual)

        # ── 메트릭 ────────────────────────────────────────────────────────────
        with torch.no_grad():
            acc1, acc5 = accuracy(output.detach().float(), targets)
        bs = samples.size(0)
        loss_m.update(loss.item(), bs)
        top1_m.update(acc1.item(), bs)
        top5_m.update(acc5.item(), bs)

        if batch_idx % log_interval == 0:
            elapsed = time.time() - start
            print(
                f"  Epoch[{epoch}] [{batch_idx:>4d}/{len(data_loader)}]  "
                f"loss={loss_m.avg:.4f}  top1={top1_m.avg:.2f}%  "
                f"top5={top5_m.avg:.2f}%  t={elapsed:.0f}s"
            )

    return {"loss": loss_m.avg, "top1": top1_m.avg, "top5": top5_m.avg}


# ── 검증 ───────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    data_loader,
    model:  nn.Module,
    device: torch.device,
    amp:    bool = True,
) -> dict[str, float]:
    """검증 루프. loss / acc1 / acc5 반환."""
    model.eval()

    criterion = nn.CrossEntropyLoss()
    loss_m    = AverageMeter()
    top1_m    = AverageMeter()
    top5_m    = AverageMeter()

    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp):
            output = model(samples)
            loss   = criterion(output, targets)

        acc1, acc5 = accuracy(output.float(), targets)
        bs = samples.size(0)
        loss_m.update(loss.item(), bs)
        top1_m.update(acc1.item(), bs)
        top5_m.update(acc5.item(), bs)

    print(
        f"  [Val] loss={loss_m.avg:.4f}  "
        f"top1={top1_m.avg:.2f}%  top5={top5_m.avg:.2f}%"
    )
    return {"loss": loss_m.avg, "acc1": top1_m.avg, "acc5": top5_m.avg}
