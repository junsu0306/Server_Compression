"""
Calibration subset (deterministic) — SPEC §9.6, §7.7.

첫 구현은 하나의 deterministic calibration subset을 score 계산·block fine-tuning·error
평가에 공용으로 쓴다 (SPEC §9.6 [DECISION]). subset sample ID는 파일로 저장, seed 고정,
random crop 대신 baseline validation preprocessing 사용. label은 필요 없다.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import timm
import timm.data
from torchvision import datasets


def _val_transform(model_name: str):
    ref = timm.create_model(model_name, pretrained=False)
    cfg = timm.data.resolve_model_data_config(ref)
    del ref
    return timm.data.create_transform(**cfg, is_training=False), cfg


def _select_indices(n_total: int, num_samples: int, seed: int) -> List[int]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    return perm[: min(num_samples, n_total)].tolist()


def build_calibration_loader(
    model_name: str, data_path: str, num_samples: int, batch_size: int,
    seed: int = 2022, split: str = "train", num_workers: int = 8,
    sample_id_file: Optional[str] = None,
):
    """deterministic calibration DataLoader. sample id를 파일에 저장(있으면 재사용)."""
    transform, data_cfg = _val_transform(model_name)   # baseline val preprocessing (aug 없음)
    ds = datasets.ImageFolder(os.path.join(data_path, split), transform=transform)

    if sample_id_file and os.path.exists(sample_id_file):
        with open(sample_id_file) as f:
            indices = [int(x) for x in f.read().split()]
    else:
        indices = _select_indices(len(ds), num_samples, seed)
        if sample_id_file:
            os.makedirs(os.path.dirname(os.path.abspath(sample_id_file)), exist_ok=True)
            with open(sample_id_file, "w") as f:
                f.write("\n".join(map(str, indices)))

    subset = torch.utils.data.Subset(ds, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return loader, data_cfg, indices
