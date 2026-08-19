"""
Deterministic reconstruction datasets.

score / block reconstruction train / acceptance eval subsets can be built independently.
Every subset stores its sample IDs so a resumed search uses exactly the same images.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import torch
import timm
import timm.data
from torchvision import datasets


def _val_transform(model_name: str):
    ref = timm.create_model(model_name, pretrained=False)
    cfg = timm.data.resolve_model_data_config(ref)
    del ref
    return timm.data.create_transform(**cfg, is_training=False), cfg


def _select_indices(n_total: int, num_samples: int, seed: int,
                    excluded_indices: Iterable[int] = ()) -> List[int]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    excluded = set(int(i) for i in excluded_indices)
    selected = [int(i) for i in perm.tolist() if int(i) not in excluded]
    return selected[: min(num_samples, len(selected))]


def build_calibration_loader(
    model_name: str, data_path: str, num_samples: int, batch_size: int,
    seed: int = 2022, split: str = "train", num_workers: int = 8,
    sample_id_file: Optional[str] = None,
    excluded_indices: Iterable[int] = (),
    shuffle: bool = False,
):
    """deterministic calibration DataLoader. sample id를 파일에 저장(있으면 재사용)."""
    transform, data_cfg = _val_transform(model_name)   # baseline val preprocessing (aug 없음)
    ds = datasets.ImageFolder(os.path.join(data_path, split), transform=transform)

    if sample_id_file and os.path.exists(sample_id_file):
        with open(sample_id_file) as f:
            indices = [int(x) for x in f.read().split()]
    else:
        indices = _select_indices(len(ds), num_samples, seed, excluded_indices)
        if sample_id_file:
            os.makedirs(os.path.dirname(os.path.abspath(sample_id_file)), exist_ok=True)
            with open(sample_id_file, "w") as f:
                f.write("\n".join(map(str, indices)))

    subset = torch.utils.data.Subset(ds, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return loader, data_cfg, indices
