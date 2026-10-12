"""Global FC1-row L2 pruning of ViT FFN hidden channels.

Compression is a fraction of all model parameters. Each hidden channel removes
its FC1 row/bias and matching FC2 column. Masks follow optimizer updates;
physical resizing is performed by the reducer.
"""
from __future__ import annotations
import torch

MIN_SURVIVE = 4


def _calc_n_prune(n_total, sparsity):
    return max(0, min(round(n_total * sparsity), n_total - MIN_SURVIVE))


def _estimate_total_removed(model, sparsity):
    return sum(_calc_n_prune(b.mlp.fc1.out_features, sparsity) *
               (b.mlp.fc1.in_features + b.mlp.fc2.out_features + int(b.mlp.fc1.bias is not None))
               for b in model.blocks)


def _find_sparsity_by_bisection(model, target_compression, max_sparsity=0.95):
    target = target_compression * sum(p.numel() for p in model.parameters())
    if target > _estimate_total_removed(model, max_sparsity):
        raise ValueError("Requested compression exceeds the FFN pruning capacity")
    lo, hi = 0.0, max_sparsity
    for _ in range(64):
        mid = (lo + hi) / 2
        if _estimate_total_removed(model, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class ViTPruner:
    def __init__(self, model, target_compression, max_sparsity=0.95,
                 index_refresh_steps=100, sparsity=None):
        if not 0 <= target_compression < 1 or not 0 <= max_sparsity < 1:
            raise ValueError("Compression and sparsity must lie in [0, 1)")
        self.target_compression = float(target_compression)
        self.max_sparsity = float(max_sparsity)
        self.index_refresh_steps = int(index_refresh_steps)
        self.sparsity = (float(sparsity) if sparsity is not None else
                         _find_sparsity_by_bisection(model, target_compression, max_sparsity))
        self._step = 0
        self._masks = []
        print(f"[ViTPruner] global L2 target={100 * target_compression:.1f}% "
              f"equivalent FFN sparsity={self.sparsity:.6f}")

    @torch.no_grad()
    def _refresh(self, model):
        scores = [b.mlp.fc1.weight.norm(dim=1) for b in model.blocks]
        sizes = [s.numel() for s in scores]
        eligible = torch.cat(scores).clone()
        starts = [sum(sizes[:i]) for i in range(len(sizes))]
        for score, start in zip(scores, starts):
            cap = _calc_n_prune(score.numel(), self.max_sparsity)
            eligible[start + torch.argsort(score)[cap:]] = float("inf")
        count = sum(_calc_n_prune(n, self.sparsity) for n in sizes)
        if count > int(torch.isfinite(eligible).sum()):
            raise ValueError("Insufficient eligible channels")
        selected = torch.topk(eligible, count, largest=False).indices
        self._masks = []
        for size, start in zip(sizes, starts):
            mask = torch.ones(size, device=eligible.device)
            local = selected[(selected >= start) & (selected < start + size)] - start
            mask[local] = 0
            self._masks.append(mask)

    @torch.no_grad()
    def apply(self, model):
        if self.sparsity <= 0:
            return
        if (not self._masks or self.index_refresh_steps <= 0 or
                self._step % self.index_refresh_steps == 0):
            self._refresh(model)
        for block, mask in zip(model.blocks, self._masks):
            mask = mask.to(block.mlp.fc1.weight.device)
            block.mlp.fc1.weight.mul_(mask[:, None])
            if block.mlp.fc1.bias is not None:
                block.mlp.fc1.bias.mul_(mask)
            block.mlp.fc2.weight.mul_(mask[None, :])
        self._step += 1

    @torch.no_grad()
    def log_sparsity(self, model):
        result = {}; total = zero = 0
        for i, block in enumerate(model.blocks):
            n = block.mlp.fc1.out_features
            dead = int((block.mlp.fc1.weight.norm(dim=1) == 0).sum())
            total += n; zero += dead
            result[f"pruning/layer/blocks/{i}/mlp"] = dead / n
            result[f"pruning/survived/blocks/{i}/mlp"] = n - dead
        result.update({"pruning/zero_filters": zero, "pruning/prunable_filters": total,
                       "pruning/actual_sparsity": zero / max(total, 1),
                       "pruning/target_sparsity": self.sparsity})
        return result

    def state_dict(self):
        return {"method": "global_l2", "sparsity": self.sparsity,
                "target_compression": self.target_compression,
                "max_sparsity": self.max_sparsity, "step": self._step,
                "index_refresh_steps": self.index_refresh_steps,
                "masks": [m.cpu() for m in self._masks]}

    def load_state_dict(self, state):
        for key in ("sparsity", "target_compression", "max_sparsity", "index_refresh_steps"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Pruner configuration mismatch: {key}")
        if state["method"] != "global_l2":
            raise ValueError("Only global L2 checkpoints are supported")
        self._step = state["step"]
        self._masks = state["masks"]
