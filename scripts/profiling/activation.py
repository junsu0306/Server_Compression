"""Memory inspection helpers; the executable entrypoint is profile_reduced_memory.py."""
from __future__ import annotations
import os
import threading
import time
import psutil
import torch
from psvit.compact import CompactPSViT


def tensor_info(value):
    tensors = []
    if torch.is_tensor(value):
        tensors.append(value)
    elif isinstance(value, (tuple, list)):
        for item in value:
            tensors.extend(tensor_info(item))
    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(tensor_info(item))
    elif hasattr(value, "x") and torch.is_tensor(value.x):
        tensors.append(value.x)
    return tensors


def describe_tensors(value):
    tensors = tensor_info(value)
    return {
        "bytes": sum(t.numel() * t.element_size() for t in tensors),
        "numel": sum(t.numel() for t in tensors),
        "shapes": [list(t.shape) for t in tensors],
        "dtypes": sorted({str(t.dtype).replace("torch.", "") for t in tensors}),
    }


class ProcessSampler:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def snapshot(self):
        info = self.process.memory_info()
        full = self.process.memory_full_info()
        vm = psutil.virtual_memory()
        return {
            "time": time.time(), "rss": info.rss, "vms": info.vms,
            "uss": getattr(full, "uss", 0), "pss": getattr(full, "pss", 0),
            "system_used": vm.used, "system_available": vm.available,
        }

    def start(self):
        self.samples = [self.snapshot()]
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.samples.append(self.snapshot())

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.samples.append(self.snapshot())
        return {
            "sample_count": len(self.samples),
            "rss_start_bytes": self.samples[0]["rss"],
            "rss_peak_bytes": max(x["rss"] for x in self.samples),
            "uss_peak_bytes": max(x["uss"] for x in self.samples),
            "pss_peak_bytes": max(x["pss"] for x in self.samples),
            "system_used_peak_bytes": max(x["system_used"] for x in self.samples),
            "system_available_min_bytes": min(x["system_available"] for x in self.samples),
        }


def analytical_blocks(model, schedule, batch, element_size):
    base = model.model if isinstance(model, CompactPSViT) else model
    rows = []
    for index, block in enumerate(base.blocks):
        n_in, n_out = int(schedule[index]), int(schedule[index + 1])
        dim = int(block.attn.qkv.in_features)
        heads = int(block.attn.num_heads)
        hidden = int(block.mlp.fc1.out_features)
        counts = {
            "input": batch * n_in * dim,
            "selected_residual": batch * n_out * dim,
            "q": batch * heads * n_out * (dim // heads),
            "k": batch * heads * n_in * (dim // heads),
            "v": batch * heads * n_in * (dim // heads),
            "attention_scores": batch * heads * n_out * n_in,
            "attention_probabilities": batch * heads * n_out * n_in,
            "attention_context": batch * n_out * dim,
            "mlp_hidden": batch * n_out * hidden,
            "block_output": batch * n_out * dim,
        }
        if (isinstance(model, CompactPSViT) and args_global_select_mode(model) == "matmul"
                and n_out != n_in and n_out != 1):
            counts["selection_matrix"] = n_out * n_in
        bytes_by_tensor = {name: count * element_size for name, count in counts.items()}
        # This is a transparent upper-bound sum, not allocator liveness peak.
        rows.append({
            "block": index, "tokens_in": n_in, "tokens_out": n_out,
            "embed_dim": dim, "heads": heads, "mlp_hidden": hidden,
            "bytes_by_tensor": bytes_by_tensor,
            "sum_tensor_bytes_upper_bound": sum(bytes_by_tensor.values()),
            "attention_matrix_bytes": bytes_by_tensor["attention_scores"] + bytes_by_tensor["attention_probabilities"],
        })
    return rows


def args_global_select_mode(model):
    return getattr(model, "select_mode", "none")
