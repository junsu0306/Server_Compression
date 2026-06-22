"""
timm ViT / DeiT 아키텍처 분석 & 메모리 프로파일링
- 모듈 구조 출력
- G_FFN / G_QKV / G_PROJ / G_HEAD / G_EMBED / G_OTHER 그룹별 파라미터 수
- target_compression 대비 예상 sparsity 표
"""

import re
import torch
import torch.nn as nn
import timm


# ── 분석 대상 모델 ─────────────────────────────────────────────────────────────
MODELS = [
    "vit_tiny_patch16_224",
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_224",
]

TARGET_COMPRESSIONS = [0.10, 0.20, 0.30, 0.50]


# ── 헬퍼 ───────────────────────────────────────────────────────────────────────

MIN_SURVIVE = 4


def _calc_n_prune(n_total: int, sparsity: float) -> int:
    n_prune = round(n_total * sparsity)
    n_prune = min(n_prune, n_total - MIN_SURVIVE)
    return max(n_prune, 0)


def _estimate_removed_ffn(mlp: nn.Module, sparsity: float) -> int:
    fc1_w = mlp.fc1.weight          # (mlp_dim, embed_dim)
    mlp_dim   = fc1_w.shape[0]
    embed_dim = fc1_w.shape[1]
    fc2_out   = mlp.fc2.weight.shape[0]

    n_prune = _calc_n_prune(mlp_dim, sparsity)
    if n_prune <= 0:
        return 0

    removed  = n_prune * embed_dim       # fc1 weight rows
    if mlp.fc1.bias is not None:
        removed += n_prune               # fc1 bias
    removed += fc2_out * n_prune         # fc2 weight cols (secondary effect)
    return removed


def _estimate_total_removed(model: nn.Module, sparsity: float) -> int:
    total = 0
    for block in model.blocks:
        total += _estimate_removed_ffn(block.mlp, sparsity)
    return total


def _find_sparsity_by_bisection(
    model: nn.Module,
    target_compression: float,
    max_sparsity: float = 0.95,
    iters: int = 64,
) -> float:
    if target_compression <= 0:
        return 0.0
    total_params  = sum(p.numel() for p in model.parameters())
    target_remove = target_compression * total_params
    lo, hi = 0.0, max_sparsity
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _estimate_total_removed(model, mid) < target_remove:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── 모듈 구조 출력 ─────────────────────────────────────────────────────────────

def print_module_structure(model: nn.Module, model_name: str) -> None:
    print(f"\n{'='*70}")
    print(f"  Module structure: {model_name}")
    print(f"{'='*70}")
    print(f"  {'name':<45} {'class':<16} {'weight shape'}")
    print(f"  {'-'*67}")
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight is not None:
            shape = tuple(module.weight.shape)
            cls   = module.__class__.__name__
            # 주요 레이어만 출력 (중간 컨테이너 제외)
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
                flag = ""
                if ".mlp.fc" in name:
                    flag = " ★ G_FFN"
                elif ".attn.qkv" in name:
                    flag = "   G_QKV"
                elif ".attn.proj" in name:
                    flag = "   G_PROJ"
                print(f"  {name:<45} {cls:<16} {str(shape):<24}{flag}")


# ── 파라미터 그룹별 분류 ────────────────────────────────────────────────────────

def measure_vit_memory(model_name: str) -> dict:
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    total = sum(p.numel() for p in model.parameters())

    groups: dict[str, list[int]] = {
        "G_FFN":   [],
        "G_QKV":   [],
        "G_PROJ":  [],
        "G_NORM":  [],
        "G_HEAD":  [],
        "G_EMBED": [],
        "G_OTHER": [],
    }
    accounted_param_ids: set[int] = set()

    for name, module in model.named_modules():
        # 직접 파라미터만 처리 (컨테이너 모듈은 skip)
        params = list(module.parameters(recurse=False))
        if not params:
            continue
        if all(id(p) in accounted_param_ids for p in params):
            continue

        def _account(grp: str) -> None:
            for p in params:
                if id(p) not in accounted_param_ids:
                    groups[grp].append(p.numel())
                    accounted_param_ids.add(id(p))

        # 이름 패턴으로 분류 (recurse=False로 받은 직접 파라미터)
        if re.search(r"blocks\.\d+\.mlp\.(fc1|fc2)$", name):
            _account("G_FFN")
        elif re.search(r"blocks\.\d+\.attn\.qkv$", name):
            _account("G_QKV")
        elif re.search(r"blocks\.\d+\.attn\.proj$", name):
            _account("G_PROJ")
        elif isinstance(module, nn.LayerNorm):
            _account("G_NORM")
        elif name in ("head", "head_dist") and isinstance(module, nn.Linear):
            _account("G_HEAD")
        elif name.startswith("patch_embed"):
            _account("G_EMBED")
        else:
            _account("G_OTHER")

    # pos_embed, cls_token, dist_token 은 Parameter로 모델에 직접 붙어있음
    for attr in ("cls_token", "pos_embed", "dist_token"):
        obj = getattr(model, attr, None)
        if obj is not None and isinstance(obj, nn.Parameter):
            if id(obj) not in accounted_param_ids:
                groups["G_EMBED"].append(obj.numel())
                accounted_param_ids.add(id(obj))

    # 분류 누락 파라미터 확인 (있으면 G_OTHER)
    for p in model.parameters():
        if id(p) not in accounted_param_ids:
            groups["G_OTHER"].append(p.numel())

    print(f"\n{'='*55}")
    print(f"  Parameter breakdown: {model_name}")
    print(f"{'='*55}")
    print(f"  {'group':<12}{'numel':>14}{'MB':>10}{'%':>9}")
    print(f"  {'-'*45}")
    for g, nums in groups.items():
        n = sum(nums)
        print(f"  {g:<12}{n:>14,}{n*4/1e6:>10.3f}{100*n/total:>8.2f}%")
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':<12}{total:>14,}{total*4/1e6:>10.3f}{'100.00':>8}%")

    ffn_numel = sum(groups["G_FFN"])
    print(f"\n  Prunable (G_FFN): {ffn_numel:,}  ({100*ffn_numel/total:.2f}%)")

    # target_compression → sparsity 표
    print(f"\n  {'target_comp':>13}  {'sparsity':>10}  {'est_removed':>13}  {'act_comp%':>10}")
    print(f"  {'-'*53}")
    for tc in TARGET_COMPRESSIONS:
        sp = _find_sparsity_by_bisection(model, tc)
        removed = _estimate_total_removed(model, sp)
        act_comp = 100.0 * removed / total
        print(f"  {tc:>13.2f}  {sp:>10.4f}  {removed:>13,}  {act_comp:>9.2f}%")

    return {"model": model_name, "total": total, "groups": groups}


# ── 메인 ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 구조 출력 (vit_base 대표)
    print("\n[1] Module structure (vit_base_patch16_224)")
    m = timm.create_model("vit_base_patch16_224", pretrained=False)
    print_module_structure(m, "vit_base_patch16_224")
    del m

    # 파라미터 분석 (모든 모델)
    print("\n\n[2] Parameter breakdown & sparsity table")
    for name in MODELS:
        try:
            measure_vit_memory(name)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")
