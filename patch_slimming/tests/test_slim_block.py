"""§16.3 — SlimBlock 수치 등가성 테스트 (가장 중요한 게이트).

all-token 출력 = 원본 block 출력, subset 출력 = 원본 출력의 subset row,
matmul 선택 = index_select 선택. Phase 2 완료 조건 (SPEC §17).
"""
import torch
import timm
from _util import run_tests

from psvit.slim_block import SlimBlock
from psvit.model_utils import get_shape

torch.manual_seed(0)
_MODEL = "deit_tiny_patch16_224"
ATOL = 1e-4


def _setup():
    model = timm.create_model(_MODEL, pretrained=False).eval()
    shp = get_shape(model)
    N, D = shp.num_global_tokens, shp.embed_dim
    x = torch.randn(2, N, D)
    all_ids = torch.arange(N)
    return model, shp, x, all_ids


def test_alltoken_equivalence():
    model, shp, x, all_ids = _setup()
    with torch.no_grad():
        for l in range(shp.num_blocks):
            ref = model.blocks[l](x)                       # 원본 full forward
            got = SlimBlock(model.blocks[l])(x, all_ids, all_ids).x
            diff = (ref - got).abs().max().item()
            assert diff <= ATOL, f"block {l}: all-token 등가성 위반 diff={diff:.2e}"


def test_subset_equivalence():
    model, shp, x, all_ids = _setup()
    N = shp.num_global_tokens
    subset = torch.tensor(sorted([0] + list(range(3, N, 5))))   # CLS + 흩어진 patch
    with torch.no_grad():
        for l in range(shp.num_blocks):
            ref_full = model.blocks[l](x)                  # [B,N,D]
            ref_sub = ref_full.index_select(1, subset)     # 원본 출력의 subset row
            got = SlimBlock(model.blocks[l])(x, all_ids, subset).x
            diff = (ref_sub - got).abs().max().item()
            assert diff <= ATOL, f"block {l}: subset 등가성 위반 diff={diff:.2e}"


def test_matmul_equals_index_select():
    model, shp, x, all_ids = _setup()
    N = shp.num_global_tokens
    subset = torch.tensor(sorted([0] + list(range(2, N, 4))))
    with torch.no_grad():
        for l in range(shp.num_blocks):
            a = SlimBlock(model.blocks[l], "index_select")(x, all_ids, subset).x
            b = SlimBlock(model.blocks[l], "matmul")(x, all_ids, subset).x
            diff = (a - b).abs().max().item()
            assert diff <= ATOL, f"block {l}: matmul≠index_select diff={diff:.2e}"


def test_attention_shape_rectangular():
    model, shp, x, all_ids = _setup()
    N = shp.num_global_tokens
    subset = torch.tensor(sorted([0] + list(range(5, N, 7))))
    out = SlimBlock(model.blocks[0])(x, all_ids, subset, return_attention=True)
    B, H, No, Ni = out.attention_probs.shape
    assert No == subset.numel() and Ni == N, (No, Ni, subset.numel(), N)


if __name__ == "__main__":
    ok = run_tests(dict(globals()))
    raise SystemExit(0 if ok else 1)
