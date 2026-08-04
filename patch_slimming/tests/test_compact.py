"""§16.9/§16.10 — CompactPSViT 등가성 및 export 관련 테스트."""
import torch
import timm
from _util import run_tests

from psvit.compact import CompactPSViT
from psvit.model_utils import get_shape

torch.manual_seed(0)
_MODEL = "deit_tiny_patch16_224"
ATOL = 1e-4


def _model():
    return timm.create_model(_MODEL, pretrained=False).eval()


def _nested_keep(N, L):
    """block 0=N tokens … block L-1=1 token(CLS) 인 nested keep_ids (arange prefix)."""
    keep = []
    for l in range(L):
        m = max(1, 1 + (N - 1) * (L - 1 - l) // (L - 1))   # l 깊을수록 작아짐, l=L-1→1
        keep.append(torch.arange(m))
    return keep


def test_compact_alltokens_equals_full():
    """모든 block이 전체 토큰 유지 → compact == 원본 timm 모델 (전체 forward 검증)."""
    model = _model()
    shp = get_shape(model)
    all_keep = [torch.arange(shp.num_global_tokens) for _ in range(shp.num_blocks)]
    compact = CompactPSViT(model, all_keep, select_mode="index_select").eval()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        ref = model(images)
        got = compact(images)
    diff = (ref - got).abs().max().item()
    assert diff <= ATOL, f"all-token compact ≠ full model diff={diff:.2e}"


def test_compact_matmul_equals_index_select():
    model = _model()
    shp = get_shape(model)
    keep = _nested_keep(shp.num_global_tokens, shp.num_blocks)
    images = torch.randn(2, 3, 224, 224)
    a = CompactPSViT(model, keep, select_mode="index_select").eval()
    b = CompactPSViT(model, keep, select_mode="matmul").eval()
    b.bake_for_export(device=torch.device("cpu"))
    with torch.no_grad():
        da = a(images); db = b(images)
    diff = (da - db).abs().max().item()
    assert diff <= ATOL, f"matmul ≠ index_select diff={diff:.2e}"


def test_compact_output_shape_and_schedule():
    model = _model()
    shp = get_shape(model)
    keep = _nested_keep(shp.num_global_tokens, shp.num_blocks)
    compact = CompactPSViT(model, keep, select_mode="index_select").eval()
    with torch.no_grad():
        out = compact(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, shp.num_classes), out.shape
    sched = compact.token_schedule()
    assert sched[0] == shp.num_global_tokens and sched[-1] == 1, sched   # 마지막은 CLS만


def test_compact_nhwc_matches_nchw():
    model = _model()
    shp = get_shape(model)
    keep = _nested_keep(shp.num_global_tokens, shp.num_blocks)
    nchw = torch.randn(2, 3, 224, 224)
    nhwc = nchw.permute(0, 2, 3, 1).contiguous()
    a = CompactPSViT(model, keep, nhwc_input=False).eval()
    b = CompactPSViT(model, keep, nhwc_input=True).eval()
    with torch.no_grad():
        diff = (a(nchw) - b(nhwc)).abs().max().item()
    assert diff <= ATOL, f"NHWC≠NCHW diff={diff:.2e}"


if __name__ == "__main__":
    ok = run_tests(dict(globals()))
    raise SystemExit(0 if ok else 1)
