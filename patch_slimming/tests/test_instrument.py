"""§16.1 — Baseline instrumentation 테스트: 계측 on/off logit 동일, attention 합≈1."""
import torch
import timm
from _util import run_tests

from psvit.instrument import capture_full, logits_match

torch.manual_seed(0)
_MODEL = "deit_tiny_patch16_224"


def _model():
    return timm.create_model(_MODEL, pretrained=False).eval()


def test_logits_unchanged():
    model = _model()
    images = torch.randn(2, 3, 224, 224)
    ok, diff = logits_match(model, images, atol=1e-4)
    assert ok, f"계측 forward가 logit을 바꿈 diff={diff:.2e}"


def test_attention_rows_sum_to_one():
    model = _model()
    images = torch.randn(2, 3, 224, 224)
    cap = capture_full(model, images, capture_attention=True)
    for l, P in enumerate(cap.attn_probs):
        s = P.sum(dim=-1)                                  # [B,H,N] 각 query row 합
        assert torch.allclose(s, torch.ones_like(s), atol=1e-4), f"block {l} attention row 합≠1"


def test_block_output_indexing():
    model = _model()
    images = torch.randn(2, 3, 224, 224)
    cap = capture_full(model, images, capture_attention=False)
    L = len(cap.z_in)
    # block l 출력 == block l+1 입력
    for l in range(L - 1):
        assert torch.equal(cap.block_output(l), cap.z_in[l + 1])
    assert torch.equal(cap.block_output(L - 1), cap.z_final)


if __name__ == "__main__":
    ok = run_tests(dict(globals()))
    raise SystemExit(0 if ok else 1)
