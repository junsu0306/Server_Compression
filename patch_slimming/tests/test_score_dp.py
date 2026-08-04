"""§16.5 — Path-energy DP score vs brute-force head-path 열거 일치 검증.

이 테스트가 통과하지 않으면 전체 mask search를 시작하지 않는다 (SPEC §7.6, §16.5).
"""
import torch
from _util import run_tests

from psvit.scoring import (_downstream_energy_dp, bruteforce_downstream_energy,
                           compute_sample_scores, scatter_attention_to_global)

torch.manual_seed(0)


def _random_attn(B, H, N):
    """행 합이 1인 임의 attention [B,H,N,N] (softmax)."""
    return torch.randn(B, H, N, N).softmax(dim=-1)


def test_dp_matches_bruteforce_2layers():
    B, H, N = 2, 2, 4
    downstream = [_random_attn(B, H, N) for _ in range(2)]
    dp = _downstream_energy_dp(downstream)
    bf = bruteforce_downstream_energy(downstream)
    diff = (dp - bf).abs().max().item()
    assert diff <= 1e-5, f"DP≠brute-force diff={diff:.2e}"


def test_dp_matches_bruteforce_3layers():
    B, H, N = 1, 3, 5
    downstream = [_random_attn(B, H, N) for _ in range(3)]  # H^m = 27 paths
    dp = _downstream_energy_dp(downstream)
    bf = bruteforce_downstream_energy(downstream)
    diff = (dp - bf).abs().max().item()
    assert diff <= 1e-5, f"DP≠brute-force diff={diff:.2e}"


def test_dp_single_layer_identity():
    # 1개 downstream, G = mean_h P^T I P. brute-force와 동일해야.
    B, H, N = 2, 4, 6
    downstream = [_random_attn(B, H, N)]
    dp = _downstream_energy_dp(downstream)
    bf = bruteforce_downstream_energy(downstream)
    assert (dp - bf).abs().max().item() <= 1e-6


def test_compute_sample_scores_shape_and_finite():
    B, H, N, D = 2, 3, 5, 8
    z_prev = torch.randn(B, N, D)
    tgt = _random_attn(B, H, N)
    downstream = [_random_attn(B, H, N) for _ in range(2)]
    s = compute_sample_scores(z_prev, tgt, downstream)
    assert s.shape == (B, N), s.shape
    assert torch.isfinite(s).all()
    assert (s >= 0).all(), "score 음수 (clamp 후엔 ≥0이어야)"


def test_scatter_places_rows_cols():
    B, H, N = 1, 1, 5
    out_ids = torch.tensor([0, 2, 4])
    in_ids = torch.tensor([0, 1, 2, 3, 4])
    local = torch.arange(3 * 5, dtype=torch.float).reshape(1, 1, 3, 5)
    g = scatter_attention_to_global(local, out_ids, in_ids, N)
    assert g.shape == (1, 1, 5, 5)
    # row 1, 3 (선택 안 된 output)은 0
    assert g[0, 0, 1].abs().sum() == 0 and g[0, 0, 3].abs().sum() == 0
    # row 2 (=out_ids[1]) == local row 1
    assert torch.equal(g[0, 0, 2], local[0, 0, 1])


if __name__ == "__main__":
    ok = run_tests(dict(globals()))
    raise SystemExit(0 if ok else 1)
