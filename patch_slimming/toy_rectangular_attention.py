"""
Patch Slimming NPU 사전 검증용 Toy 스크립트 — Rectangular Attention 컴파일 테스트.

목적
    Patch Slimming(정적 PS-ViT)을 전체 구현하기 전에, 이 기법의 유일한 NPU
    리스크 요소인 "rectangular attention (N_out × N_in, 비정사각형)"과
    "컴파일타임 상수 인덱스 gather"가 Aries2/qbcompiler에서 컴파일되는지를
    5분 안에 확인한다.

왜 이게 EViT와 다른가 (핵심)
    EViT token pruning이 NPU에서 막힌 유일한 이유는 "어떤 토큰을 남길지"가
    런타임 값(TopK 결과)이라 Gather/Equal에 런타임 인덱스가 들어갔기 때문이다
    (token_pruning_archive/TOKEN_PRUNING.md §9.2 참고).

    Patch Slimming(정적)은 남길 토큰 인덱스가 offline search로 "고정된 상수"다.
    → 이 스크립트의 keep_ids는 register_buffer로 박힌 상수이며, TopK가 전혀 없다.
    → gather는 상수 인덱스 gather(= Stage 1 채널 pruning에서 이미 통과 확인된
       Slice/Gather와 같은 성격)라 통과가 기대된다.

    남는 단 하나의 불확실성은 attention이 정사각형(N×N)이 아니라 직사각형
    (N_out × N_in)이라는 점 — 이걸 이 toy가 직접 검증한다.

무엇을 그래프에 담는가 (Patch Slimming spec §5.1 / §14.1 Mode A)
    x[1, N_in, D]
      → LayerNorm
      → q = Linear_q( x_norm[:, keep_ids] )   # 상수 인덱스로 N_out개 행만 선택
      → k = Linear_k( x_norm )                # N_in 전체
      → v = Linear_v( x_norm )                # N_in 전체
      → attn = softmax( q @ kᵀ / √d )  ∈ [H, N_out, N_in]   ← ★ 직사각형 ★
      → out  = attn @ v                ∈ [H, N_out, d]
      → proj, residual( x[:, keep_ids] ), MLP
      → x_out[1, N_out, D]

    실제 학습 weight가 아니라 랜덤 weight를 쓴다 — "수치 정확도"가 아니라
    "이 연산 구성이 컴파일되는가"만 보는 것이므로 무방하다.

사용법 (서버에서)
    # 기본: ViT-Tiny block 3 형상 (N_in=196, N_out=138, D=192, H=3)
    python patch_slimming/toy_rectangular_attention.py

    # 형상 바꿔서
    python patch_slimming/toy_rectangular_attention.py --n-in 99 --n-out 70 --dim 192 --heads 3

    # 그 다음 나온 .onnx를 qbcompiler에 넣어본다:
    #   - HL 컴파일 통과 + quantize 통과 → 논문 그대로(rectangular) 전체 구현 GO
    #   - rectangular attention에서 막히면 → spec §14.3 Mode B(정사각형)로 구현

    onnx 그래프 op 확인은 --verify (onnx/onnxruntime 설치 시).
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlimAttentionBlock(nn.Module):
    """Patch Slimming Mode A rectangular attention 한 블록 (랜덤 weight).

    keep_ids: N_in개 입력 토큰 중 출력으로 유지할 N_out개의 "상수" global index.
              register_buffer라 ONNX 그래프에 상수로 박힌다 (런타임 계산 아님).
    """

    def __init__(self, dim: int, heads: int, n_in: int, n_out: int, mlp_ratio: float = 4.0):
        super().__init__()
        assert dim % heads == 0, "dim은 heads로 나눠떨어져야 한다"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        # 표준 pre-norm ViT block 구성 (weight는 랜덤 — 컴파일 테스트용)
        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)   # Q: 출력 N_out 토큰에만 적용
        self.k_proj = nn.Linear(dim, dim)   # K: 입력 N_in 전체
        self.v_proj = nn.Linear(dim, dim)   # V: 입력 N_in 전체
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

        # ── 핵심: 유지할 토큰 인덱스를 "상수"로 고정 ──
        # 실제 Patch Slimming에서는 offline search 결과가 들어가지만, 컴파일
        # 테스트에는 아무 고정 인덱스나 쓰면 된다. 흩어진 패턴(등간격)으로 골라
        # "연속 슬라이스"로 최적화되지 않도록 해서 진짜 gather를 만든다.
        keep = torch.linspace(0, n_in - 1, steps=n_out).round().long().unique()
        # unique로 개수가 줄면 앞에서부터 채워 정확히 n_out개 보장
        if keep.numel() < n_out:
            extra = torch.tensor([i for i in range(n_in) if i not in set(keep.tolist())])
            keep = torch.cat([keep, extra[: n_out - keep.numel()]]).sort().values
        self.register_buffer("keep_ids", keep, persistent=True)  # [N_out], 상수

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_in, D]
        B, N_in, D = x.shape
        H, dh = self.heads, self.head_dim

        x_norm = self.norm1(x)

        # 상수 인덱스 gather — N_out개 토큰 선택 (런타임 값 아님)
        x_sel = x_norm.index_select(1, self.keep_ids)   # [B, N_out, D]
        N_out = x_sel.shape[1]

        q = self.q_proj(x_sel).reshape(B, N_out, H, dh).permute(0, 2, 1, 3)   # [B,H,N_out,dh]
        k = self.k_proj(x_norm).reshape(B, N_in, H, dh).permute(0, 2, 1, 3)   # [B,H,N_in ,dh]
        v = self.v_proj(x_norm).reshape(B, N_in, H, dh).permute(0, 2, 1, 3)   # [B,H,N_in ,dh]

        # ★ Rectangular attention: [B,H,N_out,dh] @ [B,H,dh,N_in] = [B,H,N_out,N_in] ★
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)                       # softmax over N_in
        out = attn @ v                                    # [B,H,N_out,dh]
        out = out.permute(0, 2, 1, 3).reshape(B, N_out, D)
        out = self.proj(out)

        # residual도 상수 인덱스로 선택된 입력에서
        residual = x.index_select(1, self.keep_ids)       # [B, N_out, D]
        x_out = residual + out
        x_out = x_out + self.mlp(self.norm2(x_out))
        return x_out                                      # [B, N_out, D]


def main():
    p = argparse.ArgumentParser("Patch Slimming rectangular-attention toy for NPU compile test")
    p.add_argument("--n-in",  type=int, default=196, help="입력 토큰 수 (CLS 제외 patch 기준 예시)")
    p.add_argument("--n-out", type=int, default=138, help="출력으로 유지할 토큰 수 (< n-in)")
    p.add_argument("--dim",   type=int, default=192, help="embed dim (ViT-Tiny=192, Small=384)")
    p.add_argument("--heads", type=int, default=3,   help="attention heads (Tiny=3, Small=6)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--output", default="",
                   help="저장 경로. 생략 시 toy_rect_attn_in<N>_out<M>_d<D>.onnx 자동 네이밍")
    p.add_argument("--verify", action="store_true", help="onnx graph check + op type 출력")
    args = p.parse_args()

    assert args.n_out < args.n_in, "n_out은 n_in보다 작아야 rectangular attention이 의미 있다"

    model = SlimAttentionBlock(args.dim, args.heads, args.n_in, args.n_out).eval()
    dummy = torch.zeros(1, args.n_in, args.dim)

    with torch.no_grad():
        out = model(dummy)
    print(f"[Toy] forward OK  in={tuple(dummy.shape)} → out={tuple(out.shape)}")
    print(f"      keep_ids(상수, 앞 8개)= {model.keep_ids[:8].tolist()} ...  총 {model.keep_ids.numel()}개")
    print(f"      attention shape = [1, {args.heads}, {args.n_out}, {args.n_in}]  (rectangular)")

    out_path = args.output or (
        f"patch_slimming/toy_rect_attn_in{args.n_in}_out{args.n_out}_d{args.dim}.onnx"
    )

    print(f"\nExporting → {out_path} (opset {args.opset}, batch=1 고정) ...")
    torch.onnx.export(
        model, dummy, out_path,
        opset_version=args.opset,
        input_names=["input"], output_names=["output"],
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
        # batch 고정 — dynamic_axes 없음. NPU는 어차피 고정 shape.
    )
    print("Export 완료.")

    if args.verify:
        try:
            import onnx
            m = onnx.load(out_path)
            onnx.checker.check_model(m)
            ops = sorted({n.op_type for n in m.graph.node})
            outs = [o.name for o in m.graph.output]
            print(f"\n[Verify] ONNX graph check OK")
            print(f"  graph outputs ({len(outs)}): {outs}")
            print(f"  op types: {ops}")
            # 이 toy가 TopK/ScatterElements/GatherElements 없이 순수 상수 gather+matmul만
            # 쓰는지 확인 (그래야 EViT와 다른 조건임이 증명됨)
            for banned in ("TopK", "ScatterElements", "GatherElements"):
                if banned in ops:
                    print(f"  ⚠ {banned}가 있다 — 이 toy가 잘못 만들어졌다 (상수여야 함)")
            print("\n  → 이 .onnx를 qbcompiler에 넣어보라. Softmax/MatMul/Gather가 모두 "
                  "Supported로 나오고 quantize까지 통과하면 rectangular attention GO.")
        except ImportError:
            print("(onnx 미설치 — pip install onnx 후 --verify)")


if __name__ == "__main__":
    main()
