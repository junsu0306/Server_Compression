"""
Patch Slimming NPU 사전 검증용 Toy 스크립트 — Rectangular Attention + 상수 토큰 선택.

목적
    Patch Slimming(정적 PS-ViT)을 전체 구현하기 전에, NPU(Aries2/qbcompiler)에서
    두 가지가 컴파일되는지 최소 그래프로 검증한다:
      (1) rectangular attention (N_out × N_in, 비정사각형)
      (2) "고정된(offline-search로 확정된) 토큰 집합"을 실제로 골라내는 연산

실측으로 확인된 사실 (2026-08, token_pruning_archive/TOKEN_PRUNING.md 참고)
    - rectangular attention: ✅ 컴파일됨. MatMul/Softmax 전부 100% Supported.
    - ONNX `Gather`(index_select): ❌ keep_ids가 **상수**여도 Unsupported.
      Aries2는 activation을 흩어진 위치로 gather하는 것 자체를 안 받고, CPU
      offload로 빠지면서 qbcompiler의 서브그래프 직렬화 버그(map::at)로 크래시.

핵심 해법 — 상수 선택행렬 MatMul
    keep_ids가 컴파일타임 상수이므로, "흩어진 N_out개 행 선택"을
        selected = P @ x         # P: 상수 [N_out, N_in],  P[i, keep_ids[i]] = 1
    로 표현할 수 있다. P는 컴파일타임 상수라 이건 그냥 "weight가 상수인 MatMul"
    = Linear/Conv와 동일하다. Gather도, 런타임 인덱스도, Equal/Cast도 없다.

    EViT(token pruning)에서 이 방식이 막혔던 이유는 onehot을 **런타임 topk
    인덱스로부터 Equal로** 만들어야 했기 때문이다. Patch Slimming은 선택이
    상수라 그 Equal이 애초에 없다 — 그래서 여기서는 통과가 기대된다.

    --select-mode matmul  (기본, 권장):  상수 선택행렬 P @ x
    --select-mode gather  (비교용):     x.index_select (= 실측 실패한 방식)

무엇을 그래프에 담는가 (Patch Slimming spec §5.1 / §14.1 Mode A)
    x[1, N_in, D]
      → LayerNorm
      → x_sel = SELECT(x_norm)                # N_out개 선택 (matmul 또는 gather)
      → q = Linear_q(x_sel)                    # [N_out]
      → k = Linear_k(x_norm), v = Linear_v(x_norm)   # [N_in]
      → attn = softmax(q @ kᵀ / √d)  ∈ [H, N_out, N_in]   ← ★ 직사각형 ★
      → out  = attn @ v
      → proj, residual( SELECT(x) ), MLP
      → x_out[1, N_out, D]

    실제 학습 weight가 아니라 랜덤 weight를 쓴다 — "이 연산 구성이 컴파일되는가"
    만 보는 것이므로 무방하다.

사용법 (서버에서, repo 루트 기준)
    # 권장: 상수 선택행렬 matmul (Gather 없음)
    python patch_slimming/toy_rectangular_attention.py --verify

    # 비교: 예전에 실패한 gather 방식
    python patch_slimming/toy_rectangular_attention.py --select-mode gather --verify

    # 형상 바꿔서 (ViT-Tiny block 6/9 등)
    python patch_slimming/toy_rectangular_attention.py --n-in 139 --n-out 98

    # 나온 .onnx를 qbcompiler에 넣어본다:
    #   matmul 모드가 HL 컴파일 + quantize 둘 다 통과 → Patch Slimming 전체 구현 GO
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn


def _make_scattered_keep(n_in: int, n_out: int) -> torch.Tensor:
    """등간격으로 흩어진 N_out개 인덱스 (연속 Slice로 최적화되지 않게)."""
    keep = torch.linspace(0, n_in - 1, steps=n_out).round().long().unique()
    if keep.numel() < n_out:
        extra = torch.tensor([i for i in range(n_in) if i not in set(keep.tolist())])
        keep = torch.cat([keep, extra[: n_out - keep.numel()]]).sort().values
    return keep


class SlimAttentionBlock(nn.Module):
    """Patch Slimming Mode A rectangular attention 한 블록 (랜덤 weight).

    select_mode:
        "matmul" — 상수 선택행렬 P[N_out,N_in] @ x  (Gather 없음, NPU 통과 기대)
        "gather" — x.index_select(1, keep_ids)      (실측 미지원 — 비교용)
    """

    def __init__(self, dim, heads, n_in, n_out, select_mode="matmul", mlp_ratio=4.0):
        super().__init__()
        assert dim % heads == 0, "dim은 heads로 나눠떨어져야 한다"
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.select_mode = select_mode

        self.norm1 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)   # Q: 출력 N_out 토큰에만
        self.k_proj = nn.Linear(dim, dim)   # K: 입력 N_in 전체
        self.v_proj = nn.Linear(dim, dim)   # V: 입력 N_in 전체
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        # offline-search 결과를 흉내낸 "고정" keep 인덱스
        keep = _make_scattered_keep(n_in, n_out)
        self.register_buffer("keep_ids", keep, persistent=True)          # [N_out] 상수

        # 상수 선택행렬 P: P[i, keep_ids[i]] = 1  → P @ x == x[keep_ids]
        P = torch.zeros(n_out, n_in)
        P[torch.arange(n_out), keep] = 1.0
        self.register_buffer("select_mat", P, persistent=True)           # [N_out,N_in] 상수

    def _select(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N_in, D] → [B, N_out, D]. 선택 방식만 다르고 결과는 동일."""
        if self.select_mode == "matmul":
            # 상수 P @ activation → ONNX MatMul (weight=상수). Gather 없음.
            return torch.matmul(self.select_mat, x)
        elif self.select_mode == "gather":
            return x.index_select(1, self.keep_ids)                      # ONNX Gather
        raise ValueError(f"select_mode: {self.select_mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N_in, D = x.shape
        H, dh = self.heads, self.head_dim

        x_norm = self.norm1(x)
        x_sel = self._select(x_norm)                # [B, N_out, D]
        N_out = x_sel.shape[1]

        q = self.q_proj(x_sel).reshape(B, N_out, H, dh).permute(0, 2, 1, 3)  # [B,H,N_out,dh]
        k = self.k_proj(x_norm).reshape(B, N_in, H, dh).permute(0, 2, 1, 3)  # [B,H,N_in ,dh]
        v = self.v_proj(x_norm).reshape(B, N_in, H, dh).permute(0, 2, 1, 3)  # [B,H,N_in ,dh]

        # ★ Rectangular attention: [B,H,N_out,dh] @ [B,H,dh,N_in] = [B,H,N_out,N_in] ★
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N_out, D)           # [B,N_out,D]
        out = self.proj(out)

        residual = self._select(x)                  # [B, N_out, D]
        x_out = residual + out
        x_out = x_out + self.mlp(self.norm2(x_out))
        return x_out                                # [B, N_out, D]


class ToyPatchSlimViT(nn.Module):
    """patch_embed(Conv2d) + SlimAttentionBlock 한 개짜리 미니 ViT.

    입력을 **NHWC [1,224,224,3]** 로 받는다 — Mobilint calibration 파이프라인이
    이미지를 [224,224,3](HWC) 레이아웃으로 넣고 자동 transpose를 안 해주기 때문.
    (NCHW [1,3,224,224]로 받으면 conv 채널 자리에 224가 들어가 quantize 단계에서
    "expected 3 channels but got 224"로 실패한다.) 내부에서 NCHW로 permute한 뒤
    conv를 태운다 — 이 시작 Transpose는 Aries2에서 100% Supported로 확인됨.

    실제 Patch Slimming 모델도 export 시 입력 레이아웃을 이 파이프라인에 맞춰야
    한다 (또는 compile 설정에서 입력 transpose를 지정).
    """

    def __init__(self, dim, heads, img_size, patch, n_out, select_mode="matmul"):
        super().__init__()
        assert img_size % patch == 0
        grid = img_size // patch
        n_in = grid * grid                                   # 224/16=14 → 196
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.block = SlimAttentionBlock(dim, heads, n_in, n_out, select_mode)

    @property
    def select_mode(self):
        return self.block.select_mode

    @select_mode.setter
    def select_mode(self, v):
        self.block.select_mode = v

    def forward(self, img: torch.Tensor) -> torch.Tensor:    # img: [B,224,224,3] NHWC
        x = img.permute(0, 3, 1, 2)                          # → [B,3,224,224] NCHW
        x = self.patch_embed(x)                              # [B, dim, grid, grid]
        x = x.flatten(2).transpose(1, 2)                     # [B, N_in, dim]
        return self.block(x)                                 # [B, N_out, dim]


def main():
    p = argparse.ArgumentParser("Patch Slimming rectangular-attention toy for NPU compile test")
    p.add_argument("--img-size", type=int, default=224, help="입력 이미지 크기")
    p.add_argument("--patch",    type=int, default=16,  help="patch 크기 → N_in = (img/patch)^2")
    p.add_argument("--n-out",    type=int, default=138, help="유지할 토큰 수 (< N_in)")
    p.add_argument("--dim",      type=int, default=192, help="embed dim (Tiny=192, Small=384)")
    p.add_argument("--heads",    type=int, default=3,   help="heads (Tiny=3, Small=6)")
    p.add_argument("--select-mode", default="matmul", choices=["matmul", "gather"],
                   help="matmul=상수 선택행렬(권장) | gather=index_select(실측 미지원, 비교용)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--output", default="")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    n_in = (args.img_size // args.patch) ** 2
    assert args.n_out < n_in, f"n_out({args.n_out}) < N_in({n_in}) 이어야 한다"

    # 이미지 입력 미니 ViT — calibration 이미지(NHWC [224,224,3]) 레이아웃에 맞춤
    model = ToyPatchSlimViT(args.dim, args.heads, args.img_size, args.patch,
                            args.n_out, args.select_mode).eval()
    dummy = torch.zeros(1, args.img_size, args.img_size, 3)  # [1,224,224,3] NHWC

    with torch.no_grad():
        out = model(dummy)
        # matmul/gather 두 선택 방식이 수치적으로 동일한지 자체 확인 (같은 weight)
        alt_mode = "gather" if args.select_mode == "matmul" else "matmul"
        model.select_mode = alt_mode
        out_alt = model(dummy)
        model.select_mode = args.select_mode
        max_diff = (out - out_alt).abs().max().item()

    print(f"[Toy] select_mode={args.select_mode}  forward OK  "
          f"in={tuple(dummy.shape)} → out={tuple(out.shape)}")
    print(f"      N_in={n_in}  N_out={args.n_out}  keep_ids(앞 8)= "
          f"{model.block.keep_ids[:8].tolist()} ...")
    print(f"      matmul vs gather 결과 최대차이 = {max_diff:.2e}  (동일해야 정상)")
    print(f"      attention shape = [1, {args.heads}, {args.n_out}, {n_in}]  (rectangular)")

    out_path = args.output or (
        f"patch_slimming/toy_rect_{args.select_mode}_in{n_in}_out{args.n_out}_d{args.dim}.onnx"
    )

    print(f"\nExporting → {out_path} (opset {args.opset}, batch=1 고정) ...")
    torch.onnx.export(
        model, dummy, out_path,
        opset_version=args.opset,
        input_names=["input"], output_names=["output"],
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
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
            if args.select_mode == "matmul":
                if "Gather" in ops:
                    print("  ⚠ matmul 모드인데 Gather가 있다 — 선택행렬이 gather로 lowering됐을 수 있음")
                else:
                    print("  ✓ Gather 없음 — 순수 MatMul/Softmax/LayerNorm만. NPU 통과 기대.")
            for banned in ("TopK", "ScatterElements", "GatherElements", "Equal"):
                if banned in ops:
                    print(f"  ⚠ {banned} 존재 — 이 toy가 잘못 만들어짐 (런타임 op 없어야 함)")
            print("\n  → 이 .onnx를 qbcompiler에 넣어보라. matmul 모드가 quantize까지 "
                  "통과하면 Patch Slimming 전체 구현 GO.")
        except ImportError:
            print("(onnx 미설치 — pip install onnx 후 --verify)")


if __name__ == "__main__":
    main()
