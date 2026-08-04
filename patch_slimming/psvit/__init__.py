"""
psvit — 정적 Patch Slimming (PS-ViT) 구현 패키지.

레퍼런스: Tang et al., "Patch Slimming for Efficient Vision Transformers", CVPR 2022.
구현 명세: patch_slimming/PATCH_SLIMMING_IMPLEMENTATION_SPEC.md (이하 SPEC).

모듈 구성:
    ids.py          §3  global/local token ID, mask invariant
    model_utils.py      timm ViT adapter, 검증, block 접근
    data.py             calibration subset (deterministic)
    slim_block.py   §5  SlimBlock (rectangular attention)
    instrument.py   §6  attention/feature 계측
    scoring.py      §7  significance score (path-energy DP)
    reconstruct.py  §9  block reconstruction fine-tuning
    search.py     §8,11 Algorithm 1 top-down mask search
    architecture.py §4  architecture spec 직렬화
    compact.py    §12,14 compact PS-ViT (NPU-safe 상수 선택행렬)

주의: 이 패키지는 서버(GPU)에서 실행/검증한다. 논문에 없는 세부는 SPEC의
[DECISION]을 따르며, 각 Phase는 SPEC §16의 테스트로 게이트한다.
"""
