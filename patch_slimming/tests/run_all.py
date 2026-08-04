"""전체 게이팅 테스트 실행 (SPEC §16). Phase 1~2 및 scoring 정확성 검증.

    python patch_slimming/tests/run_all.py

ImageNet 없이 CPU에서 수 초 내 실행된다 (timm 모델은 pretrained=False, 랜덤 weight).
전부 PASS해야 run_search로 넘어간다 (특히 test_slim_block, test_score_dp).
"""
import os
import sys
import runpy

_TESTS = os.path.dirname(os.path.abspath(__file__))
MODULES = ["test_ids", "test_slim_block", "test_instrument", "test_score_dp", "test_compact"]


def main():
    failed = []
    for m in MODULES:
        print(f"\n{'#'*55}\n# {m}\n{'#'*55}")
        try:
            runpy.run_path(os.path.join(_TESTS, m + ".py"), run_name="__main__")
        except SystemExit as e:
            if e.code not in (0, None):
                failed.append(m)
        except Exception as e:
            print(f"  MODULE ERROR {m}: {type(e).__name__}: {e}")
            failed.append(m)
    print(f"\n{'='*55}")
    if failed:
        print(f"FAILED modules: {failed}")
        raise SystemExit(1)
    print("ALL TEST MODULES PASSED — run_search 진행 가능")


if __name__ == "__main__":
    main()
