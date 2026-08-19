"""Backward-compatible entry point. Implementation: scripts/stage_ablation.py."""
from scripts.stage_ablation import POLICIES, _rank_ids, main, transformer_macs

if __name__ == "__main__":
    main()
