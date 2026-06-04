from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys

# Allow `python scripts/train_pn_flowse.py` without installing the package first.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sia_fm_tse.training.pn_flowse import train


def main() -> None:
    parser = ArgumentParser(description="Train PN-conditioned FlowSE for TSE.")
    parser.add_argument("--config", default="configs/pn_flowse.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    train(args.config, device_name=args.device, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
