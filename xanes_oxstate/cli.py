"""Entry points: python -m xanes_oxstate.<cmd>."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data.run import build_element_dataset


def cmd_build_data(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m xanes_oxstate.build_data")
    p.add_argument("--element", required=True)
    p.add_argument("--raw-dir", default="data/raw", type=Path)
    p.add_argument("--processed-dir", default="data/processed", type=Path)
    p.add_argument("--report-dir", default="data/reports", type=Path)
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args(argv)

    paths = build_element_dataset(
        args.element,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        report_dir=args.report_dir,
        seed=args.seed,
    )
    print(f"[{args.element}] {paths}")


if __name__ == "__main__":
    cmd_build_data()
