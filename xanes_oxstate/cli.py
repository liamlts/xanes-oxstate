"""Entry points: python -m xanes_oxstate.cli."""
from __future__ import annotations

import argparse
from pathlib import Path


def cmd_build_data(args) -> None:
    from .data.run import build_element_dataset
    paths = build_element_dataset(
        args.element,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        report_dir=args.report_dir,
        seed=args.seed,
    )
    print(f"[{args.element}] {paths}")


def cmd_evaluate(args) -> None:
    import yaml
    from .eval.run import evaluate_element
    epochs = args.epochs
    seeds = (0, 1, 2, 3, 4)
    if args.config is not None:
        cfg = yaml.safe_load(args.config.read_text())
        epochs = cfg["training"]["epochs"]
        seeds = tuple(cfg["training"]["seeds"])
    metrics = evaluate_element(
        args.element,
        processed_dir=args.processed_dir,
        ckpt_dir=args.ckpt_dir,
        metrics_dir=args.metrics_dir,
        epochs=epochs,
        seeds=seeds,
    )
    print(f"[{args.element}] CNN acc = {metrics['accuracy']['cnn']:.3f}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m xanes_oxstate.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("build-data")
    bp.add_argument("--element", required=True)
    bp.add_argument("--raw-dir", default="data/raw", type=Path)
    bp.add_argument("--processed-dir", default="data/processed", type=Path)
    bp.add_argument("--report-dir", default="data/reports", type=Path)
    bp.add_argument("--seed", default=42, type=int)
    bp.set_defaults(func=cmd_build_data)

    ep = sub.add_parser("evaluate")
    ep.add_argument("--element", required=True)
    ep.add_argument("--processed-dir", default="data/processed", type=Path)
    ep.add_argument("--ckpt-dir", default="checkpoints", type=Path)
    ep.add_argument("--metrics-dir", default="metrics", type=Path)
    ep.add_argument("--epochs", default=50, type=int)
    ep.add_argument("--config", type=Path,
                    help="YAML config; overrides --epochs if present")
    ep.set_defaults(func=cmd_evaluate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
