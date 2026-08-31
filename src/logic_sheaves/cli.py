from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import pilot_config, run_experiment, smoke_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run controlled Boolean-representation coherence experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("smoke", "Run a fast two-condition pipeline check."),
        ("pilot", "Run the first multi-seed pilot experiment."),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--output", type=Path, default=Path("runs") / name)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        command.add_argument("--steps", type=int, default=None)
        command.add_argument("--seeds", type=int, nargs="+", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    default_seeds = (0,) if args.command == "smoke" else (0, 1, 2)
    seeds = tuple(args.seeds) if args.seeds else default_seeds
    if args.command == "smoke":
        config = smoke_config(args.output, device=args.device, steps=args.steps, seeds=seeds)
    else:
        config = pilot_config(args.output, device=args.device, steps=args.steps, seeds=seeds)
    rows = run_experiment(config)
    print(f"Wrote {len(rows)} runs to {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
