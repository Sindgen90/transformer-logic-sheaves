from __future__ import annotations

import argparse
from pathlib import Path

from .complex_experiment import default_complex_config, run_complex_experiment
from .depth_sweep import default_depth_sweep_config, run_depth_sweep
from .experiment import pilot_config, run_experiment, smoke_config
from .holonomy_audit import run_holonomy_audit


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
    depth_sweep = subparsers.add_parser(
        "depth-sweep",
        help="Sweep 1-6 Transformer layers with dynamics, patching, and complete plots.",
    )
    depth_sweep.add_argument("--output-root", type=Path, default=Path("runs") / "depth_sweeps")
    depth_sweep.add_argument("--run-name", default=None)
    depth_sweep.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    depth_sweep.add_argument("--layers", type=int, nargs="+", choices=range(1, 7), default=None)
    depth_sweep.add_argument("--steps", type=int, default=3_000)
    depth_sweep.add_argument("--seeds", type=int, nargs="+", default=None)
    depth_sweep.add_argument("--dynamics-every", type=int, default=500)
    depth_sweep.add_argument("--validation-size", type=int, default=1_000)
    depth_sweep.add_argument("--test-size", type=int, default=1_000)
    depth_sweep.add_argument("--cycle-count", type=int, default=512)
    depth_sweep.add_argument("--dynamics-cycle-count", type=int, default=256)
    depth_sweep.add_argument("--patch-count", type=int, default=256)
    depth_sweep.add_argument("--dynamics-patch-count", type=int, default=64)
    depth_sweep.add_argument("--dynamics-analysis-size", type=int, default=256)
    complex_sweep = subparsers.add_parser(
        "complex-sweep",
        help="Run the symbolic, higher-arity, multi-diagram equivalence experiment.",
    )
    complex_sweep.add_argument(
        "--output-root", type=Path, default=Path("runs") / "equivalence_complexes"
    )
    complex_sweep.add_argument("--run-name", default=None)
    complex_sweep.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    complex_sweep.add_argument("--layers", type=int, nargs="+", choices=range(1, 9), default=None)
    complex_sweep.add_argument("--steps", type=int, default=2_000)
    complex_sweep.add_argument("--seeds", type=int, nargs="+", default=None)
    complex_sweep.add_argument("--validation-size", type=int, default=1_000)
    complex_sweep.add_argument("--test-size", type=int, default=1_000)
    complex_sweep.add_argument("--calibration-pairs", type=int, default=192)
    complex_sweep.add_argument("--diagrams-per-family", type=int, default=96)
    holonomy_audit = subparsers.add_parser(
        "holonomy-audit",
        help="Audit a completed complex sweep with operator diagnostics and null connections.",
    )
    holonomy_audit.add_argument("run_directory", type=Path)
    holonomy_audit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "holonomy-audit":
        audit_directory = run_holonomy_audit(args.run_directory, device=args.device)
        print(f"Holonomy-audit artifacts: {audit_directory.resolve()}")
        return
    if args.command == "complex-sweep":
        config = default_complex_config(
            args.output_root,
            run_name=args.run_name,
            layers=tuple(args.layers or (2, 4, 6)),
            seeds=tuple(args.seeds or (0, 1, 2)),
            steps=args.steps,
            device=args.device,
            validation_size=args.validation_size,
            test_size=args.test_size,
            calibration_pairs_per_rewrite=args.calibration_pairs,
            diagrams_per_family=args.diagrams_per_family,
        )
        run_directory = run_complex_experiment(config)
        print(f"Equivalence-complex artifacts: {run_directory.resolve()}")
        return
    if args.command == "depth-sweep":
        config = default_depth_sweep_config(
            args.output_root,
            run_name=args.run_name,
            layers=tuple(args.layers or range(1, 7)),
            seeds=tuple(args.seeds or (0, 1, 2)),
            steps=args.steps,
            dynamics_every=args.dynamics_every,
            device=args.device,
            validation_size=args.validation_size,
            test_size=args.test_size,
            cycle_count=args.cycle_count,
            dynamics_cycle_count=args.dynamics_cycle_count,
            patch_count=args.patch_count,
            dynamics_patch_count=args.dynamics_patch_count,
            dynamics_analysis_size=args.dynamics_analysis_size,
        )
        run_directory = run_depth_sweep(config)
        print(f"Depth sweep artifacts: {run_directory.resolve()}")
        return
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
