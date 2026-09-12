from __future__ import annotations

import argparse
from pathlib import Path

from .complex_experiment import default_complex_config, run_complex_experiment
from .contextual_holonomy import run_contextual_holonomy
from .depth_sweep import default_depth_sweep_config, run_depth_sweep
from .experiment import pilot_config, run_experiment, smoke_config
from .gauge_experiment import ATLAS_COMPONENTS, ATLAS_SCOPES, run_gauge_atlas
from .holonomy_audit import run_holonomy_audit
from .local_global_experiment import run_local_global_experiment
from .path_patching import run_path_patching
from .qkv_holonomy import run_qkv_holonomy
from .qkv_patching import run_qkv_patching
from .typed_gauge_experiment import run_typed_gauge_experiment


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
    qkv_holonomy = subparsers.add_parser(
        "qkv-holonomy",
        help="Measure balanced Q/K/V rewrite transport and holonomy at every layer.",
    )
    qkv_holonomy.add_argument("run_directory", type=Path)
    qkv_holonomy.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    qkv_patching = subparsers.add_parser(
        "qkv-patching",
        help="Causally patch pre-attention Q/K/V activations in a completed complex sweep.",
    )
    qkv_patching.add_argument("run_directory", type=Path)
    qkv_patching.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    qkv_patching.add_argument("--count-per-operator", type=int, default=16)
    contextual = subparsers.add_parser(
        "contextual-holonomy",
        help="Fit bidirectional context-conditioned subtree transports and Q/K geometry.",
    )
    contextual.add_argument("run_directory", type=Path)
    contextual.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    contextual.add_argument("--calibration-pairs", type=int, default=96)
    path_patching = subparsers.add_parser(
        "path-patching",
        help="Causally compare Q/K/V patches along equivalent rewrite paths.",
    )
    path_patching.add_argument("run_directory", type=Path)
    path_patching.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    gauge_atlas = subparsers.add_parser(
        "gauge-atlas",
        help="Fit paper-style local charts, defect connections, and fundamental-cycle holonomy.",
    )
    gauge_atlas.add_argument("run_directory", type=Path)
    gauge_atlas.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    gauge_atlas.add_argument("--chart-dimension", type=int, default=32)
    gauge_atlas.add_argument("--ridge", type=float, default=1e-2)
    gauge_atlas.add_argument("--persistence", type=float, default=0.0)
    gauge_atlas.add_argument("--fit-fraction", type=float, default=0.5)
    gauge_atlas.add_argument("--components", nargs="+", choices=ATLAS_COMPONENTS, default=None)
    gauge_atlas.add_argument("--scopes", nargs="+", choices=ATLAS_SCOPES, default=None)
    gauge_atlas.add_argument(
        "--heads",
        type=int,
        nargs="+",
        default=None,
        help="Restrict Q/K/V fibers to a causally nominated set of attention heads.",
    )
    gauge_atlas.add_argument(
        "--bit-flips",
        nargs="+",
        choices=("x0", "x1", "x2", "x3"),
        default=None,
        help="Repeat on paired assignment interventions without changing diagram syntax.",
    )
    typed_gauge = subparsers.add_parser(
        "typed-gauge",
        help="Run type-correct Q/P/relative holonomy with persistence and circuit controls.",
    )
    typed_gauge.add_argument("run_directory", type=Path)
    typed_gauge.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    typed_gauge.add_argument(
        "--conditions",
        nargs="+",
        choices=("higher_diversity", "low_diversity"),
        default=None,
    )
    typed_gauge.add_argument("--seeds", type=int, nargs="+", default=None)
    typed_gauge.add_argument("--chart-dimensions", type=int, nargs="+", default=None)
    typed_gauge.add_argument("--thresholds", type=float, nargs="+", default=None)
    typed_gauge.add_argument("--ridge", type=float, default=1e-2)
    typed_gauge.add_argument("--min-condition-ratio", type=float, default=0.02)
    typed_gauge.add_argument("--max-bootstrap-stability", type=float, default=0.35)
    typed_gauge.add_argument("--bootstrap-samples", type=int, default=16)
    typed_gauge.add_argument("--bitflip-variable", choices=("x0", "x1", "x2", "x3"), default="x0")
    local_global = subparsers.add_parser(
        "local-global",
        help="Compare global/family/local charts and correct/incorrect logical predictions.",
    )
    local_global.add_argument("run_directory", type=Path)
    local_global.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    local_global.add_argument(
        "--conditions",
        nargs="+",
        choices=("higher_diversity", "low_diversity"),
        default=None,
    )
    local_global.add_argument("--seeds", type=int, nargs="+", default=None)
    local_global.add_argument("--chart-dimensions", type=int, nargs="+", default=None)
    local_global.add_argument("--bootstrap-samples", type=int, default=8)
    local_global.add_argument("--min-sigma", type=float, default=0.0)
    local_global.add_argument("--min-condition-ratio", type=float, default=0.0)
    local_global.add_argument("--max-bootstrap-stability", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "local-global":
        result_directory = run_local_global_experiment(
            args.run_directory,
            device=args.device,
            conditions=tuple(args.conditions or ("higher_diversity",)),
            seeds=None if args.seeds is None else tuple(sorted(set(args.seeds))),
            chart_dimensions=tuple(args.chart_dimensions or (8,)),
            bootstrap_samples=args.bootstrap_samples,
            min_sigma=args.min_sigma,
            min_condition_ratio=args.min_condition_ratio,
            max_bootstrap_stability=args.max_bootstrap_stability,
        )
        print(f"Local/global artifacts: {result_directory.resolve()}")
        return
    if args.command == "typed-gauge":
        typed_directory = run_typed_gauge_experiment(
            args.run_directory,
            device=args.device,
            conditions=tuple(args.conditions or ("higher_diversity",)),
            seeds=None if args.seeds is None else tuple(sorted(set(args.seeds))),
            chart_dimensions=tuple(args.chart_dimensions or (8, 16)),
            thresholds=tuple(args.thresholds or (0.0, 0.015, 0.03, 0.06, 0.12, 0.24)),
            ridge=args.ridge,
            min_condition_ratio=args.min_condition_ratio,
            max_bootstrap_stability=args.max_bootstrap_stability,
            bootstrap_samples=args.bootstrap_samples,
            bitflip_variable=args.bitflip_variable,
        )
        print(f"Typed-gauge artifacts: {typed_directory.resolve()}")
        return
    if args.command == "gauge-atlas":
        atlas_directory = run_gauge_atlas(
            args.run_directory,
            device=args.device,
            chart_dimension=args.chart_dimension,
            ridge=args.ridge,
            persistence=args.persistence,
            fit_fraction=args.fit_fraction,
            components=tuple(args.components or ("query", "key", "value", "coupled_qk")),
            scopes=tuple(args.scopes or ("cls", "expression_root")),
            heads=None if args.heads is None else tuple(sorted(set(args.heads))),
            bit_flips=tuple(args.bit_flips or ()),
        )
        print(f"Gauge-atlas artifacts: {atlas_directory.resolve()}")
        return
    if args.command == "path-patching":
        path_directory = run_path_patching(args.run_directory, device=args.device)
        print(f"Path-patching artifacts: {path_directory.resolve()}")
        return
    if args.command == "contextual-holonomy":
        contextual_directory = run_contextual_holonomy(
            args.run_directory,
            device=args.device,
            calibration_pairs=args.calibration_pairs,
        )
        print(f"Contextual-holonomy artifacts: {contextual_directory.resolve()}")
        return
    if args.command == "qkv-patching":
        patch_directory = run_qkv_patching(
            args.run_directory,
            device=args.device,
            count_per_operator=args.count_per_operator,
        )
        print(f"QKV-patching artifacts: {patch_directory.resolve()}")
        return
    if args.command == "qkv-holonomy":
        qkv_directory = run_qkv_holonomy(args.run_directory, device=args.device)
        print(f"QKV-holonomy artifacts: {qkv_directory.resolve()}")
        return
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
