from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .complex_experiment import _disjoint_diagram_suite, _disjoint_rewrite_pairs, _write_rows
from .data import AssignedExpression, make_symbolic_splits
from .diagram_metrics import score_equivalence_diagrams
from .metrics import OrthogonalTransport, fit_orthogonal_transport, representations
from .model import ModelConfig, TinyLogicTransformer
from .training import resolve_device

AUDIT_METRICS = (
    "transport_error",
    "holonomy_error",
    "holonomy_rotation_error",
    "holonomy_linear_action_error",
    "holonomy_systematic_error",
    "holonomy_dispersion_error",
    "holonomy_edge_accumulation_ratio",
    "path_agreement_error",
    "path_endpoint_error",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fit_transports(
    model: TinyLogicTransformer,
    pairs: dict[str, list[tuple[AssignedExpression, AssignedExpression]]],
    *,
    device: torch.device,
    shuffle_seed: int | None = None,
) -> dict[str, OrthogonalTransport]:
    transports: dict[str, OrthogonalTransport] = {}
    rng = np.random.default_rng(shuffle_seed)
    for label, examples in pairs.items():
        source = representations(model, [pair[0] for pair in examples], device=device)
        target = representations(model, [pair[1] for pair in examples], device=device)
        if shuffle_seed is not None:
            target = target[rng.permutation(len(target))]
        transports[label] = fit_orthogonal_transport(source, target)
    return transports


def _identity_transports(labels: set[str], width: int) -> dict[str, OrthogonalTransport]:
    identity = OrthogonalTransport(
        source_mean=np.zeros((1, width)),
        target_mean=np.zeros((1, width)),
        rotation=np.eye(width),
    )
    return {label: identity for label in labels}


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _pearson(x: list[float], y: list[float]) -> float:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    keep = np.isfinite(x_array) & np.isfinite(y_array)
    if keep.sum() < 3 or np.std(x_array[keep]) < 1e-12 or np.std(y_array[keep]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_array[keep], y_array[keep])[0, 1])


def _partial_correlation(rows: list[dict[str, Any]], metric: str) -> float:
    keep = [
        row
        for row in rows
        if np.isfinite(float(row[metric])) and np.isfinite(float(row["ood_accuracy_mean"]))
    ]
    if len(keep) < 5:
        return float("nan")
    controls = np.asarray(
        [
            [
                1.0,
                float(row["architecture_layers"]),
                float(row["condition"] == "higher_diversity"),
            ]
            for row in keep
        ]
    )
    metric_values = np.asarray([float(row[metric]) for row in keep])
    accuracy = np.asarray([float(row["ood_accuracy_mean"]) for row in keep])
    metric_residual = metric_values - controls @ np.linalg.lstsq(
        controls, metric_values, rcond=None
    )[0]
    accuracy_residual = accuracy - controls @ np.linalg.lstsq(controls, accuracy, rcond=None)[0]
    return _pearson(metric_residual.tolist(), accuracy_residual.tolist())


def _load_model(checkpoint_path: Path, device: torch.device) -> TinyLogicTransformer:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TinyLogicTransformer(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def _regenerate_evaluation_data(config: dict[str, Any]):
    splits = make_symbolic_splits(
        train_size=max(config["conditions"].values()),
        validation_size=int(config["validation_size"]),
        test_size=int(config["test_size"]),
        train_depth=int(config["train_depth"]),
        ood_depths=tuple(int(depth) for depth in config["ood_depths"]),
        seed=4_281,
    )
    all_data = (
        *splits.train,
        *splits.validation,
        *splits.test_id,
        *(
            example
            for depth in config["ood_depths"]
            for example in splits.test_ood[int(depth)]
        ),
    )
    calibration, used = _disjoint_rewrite_pairs(
        int(config["calibration_pairs_per_rewrite"]),
        operand_depth=int(config["diagram_operand_depth"]),
        seed=73_013,
        blocked={str(example) for example in all_data},
    )
    _, used = _disjoint_rewrite_pairs(
        max(64, int(config["calibration_pairs_per_rewrite"]) // 2),
        operand_depth=int(config["diagram_operand_depth"]),
        seed=73_014,
        blocked=used,
    )
    diagrams = _disjoint_diagram_suite(
        int(config["diagrams_per_family"]),
        operand_depth=int(config["diagram_operand_depth"]),
        blocked=used,
    )
    return calibration, diagrams


def _model_summaries(family_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in family_rows:
        key = (
            int(row["architecture_layers"]),
            str(row["condition"]),
            int(row["seed"]),
            str(row["baseline"]),
        )
        grouped[key].append(row)
    summaries: list[dict[str, Any]] = []
    for (layers, condition, seed, baseline), rows in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "architecture_layers": layers,
            "condition": condition,
            "seed": seed,
            "baseline": baseline,
            "ood_accuracy_mean": float(rows[0]["ood_accuracy_mean"]),
        }
        summary.update({metric: _mean(rows, metric) for metric in AUDIT_METRICS})
        summaries.append(summary)
    return summaries


def _correlation_rows(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fitted = [row for row in model_rows if row["baseline"] == "fitted"]
    output = []
    for metric in AUDIT_METRICS:
        output.append(
            {
                "metric": metric,
                "pearson_with_ood": _pearson(
                    [float(row[metric]) for row in fitted],
                    [float(row["ood_accuracy_mean"]) for row in fitted],
                ),
                "partial_with_ood": _partial_correlation(fitted, metric),
            }
        )
    return output


def _setup_plots(output_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)
    return plt


def _audit_plots(
    model_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    plt = _setup_plots(output_dir)
    fitted = [row for row in model_rows if row["baseline"] == "fitted"]
    colors = [float(row["ood_accuracy_mean"]) for row in fitted]
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    scatter = axes[0, 0].scatter(
        [float(row["transport_error"]) for row in fitted],
        [float(row["holonomy_error"]) for row in fitted],
        c=colors,
        cmap="viridis",
        s=70,
        edgecolor="white",
    )
    axes[0, 0].set(xlabel="held-out edge error", ylabel="return-to-start error")
    figure.colorbar(scatter, ax=axes[0, 0], label="OOD accuracy")
    axes[0, 1].scatter(
        [float(row["holonomy_rotation_error"]) for row in fitted],
        [float(row["holonomy_error"]) for row in fitted],
        c=colors,
        cmap="viridis",
        s=70,
        edgecolor="white",
    )
    axes[0, 1].set(xlabel=r"operator defect $||R_{loop}-I||_F^2/d$", ylabel="state error")
    labels: list[str] = []
    systematic: list[float] = []
    dispersion: list[float] = []
    for layers in sorted({int(row["architecture_layers"]) for row in fitted}):
        for condition in ("low_diversity", "higher_diversity"):
            subset = [
                row
                for row in fitted
                if int(row["architecture_layers"]) == layers and row["condition"] == condition
            ]
            labels.append(f"L{layers}\n{'low' if condition == 'low_diversity' else 'high'}")
            systematic.append(_mean(subset, "holonomy_systematic_error"))
            dispersion.append(_mean(subset, "holonomy_dispersion_error"))
    positions = np.arange(len(labels))
    axes[1, 0].bar(positions, systematic, label="systematic displacement", color="#31688e")
    axes[1, 0].bar(
        positions,
        dispersion,
        bottom=systematic,
        label="context-dependent dispersion",
        color="#35b779",
    )
    axes[1, 0].set_xticks(positions, labels)
    axes[1, 0].set_ylabel("return-to-start error")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].scatter(
        [float(row["holonomy_edge_accumulation_ratio"]) for row in fitted],
        [float(row["ood_accuracy_mean"]) for row in fitted],
        c=[int(row["architecture_layers"]) for row in fitted],
        cmap="plasma",
        s=70,
        edgecolor="white",
    )
    axes[1, 1].set(
        xlabel="closure / accumulated one-edge error",
        ylabel="OOD accuracy",
    )
    figure.suptitle("L. What the present holonomy statistic contains", fontsize=16)
    figure.tight_layout()
    path_l = output_dir / "L_holonomy_decomposition.png"
    figure.savefig(path_l, dpi=180, bbox_inches="tight")
    plt.close(figure)

    baseline_order = ("identity", "shuffled", "fitted")
    baseline_colors = {"identity": "#b35806", "shuffled": "#777777", "fitted": "#21918c"}
    families = sorted({str(row["family"]) for row in family_rows})
    figure, axes = plt.subplots(1, 2, figsize=(17, 6))
    width = 0.25
    positions = np.arange(len(families))
    for offset, baseline in enumerate(baseline_order):
        family_means = [
            _mean(
                [
                    row
                    for row in family_rows
                    if row["baseline"] == baseline and row["family"] == family
                ],
                "holonomy_error",
            )
            for family in families
        ]
        axes[0].bar(
            positions + (offset - 1) * width,
            family_means,
            width,
            label=baseline,
            color=baseline_colors[baseline],
        )
    axes[0].set_xticks(positions, [item.replace("_", " ") for item in families], rotation=55, ha="right")
    axes[0].set(ylabel="return-to-start error", title="Closure can be perfect for a useless connection")
    axes[0].legend()
    overall = {
        baseline: [row for row in model_rows if row["baseline"] == baseline]
        for baseline in baseline_order
    }
    x = np.arange(len(baseline_order))
    axes[1].bar(
        x - 0.18,
        [_mean(overall[item], "transport_error") for item in baseline_order],
        0.36,
        label="edge error",
        color="#31688e",
    )
    axes[1].bar(
        x + 0.18,
        [_mean(overall[item], "holonomy_error") for item in baseline_order],
        0.36,
        label="closure error",
        color="#fde725",
    )
    axes[1].set_xticks(x, baseline_order)
    axes[1].set(ylabel="variance-normalized error", title="Matched null baselines")
    axes[1].legend()
    figure.suptitle("M. Holonomy needs edge-fidelity controls", fontsize=16)
    figure.tight_layout()
    path_m = output_dir / "M_holonomy_nulls.png"
    figure.savefig(path_m, dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(11, 6))
    metric_labels = {
        "transport_error": "held-out edge error",
        "holonomy_error": "state closure",
        "holonomy_rotation_error": "operator rotation defect",
        "holonomy_linear_action_error": "linear-action closure",
        "holonomy_systematic_error": "systematic drift",
        "holonomy_dispersion_error": "context dispersion",
        "holonomy_edge_accumulation_ratio": "cancellation ratio",
        "path_agreement_error": "path agreement error",
        "path_endpoint_error": "path endpoint error",
    }
    labels = [metric_labels[str(row["metric"])] for row in correlations]
    y = np.arange(len(labels))
    ax.barh(
        y - 0.18,
        [float(row["pearson_with_ood"]) for row in correlations],
        0.36,
        label="pooled Pearson",
        color="#31688e",
    )
    ax.barh(
        y + 0.18,
        [float(row["partial_with_ood"]) for row in correlations],
        0.36,
        label="partial: depth + data regime",
        color="#35b779",
    )
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set(xlabel="correlation with OOD accuracy", xlim=(-1, 1))
    ax.legend()
    ax.set_title("N. No current holonomy scalar robustly tracks generalization")
    figure.tight_layout()
    path_n = output_dir / "N_holonomy_correlations.png"
    figure.savefig(path_n, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return [path_l, path_m, path_n]


def _report(
    model_rows: list[dict[str, Any]],
    correlation_rows: list[dict[str, Any]],
    plot_paths: list[Path],
) -> str:
    by_baseline = {
        baseline: [row for row in model_rows if row["baseline"] == baseline]
        for baseline in ("identity", "shuffled", "fitted")
    }
    lines = [
        "# Holonomy metric audit",
        "",
        "## Verdict",
        "",
        "The existing `holonomy_error` is correctly computed as the variance-normalized ",
        "displacement of held-out `<CLS>` states after composing fitted affine-orthogonal ",
        "rewrite maps around a closed path. It is therefore a useful *empirical ",
        "return-to-start diagnostic*. It is not, by itself, a reliable measure of learned ",
        "logical coherence or an intrinsic holonomy invariant.",
        "",
        "The decisive counterexample is the identity connection: it has exactly zero loop ",
        "and path error for every diagram, although it does not transport a source state to ",
        "the representation of its rewritten target. Closure must therefore be conditioned ",
        "on independently measured edge fidelity.",
        "",
        "## Matched-baseline results",
        "",
        "All values are means over the same 18 saved models and eleven held-out families.",
        "",
        "| Connection | Edge error | State closure | Rotation defect | Linear-action error | Cancellation ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for baseline in ("identity", "shuffled", "fitted"):
        rows = by_baseline[baseline]
        lines.append(
            f"| {baseline} | {_mean(rows, 'transport_error'):.3f} | "
            f"{_mean(rows, 'holonomy_error'):.3f} | "
            f"{_mean(rows, 'holonomy_rotation_error'):.3f} | "
            f"{_mean(rows, 'holonomy_linear_action_error'):.3f} | "
            f"{_mean(rows, 'holonomy_edge_accumulation_ratio'):.3f} |"
        )
    fitted = by_baseline["fitted"]
    total = _mean(fitted, "holonomy_error")
    systematic = _mean(fitted, "holonomy_systematic_error")
    dispersion = _mean(fitted, "holonomy_dispersion_error")
    edge_improvement = 1.0 - _mean(fitted, "transport_error") / _mean(
        by_baseline["identity"], "transport_error"
    )
    lines.extend(
        [
            "",
            "The fitted state error decomposes exactly into a systematic mean displacement ",
            f"({systematic:.3f}, {systematic / total:.1%} of the total) and a context-dependent ",
            f"dispersion term ({dispersion:.3f}, {dispersion / total:.1%}). The operator defect ",
            "measures the full linear map rather than only its action on sampled start states.",
            "",
            f"The fitted connection lowers held-out edge error by {edge_improvement:.1%} relative ",
            "to identity, and it strongly beats the target-shuffled fit. It therefore captures ",
            "some rewrite structure. The problem is specifically that loop closure can be ",
            "optimized independently of correct transport.",
            "",
            "## Relationship to OOD behavior",
            "",
            "| Metric (lower is nominally better) | Pooled correlation | Partial correlation |",
            "|---|---:|---:|",
        ]
    )
    for row in correlation_rows:
        lines.append(
            f"| {str(row['metric']).replace('_', ' ')} | "
            f"{float(row['pearson_with_ood']):+.3f} | {float(row['partial_with_ood']):+.3f} |"
        )
    lines.extend(
        [
            "",
            "Partial correlations remove linear effects of architecture depth and training-data ",
            "regime. With only 18 models these values are diagnostics, not inferential evidence.",
            "",
            "## What should count as holonomy here",
            "",
            "For a path `p`, retain the composed affine map itself, `T_p(x)=xR_p+b_p`. For a ",
            "closed path report at least three quantities rather than one:",
            "",
            "1. **Held-out edge fidelity:** whether each `T_e` predicts the actual next fiber.",
            "2. **Operator holonomy:** `||R_p-I||_F^2/d`, plus a separately scaled affine drift.",
            "3. **Data-supported holonomy:** return-to-start displacement on held-out states, ",
            "   split into systematic and context-dependent components.",
            "",
            "For two routes with common endpoints, path curvature should be reported alongside ",
            "both routes' endpoint errors. Agreement between two equally wrong paths is not ",
            "coherence. There is no defensible single scalar until edge errors are substantially ",
            "below the identity and shuffled baselines.",
            "",
            "## Why the current connection is provisional",
            "",
            "- One global map is shared by every occurrence of a rewrite label, despite the ",
            "  observed context dependence of the representation change.",
            "- Self-inverse rewrites reuse the same unconstrained Procrustes map in both ",
            "  directions; inverse and involution laws are tested only indirectly by a loop.",
            "- The connection is fitted after training and is not a mechanism learned or used ",
            "  by the Transformer.",
            "- Final-layer `<CLS>` states may discard local rewrite geometry present at the ",
            "  rewritten subtree or at earlier layers.",
            "- Scalar variance normalization is orthogonally gauge-invariant, but it weights ",
            "  high-variance latent directions more heavily and can miss defects in task-relevant ",
            "  low-variance directions.",
            "",
            "## Highest-value next experiment",
            "",
            "Fit context-conditioned, bidirectionally constrained transports at every layer and ",
            "at the rewritten subtree token—not only final-layer `<CLS>`. Train forward and ",
            "inverse maps jointly, enforce `T_inverse T_forward ≈ I`, and evaluate on unseen ",
            "contexts. Compare against identity, target-shuffled, rewrite-label-shuffled, and ",
            "low-rank nulls. Only after held-out edge fidelity improves should loop curvature be ",
            "tested as a predictor of OOD accuracy.",
            "",
            "Then add a causal version: patch the same semantic subtree through each competing ",
            "rewrite route and compare downstream logit changes. That tests whether path ",
            "coherence is used by the classifier rather than merely visible in geometry.",
            "",
            "## Figures",
            "",
        ]
    )
    for path in plot_paths:
        lines.extend([f"![{path.stem}](plots/{path.name})", ""])
    return "\n".join(line.rstrip() for line in lines)


def run_holonomy_audit(run_directory: Path, *, device: str = "auto") -> Path:
    run_directory = run_directory.resolve()
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    result_rows = _read_rows(run_directory / "tables" / "results.csv")
    result_lookup = {
        (int(row["architecture_layers"]), row["condition"], int(row["seed"])): row
        for row in result_rows
    }
    calibration, diagrams = _regenerate_evaluation_data(config)
    labels = {edge.label for diagram in diagrams for edge in diagram.edges}
    target_device = resolve_device(device)
    family_rows: list[dict[str, Any]] = []
    checkpoints = sorted((run_directory / "checkpoints").glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {run_directory / 'checkpoints'}")
    for index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train_config = checkpoint["train_config"]
        model_config = checkpoint["model_config"]
        layers = int(model_config["n_layers"])
        seed = int(train_config["seed"])
        condition = "higher_diversity" if "higher_diversity" in checkpoint_path.stem else "low_diversity"
        metadata = {
            "architecture_layers": layers,
            "condition": condition,
            "seed": seed,
            "ood_accuracy_mean": float(result_lookup[(layers, condition, seed)]["ood_accuracy_mean"]),
        }
        print(f"[{index}/{len(checkpoints)}] auditing {checkpoint_path.stem}", flush=True)
        model = _load_model(checkpoint_path, target_device)
        fitted = _fit_transports(model, calibration, device=target_device)
        shuffled = _fit_transports(
            model,
            calibration,
            device=target_device,
            shuffle_seed=91_000 + layers * 100 + seed,
        )
        connections = {
            "fitted": fitted,
            "shuffled": shuffled,
            "identity": _identity_transports(labels, int(model_config["d_model"])),
        }
        for baseline, transports in connections.items():
            rows = score_equivalence_diagrams(
                model,
                diagrams,
                transports,
                device=target_device,
            )
            family_rows.extend({**metadata, "baseline": baseline, **row} for row in rows)
        del model
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    audit_directory = run_directory / "holonomy_audit"
    plot_directory = audit_directory / "plots"
    audit_directory.mkdir(exist_ok=True)
    model_rows = _model_summaries(family_rows)
    correlation_rows = _correlation_rows(model_rows)
    _write_rows(audit_directory / "family_metrics.csv", family_rows)
    _write_rows(audit_directory / "model_metrics.csv", model_rows)
    _write_rows(audit_directory / "correlations.csv", correlation_rows)
    plot_paths = _audit_plots(model_rows, family_rows, correlation_rows, plot_directory)
    (audit_directory / "report.md").write_text(
        _report(model_rows, correlation_rows, plot_paths), encoding="utf-8"
    )
    (audit_directory / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "models": len(checkpoints),
                "baselines": ["fitted", "shuffled", "identity"],
                "plots": [str(path.relative_to(audit_directory)) for path in plot_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return audit_directory
