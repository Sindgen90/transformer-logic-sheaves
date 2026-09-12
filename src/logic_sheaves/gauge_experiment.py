from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import AssignedExpression, ExpressionDataset, collate_expressions
from .equivalence import EquivalenceDiagram
from .gauge_atlas import (
    AtlasAnalysis,
    component_cycles,
    fit_atlas,
    rank_matched_null,
    replace_defects,
)
from .holonomy_audit import _load_model, _regenerate_evaluation_data
from .training import resolve_device

ATLAS_COMPONENTS = ("residual", "query", "key", "value", "coupled_qk")
ATLAS_SCOPES = ("cls", "expression_root", "expression_mean")
ATLAS_CONTROLS = (
    "learned",
    "identity",
    "target_shuffled",
    "assignment_shuffled",
    "label_shuffled",
    "random_basis",
    "rank_matched",
)


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append one checkpoint buffer without retaining the full run in memory."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _head_slice(tensor: torch.Tensor, heads: tuple[int, ...] | None) -> torch.Tensor:
    return tensor if heads is None else tensor[:, :, list(heads)]


@torch.inference_mode()
def atlas_features(
    model,
    expressions: Sequence[AssignedExpression],
    *,
    device: torch.device,
    batch_size: int,
    components: tuple[str, ...],
    scopes: tuple[str, ...],
    heads: tuple[int, ...] | None,
) -> dict[tuple[int, str, str], np.ndarray]:
    """Extract atlas fibers for all layers in flat expression order."""

    model.eval()
    output: dict[tuple[int, str, str], list[np.ndarray]] = defaultdict(list)
    dataset = ExpressionDataset(expressions)
    for start in range(0, len(dataset), batch_size):
        batch_examples = dataset.expressions[start : start + batch_size]
        batch = collate_expressions(batch_examples)
        tokens = batch.tokens.to(device)
        padding_mask = batch.padding_mask.to(device)
        stages = model.stage_representations(tokens, padding_mask)
        qkv = model.qkv_projections(tokens, padding_mask)
        rows = torch.arange(len(batch_examples), device=device)
        roots = torch.tensor(
            [len(example.assignment) + 3 for example in batch_examples], device=device
        )
        positions = torch.arange(tokens.shape[1], device=device)[None, :]
        expression_mask = (~padding_mask) & (positions >= roots[:, None])
        denominator = expression_mask.sum(dim=1).clamp_min(1).reshape(-1, 1)

        for layer in range(1, model.config.n_layers + 1):
            tensors: dict[str, torch.Tensor] = {}
            if "residual" in components:
                tensors["residual"] = stages[layer]
            query, key, value = qkv[layer - 1]
            selected = {
                "query": _head_slice(query, heads).flatten(2),
                "key": _head_slice(key, heads).flatten(2),
                "value": _head_slice(value, heads).flatten(2),
            }
            for component in ("query", "key", "value"):
                if component in components:
                    tensors[component] = selected[component]
            if "coupled_qk" in components:
                tensors["coupled_qk"] = torch.cat((selected["query"], selected["key"]), dim=-1)
            for component, tensor in tensors.items():
                if "cls" in scopes:
                    output[(layer, "cls", component)].append(tensor[:, 0].float().cpu().numpy())
                if "expression_root" in scopes:
                    output[(layer, "expression_root", component)].append(
                        tensor[rows, roots].float().cpu().numpy()
                    )
                if "expression_mean" in scopes:
                    mean = (tensor * expression_mask[:, :, None]).sum(dim=1) / denominator
                    output[(layer, "expression_mean", component)].append(mean.float().cpu().numpy())
    return {key: np.concatenate(chunks) for key, chunks in output.items()}


def _flip_diagrams(
    diagrams: Sequence[EquivalenceDiagram], variable: str
) -> list[EquivalenceDiagram]:
    output: list[EquivalenceDiagram] = []
    for diagram in diagrams:
        vertices = tuple(
            AssignedExpression(
                vertex.expression,
                tuple(
                    (name, 1 - value if name == variable else value)
                    for name, value in vertex.assignment
                ),
            )
            for vertex in diagram.vertices
        )
        flipped = replace(diagram, vertices=vertices)
        flipped.validate()
        output.append(flipped)
    return output


def _family_tensors(
    diagrams: Sequence[EquivalenceDiagram], features: np.ndarray
) -> dict[str, tuple[list[EquivalenceDiagram], np.ndarray]]:
    grouped_diagrams: dict[str, list[EquivalenceDiagram]] = defaultdict(list)
    grouped_features: dict[str, list[np.ndarray]] = defaultdict(list)
    cursor = 0
    for diagram in diagrams:
        next_cursor = cursor + len(diagram.vertices)
        grouped_diagrams[diagram.family].append(diagram)
        grouped_features[diagram.family].append(features[cursor:next_cursor])
        cursor = next_cursor
    return {
        family: (items, np.stack(grouped_features[family]))
        for family, items in grouped_diagrams.items()
    }


def _bitflip_effect_masks(
    original: Sequence[EquivalenceDiagram],
    flipped: Sequence[EquivalenceDiagram],
) -> dict[str, np.ndarray]:
    original_by_family: dict[str, list[int]] = defaultdict(list)
    flipped_by_family: dict[str, list[int]] = defaultdict(list)
    for left, right in zip(original, flipped, strict=True):
        if left.family != right.family:
            raise ValueError("Bit-flip diagrams must preserve family order")
        original_by_family[left.family].append(left.vertices[0].value)
        flipped_by_family[right.family].append(right.vertices[0].value)
    return {
        family: np.asarray(original_by_family[family]) != np.asarray(flipped_by_family[family])
        for family in original_by_family
    }


def variable_influence_class(diagram: EquivalenceDiagram, variable: str) -> str:
    """Classify a flip as absent, globally irrelevant, preserved, or changing."""

    expression = diagram.vertices[0].expression
    present = {node.op for node in expression.nodes_prefix() if node.is_variable}
    if variable not in present:
        return "absent"
    assignment = dict(diagram.vertices[0].assignment)
    original = expression.evaluate(assignment)
    flipped_assignment = {**assignment, variable: 1 - assignment[variable]}
    if expression.evaluate(flipped_assignment) != original:
        return "value_changed"
    other_variables = sorted(present - {variable})
    globally_influential = False
    for mask in range(1 << len(other_variables)):
        probe = dict(assignment)
        for index, name in enumerate(other_variables):
            probe[name] = int(bool(mask & (1 << index)))
        probe[variable] = 0
        zero = expression.evaluate(probe)
        probe[variable] = 1
        if expression.evaluate(probe) != zero:
            globally_influential = True
            break
    return "value_preserved" if globally_influential else "globally_irrelevant"


def _derangement(length: int, rng: np.random.Generator) -> np.ndarray:
    if length < 2:
        return np.arange(length)
    return np.roll(np.arange(length), int(rng.integers(1, length)))


def _orders(
    edge_pairs: Sequence[tuple[int, int]],
    labels: np.ndarray,
    rng: np.random.Generator,
    *,
    preserve_label: bool,
) -> dict[tuple[int, int], np.ndarray]:
    output: dict[tuple[int, int], np.ndarray] = {}
    for edge in sorted({tuple(sorted(edge)) for edge in edge_pairs}):
        if not preserve_label:
            output[edge] = _derangement(len(labels), rng)
            continue
        order = np.arange(len(labels))
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            order[members] = members[_derangement(len(members), rng)]
        output[edge] = order
    return output


def _operator_control(
    learned: AtlasAnalysis, control: str, rng: np.random.Generator
) -> AtlasAnalysis:
    if control == "identity":
        defects = [np.eye(edge.defect.shape[0]) for _, edge in sorted(learned.edges.items())]
        edges = replace_defects(learned.edges, defects)
    elif control == "label_shuffled":
        defects = [edge.defect for _, edge in sorted(learned.edges.items())]
        edges = replace_defects(learned.edges, defects[1:] + defects[:1])
    elif control == "rank_matched":
        edges = rank_matched_null(learned.edges, rng)
    else:
        raise ValueError(control)
    return AtlasAnalysis(
        edges=edges,
        cycles=component_cycles(edges),
        edge_fidelity=float("nan"),
        state_return_error=float("nan"),
        cycle_coverage=len(component_cycles(edges)),
    )


def _analysis_summary(result: AtlasAnalysis) -> dict[str, Any]:
    cycles = result.cycles
    edges = list(result.edges.values())
    return {
        "cycle_coverage": len(cycles),
        "edge_coverage": len(edges),
        "paper_holonomy_mean": float(np.mean([item.paper_distance for item in cycles]))
        if cycles
        else float("nan"),
        "paper_holonomy_max": float(np.max([item.paper_distance for item in cycles]))
        if cycles
        else float("nan"),
        "unit_holonomy_mean": float(np.mean([item.unit_distance for item in cycles]))
        if cycles
        else float("nan"),
        "max_eigenphase_mean": float(np.mean([item.max_abs_eigenphase for item in cycles]))
        if cycles
        else float("nan"),
        "mean_eigenphase_mean": float(np.mean([item.mean_abs_eigenphase for item in cycles]))
        if cycles
        else float("nan"),
        "edge_fidelity": result.edge_fidelity,
        "state_return_error": result.state_return_error,
        "shear_mean": float(np.mean([item.shear for item in edges])) if edges else float("nan"),
        "sigma_min_mean": float(np.mean([item.sigma_min for item in edges]))
        if edges
        else float("nan"),
        "sigma_min_min": float(np.min([item.sigma_min for item in edges]))
        if edges
        else float("nan"),
        "transfer_mismatch_mean": float(np.mean([item.transfer_mismatch for item in edges]))
        if edges
        else float("nan"),
        "transfer_lower_bound_mean": float(np.mean([item.transfer_lower_bound for item in edges]))
        if edges
        else float("nan"),
    }


def _bitflip_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "paper_holonomy_mean",
        "unit_holonomy_mean",
        "edge_fidelity",
        "state_return_error",
        "shear_mean",
        "sigma_min_mean",
    )
    identity_fields = (
        "checkpoint",
        "architecture_layers",
        "training_condition",
        "seed",
        "family",
        "measured_layer",
        "scope",
        "component",
        "heads",
        "flip_variable",
        "flip_effect",
        "control",
    )
    original = {
        tuple(row[field] for field in identity_fields): row
        for row in rows
        if row["assignment_condition"] == "original" and row["flip_variable"] != "none"
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        if not str(row["assignment_condition"]).startswith("bitflip_"):
            continue
        key = tuple(row[field] for field in identity_fields)
        baseline = original.get(key)
        if baseline is None:
            continue
        result = {field: row[field] for field in identity_fields}
        result["original_assignment_condition"] = "original"
        result["intervention_assignment_condition"] = row["assignment_condition"]
        for metric in metrics:
            result[f"original_{metric}"] = baseline[metric]
            result[f"intervention_{metric}"] = row[metric]
            result[f"delta_{metric}"] = float(row[metric]) - float(baseline[metric])
        output.append(result)
    return output


def _plot(rows: list[dict[str, Any]], output: Path) -> list[Path]:
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(output.parent / "matplotlib_cache"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    output.mkdir(parents=True, exist_ok=True)
    learned = [
        row
        for row in rows
        if row["control"] == "learned"
        and row["assignment_condition"] == "original"
        and row["flip_variable"] == "none"
        and row["flip_effect"] == "all"
    ]
    paths: list[Path] = []
    if learned:
        labels = sorted({f"{row['scope']}:{row['component']}" for row in learned})
        layers = sorted({int(row["measured_layer"]) for row in learned})
        matrix = np.full((len(labels), len(layers)), np.nan)
        for row_index, label in enumerate(labels):
            for column, layer in enumerate(layers):
                values = [
                    float(row["paper_holonomy_mean"])
                    for row in learned
                    if f"{row['scope']}:{row['component']}" == label
                    and int(row["measured_layer"]) == layer
                ]
                matrix[row_index, column] = np.nanmean(values) if values else np.nan
        fig, axis = plt.subplots(figsize=(max(7, len(layers)), max(4, 0.4 * len(labels))))
        image = axis.imshow(matrix, aspect="auto", cmap="magma")
        axis.set_xticks(range(len(layers)), layers)
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("Layer")
        axis.set_title("Paper-normalized fundamental-cycle holonomy")
        fig.colorbar(image, ax=axis, label=r"$||h-I||_F / \sqrt{2k}$")
        fig.tight_layout()
        path = output / "holonomy_layer_feature_heatmap.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        x = np.asarray([float(row["paper_holonomy_mean"]) for row in learned])
        y = np.asarray([float(row["state_return_error"]) for row in learned])
        keep = np.isfinite(x) & np.isfinite(y)
        fig, axis = plt.subplots(figsize=(6.5, 5))
        axis.scatter(x[keep], y[keep], alpha=0.55)
        axis.set_xlabel("Operator holonomy (paper normalization)")
        axis.set_ylabel("Held-out state-return error")
        axis.set_title("Operator defect versus operational return error")
        axis.grid(alpha=0.25)
        fig.tight_layout()
        path = output / "operator_vs_state_return.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    controls = sorted({str(row["control"]) for row in rows})
    means = []
    for control in controls:
        values = np.asarray(
            [
                float(row["paper_holonomy_mean"])
                for row in rows
                if row["control"] == control
                and row["assignment_condition"] == "original"
                and row["flip_variable"] == "none"
                and row["flip_effect"] == "all"
            ]
        )
        means.append(float(np.nanmean(values)) if np.isfinite(values).any() else np.nan)
    if controls:
        fig, axis = plt.subplots(figsize=(8, 4.8))
        axis.bar(controls, means)
        axis.tick_params(axis="x", rotation=35)
        axis.set_ylabel("Mean paper-normalized holonomy")
        axis.set_title("Connection controls")
        fig.tight_layout()
        path = output / "control_comparison.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    deltas = [row for row in _bitflip_deltas(rows) if row["control"] == "learned"]
    if deltas:
        groups = sorted({(row["flip_variable"], row["flip_effect"]) for row in deltas})
        values = [
            float(
                np.nanmean(
                    [
                        float(row["delta_paper_holonomy_mean"])
                        for row in deltas
                        if (row["flip_variable"], row["flip_effect"]) == group
                    ]
                )
            )
            for group in groups
        ]
        labels = [f"{variable}\n{effect.replace('_', ' ')}" for variable, effect in groups]
        fig, axis = plt.subplots(figsize=(max(7, 1.2 * len(groups)), 4.8))
        axis.bar(labels, values)
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_ylabel("Bit flip - original holonomy")
        axis.set_title("Paired assignment intervention")
        fig.tight_layout()
        path = output / "bitflip_holonomy_delta.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def finalize_gauge_atlas(output: Path, settings: dict[str, Any]) -> Path:
    """Finalize plots and manifests from checkpoint-streamed tables."""

    output = output.resolve()
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    plot_rows: list[dict[str, Any]] = []
    with (output / "tables" / "summary.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["control"] == "learned" or (
                row["assignment_condition"] == "original"
                and row["flip_variable"] == "none"
                and row["flip_effect"] == "all"
            ):
                plot_rows.append(row)
    settings = {
        **settings,
        "summary_rows": int(progress["summary_rows"]),
    }
    plots = _plot(plot_rows, output / "plots")
    (output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output / "report.md").write_text(_report(settings, plot_rows, plots), encoding="utf-8")
    status = {
        **progress,
        "status": "complete",
        "plots": [str(path.relative_to(output)) for path in plots],
    }
    (output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (output / "progress.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return output


def _report(settings: dict[str, Any], rows: list[dict[str, Any]], plots: list[Path]) -> str:
    return "\n".join(
        [
            "# Gauge-atlas holonomy analysis",
            "",
            "This analysis implements the atlas construction from *A Gauge Theory of ",
            "Superposition* on logical-equivalence diagrams. Each logical vertex is a chart; ",
            "matched diagram instances are its overlap samples. PCA bases, ridge transports, ",
            "orthogonal polar factors, basis-overlap proxies, and edge defects are fit only on ",
            "the calibration partition. Edge fidelity and state-return error use a disjoint ",
            "evaluation partition.",
            "",
            "The paper-compatible metric is `||h-I||_F/sqrt(2k)`. Its true mathematical range ",
            "on O(k) is `[0, sqrt(2)]`, despite the paper's `[0,1]` claim. The tables also report ",
            "`||h-I||_F/(2 sqrt(k))`, which really is unit-normalized.",
            "",
            "`paper_holonomy_*` is operator holonomy of the defect connection. ",
            "`state_return_error` is the older operational idea: compose affine transports ",
            "around the same fundamental cycle and measure normalized held-out activation MSE. ",
            "They answer different questions and are deliberately kept side by side.",
            "",
            f"- Rows: {settings['summary_rows']}",
            f"- Chart dimension requested: {settings['chart_dimension']}",
            f"- Ridge: {settings['ridge']}",
            f"- Persistence threshold: {settings['persistence']}",
            f"- Head-restricted circuit: {settings['heads']}",
            f"- Assignment conditions: {', '.join(settings['assignment_conditions'])}",
            "",
            "Controls are identity, arbitrary target matching, truth-preserving assignment ",
            "shuffle, topology-edge/label shuffle, random chart bases, and a spectrum/rank- ",
            "matched random conjugation null. Always interpret holonomy conditional on held-out ",
            "edge fidelity, minimum singular value, and retained cycle coverage.",
            "",
            "## Figures",
            "",
            *[f"![{path.stem}](plots/{path.name})" for path in plots],
            "",
            "## Circuit and bit-flip interpretation",
            "",
            "Use Q/K/V causal patching to nominate a layer and heads, then rerun this command ",
            "with `--heads`. This measures the same connection inside that circuit-defined ",
            "fiber. `--bit-flips x0 ...` reruns the atlas on exactly the same expressions after ",
            "paired assignment interventions; differences therefore localize sensitivity to ",
            "the intervened logical variable rather than changes in syntax or graph topology.",
        ]
    )


def run_gauge_atlas(
    run_directory: Path,
    *,
    device: str = "auto",
    chart_dimension: int = 32,
    ridge: float = 1e-2,
    persistence: float = 0.0,
    fit_fraction: float = 0.5,
    components: tuple[str, ...] = ("query", "key", "value", "coupled_qk"),
    scopes: tuple[str, ...] = ("cls", "expression_root"),
    heads: tuple[int, ...] | None = None,
    bit_flips: tuple[str, ...] = (),
) -> Path:
    run_directory = run_directory.resolve()
    if not 0.2 <= fit_fraction <= 0.8:
        raise ValueError("fit_fraction must be between 0.2 and 0.8")
    invalid_components = set(components) - set(ATLAS_COMPONENTS)
    invalid_scopes = set(scopes) - set(ATLAS_SCOPES)
    if invalid_components or invalid_scopes:
        raise ValueError(
            f"Invalid components/scopes: {sorted(invalid_components | invalid_scopes)}"
        )
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    _, diagrams = _regenerate_evaluation_data(config)
    conditions = {"original": diagrams}
    conditions.update({f"bitflip_{name}": _flip_diagrams(diagrams, name) for name in bit_flips})
    effect_masks = {
        name: _bitflip_effect_masks(diagrams, conditions[f"bitflip_{name}"]) for name in bit_flips
    }
    target_device = resolve_device(device)
    stamp = datetime.now(timezone.utc).strftime("gauge_atlas_%Y%m%d_%H%M%S_%fZ")
    output = run_directory / "gauge_atlas" / stamp
    output.mkdir(parents=True)
    plot_rows: list[dict[str, Any]] = []
    total_summary_rows = 0
    total_edge_rows = 0
    total_cycle_rows = 0
    total_delta_rows = 0

    for checkpoint_index, checkpoint_path in enumerate(
        sorted((run_directory / "checkpoints").glob("*.pt")), start=1
    ):
        result_rows: list[dict[str, Any]] = []
        edge_rows: list[dict[str, Any]] = []
        cycle_rows: list[dict[str, Any]] = []
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        layers = int(checkpoint["model_config"]["n_layers"])
        model_heads = int(checkpoint["model_config"]["n_heads"])
        if heads is not None and (min(heads) < 0 or max(heads) >= model_heads):
            raise ValueError(f"Heads must be between 0 and {model_heads - 1}")
        seed = int(checkpoint["train_config"]["seed"])
        training_condition = (
            "higher_diversity" if "higher_diversity" in checkpoint_path.stem else "low_diversity"
        )
        print(f"[{checkpoint_index}] atlas {checkpoint_path.stem} ({layers} layers)", flush=True)
        model = _load_model(checkpoint_path, target_device)
        for assignment_condition, condition_diagrams in conditions.items():
            expressions = [vertex for diagram in condition_diagrams for vertex in diagram.vertices]
            features = atlas_features(
                model,
                expressions,
                device=target_device,
                batch_size=int(config["batch_size"]),
                components=components,
                scopes=scopes,
                heads=heads,
            )
            for (measured_layer, scope, component), matrix in features.items():
                families = _family_tensors(condition_diagrams, matrix)
                for family_index, (family, (family_diagrams, tensor)) in enumerate(
                    sorted(families.items())
                ):
                    strata: list[tuple[str, str, np.ndarray]] = [
                        (
                            "none"
                            if assignment_condition == "original"
                            else assignment_condition[8:],
                            "all",
                            np.ones(len(tensor), dtype=bool),
                        )
                    ]
                    if assignment_condition == "original":
                        for variable in bit_flips:
                            changed = effect_masks[variable][family]
                            strata.extend(
                                (
                                    (variable, "value_changed", changed),
                                    (variable, "value_preserved", ~changed),
                                )
                            )
                    else:
                        variable = assignment_condition[8:]
                        changed = effect_masks[variable][family]
                        strata.extend(
                            (
                                (variable, "value_changed", changed),
                                (variable, "value_preserved", ~changed),
                            )
                        )
                    for flip_variable, flip_effect, keep in strata:
                        selected_tensor = tensor[keep]
                        selected_diagrams = [
                            diagram
                            for diagram, include in zip(family_diagrams, keep, strict=True)
                            if include
                        ]
                        if len(selected_diagrams) < 4:
                            continue
                        topology = selected_diagrams[0]
                        edge_pairs = [(edge.source, edge.target) for edge in topology.edges]
                        rng = np.random.default_rng(
                            910_000
                            + 10_000 * seed
                            + 1_000 * measured_layer
                            + 100 * family_index
                            + sum(
                                map(
                                    ord,
                                    scope + component + flip_variable + flip_effect,
                                )
                            )
                        )
                        permutation = rng.permutation(len(selected_tensor))
                        fit_count = max(
                            2,
                            min(
                                len(selected_tensor) - 2,
                                round(len(selected_tensor) * fit_fraction),
                            ),
                        )
                        fit_indices = np.sort(permutation[:fit_count])
                        evaluation_indices = np.sort(permutation[fit_count:])
                        fit_labels = np.asarray(
                            [selected_diagrams[index].vertices[0].value for index in fit_indices]
                        )
                        analyses: dict[str, AtlasAnalysis] = {}
                        analyses["learned"] = fit_atlas(
                            selected_tensor,
                            edge_pairs,
                            fit_indices,
                            evaluation_indices,
                            dimension=chart_dimension,
                            ridge=ridge,
                            persistence=persistence,
                        )
                        analyses["target_shuffled"] = fit_atlas(
                            selected_tensor,
                            edge_pairs,
                            fit_indices,
                            evaluation_indices,
                            dimension=chart_dimension,
                            ridge=ridge,
                            persistence=persistence,
                            target_orders=_orders(
                                edge_pairs, fit_labels, rng, preserve_label=False
                            ),
                        )
                        analyses["assignment_shuffled"] = fit_atlas(
                            selected_tensor,
                            edge_pairs,
                            fit_indices,
                            evaluation_indices,
                            dimension=chart_dimension,
                            ridge=ridge,
                            persistence=persistence,
                            target_orders=_orders(edge_pairs, fit_labels, rng, preserve_label=True),
                        )
                        analyses["random_basis"] = fit_atlas(
                            selected_tensor,
                            edge_pairs,
                            fit_indices,
                            evaluation_indices,
                            dimension=chart_dimension,
                            ridge=ridge,
                            persistence=persistence,
                            random_basis_seed=int(rng.integers(0, 2**31)),
                        )
                        for control in ("identity", "label_shuffled", "rank_matched"):
                            analyses[control] = _operator_control(analyses["learned"], control, rng)

                        base = {
                            "checkpoint": checkpoint_path.name,
                            "architecture_layers": layers,
                            "training_condition": training_condition,
                            "seed": seed,
                            "assignment_condition": assignment_condition,
                            "flip_variable": flip_variable,
                            "flip_effect": flip_effect,
                            "family": family,
                            "measured_layer": measured_layer,
                            "scope": scope,
                            "component": component,
                            "heads": "all" if heads is None else "+".join(map(str, heads)),
                            "ambient_dimension": selected_tensor.shape[2],
                            "chart_dimension": min(
                                chart_dimension, fit_count - 1, selected_tensor.shape[2]
                            ),
                            "fit_instances": len(fit_indices),
                            "evaluation_instances": len(evaluation_indices),
                        }
                        for control in ATLAS_CONTROLS:
                            analysis = analyses[control]
                            result_rows.append(
                                {**base, "control": control, **_analysis_summary(analysis)}
                            )
                            if control in {
                                "learned",
                                "target_shuffled",
                                "assignment_shuffled",
                                "random_basis",
                            }:
                                for edge_key, edge in sorted(analysis.edges.items()):
                                    edge_rows.append(
                                        {
                                            **base,
                                            "control": control,
                                            "source": edge_key[0],
                                            "target": edge_key[1],
                                            "sigma_min": edge.sigma_min,
                                            "shear": edge.shear,
                                            "transfer_mismatch": edge.transfer_mismatch,
                                            "transfer_lower_bound": edge.transfer_lower_bound,
                                        }
                                    )
                            for cycle_index, cycle in enumerate(analysis.cycles):
                                cycle_rows.append(
                                    {
                                        **base,
                                        "control": control,
                                        "cycle_index": cycle_index,
                                        "chord": f"{cycle.chord[0]}-{cycle.chord[1]}",
                                        "length": len(cycle.traversals),
                                        "paper_holonomy": cycle.paper_distance,
                                        "unit_holonomy": cycle.unit_distance,
                                        "max_abs_eigenphase": cycle.max_abs_eigenphase,
                                        "mean_abs_eigenphase": cycle.mean_abs_eigenphase,
                                        "eigenphases_radians": json.dumps(
                                            np.angle(np.linalg.eigvals(cycle.holonomy)).tolist()
                                        ),
                                    }
                                )
            del features
        del model
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
        delta_rows = _bitflip_deltas(result_rows)
        _append_rows(output / "tables" / "summary.csv", result_rows)
        _append_rows(output / "tables" / "edges.csv", edge_rows)
        _append_rows(output / "tables" / "cycles.csv", cycle_rows)
        _append_rows(output / "tables" / "bitflip_deltas.csv", delta_rows)
        plot_rows.extend(
            row
            for row in result_rows
            if row["control"] == "learned"
            or (
                row["assignment_condition"] == "original"
                and row["flip_variable"] == "none"
                and row["flip_effect"] == "all"
            )
        )
        total_summary_rows += len(result_rows)
        total_edge_rows += len(edge_rows)
        total_cycle_rows += len(cycle_rows)
        total_delta_rows += len(delta_rows)
        (output / "progress.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "completed_checkpoints": checkpoint_index,
                    "summary_rows": total_summary_rows,
                    "edge_rows": total_edge_rows,
                    "cycle_rows": total_cycle_rows,
                    "bitflip_delta_rows": total_delta_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    plots = _plot(plot_rows, output / "plots")
    settings = {
        "chart_dimension": chart_dimension,
        "ridge": ridge,
        "persistence": persistence,
        "fit_fraction": fit_fraction,
        "components": list(components),
        "scopes": list(scopes),
        "heads": "all" if heads is None else list(heads),
        "assignment_conditions": list(conditions),
        "controls": list(ATLAS_CONTROLS),
        "summary_rows": total_summary_rows,
    }
    (output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output / "report.md").write_text(_report(settings, plot_rows, plots), encoding="utf-8")
    (output / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "summary_rows": total_summary_rows,
                "edge_rows": total_edge_rows,
                "cycle_rows": total_cycle_rows,
                "bitflip_delta_rows": total_delta_rows,
                "plots": [str(path.relative_to(output)) for path in plots],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
