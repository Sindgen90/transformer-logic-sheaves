from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .circuit_selection import CircuitNomination, nominate_circuits
from .gauge_atlas import (
    AtlasEdge,
    fit_atlas_edges,
    heldout_edge_fidelities,
    heldout_edge_fidelity,
    heldout_state_return_error,
    largest_connected_component,
    proxy_connection_control,
    spectrum_matched_transport_null,
    stable_subgraph,
    topology_shuffled_transport_null,
    typed_component_cycles,
)
from .gauge_experiment import (
    _family_tensors,
    _flip_diagrams,
    _orders,
    atlas_features,
    variable_influence_class,
)
from .holonomy_audit import _load_model, _regenerate_evaluation_data
from .training import resolve_device

TYPED_CONTROLS = (
    "learned",
    "proxy_connection",
    "target_shuffled",
    "assignment_shuffled",
    "random_basis",
    "topology_shuffled",
    "spectrum_matched",
)


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if header:
            writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def _condition_from_checkpoint(path: Path) -> str:
    return "higher_diversity" if "higher_diversity" in path.stem else "low_diversity"


def _circuit_specs(nominations: list[CircuitNomination], heads: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, str], set[int]] = defaultdict(set)
    scores: dict[tuple[int, str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for item in nominations:
        scope = "cls" if item.site == "cls" else "expression_root"
        key = (item.layer, scope, item.component, item.status)
        grouped[key].update(item.heads)
        scores[key].append((item.discovery_specificity, item.confirmation_specificity))
    output: list[dict[str, Any]] = []
    all_heads = tuple(range(heads))
    for (layer, scope, nomination_component, status), selected in sorted(grouped.items()):
        selected_heads = tuple(sorted(selected))
        complement = tuple(head for head in all_heads if head not in selected_heads)
        discovery = _mean([item[0] for item in scores[(layer, scope, status)]])
        confirmation = _mean([item[1] for item in scores[(layer, scope, status)]])
        for mode, mode_heads in (
            ("circuit", selected_heads),
            ("complement", complement),
            ("all", all_heads),
        ):
            if not mode_heads:
                continue
            output.append(
                {
                    "layer": layer,
                    "scope": scope,
                    "nomination_component": nomination_component,
                    "nomination_status": status,
                    "circuit_mode": mode,
                    "heads": mode_heads,
                    "discovery_specificity": discovery,
                    "confirmation_specificity": confirmation,
                }
            )
    return output


def _select_heads(matrix: np.ndarray, heads: tuple[int, ...], model_heads: int) -> np.ndarray:
    reshaped = matrix.reshape(len(matrix), model_heads, matrix.shape[1] // model_heads)
    return reshaped[:, list(heads)].reshape(len(matrix), -1)


def _feature_matrix(
    features: dict[tuple[int, str, str], np.ndarray],
    *,
    layer: int,
    scope: str,
    component: str,
    heads: tuple[int, ...],
    model_heads: int,
) -> np.ndarray:
    if component == "coupled_qk":
        query = _select_heads(features[(layer, scope, "query")], heads, model_heads)
        key = _select_heads(features[(layer, scope, "key")], heads, model_heads)
        return np.concatenate((query, key), axis=1)
    return _select_heads(features[(layer, scope, component)], heads, model_heads)


def _split(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 8:
        raise ValueError("A typed atlas stratum requires at least eight instances")
    permutation = np.random.default_rng(seed).permutation(count)
    fit_count = max(4, min(count - 4, count // 2))
    return np.sort(permutation[:fit_count]), np.sort(permutation[fit_count:])


def _edge_subset(
    edges: dict[tuple[int, int], AtlasEdge], keys: set[tuple[int, int]]
) -> dict[tuple[int, int], AtlasEdge]:
    return largest_connected_component({key: edge for key, edge in edges.items() if key in keys})


def _typed_summary(
    edges: dict[tuple[int, int], AtlasEdge],
    tensor: np.ndarray,
    evaluation_indices: np.ndarray,
) -> dict[str, Any]:
    cycles = typed_component_cycles(edges)
    return {
        "vertices_retained": len({vertex for key in edges for vertex in key}),
        "edges_retained": len(edges),
        "cycles_retained": len(cycles),
        "transport_paper_mean": _mean([cycle.transport_paper_distance for cycle in cycles]),
        "transport_unit_mean": _mean([cycle.transport_unit_distance for cycle in cycles]),
        "proxy_paper_mean": _mean([cycle.proxy_paper_distance for cycle in cycles]),
        "proxy_unit_mean": _mean([cycle.proxy_unit_distance for cycle in cycles]),
        "relative_paper_mean": _mean([cycle.relative_paper_distance for cycle in cycles]),
        "relative_unit_mean": _mean([cycle.relative_unit_distance for cycle in cycles]),
        "relative_max_eigenphase_mean": _mean(
            [max(np.abs(cycle.relative_eigenphases), default=0.0) for cycle in cycles]
        ),
        "edge_fidelity": heldout_edge_fidelity(edges, tensor, evaluation_indices),
        "state_return_error": heldout_state_return_error(edges, tensor, evaluation_indices),
        "sigma_min_mean": _mean([edge.sigma_min for edge in edges.values()]),
        "condition_ratio_mean": _mean([edge.condition_ratio for edge in edges.values()]),
        "bootstrap_stability_mean": _mean([edge.bootstrap_stability for edge in edges.values()]),
    }


def _primary_controls(
    tensor: np.ndarray,
    edge_pairs: list[tuple[int, int]],
    fit_indices: np.ndarray,
    labels: np.ndarray,
    *,
    dimension: int,
    ridge: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, dict[tuple[int, int], AtlasEdge]]:
    rng = np.random.default_rng(seed)
    learned = fit_atlas_edges(
        tensor,
        edge_pairs,
        fit_indices,
        dimension=dimension,
        ridge=ridge,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=seed + 100,
    )
    target = fit_atlas_edges(
        tensor,
        edge_pairs,
        fit_indices,
        dimension=dimension,
        ridge=ridge,
        target_orders=_orders(edge_pairs, labels, rng, preserve_label=False),
    )
    assignment = fit_atlas_edges(
        tensor,
        edge_pairs,
        fit_indices,
        dimension=dimension,
        ridge=ridge,
        target_orders=_orders(edge_pairs, labels, rng, preserve_label=True),
    )
    random_basis = fit_atlas_edges(
        tensor,
        edge_pairs,
        fit_indices,
        dimension=dimension,
        ridge=ridge,
        random_basis_seed=seed + 200,
    )
    vertices = tensor.shape[1]
    return {
        "learned": learned,
        "proxy_connection": proxy_connection_control(learned),
        "target_shuffled": target,
        "assignment_shuffled": assignment,
        "random_basis": random_basis,
        "topology_shuffled": topology_shuffled_transport_null(vertices, learned),
        "spectrum_matched": spectrum_matched_transport_null(learned, rng),
    }


def _paired_delta_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    paired = [row for row in rows if row["filter_mode"] == "paired_shared_mask"]
    identity_fields = (
        "checkpoint",
        "training_condition",
        "seed",
        "influence_class",
        "layer",
        "scope",
        "nomination_component",
        "nomination_status",
        "circuit_mode",
        "heads",
        "component",
        "family",
        "chart_dimension",
        "min_sigma",
    )
    by_identity: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in paired:
        key = tuple(row[field] for field in identity_fields)
        by_identity[key][row["assignment_condition"]] = row
    output: list[dict[str, Any]] = []
    metrics = (
        "transport_unit_mean",
        "proxy_unit_mean",
        "relative_unit_mean",
        "edge_fidelity",
        "state_return_error",
    )
    for key, conditions in by_identity.items():
        if "original" not in conditions:
            continue
        original = conditions["original"]
        for condition, flipped in conditions.items():
            if condition == "original":
                continue
            row: dict[str, Any] = dict(zip(identity_fields, key, strict=True))
            row["intervention"] = condition
            for metric in metrics:
                before = float(original[metric])
                after = float(flipped[metric])
                row[f"original_{metric}"] = before
                row[f"intervened_{metric}"] = after
                row[f"delta_{metric}"] = after - before
            output.append(row)
    return output


def _write_report(
    rows: list[dict[str, str]],
    deltas: list[dict[str, Any]],
    nominations: list[CircuitNomination],
    output: Path,
    settings: dict[str, Any],
) -> None:
    def finite_mean(items: list[dict[str, str]], field: str) -> float:
        numbers = [float(item[field]) for item in items]
        return _mean([number for number in numbers if np.isfinite(number)])

    thresholds = list(settings["thresholds"])
    reference = min(thresholds, key=lambda value: abs(value - 0.015))
    primary = [
        row
        for row in rows
        if row["filter_mode"] == "learned_mask"
        and row["control"] == "learned"
        and row["assignment_condition"] == "original"
        and row["influence_class"] == "all"
        and row["nomination_status"] == "confirmed"
        and row["circuit_mode"] == "circuit"
        and float(row["min_sigma"]) == reference
    ]
    control_rows = [
        row
        for row in rows
        if row["filter_mode"] == "learned_mask"
        and row["assignment_condition"] == "original"
        and row["influence_class"] == "all"
        and row["nomination_status"] == "confirmed"
        and row["circuit_mode"] == "circuit"
        and float(row["min_sigma"]) == reference
    ]
    finite_primary = [row for row in primary if np.isfinite(float(row["relative_unit_mean"]))]

    def control_mean(control: str) -> float:
        items = [row for row in control_rows if row["control"] == control]
        return finite_mean(items, "relative_unit_mean")

    def mode_mean(mode: str) -> float:
        items = [
            row
            for row in rows
            if row["filter_mode"] == "learned_mask"
            and row["control"] == "learned"
            and row["assignment_condition"] == "original"
            and row["influence_class"] == "all"
            and row["nomination_status"] == "confirmed"
            and row["circuit_mode"] == mode
            and float(row["min_sigma"]) == reference
        ]
        return finite_mean(items, "relative_unit_mean")

    highest_threshold = max(thresholds)
    highest_rows = [
        row
        for row in rows
        if row["filter_mode"] == "learned_mask"
        and row["control"] == "learned"
        and row["assignment_condition"] == "original"
        and row["influence_class"] == "all"
        and row["nomination_status"] == "confirmed"
        and row["circuit_mode"] == "circuit"
        and float(row["min_sigma"]) == highest_threshold
    ]
    finite_highest = [row for row in highest_rows if np.isfinite(float(row["relative_unit_mean"]))]
    lines = [
        "# Type-correct circuit gauge experiment",
        "",
        "## What changed",
        "",
        (
            "The earlier literal edge defect `g = P^T Q` is an endomorphism of the "
            "edge's source fiber. Defects based at different vertices therefore cannot be "
            "multiplied directly. This experiment instead composes the learned transports "
            "`Q` and chart-overlap transports `P` separately around each loop and compares "
            "them only after both return to the same base fiber:"
        ),
        "",
        ("`H_Q = Q_n ... Q_1`, `H_P = P_n ... P_1`, and `H_rel = H_P^{-1} H_Q`."),
        "",
        (
            "Under independent orthogonal chart changes, all three loop operators transform "
            "by conjugation at the loop base. Their spectra, eigenphases, and normalized "
            "Frobenius distances from identity are consequently gauge invariant."
        ),
        "",
        "## Design",
        "",
        f"- Checkpoints: {len({row['checkpoint'] for row in rows})} six-layer models.",
        f"- Chart dimensions: {', '.join(map(str, settings['chart_dimensions']))}.",
        f"- Singular-value thresholds: {', '.join(map(str, thresholds))}.",
        f"- Bootstrap resamples per learned edge: {settings['bootstrap_samples']}.",
        f"- Maximum bootstrap instability: {settings['max_bootstrap_stability']}.",
        f"- Reference threshold used below: {reference}.",
        (
            f"- Cycle-bearing confirmed-circuit strata at that threshold: "
            f"{len(finite_primary)}/{len(primary)}."
        ),
        (
            "- An edge survives only if it meets the singular-value, condition-ratio, "
            "bootstrap-stability, and learned-better-than-target-shuffle fidelity gates."
        ),
        (
            "- After filtering, cycles are recomputed on the largest connected component; "
            "the nulls use the same learned edge mask."
        ),
        "- Q, K, V, and coupled Q-K fibers are measured separately.",
        (
            "- The x0 intervention uses the same diagrams, split, circuit, and shared "
            "original/intervened survivor mask."
        ),
        "",
        "## Frozen causal nominations",
        "",
        (
            "Discovery uses patching seeds 0-1 and confirmation uses seed 2. A discovery-only "
            "failure is retained as a negative reference."
        ),
        "",
        "| Layer | Site | Patch component | Heads | Discovery specificity | Confirmation specificity | Status |",
        "|---:|---|---|---|---:|---:|---|",
    ]
    for item in nominations:
        lines.append(
            f"| {item.layer} | {item.site} | {item.component} | "
            f"{'+'.join(map(str, item.heads))} | {item.discovery_specificity:.5f} | "
            f"{item.confirmation_specificity:.5f} | {item.status} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate results at the reference threshold",
            "",
            "| Fiber | H_Q | H_P | H_rel | Edge error | State-return error | Mean cycles |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for component in ("query", "key", "value", "coupled_qk"):
        items = [row for row in primary if row["component"] == component]
        finite_items = [row for row in items if np.isfinite(float(row["relative_unit_mean"]))]
        lines.append(
            f"| {component} ({len(finite_items)}/{len(items)} strata) | "
            f"{finite_mean(items, 'transport_unit_mean'):.4f} | "
            f"{finite_mean(items, 'proxy_unit_mean'):.4f} | "
            f"{finite_mean(items, 'relative_unit_mean'):.4f} | "
            f"{finite_mean(items, 'edge_fidelity'):.4f} | "
            f"{finite_mean(items, 'state_return_error'):.4f} | "
            f"{finite_mean(items, 'cycles_retained'):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Same-mask null comparison",
            "",
            "| Control | Relative unit holonomy |",
            "|---|---:|",
        ]
    )
    for control in TYPED_CONTROLS:
        items = [row for row in control_rows if row["control"] == control]
        lines.append(f"| {control} | {finite_mean(items, 'relative_unit_mean'):.4f} |")
    lines.extend(
        [
            "",
            "## Main findings",
            "",
            (
                f"At the reference threshold, learned relative holonomy is "
                f"{control_mean('learned'):.4f}, versus "
                f"{control_mean('topology_shuffled'):.4f} for the topology shuffle and "
                f"{control_mean('spectrum_matched'):.4f} for the spectrum-matched null. "
                "This is evidence for cross-edge organization beyond marginal edge spectra, "
                "conditional on the retained graph."
            ),
            "",
            (
                f"Localization is weak: confirmed circuits score {mode_mean('circuit'):.4f}, "
                f"their complements {mode_mean('complement'):.4f}, and all heads "
                f"{mode_mean('all'):.4f}. The causal screen therefore identifies sites with "
                "specific patch effects, but this atlas statistic is not confined to those "
                "heads."
            ),
            "",
            (
                f"At the most selective singular-value threshold ({highest_threshold:g}), "
                f"mean relative holonomy is "
                f"{finite_mean(highest_rows, 'relative_unit_mean'):.4f}, but only "
                f"{len(finite_highest)}/{len(highest_rows)} strata retain a cycle, versus "
                f"{len(finite_primary)}/{len(primary)} at the reference threshold. The "
                "downward persistence curve is therefore inseparable from coverage loss."
            ),
            "",
            "## Paired bit-flip result",
            "",
            (
                "The table reports `x0-flipped minus original` relative holonomy. Empty or "
                "acyclic strata are excluded from finite means."
            ),
            "",
            "| Influence class | Fiber | Finite strata | Mean delta H_rel |",
            "|---|---|---:|---:|",
        ]
    )
    for influence in ("absent", "globally_irrelevant", "value_preserved", "value_changed"):
        for component in ("query", "key", "value", "coupled_qk"):
            numbers = [
                float(row["delta_relative_unit_mean"])
                for row in deltas
                if row["influence_class"] == influence
                and row["component"] == component
                and float(row["min_sigma"]) == reference
            ]
            finite = [number for number in numbers if np.isfinite(number)]
            lines.append(f"| {influence} | {component} | {len(finite)} | {_mean(finite):.5f} |")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            (
                "A low `H_rel` is evidence of agreement between two fitted connections, not "
                "by itself evidence that the classifier uses that geometry. The causal "
                "nomination is therefore kept separate from the atlas measurement. Claims "
                "should require seed stability, nonzero cycle coverage, edge fidelity better "
                "than shuffled matching, and separation from topology- and spectrum-matched "
                "controls. The rejected `<CLS>` nomination is an explicit selection-bias check."
            ),
            "",
            "## Figures",
            "",
        ]
    )
    for path in sorted((output / "plots").glob("[A-H]_*.png")):
        lines.append(f"![{path.stem}](plots/{path.name})")
        lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _plot(rows: list[dict[str, str]], output: Path) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def values(items: list[dict[str, str]], column: str) -> list[float]:
        result = [float(item[column]) for item in items]
        return [value for value in result if np.isfinite(value)]

    def save(figure: Any, name: str) -> None:
        figure.tight_layout()
        path = output / name
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

    primary = [
        row
        for row in rows
        if row["filter_mode"] == "learned_mask"
        and row["control"] == "learned"
        and row["assignment_condition"] == "original"
        and row["influence_class"] == "all"
        and row["nomination_status"] == "confirmed"
        and row["circuit_mode"] == "circuit"
    ]
    thresholds = sorted({float(row["min_sigma"]) for row in primary})
    if primary:
        figure, axis = plt.subplots(figsize=(7.5, 5))
        for component in ("query", "key", "value", "coupled_qk"):
            means = [
                _mean(
                    [
                        float(row["relative_unit_mean"])
                        for row in primary
                        if row["component"] == component and float(row["min_sigma"]) == threshold
                    ]
                )
                for threshold in thresholds
            ]
            axis.plot(thresholds, means, marker="o", label=component)
        axis.set(
            xlabel=r"minimum $\sigma_{min}$",
            ylabel="relative unit holonomy",
            title="Type-correct circuit holonomy persistence",
        )
        axis.legend()
        axis.grid(alpha=0.25)
        save(figure, "A_relative_holonomy_persistence.png")

        reference_threshold = min(thresholds, key=lambda value: abs(value - 0.015))
        modes = ("circuit", "complement", "all")
        mode_values = []
        for mode in modes:
            candidates = [
                float(row["relative_unit_mean"])
                for row in rows
                if row["filter_mode"] == "learned_mask"
                and row["control"] == "learned"
                and row["assignment_condition"] == "original"
                and row["influence_class"] == "all"
                and row["nomination_status"] == "confirmed"
                and row["circuit_mode"] == mode
                and float(row["min_sigma"]) == reference_threshold
            ]
            mode_values.append(_mean(candidates))
        figure, axis = plt.subplots(figsize=(6, 4.5))
        axis.bar(modes, mode_values)
        axis.set(
            ylabel="relative unit holonomy",
            title=f"Circuit comparison at $\\sigma_{{min}}\\geq${reference_threshold:g}",
        )
        save(figure, "B_circuit_comparison.png")

        control_rows = [
            row
            for row in rows
            if row["filter_mode"] == "learned_mask"
            and row["assignment_condition"] == "original"
            and row["influence_class"] == "all"
            and row["nomination_status"] == "confirmed"
            and row["circuit_mode"] == "circuit"
            and float(row["min_sigma"]) == reference_threshold
        ]
        controls = list(TYPED_CONTROLS)
        figure, axis = plt.subplots(figsize=(9, 4.8))
        axis.bar(
            controls,
            [
                _mean(
                    values(
                        [row for row in control_rows if row["control"] == item],
                        "relative_unit_mean",
                    )
                )
                for item in controls
            ],
        )
        axis.set(ylabel="relative unit holonomy", title="Learned-mask null comparison")
        axis.tick_params(axis="x", rotation=35)
        save(figure, "C_null_control_comparison.png")

        figure, axis = plt.subplots(figsize=(7, 4.8))
        kinds = ("transport", "proxy", "relative")
        reference_primary = [
            row for row in primary if float(row["min_sigma"]) == reference_threshold
        ]
        axis.bar(
            kinds,
            [_mean(values(reference_primary, f"{kind}_unit_mean")) for kind in kinds],
            color=("#4c78a8", "#f58518", "#54a24b"),
        )
        axis.set(
            ylabel="unit-normalized loop distance",
            title="Separately typed loop holonomies",
        )
        save(figure, "D_transport_proxy_relative.png")

        figure, first_axis = plt.subplots(figsize=(7.5, 4.8))
        edge_means = [
            _mean(
                values(
                    [row for row in primary if float(row["min_sigma"]) == threshold],
                    "edges_retained",
                )
            )
            for threshold in thresholds
        ]
        cycle_means = [
            _mean(
                values(
                    [row for row in primary if float(row["min_sigma"]) == threshold],
                    "cycles_retained",
                )
            )
            for threshold in thresholds
        ]
        first_axis.plot(thresholds, edge_means, marker="o", color="#4c78a8", label="edges")
        first_axis.set(xlabel=r"minimum $\sigma_{min}$", ylabel="mean retained edges")
        second_axis = first_axis.twinx()
        second_axis.plot(thresholds, cycle_means, marker="s", color="#e45756", label="cycles")
        second_axis.set_ylabel("mean independent cycles")
        first_axis.set_title("Persistence coverage")
        save(figure, "E_persistence_coverage.png")

        scatter_rows = [
            row
            for row in primary
            if float(row["min_sigma"]) == reference_threshold
            and np.isfinite(float(row["relative_unit_mean"]))
        ]
        figure, axis = plt.subplots(figsize=(7, 5))
        for component in ("query", "key", "value", "coupled_qk"):
            items = [row for row in scatter_rows if row["component"] == component]
            axis.scatter(
                [float(row["edge_fidelity"]) for row in items],
                [float(row["relative_unit_mean"]) for row in items],
                label=component,
                alpha=0.72,
            )
        axis.set(
            xlabel="held-out normalized edge error",
            ylabel="relative unit holonomy",
            title="Holonomy conditional on edge fidelity",
        )
        axis.legend()
        axis.grid(alpha=0.2)
        save(figure, "F_fidelity_vs_holonomy.png")

        paired = [
            row
            for row in rows
            if row["filter_mode"] == "paired_shared_mask"
            and float(row["min_sigma"]) == reference_threshold
        ]
        pair_fields = (
            "checkpoint",
            "influence_class",
            "layer",
            "scope",
            "nomination_component",
            "heads",
            "component",
            "family",
            "chart_dimension",
            "min_sigma",
        )
        paired_index = {
            tuple(row[field] for field in pair_fields) + (row["assignment_condition"],): row
            for row in paired
        }
        delta_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
        flipped_names = sorted(
            {
                row["assignment_condition"]
                for row in paired
                if row["assignment_condition"] != "original"
            }
        )
        if flipped_names:
            flipped_name = flipped_names[0]
            for row in paired:
                if row["assignment_condition"] != "original":
                    continue
                partner = paired_index.get(
                    tuple(row[field] for field in pair_fields) + (flipped_name,)
                )
                if partner is None:
                    continue
                before = float(row["relative_unit_mean"])
                after = float(partner["relative_unit_mean"])
                if np.isfinite(before) and np.isfinite(after):
                    delta_groups[(row["influence_class"], row["component"])].append(after - before)
            classes = ("globally_irrelevant", "value_preserved", "value_changed")
            components = ("query", "key", "value", "coupled_qk")
            x = np.arange(len(classes))
            width = 0.19
            figure, axis = plt.subplots(figsize=(9, 4.8))
            for index, component in enumerate(components):
                axis.bar(
                    x + (index - 1.5) * width,
                    [_mean(delta_groups[(item, component)]) for item in classes],
                    width,
                    label=component,
                )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_xticks(x, classes, rotation=15)
            axis.set(
                ylabel="flipped - original relative holonomy",
                title="Paired bit-flip response by causal influence",
            )
            axis.legend(ncol=2)
            save(figure, "G_bitflip_influence_delta.png")

        dimensions = sorted({int(row["chart_dimension"]) for row in primary})
        figure, axis = plt.subplots(figsize=(7, 4.8))
        for component in ("query", "key", "value", "coupled_qk"):
            axis.plot(
                dimensions,
                [
                    _mean(
                        values(
                            [
                                row
                                for row in primary
                                if row["component"] == component
                                and int(row["chart_dimension"]) == dimension
                                and float(row["min_sigma"]) == reference_threshold
                            ],
                            "relative_unit_mean",
                        )
                    )
                    for dimension in dimensions
                ],
                marker="o",
                label=component,
            )
        axis.set(
            xlabel="chart dimension",
            ylabel="relative unit holonomy",
            title="Chart-dimension sensitivity",
        )
        axis.legend()
        axis.grid(alpha=0.2)
        save(figure, "H_chart_dimension_sensitivity.png")
    return paths


def run_typed_gauge_experiment(
    run_directory: Path,
    *,
    device: str = "auto",
    conditions: tuple[str, ...] = ("higher_diversity",),
    seeds: tuple[int, ...] | None = None,
    chart_dimensions: tuple[int, ...] = (8, 16),
    thresholds: tuple[float, ...] = (0.0, 0.015, 0.03, 0.06, 0.12, 0.24),
    ridge: float = 1e-2,
    min_condition_ratio: float = 0.02,
    max_bootstrap_stability: float = 0.35,
    bootstrap_samples: int = 16,
    bitflip_variable: str = "x0",
) -> Path:
    run_directory = run_directory.resolve()
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    _, diagrams = _regenerate_evaluation_data(config)
    flipped_diagrams = _flip_diagrams(diagrams, bitflip_variable)
    patch_metrics = run_directory / "qkv_patching" / "patch_metrics.csv"
    nominations = nominate_circuits(patch_metrics)
    circuit_specs = _circuit_specs(nominations, int(config["n_heads"]))
    confirmed_specs = [item for item in circuit_specs if item["nomination_status"] == "confirmed"]
    scopes = tuple(sorted({str(item["scope"]) for item in circuit_specs}))
    stamp = datetime.now(timezone.utc).strftime("typed_gauge_%Y%m%d_%H%M%S_%fZ")
    output = run_directory / "typed_gauge" / stamp
    output.mkdir(parents=True)
    (output / "nominations.json").write_text(
        json.dumps([item.to_dict() for item in nominations], indent=2), encoding="utf-8"
    )
    target_device = resolve_device(device)
    checkpoints = [
        path
        for path in sorted((run_directory / "checkpoints").glob("layers_6_*.pt"))
        if _condition_from_checkpoint(path) in conditions
        and (
            seeds is None
            or int(torch.load(path, map_location="cpu", weights_only=False)["train_config"]["seed"])
            in seeds
        )
    ]
    if not checkpoints:
        raise ValueError("No six-layer checkpoints match the requested conditions and seed filter")
    total_rows = 0
    total_cycle_rows = 0

    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_heads = int(checkpoint["model_config"]["n_heads"])
        seed = int(checkpoint["train_config"]["seed"])
        training_condition = _condition_from_checkpoint(checkpoint_path)
        print(
            f"[{checkpoint_index}/{len(checkpoints)}] typed gauge {checkpoint_path.stem}",
            flush=True,
        )
        model = _load_model(checkpoint_path, target_device)
        feature_sets: dict[str, tuple[list, dict]] = {}
        for assignment_condition, condition_diagrams in (
            ("original", diagrams),
            (f"bitflip_{bitflip_variable}", flipped_diagrams),
        ):
            expressions = [vertex for diagram in condition_diagrams for vertex in diagram.vertices]
            feature_sets[assignment_condition] = (
                condition_diagrams,
                atlas_features(
                    model,
                    expressions,
                    device=target_device,
                    batch_size=int(config["batch_size"]),
                    components=("query", "key", "value"),
                    scopes=scopes,
                    heads=None,
                ),
            )

        summary_rows: list[dict[str, Any]] = []
        cycle_rows: list[dict[str, Any]] = []
        edge_rows: list[dict[str, Any]] = []
        for spec_index, spec in enumerate(circuit_specs):
            for component in ("query", "key", "value", "coupled_qk"):
                original_features = _feature_matrix(
                    feature_sets["original"][1],
                    layer=int(spec["layer"]),
                    scope=str(spec["scope"]),
                    component=component,
                    heads=spec["heads"],
                    model_heads=model_heads,
                )
                original_families = _family_tensors(diagrams, original_features)
                for family_index, (family, (family_diagrams, tensor)) in enumerate(
                    sorted(original_families.items())
                ):
                    edge_pairs = [(edge.source, edge.target) for edge in family_diagrams[0].edges]
                    split_seed = (
                        8_100_000
                        + seed * 100_000
                        + spec_index * 10_000
                        + family_index * 100
                        + sum(map(ord, component))
                    )
                    fit_indices, evaluation_indices = _split(len(tensor), split_seed)
                    fit_labels = np.asarray(
                        [family_diagrams[index].vertices[0].value for index in fit_indices]
                    )
                    for dimension in chart_dimensions:
                        controls = _primary_controls(
                            tensor,
                            edge_pairs,
                            fit_indices,
                            fit_labels,
                            dimension=dimension,
                            ridge=ridge,
                            bootstrap_samples=bootstrap_samples,
                            seed=split_seed + dimension,
                        )
                        learned_fidelity = heldout_edge_fidelities(
                            controls["learned"], tensor, evaluation_indices
                        )
                        shuffled_fidelity = heldout_edge_fidelities(
                            controls["target_shuffled"], tensor, evaluation_indices
                        )
                        fidelity_keys = {
                            key
                            for key in learned_fidelity
                            if learned_fidelity[key] < shuffled_fidelity[key]
                        }
                        for edge_key, edge in sorted(controls["learned"].items()):
                            edge_rows.append(
                                {
                                    "checkpoint": checkpoint_path.name,
                                    "training_condition": training_condition,
                                    "seed": seed,
                                    **spec,
                                    "heads": "+".join(map(str, spec["heads"])),
                                    "component": component,
                                    "family": family,
                                    "chart_dimension": dimension,
                                    "source": edge_key[0],
                                    "target": edge_key[1],
                                    "sigma_min": edge.sigma_min,
                                    "sigma_max": edge.sigma_max,
                                    "condition_ratio": edge.condition_ratio,
                                    "bootstrap_stability": edge.bootstrap_stability,
                                    "heldout_fidelity": learned_fidelity[edge_key],
                                    "shuffled_fidelity": shuffled_fidelity[edge_key],
                                    "fidelity_improvement": shuffled_fidelity[edge_key]
                                    - learned_fidelity[edge_key],
                                }
                            )
                        for threshold in thresholds:
                            stable = stable_subgraph(
                                controls["learned"],
                                min_sigma=threshold,
                                min_condition_ratio=min_condition_ratio,
                                max_bootstrap_stability=max_bootstrap_stability,
                            )
                            learned_keys = set(stable) & fidelity_keys
                            stable = _edge_subset(controls["learned"], learned_keys)
                            learned_keys = set(stable)
                            for filter_mode in ("learned_mask", "intrinsic"):
                                for control_name in TYPED_CONTROLS:
                                    control_edges = controls[control_name]
                                    if filter_mode == "learned_mask" or control_name in {
                                        "proxy_connection",
                                        "topology_shuffled",
                                        "spectrum_matched",
                                    }:
                                        retained = _edge_subset(control_edges, learned_keys)
                                    else:
                                        retained = stable_subgraph(
                                            control_edges,
                                            min_sigma=threshold,
                                            min_condition_ratio=min_condition_ratio,
                                            max_bootstrap_stability=float("inf"),
                                        )
                                    base = {
                                        "checkpoint": checkpoint_path.name,
                                        "training_condition": training_condition,
                                        "seed": seed,
                                        "assignment_condition": "original",
                                        "influence_class": "all",
                                        **spec,
                                        "heads": "+".join(map(str, spec["heads"])),
                                        "component": component,
                                        "family": family,
                                        "chart_dimension": dimension,
                                        "min_sigma": threshold,
                                        "min_condition_ratio": min_condition_ratio,
                                        "max_bootstrap_stability": max_bootstrap_stability,
                                        "filter_mode": filter_mode,
                                        "control": control_name,
                                    }
                                    summary_rows.append(
                                        {
                                            **base,
                                            **_typed_summary(retained, tensor, evaluation_indices),
                                        }
                                    )
                                    for cycle_index, cycle in enumerate(
                                        typed_component_cycles(retained)
                                    ):
                                        cycle_rows.append(
                                            {
                                                **base,
                                                "cycle_index": cycle_index,
                                                "chord": f"{cycle.chord[0]}-{cycle.chord[1]}",
                                                "length": len(cycle.traversals),
                                                "transport_unit": cycle.transport_unit_distance,
                                                "proxy_unit": cycle.proxy_unit_distance,
                                                "relative_unit": cycle.relative_unit_distance,
                                                "transport_eigenphases": json.dumps(
                                                    cycle.transport_eigenphases
                                                ),
                                                "proxy_eigenphases": json.dumps(
                                                    cycle.proxy_eigenphases
                                                ),
                                                "relative_eigenphases": json.dumps(
                                                    cycle.relative_eigenphases
                                                ),
                                            }
                                        )

        # Paired x0 intervention is restricted to confirmed circuit fibers.
        for spec_index, spec in enumerate(confirmed_specs):
            if spec["circuit_mode"] != "circuit":
                continue
            for component in ("query", "key", "value", "coupled_qk"):
                matrices = {}
                for assignment_condition in ("original", f"bitflip_{bitflip_variable}"):
                    condition_diagrams, features = feature_sets[assignment_condition]
                    matrix = _feature_matrix(
                        features,
                        layer=int(spec["layer"]),
                        scope=str(spec["scope"]),
                        component=component,
                        heads=spec["heads"],
                        model_heads=model_heads,
                    )
                    matrices[assignment_condition] = _family_tensors(condition_diagrams, matrix)
                for family_index, family in enumerate(sorted(matrices["original"])):
                    original_diagrams, original_tensor = matrices["original"][family]
                    _, flipped_tensor = matrices[f"bitflip_{bitflip_variable}"][family]
                    classes = np.asarray(
                        [
                            variable_influence_class(diagram, bitflip_variable)
                            for diagram in original_diagrams
                        ]
                    )
                    edge_pairs = [(edge.source, edge.target) for edge in original_diagrams[0].edges]
                    for influence_class in (
                        "all",
                        "absent",
                        "globally_irrelevant",
                        "value_preserved",
                        "value_changed",
                    ):
                        keep = (
                            np.ones(len(classes), dtype=bool)
                            if influence_class == "all"
                            else classes == influence_class
                        )
                        if keep.sum() < 16:
                            continue
                        selected = {
                            "original": original_tensor[keep],
                            f"bitflip_{bitflip_variable}": flipped_tensor[keep],
                        }
                        split_seed = (
                            9_200_000
                            + seed * 100_000
                            + spec_index * 10_000
                            + family_index * 100
                            + sum(map(ord, component + influence_class))
                        )
                        fit_indices, evaluation_indices = _split(int(keep.sum()), split_seed)
                        for dimension in chart_dimensions:
                            condition_edges = {
                                condition: fit_atlas_edges(
                                    tensor,
                                    edge_pairs,
                                    fit_indices,
                                    dimension=dimension,
                                    ridge=ridge,
                                    bootstrap_samples=max(4, bootstrap_samples // 4),
                                    bootstrap_seed=split_seed + dimension,
                                )
                                for condition, tensor in selected.items()
                            }
                            for threshold in thresholds:
                                survivor_sets = []
                                for edges in condition_edges.values():
                                    survivor_sets.append(
                                        set(
                                            stable_subgraph(
                                                edges,
                                                min_sigma=threshold,
                                                min_condition_ratio=min_condition_ratio,
                                                max_bootstrap_stability=max_bootstrap_stability,
                                            )
                                        )
                                    )
                                shared = set.intersection(*survivor_sets)
                                for assignment_condition, edges in condition_edges.items():
                                    retained = _edge_subset(edges, shared)
                                    summary_rows.append(
                                        {
                                            "checkpoint": checkpoint_path.name,
                                            "training_condition": training_condition,
                                            "seed": seed,
                                            "assignment_condition": assignment_condition,
                                            "influence_class": influence_class,
                                            **spec,
                                            "heads": "+".join(map(str, spec["heads"])),
                                            "component": component,
                                            "family": family,
                                            "chart_dimension": dimension,
                                            "min_sigma": threshold,
                                            "min_condition_ratio": min_condition_ratio,
                                            "max_bootstrap_stability": max_bootstrap_stability,
                                            "filter_mode": "paired_shared_mask",
                                            "control": "learned",
                                            **_typed_summary(
                                                retained,
                                                selected[assignment_condition],
                                                evaluation_indices,
                                            ),
                                        }
                                    )

        _append_rows(output / "tables" / "summary.csv", summary_rows)
        _append_rows(output / "tables" / "cycles.csv", cycle_rows)
        _append_rows(output / "tables" / "edges.csv", edge_rows)
        total_rows += len(summary_rows)
        total_cycle_rows += len(cycle_rows)
        (output / "progress.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "completed_checkpoints": checkpoint_index,
                    "summary_rows": total_rows,
                    "cycle_rows": total_cycle_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        del model, feature_sets
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    with (output / "tables" / "summary.csv").open(newline="", encoding="utf-8") as handle:
        plot_rows = list(csv.DictReader(handle))
    plots = _plot(plot_rows, output / "plots")
    settings = {
        "conditions": list(conditions),
        "seeds": None if seeds is None else list(seeds),
        "chart_dimensions": list(chart_dimensions),
        "thresholds": list(thresholds),
        "ridge": ridge,
        "min_condition_ratio": min_condition_ratio,
        "max_bootstrap_stability": max_bootstrap_stability,
        "bootstrap_samples": bootstrap_samples,
        "bitflip_variable": bitflip_variable,
        "controls": list(TYPED_CONTROLS),
    }
    delta_rows = _paired_delta_rows(plot_rows)
    _append_rows(output / "tables" / "bitflip_deltas.csv", delta_rows)
    (output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    _write_report(plot_rows, delta_rows, nominations, output, settings)
    (output / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "checkpoints": len(checkpoints),
                "summary_rows": total_rows,
                "cycle_rows": total_cycle_rows,
                "plots": [str(path.relative_to(output)) for path in plots],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
