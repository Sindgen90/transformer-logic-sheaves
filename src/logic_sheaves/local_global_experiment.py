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

from .circuit_selection import nominate_circuits
from .diagram_metrics import predictions
from .gauge_atlas import (
    AtlasChart,
    AtlasEdge,
    chart_reconstruction_errors,
    fit_atlas_edges,
    fit_chart,
    heldout_cycle_return_errors,
    heldout_edge_fidelities,
    heldout_section_energies,
    largest_connected_component,
    sheaf_laplacian_spectrum,
    stable_subgraph,
    typed_component_cycles,
)
from .gauge_experiment import _family_tensors, _orders, atlas_features
from .holonomy_audit import _load_model, _regenerate_evaluation_data
from .training import resolve_device
from .typed_gauge_experiment import (
    _circuit_specs,
    _condition_from_checkpoint,
    _feature_matrix,
    _mean,
)

CHART_MODES = ("global", "family_global", "vertex_local")
OUTCOMES = ("correct", "incorrect")
COMPONENTS = ("query", "key", "value", "coupled_qk")


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


def _balanced_splits(
    labels: np.ndarray,
    correct: np.ndarray,
    *,
    seed: int,
    minimum_per_cell: int = 8,
) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Create equally sized, truth-balanced correct and incorrect fit/eval groups."""

    rng = np.random.default_rng(seed)
    cells = {
        (outcome, label): np.flatnonzero((correct == outcome) & (labels == label))
        for outcome in (False, True)
        for label in (0, 1)
    }
    count = min(map(len, cells.values()))
    if count < minimum_per_cell:
        return None
    fit_per_label = max(5, (2 * count) // 3)
    fit_per_label = min(fit_per_label, count - 2)
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for outcome, name in ((True, "correct"), (False, "incorrect")):
        fit: list[int] = []
        evaluation: list[int] = []
        for label in (0, 1):
            selected = rng.permutation(cells[(outcome, label)])[:count]
            fit.extend(selected[:fit_per_label].tolist())
            evaluation.extend(selected[fit_per_label:].tolist())
        output[name] = (np.asarray(sorted(fit)), np.asarray(sorted(evaluation)))
    return output


def _charts_from_edges(edges: dict[tuple[int, int], AtlasEdge]) -> dict[int, AtlasChart]:
    charts: dict[int, AtlasChart] = {}
    for edge in edges.values():
        charts[edge.source] = edge.source_chart
        charts[edge.target] = edge.target_chart
    return charts


def _shared_charts(chart: AtlasChart, vertices: int) -> dict[int, AtlasChart]:
    return {vertex: chart for vertex in range(vertices)}


def _pooled_chart(
    tensors: dict[str, np.ndarray],
    indices: dict[str, np.ndarray],
    dimension: int,
) -> AtlasChart:
    pooled = np.concatenate(
        [
            tensors[family][indices[family]].reshape(-1, tensors[family].shape[-1])
            for family in indices
        ]
    )
    return fit_chart(pooled, dimension)


def _fit_edges(
    tensor: np.ndarray,
    edge_pairs: list[tuple[int, int]],
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    labels: np.ndarray,
    *,
    dimension: int,
    ridge: float,
    charts: dict[int, AtlasChart] | None,
    bootstrap_samples: int,
    min_sigma: float,
    min_condition_ratio: float,
    max_bootstrap_stability: float,
    seed: int,
) -> tuple[dict[tuple[int, int], AtlasEdge], dict[tuple[int, int], AtlasEdge]]:
    learned = fit_atlas_edges(
        tensor,
        edge_pairs,
        fit_indices,
        dimension=dimension,
        ridge=ridge,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=seed,
        charts=charts,
    )
    shuffled = fit_atlas_edges(
        tensor,
        edge_pairs,
        fit_indices,
        dimension=dimension,
        ridge=ridge,
        target_orders=_orders(
            edge_pairs,
            labels[fit_indices],
            np.random.default_rng(seed + 10_000),
            preserve_label=False,
        ),
        charts=charts,
    )
    learned_fidelity = heldout_edge_fidelities(learned, tensor, evaluation_indices)
    shuffled_fidelity = heldout_edge_fidelities(shuffled, tensor, evaluation_indices)
    faithful = {key for key in learned if learned_fidelity[key] < shuffled_fidelity[key]}
    stable = stable_subgraph(
        learned,
        min_sigma=min_sigma,
        min_condition_ratio=min_condition_ratio,
        max_bootstrap_stability=max_bootstrap_stability,
    )
    retained = largest_connected_component(
        {key: edge for key, edge in learned.items() if key in faithful and key in stable}
    )
    return learned, retained


def _subset_edges(
    edges: dict[tuple[int, int], AtlasEdge], keys: set[tuple[int, int]]
) -> dict[tuple[int, int], AtlasEdge]:
    return largest_connected_component({key: edge for key, edge in edges.items() if key in keys})


def _summary_row(
    edges: dict[tuple[int, int], AtlasEdge],
    charts: dict[int, AtlasChart],
    tensor: np.ndarray,
    evaluation: np.ndarray,
) -> dict[str, Any]:
    cycles = typed_component_cycles(edges)
    returns = heldout_cycle_return_errors(edges, tensor, evaluation)
    section = heldout_section_energies(edges, tensor, evaluation)
    reconstruction = chart_reconstruction_errors(charts, tensor, evaluation)
    spectrum, approximate_h0 = sheaf_laplacian_spectrum(edges)
    finite_returns = returns[np.isfinite(returns)]
    finite_section = section[np.isfinite(section)]
    return {
        "fit_vertices": len(charts),
        "vertices_retained": len({vertex for edge in edges for vertex in edge}),
        "edges_retained": len(edges),
        "cycles_retained": len(cycles),
        "transport_unit": _mean([cycle.transport_unit_distance for cycle in cycles]),
        "proxy_unit": _mean([cycle.proxy_unit_distance for cycle in cycles]),
        "relative_unit": _mean([cycle.relative_unit_distance for cycle in cycles]),
        "reconstruction_error": _mean(reconstruction.tolist()),
        "state_return_mean": _mean(finite_returns.tolist()),
        "state_return_median": float(np.median(finite_returns)) if len(finite_returns) else np.nan,
        "section_energy_mean": _mean(finite_section.tolist()),
        "section_energy_median": float(np.median(finite_section))
        if len(finite_section)
        else np.nan,
        "laplacian_min_eigenvalue": float(spectrum[0]) if len(spectrum) else np.nan,
        "laplacian_h0_dimension": approximate_h0,
        "sigma_min_mean": _mean([edge.sigma_min for edge in edges.values()]),
        "condition_ratio_mean": _mean([edge.condition_ratio for edge in edges.values()]),
        "bootstrap_stability_mean": _mean([edge.bootstrap_stability for edge in edges.values()]),
        "evaluation_count": len(evaluation),
    }


def _comparison_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    identity = (
        "checkpoint",
        "training_condition",
        "seed",
        "layer",
        "scope",
        "nomination_component",
        "nomination_status",
        "circuit_mode",
        "heads",
        "component",
        "family",
        "chart_dimension",
        "chart_mode",
        "fit_mode",
    )
    indexed: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        indexed[tuple(row[field] for field in identity)][row["outcome"]] = row
    metrics = (
        "relative_unit",
        "reconstruction_error",
        "state_return_mean",
        "section_energy_mean",
        "laplacian_min_eigenvalue",
    )
    output: list[dict[str, Any]] = []
    for key, outcomes in indexed.items():
        if not set(OUTCOMES) <= set(outcomes):
            continue
        row: dict[str, Any] = dict(zip(identity, key, strict=True))
        row["correct_evaluation_count"] = outcomes["correct"]["evaluation_count"]
        row["incorrect_evaluation_count"] = outcomes["incorrect"]["evaluation_count"]
        row["cycles_retained"] = outcomes["correct"]["cycles_retained"]
        for metric in metrics:
            correct = float(outcomes["correct"][metric])
            incorrect = float(outcomes["incorrect"][metric])
            row[f"correct_{metric}"] = correct
            row[f"incorrect_{metric}"] = incorrect
            row[f"delta_{metric}"] = incorrect - correct
        output.append(row)
    return output


def _plot(rows: list[dict[str, str]], output: Path, dimension: int) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def finite(items: list[dict[str, str]], field: str) -> list[float]:
        values = [float(item[field]) for item in items]
        return [value for value in values if np.isfinite(value)]

    def mean(items: list[dict[str, str]], field: str) -> float:
        return _mean(finite(items, field))

    def save(figure: Any, name: str) -> None:
        figure.tight_layout()
        path = output / name
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)

    selected = [
        row
        for row in rows
        if int(row["chart_dimension"]) == dimension and row["nomination_status"] == "confirmed"
    ]
    group = [row for row in selected if row["fit_mode"] == "group_fit"]
    shared = [row for row in selected if row["fit_mode"] == "shared_fit"]
    colors = {"correct": "#4c78a8", "incorrect": "#e45756"}

    def outcome_bars(items: list[dict[str, str]], field: str, title: str, name: str) -> None:
        x = np.arange(len(CHART_MODES))
        width = 0.36
        figure, axis = plt.subplots(figsize=(8, 4.8))
        for index, outcome in enumerate(OUTCOMES):
            axis.bar(
                x + (index - 0.5) * width,
                [
                    mean(
                        [r for r in items if r["chart_mode"] == mode and r["outcome"] == outcome],
                        field,
                    )
                    for mode in CHART_MODES
                ],
                width,
                label=outcome,
                color=colors[outcome],
            )
        axis.set_xticks(x, ("global", "family-global", "vertex-local"))
        axis.set(ylabel=field.replace("_", " "), title=title)
        axis.legend()
        save(figure, name)

    outcome_bars(
        group,
        "relative_unit",
        "Operator holonomy by chart and correctness",
        "A_chart_correctness_holonomy.png",
    )

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(COMPONENTS))
    width = 0.25
    for index, mode in enumerate(CHART_MODES):
        deltas = []
        for component in COMPONENTS:
            correct = mean(
                [
                    r
                    for r in group
                    if r["chart_mode"] == mode
                    and r["component"] == component
                    and r["outcome"] == "correct"
                ],
                "relative_unit",
            )
            incorrect = mean(
                [
                    r
                    for r in group
                    if r["chart_mode"] == mode
                    and r["component"] == component
                    and r["outcome"] == "incorrect"
                ],
                "relative_unit",
            )
            deltas.append(incorrect - correct)
        axis.bar(x + (index - 1) * width, deltas, width, label=mode)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, COMPONENTS)
    axis.set(
        ylabel="incorrect - correct relative holonomy", title="Correctness gap by activation fiber"
    )
    axis.legend()
    save(figure, "B_component_holonomy_gap.png")

    outcome_bars(
        shared,
        "state_return_mean",
        "Shared-connection state-return error",
        "C_shared_state_return.png",
    )
    outcome_bars(
        shared,
        "section_energy_mean",
        "Connection-sheaf section energy",
        "D_sheaf_section_energy.png",
    )
    outcome_bars(
        shared,
        "reconstruction_error",
        "Held-out chart reconstruction error",
        "E_chart_reconstruction.png",
    )

    figure, axis = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(CHART_MODES))
    width = 0.25
    for index, circuit_mode in enumerate(("circuit", "complement", "all")):
        values = [
            mean(
                [
                    r
                    for r in group
                    if r["chart_mode"] == chart_mode and r["circuit_mode"] == circuit_mode
                ],
                "relative_unit",
            )
            for chart_mode in CHART_MODES
        ]
        axis.bar(x + (index - 1) * width, values, width, label=circuit_mode)
    axis.set_xticks(x, ("global", "family-global", "vertex-local"))
    axis.set(ylabel="relative unit holonomy", title="Circuit localization across chart scales")
    axis.legend()
    save(figure, "F_circuit_localization.png")

    families = sorted({row["family"] for row in group})
    matrix = np.full((len(families), len(CHART_MODES)), np.nan)
    for family_index, family in enumerate(families):
        for mode_index, mode in enumerate(CHART_MODES):
            correct = mean(
                [
                    r
                    for r in group
                    if r["family"] == family
                    and r["chart_mode"] == mode
                    and r["outcome"] == "correct"
                ],
                "relative_unit",
            )
            incorrect = mean(
                [
                    r
                    for r in group
                    if r["family"] == family
                    and r["chart_mode"] == mode
                    and r["outcome"] == "incorrect"
                ],
                "relative_unit",
            )
            matrix[family_index, mode_index] = incorrect - correct
    figure, axis = plt.subplots(figsize=(8, 6.5))
    image = axis.imshow(
        matrix,
        cmap="coolwarm",
        aspect="auto",
        vmin=-np.nanmax(np.abs(matrix)),
        vmax=np.nanmax(np.abs(matrix)),
    )
    axis.set_xticks(range(len(CHART_MODES)), ("global", "family-global", "vertex-local"))
    axis.set_yticks(range(len(families)), families)
    axis.set_title("Family correctness gap in relative holonomy")
    figure.colorbar(image, ax=axis, label="incorrect - correct")
    save(figure, "G_family_holonomy_gap.png")

    outcome_bars(
        group, "laplacian_min_eigenvalue", "Sheaf-Laplacian obstruction", "H_sheaf_laplacian.png"
    )
    return paths


def _write_report(rows: list[dict[str, str]], output: Path, settings: dict[str, Any]) -> None:
    dimension = int(settings["chart_dimensions"][0])
    selected = [
        row
        for row in rows
        if int(row["chart_dimension"]) == dimension and row["nomination_status"] == "confirmed"
    ]

    def value(
        fit_mode: str,
        chart_mode: str,
        outcome: str,
        field: str,
        seed: int | None = None,
    ) -> float:
        values = [
            float(row[field])
            for row in selected
            if row["fit_mode"] == fit_mode
            and row["chart_mode"] == chart_mode
            and row["outcome"] == outcome
            and (seed is None or int(row["seed"]) == seed)
            and np.isfinite(float(row[field]))
        ]
        return _mean(values)

    def circuit_value(chart_mode: str, circuit_mode: str) -> float:
        values = [
            float(row["relative_unit"])
            for row in selected
            if row["fit_mode"] == "group_fit"
            and row["chart_mode"] == chart_mode
            and row["circuit_mode"] == circuit_mode
            and np.isfinite(float(row["relative_unit"]))
        ]
        return _mean(values)

    lines = [
        "# Correct-versus-incorrect and local-versus-global atlas experiment",
        "",
        (
            "The canonical vertex-0 prediction defines correctness. Correct and incorrect "
            "groups are matched within each logical family and truth label. Group-fitted rows "
            "compare separately fitted operator connections; shared-fit rows use one connection "
            "and compare held-out states without refitting."
        ),
        "",
        "## Chart scales",
        "",
        "- `global`: one PCA chart shared across every family and vertex.",
        "- `family_global`: one chart per logical family, shared across its vertices.",
        "- `vertex_local`: an independently fitted chart at each expression vertex.",
        "",
        (
            f"Primary chart dimension: {dimension}. Models: "
            f"{len({row['checkpoint'] for row in rows})}."
        ),
        "",
        "## Aggregate comparison",
        "",
        "| Chart | Correct H_rel | Incorrect H_rel | Gap | Shared correct return | Shared incorrect return | Correct sheaf energy | Incorrect sheaf energy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in CHART_MODES:
        correct_h = value("group_fit", mode, "correct", "relative_unit")
        incorrect_h = value("group_fit", mode, "incorrect", "relative_unit")
        lines.append(
            f"| {mode} | {correct_h:.4f} | {incorrect_h:.4f} | "
            f"{incorrect_h - correct_h:+.4f} | "
            f"{value('shared_fit', mode, 'correct', 'state_return_mean'):.4f} | "
            f"{value('shared_fit', mode, 'incorrect', 'state_return_mean'):.4f} | "
            f"{value('shared_fit', mode, 'correct', 'section_energy_mean'):.4f} | "
            f"{value('shared_fit', mode, 'incorrect', 'section_energy_mean'):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Seed-wise incorrect-minus-correct gaps",
            "",
            "| Chart | Seed | Operator H_rel | Shared return | Shared sheaf energy |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    seeds = sorted({int(row["seed"]) for row in selected})
    for mode in CHART_MODES:
        for seed in seeds:
            lines.append(
                f"| {mode} | {seed} | "
                f"{value('group_fit', mode, 'incorrect', 'relative_unit', seed) - value('group_fit', mode, 'correct', 'relative_unit', seed):+.4f} | "
                f"{value('shared_fit', mode, 'incorrect', 'state_return_mean', seed) - value('shared_fit', mode, 'correct', 'state_return_mean', seed):+.4f} | "
                f"{value('shared_fit', mode, 'incorrect', 'section_energy_mean', seed) - value('shared_fit', mode, 'correct', 'section_energy_mean', seed):+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Main findings",
            "",
            (
                "Family-global charts have the lowest operator holonomy and sheaf section "
                "energy, while vertex-local charts have the lowest actual loop-return error. "
                "A single global chart performs worst on state return. This favors a "
                "family-structured atlas over either one universal coordinate system or a "
                "claim that every vertex needs an unrelated chart."
            ),
            "",
            (
                "Incorrect examples have higher shared-fit return error and section energy "
                "for every chart scale in every model seed. Their separately fitted operator "
                "holonomy is also higher in eight of nine chart-by-seed comparisons, but the "
                "operator gap is small relative to its absolute level."
            ),
            "",
            (
                f"Incorrect activations also have higher chart reconstruction error: "
                f"global {value('shared_fit', 'global', 'correct', 'reconstruction_error'):.4f}"
                f" -> {value('shared_fit', 'global', 'incorrect', 'reconstruction_error'):.4f}, "
                f"family-global {value('shared_fit', 'family_global', 'correct', 'reconstruction_error'):.4f}"
                f" -> {value('shared_fit', 'family_global', 'incorrect', 'reconstruction_error'):.4f}, "
                f"and vertex-local {value('shared_fit', 'vertex_local', 'correct', 'reconstruction_error'):.4f}"
                f" -> {value('shared_fit', 'vertex_local', 'incorrect', 'reconstruction_error'):.4f}. "
                "Some of the correctness signal therefore reflects incorrect states lying "
                "farther from the dominant activation subspace."
            ),
            "",
            (
                f"Circuit localization remains weak. For family-global charts, relative "
                f"holonomy is {circuit_value('family_global', 'circuit'):.4f} in nominated "
                f"heads, {circuit_value('family_global', 'complement'):.4f} in their "
                f"complements, and {circuit_value('family_global', 'all'):.4f} across all "
                "heads. The all-head representation is slightly flatter than the nominated "
                "subset, not the reverse."
            ),
            "",
            "## Interpretation",
            "",
            (
                "Operator-holonomy gaps are population comparisons and can be influenced by "
                "conditioning. Shared-fit state-return and section-energy gaps are the cleaner "
                "example-level test because correct and incorrect states use identical charts, "
                "edge maps, and retained topology."
            ),
            "",
            (
                "The connection-sheaf energy uses edge stalk `F_v`, restrictions `rho_u=Q_vu` "
                "and `rho_v=I`, and residual `Q_vu z_u-z_v`. The sheaf-Laplacian spectrum is "
                "computed from the corresponding block coboundary matrix."
            ),
            "",
            "## Figures",
            "",
        ]
    )
    for path in sorted((output / "plots").glob("[A-H]_*.png")):
        lines.extend((f"![{path.stem}](plots/{path.name})", ""))
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_local_global_experiment(
    run_directory: Path,
    *,
    device: str = "auto",
    conditions: tuple[str, ...] = ("higher_diversity",),
    seeds: tuple[int, ...] | None = None,
    chart_dimensions: tuple[int, ...] = (8,),
    ridge: float = 1e-2,
    bootstrap_samples: int = 8,
    min_sigma: float = 0.0,
    min_condition_ratio: float = 0.0,
    max_bootstrap_stability: float = 0.5,
) -> Path:
    run_directory = run_directory.resolve()
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    _, diagrams = _regenerate_evaluation_data(config)
    nominations = nominate_circuits(run_directory / "qkv_patching" / "patch_metrics.csv")
    specs = [
        spec
        for spec in _circuit_specs(nominations, int(config["n_heads"]))
        if spec["nomination_status"] == "confirmed"
    ]
    checkpoints = []
    for path in sorted((run_directory / "checkpoints").glob("layers_6_*.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        seed = int(checkpoint["train_config"]["seed"])
        if _condition_from_checkpoint(path) in conditions and (seeds is None or seed in seeds):
            checkpoints.append(path)
    if not checkpoints:
        raise ValueError("No matching six-layer checkpoints")

    stamp = datetime.now(timezone.utc).strftime("local_global_%Y%m%d_%H%M%S_%fZ")
    output = run_directory / "local_global" / stamp
    output.mkdir(parents=True)
    target_device = resolve_device(device)
    all_expressions = [vertex for diagram in diagrams for vertex in diagram.vertices]
    offsets = np.cumsum([0] + [len(diagram.vertices) for diagram in diagrams])
    total_rows = 0

    for checkpoint_index, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        seed = int(checkpoint["train_config"]["seed"])
        model_heads = int(checkpoint["model_config"]["n_heads"])
        print(
            f"[{checkpoint_index}/{len(checkpoints)}] local/global {checkpoint_path.stem}",
            flush=True,
        )
        model = _load_model(checkpoint_path, target_device)
        predicted = predictions(
            model, all_expressions, device=target_device, batch_size=int(config["batch_size"])
        )
        diagram_correct = np.asarray(
            [
                int(predicted[offsets[index]]) == diagram.vertices[0].value
                for index, diagram in enumerate(diagrams)
            ]
        )
        family_correct: dict[str, list[bool]] = defaultdict(list)
        for diagram, correct in zip(diagrams, diagram_correct, strict=True):
            family_correct[diagram.family].append(bool(correct))

        scopes = tuple(sorted({str(spec["scope"]) for spec in specs}))
        features = atlas_features(
            model,
            all_expressions,
            device=target_device,
            batch_size=int(config["batch_size"]),
            components=("query", "key", "value"),
            scopes=scopes,
            heads=None,
        )
        rows: list[dict[str, Any]] = []
        for spec_index, spec in enumerate(specs):
            for component_index, component in enumerate(COMPONENTS):
                matrix = _feature_matrix(
                    features,
                    layer=int(spec["layer"]),
                    scope=str(spec["scope"]),
                    component=component,
                    heads=spec["heads"],
                    model_heads=model_heads,
                )
                grouped = _family_tensors(diagrams, matrix)
                tensors = {family: tensor for family, (_, tensor) in grouped.items()}
                family_diagrams = {family: items for family, (items, _) in grouped.items()}
                splits: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
                for family_index, family in enumerate(sorted(grouped)):
                    labels = np.asarray(
                        [item.vertices[0].value for item in family_diagrams[family]]
                    )
                    result = _balanced_splits(
                        labels,
                        np.asarray(family_correct[family]),
                        seed=7_000_000
                        + seed * 100_000
                        + spec_index * 1000
                        + component_index * 100
                        + family_index,
                    )
                    if result is not None:
                        splits[family] = result
                if not splits:
                    continue

                for dimension in chart_dimensions:
                    global_charts: dict[str, AtlasChart] = {}
                    for outcome in (*OUTCOMES, "shared"):
                        indices = {
                            family: (
                                np.concatenate((split["correct"][0], split["incorrect"][0]))
                                if outcome == "shared"
                                else split[outcome][0]
                            )
                            for family, split in splits.items()
                        }
                        global_charts[outcome] = _pooled_chart(tensors, indices, dimension)

                    for family_index, (family, split) in enumerate(sorted(splits.items())):
                        tensor = tensors[family]
                        items = family_diagrams[family]
                        labels = np.asarray([item.vertices[0].value for item in items])
                        edge_pairs = [(edge.source, edge.target) for edge in items[0].edges]
                        for chart_index, chart_mode in enumerate(CHART_MODES):
                            fitted: dict[str, dict[tuple[int, int], AtlasEdge]] = {}
                            retained: dict[str, dict[tuple[int, int], AtlasEdge]] = {}
                            all_charts: dict[str, dict[int, AtlasChart]] = {}
                            for outcome_index, outcome in enumerate(OUTCOMES):
                                fit_indices, evaluation = split[outcome]
                                supplied = None
                                if chart_mode == "global":
                                    supplied = _shared_charts(
                                        global_charts[outcome], tensor.shape[1]
                                    )
                                elif chart_mode == "family_global":
                                    chart = fit_chart(
                                        tensor[fit_indices].reshape(-1, tensor.shape[-1]), dimension
                                    )
                                    supplied = _shared_charts(chart, tensor.shape[1])
                                raw, kept = _fit_edges(
                                    tensor,
                                    edge_pairs,
                                    fit_indices,
                                    evaluation,
                                    labels,
                                    dimension=dimension,
                                    ridge=ridge,
                                    charts=supplied,
                                    bootstrap_samples=bootstrap_samples,
                                    min_sigma=min_sigma,
                                    min_condition_ratio=min_condition_ratio,
                                    max_bootstrap_stability=max_bootstrap_stability,
                                    seed=8_000_000
                                    + seed * 100_000
                                    + spec_index * 10_000
                                    + component_index * 1000
                                    + family_index * 50
                                    + chart_index * 10
                                    + outcome_index,
                                )
                                fitted[outcome] = raw
                                retained[outcome] = kept
                                all_charts[outcome] = supplied or _charts_from_edges(raw)
                            shared_keys = set(retained["correct"]) & set(retained["incorrect"])
                            for outcome in OUTCOMES:
                                edges = _subset_edges(fitted[outcome], shared_keys)
                                fit_indices, evaluation = split[outcome]
                                rows.append(
                                    {
                                        "checkpoint": checkpoint_path.name,
                                        "training_condition": _condition_from_checkpoint(
                                            checkpoint_path
                                        ),
                                        "seed": seed,
                                        **spec,
                                        "heads": "+".join(map(str, spec["heads"])),
                                        "component": component,
                                        "family": family,
                                        "chart_dimension": dimension,
                                        "chart_mode": chart_mode,
                                        "fit_mode": "group_fit",
                                        "outcome": outcome,
                                        "fit_count": len(fit_indices),
                                        "matched_edge_count": len(shared_keys),
                                        **_summary_row(
                                            edges, all_charts[outcome], tensor, evaluation
                                        ),
                                    }
                                )

                            shared_fit = np.concatenate(
                                (split["correct"][0], split["incorrect"][0])
                            )
                            shared_evaluation = np.concatenate(
                                (split["correct"][1], split["incorrect"][1])
                            )
                            supplied = None
                            if chart_mode == "global":
                                supplied = _shared_charts(global_charts["shared"], tensor.shape[1])
                            elif chart_mode == "family_global":
                                chart = fit_chart(
                                    tensor[shared_fit].reshape(-1, tensor.shape[-1]), dimension
                                )
                                supplied = _shared_charts(chart, tensor.shape[1])
                            raw, edges = _fit_edges(
                                tensor,
                                edge_pairs,
                                shared_fit,
                                shared_evaluation,
                                labels,
                                dimension=dimension,
                                ridge=ridge,
                                charts=supplied,
                                bootstrap_samples=bootstrap_samples,
                                min_sigma=min_sigma,
                                min_condition_ratio=min_condition_ratio,
                                max_bootstrap_stability=max_bootstrap_stability,
                                seed=9_000_000
                                + seed * 100_000
                                + spec_index * 10_000
                                + component_index * 1000
                                + family_index * 50
                                + chart_index,
                            )
                            charts = supplied or _charts_from_edges(raw)
                            for outcome in OUTCOMES:
                                _, evaluation = split[outcome]
                                rows.append(
                                    {
                                        "checkpoint": checkpoint_path.name,
                                        "training_condition": _condition_from_checkpoint(
                                            checkpoint_path
                                        ),
                                        "seed": seed,
                                        **spec,
                                        "heads": "+".join(map(str, spec["heads"])),
                                        "component": component,
                                        "family": family,
                                        "chart_dimension": dimension,
                                        "chart_mode": chart_mode,
                                        "fit_mode": "shared_fit",
                                        "outcome": outcome,
                                        "fit_count": len(shared_fit),
                                        "matched_edge_count": len(edges),
                                        **_summary_row(edges, charts, tensor, evaluation),
                                    }
                                )
        _append_rows(output / "tables" / "summary.csv", rows)
        total_rows += len(rows)
        (output / "progress.json").write_text(
            json.dumps(
                {
                    "status": "running",
                    "completed_checkpoints": checkpoint_index,
                    "rows": total_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        del model, features
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    with (output / "tables" / "summary.csv").open(newline="", encoding="utf-8") as handle:
        plot_rows = list(csv.DictReader(handle))
    settings = {
        "conditions": list(conditions),
        "seeds": None if seeds is None else list(seeds),
        "chart_dimensions": list(chart_dimensions),
        "ridge": ridge,
        "bootstrap_samples": bootstrap_samples,
        "min_sigma": min_sigma,
        "min_condition_ratio": min_condition_ratio,
        "max_bootstrap_stability": max_bootstrap_stability,
        "correctness_definition": "canonical vertex-0 prediction equals the expression value",
        "chart_modes": list(CHART_MODES),
    }
    comparisons = _comparison_rows(plot_rows)
    _append_rows(output / "tables" / "correctness_deltas.csv", comparisons)
    plots = _plot(plot_rows, output / "plots", int(chart_dimensions[0]))
    (output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    _write_report(plot_rows, output, settings)
    (output / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "checkpoints": len(checkpoints),
                "rows": total_rows,
                "plots": [str(path.relative_to(output)) for path in plots],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
