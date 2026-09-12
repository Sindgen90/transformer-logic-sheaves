from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .complex_experiment import _write_rows
from .data import AssignedExpression, collate_expressions
from .equivalence import EquivalenceDiagram
from .holonomy_audit import _load_model, _regenerate_evaluation_data
from .qkv_patching import COMPONENT_GROUPS
from .training import resolve_device


@dataclass(frozen=True)
class PathPatchCase:
    family: str
    label: str
    start: AssignedExpression
    left_donor: AssignedExpression
    right_donor: AssignedExpression


def _path_cases(diagrams: list[EquivalenceDiagram]) -> list[PathPatchCase]:
    cases: list[PathPatchCase] = []
    for diagram in diagrams:
        for pair in diagram.path_pairs:
            left_vertex = diagram.edges[pair.left[-1]].source
            right_vertex = diagram.edges[pair.right[-1]].source
            left = diagram.vertices[left_vertex]
            right = diagram.vertices[right_vertex]
            if left == right:
                continue
            cases.append(
                PathPatchCase(
                    family=diagram.family,
                    label=pair.label,
                    start=diagram.vertices[pair.start],
                    left_donor=left,
                    right_donor=right,
                )
            )
    return cases


def _matched_shuffle(cases: list[PathPatchCase]) -> np.ndarray:
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, case in enumerate(cases):
        groups[(case.family, case.start.value)].append(index)
    permutation = np.arange(len(cases))
    for indices in groups.values():
        if len(indices) < 2:
            continue
        shifted = indices[1:] + indices[:1]
        permutation[indices] = shifted
    return permutation


def _true_margin(logits: torch.Tensor, labels: np.ndarray) -> np.ndarray:
    difference = (logits[:, 1] - logits[:, 0]).float().cpu().numpy()
    return difference * (2 * labels - 1)


@torch.inference_mode()
def _model_rows(
    model,
    cases: list[PathPatchCase],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, Any]], float]:
    permutation = _matched_shuffle(cases)
    labels = np.asarray([case.start.value for case in cases], dtype=np.int64)
    families = np.asarray([case.family for case in cases])
    clean_chunks: list[torch.Tensor] = []
    result_chunks: dict[tuple[int, str, str, str], list[torch.Tensor]] = defaultdict(list)
    self_errors: list[float] = []
    head_modes: tuple[tuple[str, int | None], ...] = (("all", None), ("head_2", 2))

    for start_index in range(0, len(cases), batch_size):
        stop = min(start_index + batch_size, len(cases))
        indices = np.arange(start_index, stop)
        shuffled = permutation[indices]
        start_batch = collate_expressions([cases[index].start for index in indices])
        left_batch = collate_expressions([cases[index].left_donor for index in indices])
        right_batch = collate_expressions([cases[index].right_donor for index in indices])
        shuffled_left_batch = collate_expressions([cases[index].left_donor for index in shuffled])
        shuffled_right_batch = collate_expressions([cases[index].right_donor for index in shuffled])
        tokens = start_batch.tokens.to(device)
        mask = start_batch.padding_mask.to(device)
        positions = torch.zeros(len(indices), dtype=torch.long, device=device)
        clean = model(tokens, mask)
        clean_chunks.append(clean.cpu())

        donors = {
            "matched_left": model.qkv_projections(
                left_batch.tokens.to(device), left_batch.padding_mask.to(device)
            ),
            "matched_right": model.qkv_projections(
                right_batch.tokens.to(device), right_batch.padding_mask.to(device)
            ),
            "shuffled_left": model.qkv_projections(
                shuffled_left_batch.tokens.to(device),
                shuffled_left_batch.padding_mask.to(device),
            ),
            "shuffled_right": model.qkv_projections(
                shuffled_right_batch.tokens.to(device),
                shuffled_right_batch.padding_mask.to(device),
            ),
        }
        names = ("query", "key", "value")
        for layer in (5, 6):
            layer_values = {
                donor_name: dict(zip(names, projections[layer - 1], strict=True))
                for donor_name, projections in donors.items()
            }
            for component, component_names in COMPONENT_GROUPS.items():
                for head_name, head in head_modes:
                    for condition in ("matched", "shuffled"):
                        for route in ("left", "right"):
                            values = layer_values[f"{condition}_{route}"]
                            patch_values = {name: values[name][:, 0] for name in component_names}
                            patched = model.forward_qkv_patched(
                                tokens,
                                mask,
                                patch_layer=layer,
                                patch_positions=positions,
                                patch_values=patch_values,
                                patch_head=head,
                            )
                            result_chunks[
                                (layer, component, head_name, f"{condition}_{route}")
                            ].append(patched.cpu())
            own = model.qkv_projections(tokens, mask)[4][0][:, 0]
            self_patched = model.forward_qkv_patched(
                tokens,
                mask,
                patch_layer=5,
                patch_positions=positions,
                patch_values={"query": own},
            )
            self_errors.append(float((self_patched - clean).abs().max().item()))

    clean_logits = torch.cat(clean_chunks)
    clean_margin = _true_margin(clean_logits, labels)
    clean_prediction = clean_logits.argmax(dim=-1).numpy()
    results = {key: torch.cat(value) for key, value in result_chunks.items()}
    rows: list[dict[str, Any]] = []
    for layer in (5, 6):
        for component in COMPONENT_GROUPS:
            for head_name, _ in head_modes:
                matched_left = results[(layer, component, head_name, "matched_left")]
                matched_right = results[(layer, component, head_name, "matched_right")]
                shuffled_left = results[(layer, component, head_name, "shuffled_left")]
                shuffled_right = results[(layer, component, head_name, "shuffled_right")]
                margins = {
                    "matched_left": _true_margin(matched_left, labels),
                    "matched_right": _true_margin(matched_right, labels),
                    "shuffled_left": _true_margin(shuffled_left, labels),
                    "shuffled_right": _true_margin(shuffled_right, labels),
                }
                predictions = {
                    "matched_left": matched_left.argmax(dim=-1).numpy(),
                    "matched_right": matched_right.argmax(dim=-1).numpy(),
                    "shuffled_left": shuffled_left.argmax(dim=-1).numpy(),
                    "shuffled_right": shuffled_right.argmax(dim=-1).numpy(),
                }
                for family in ("ALL", *sorted(set(families.tolist()))):
                    selected = (
                        np.ones(len(cases), dtype=bool) if family == "ALL" else families == family
                    )
                    matched_disagreement = np.abs(
                        margins["matched_left"][selected] - margins["matched_right"][selected]
                    )
                    shuffled_disagreement = np.abs(
                        margins["shuffled_left"][selected] - margins["shuffled_right"][selected]
                    )
                    rows.append(
                        {
                            "measured_layer": layer,
                            "component": component,
                            "head": head_name,
                            "family": family,
                            "count": int(selected.sum()),
                            "clean_accuracy": float(
                                (clean_prediction[selected] == labels[selected]).mean()
                            ),
                            "matched_left_accuracy": float(
                                (predictions["matched_left"][selected] == labels[selected]).mean()
                            ),
                            "matched_right_accuracy": float(
                                (predictions["matched_right"][selected] == labels[selected]).mean()
                            ),
                            "matched_route_disagreement": float(np.mean(matched_disagreement)),
                            "shuffled_route_disagreement": float(np.mean(shuffled_disagreement)),
                            "route_specificity": float(
                                np.mean(shuffled_disagreement) - np.mean(matched_disagreement)
                            ),
                            "matched_left_abs_effect": float(
                                np.mean(
                                    np.abs(
                                        margins["matched_left"][selected] - clean_margin[selected]
                                    )
                                )
                            ),
                            "matched_right_abs_effect": float(
                                np.mean(
                                    np.abs(
                                        margins["matched_right"][selected] - clean_margin[selected]
                                    )
                                )
                            ),
                            "matched_prediction_disagreement": float(
                                np.mean(
                                    predictions["matched_left"][selected]
                                    != predictions["matched_right"][selected]
                                )
                            ),
                        }
                    )
    return rows, max(self_errors, default=0.0)


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("measured_layer", "component", "head", "family")
    excluded = {*keys, "condition", "seed", "model", "architecture_layers"}
    metrics = [
        key
        for key, value in rows[0].items()
        if key not in excluded and isinstance(value, (int, float, np.number))
    ]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group_key, group in groups.items():
        item = dict(zip(keys, group_key, strict=True))
        item["models"] = len(group)
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def _plot(rows: list[dict[str, Any]], output: Path) -> Path:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    components = tuple(COMPONENT_GROUPS)
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for axis, head in zip(axes, ("all", "head_2"), strict=True):
        selected = {
            (int(row["measured_layer"]), row["component"]): row
            for row in rows
            if row["family"] == "ALL" and row["head"] == head
        }
        x = np.arange(len(components))
        width = 0.2
        for index, (layer, condition, color) in enumerate(
            (
                (5, "matched_route_disagreement_mean", "#4472c4"),
                (5, "shuffled_route_disagreement_mean", "#9dc3e6"),
                (6, "matched_route_disagreement_mean", "#c55a11"),
                (6, "shuffled_route_disagreement_mean", "#f4b183"),
            )
        ):
            axis.bar(
                x + (index - 1.5) * width,
                [float(selected[(layer, component)][condition]) for component in components],
                width,
                color=color,
                label=f"layer {layer} {'matched' if 'matched_' in condition else 'shuffled'}",
            )
        axis.set_xticks(x, components)
        axis.set(title=head.replace("_", " "), ylabel="absolute true-margin disagreement")
        axis.legend(fontsize=8)
    figure.suptitle("AE. Causal Q/K/V patching along two equivalent rewrite paths")
    figure.tight_layout()
    path = output / "AE_causal_path_patching.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def run_path_patching(run_directory: Path, *, device: str = "auto") -> Path:
    run_directory = run_directory.resolve()
    config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
    _, diagrams = _regenerate_evaluation_data(config)
    cases = _path_cases(diagrams)
    output = run_directory / "path_patching"
    output.mkdir(exist_ok=True)
    target_device = resolve_device(device)
    rows: list[dict[str, Any]] = []
    self_errors: list[float] = []
    checkpoints = sorted((run_directory / "checkpoints").glob("layers_6_higher_diversity*.pt"))
    previous_fastpath = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        for index, checkpoint_path in enumerate(checkpoints, start=1):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            seed = int(checkpoint["train_config"]["seed"])
            print(f"[{index}/{len(checkpoints)}] causal paths {checkpoint_path.stem}", flush=True)
            model = _load_model(checkpoint_path, target_device)
            model_rows, error = _model_rows(
                model,
                cases,
                device=target_device,
                batch_size=int(config["batch_size"]),
            )
            rows.extend(
                {
                    "architecture_layers": 6,
                    "condition": "higher_diversity",
                    "seed": seed,
                    "model": checkpoint_path.stem,
                    **row,
                }
                for row in model_rows
            )
            self_errors.append(error)
            del model
            if target_device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        torch.backends.mha.set_fastpath_enabled(previous_fastpath)
    aggregate = _aggregate(rows)
    _write_rows(output / "path_patching_metrics.csv", rows)
    _write_rows(output / "aggregate_path_patching_metrics.csv", aggregate)
    plot = _plot(aggregate, output / "plots")
    (output / "report.md").write_text(
        "\n".join(
            (
                "# Causal Q/K/V patching across equivalent rewrite paths",
                "",
                f"Cases: {len(cases)} path pairs from {len({case.family for case in cases})} diagram families.",
                "",
                "For each path pair, the donor is the penultimate expression on each route. Q, K, V, or all three are patched at `<CLS>` into the common start expression at layers 5 and 6. The primary statistic is the absolute difference between the two patched true-class margins. A family-and-truth-matched donor derangement is the context-shuffled control.",
                "",
                f"Maximum self-patch logit error: {max(self_errors, default=0.0):.3e}.",
                "",
                f"![causal path patching](plots/{plot.name})",
                "",
            )
        ),
        encoding="utf-8",
    )
    (output / "status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "models": len(checkpoints),
                "cases": len(cases),
                "layers": [5, 6],
                "components": list(COMPONENT_GROUPS),
                "heads": ["all", "head_2"],
                "max_self_patch_logit_error": max(self_errors, default=0.0),
                "plot": str(plot.relative_to(output)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output
