from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CircuitNomination:
    layer: int
    site: str
    component: str
    heads: tuple[int, ...]
    discovery_specificity: float
    confirmation_specificity: float
    discovery_matched_effect: float
    status: str = "confirmed"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def nominate_circuits(
    patch_metrics: Path,
    *,
    discovery_seeds: tuple[int, ...] = (0, 1),
    confirmation_seeds: tuple[int, ...] = (2,),
    layers: tuple[int, ...] = (5, 6),
    min_discovery_specificity: float = 1e-3,
    max_groups: int = 4,
) -> list[CircuitNomination]:
    """Freeze causal head sets using model-seed discovery and confirmation splits."""

    with patch_metrics.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["architecture_layers"]) == 6
            and row["condition"] == "higher_diversity"
            and row["subset"] == "eligible"
            and row["operator"] == "ALL"
            and row["head"] != "all"
            and int(row["patch_layer"]) in layers
            and row["component"] in {"query", "key", "value"}
            and row["donor_type"] in {"counterfactual", "counterfactual_shuffled"}
        ]
    by_seed: dict[tuple[int, str, str, int, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (
            int(row["patch_layer"]),
            row["site"],
            row["component"],
            int(row["head"]),
            int(row["seed"]),
        )
        by_seed[key][row["donor_type"]] = float(row["effect_fraction"])

    individual: list[CircuitNomination] = []
    sites = sorted({key[:4] for key in by_seed})
    for layer, site, component, head in sites:
        specificity: dict[int, float] = {}
        matched: dict[int, float] = {}
        for seed in (*discovery_seeds, *confirmation_seeds):
            values = by_seed.get((layer, site, component, head, seed), {})
            if {"counterfactual", "counterfactual_shuffled"} <= set(values):
                specificity[seed] = values["counterfactual"] - values["counterfactual_shuffled"]
                matched[seed] = values["counterfactual"]
        discovery = _mean([specificity[seed] for seed in discovery_seeds if seed in specificity])
        confirmation = _mean(
            [specificity[seed] for seed in confirmation_seeds if seed in specificity]
        )
        individual.append(
            CircuitNomination(
                layer=layer,
                site=site,
                component=component,
                heads=(head,),
                discovery_specificity=discovery,
                confirmation_specificity=confirmation,
                discovery_matched_effect=_mean(
                    [matched[seed] for seed in discovery_seeds if seed in matched]
                ),
                status=(
                    "confirmed"
                    if discovery >= min_discovery_specificity and confirmation > 0.0
                    else "rejected"
                ),
            )
        )

    confirmed = [item for item in individual if item.status == "confirmed"]
    grouped: dict[tuple[int, str, str], list[CircuitNomination]] = defaultdict(list)
    for item in confirmed:
        grouped[(item.layer, item.site, item.component)].append(item)
    combined = [
        CircuitNomination(
            layer=key[0],
            site=key[1],
            component=key[2],
            heads=tuple(sorted(item.heads[0] for item in values)),
            discovery_specificity=float(np.mean([item.discovery_specificity for item in values])),
            confirmation_specificity=float(
                np.mean([item.confirmation_specificity for item in values])
            ),
            discovery_matched_effect=float(
                np.mean([item.discovery_matched_effect for item in values])
            ),
        )
        for key, values in grouped.items()
    ]
    combined.sort(key=lambda item: item.discovery_specificity, reverse=True)

    # Preserve the strongest discovery-only failure as an explicit negative reference.
    rejected = sorted(
        (item for item in individual if item.status == "rejected"),
        key=lambda item: item.discovery_specificity,
        reverse=True,
    )
    output = combined[:max_groups]
    if rejected:
        output.append(
            CircuitNomination(
                **{
                    **rejected[0].to_dict(),
                    "status": "rejected_reference",
                }
            )
        )
    return output
