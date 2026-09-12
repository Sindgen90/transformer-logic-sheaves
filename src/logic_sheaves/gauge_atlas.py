from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np


def orthogonal_polar(matrix: np.ndarray) -> np.ndarray:
    """Return the nearest orthogonal matrix in Frobenius norm."""

    left, _, right_t = np.linalg.svd(matrix, full_matrices=False)
    return left @ right_t


@dataclass(frozen=True)
class AtlasChart:
    mean: np.ndarray
    basis: np.ndarray

    def coordinates(self, samples: np.ndarray) -> np.ndarray:
        return (samples - self.mean) @ self.basis

    def reconstruct(self, coordinates: np.ndarray) -> np.ndarray:
        return self.mean + coordinates @ self.basis.T


@dataclass(frozen=True)
class AtlasEdge:
    """A fitted transport on canonical orientation ``source < target``.

    ``transport`` and ``proxy`` use column-vector convention and map source
    chart coordinates to target chart coordinates.  ``defect`` follows the
    paper's literal definition ``proxy.T @ transport`` and is treated as the
    orthogonal connection assigned to this oriented graph edge.
    """

    source: int
    target: int
    source_chart: AtlasChart
    target_chart: AtlasChart
    ridge_map: np.ndarray
    transport: np.ndarray
    proxy: np.ndarray
    defect: np.ndarray
    sigma_min: float
    shear: float
    transfer_mismatch: float
    transfer_lower_bound: float
    sigma_max: float = float("nan")
    condition_ratio: float = float("nan")
    bootstrap_stability: float = float("nan")

    def apply(self, samples: np.ndarray, *, reverse: bool = False) -> np.ndarray:
        if reverse:
            coordinates = self.target_chart.coordinates(samples)
            return self.source_chart.reconstruct(coordinates @ self.transport)
        coordinates = self.source_chart.coordinates(samples)
        return self.target_chart.reconstruct(coordinates @ self.transport.T)


@dataclass(frozen=True)
class FundamentalCycle:
    chord: tuple[int, int]
    traversals: tuple[tuple[int, int], ...]
    holonomy: np.ndarray
    paper_distance: float
    unit_distance: float
    max_abs_eigenphase: float
    mean_abs_eigenphase: float


@dataclass(frozen=True)
class TypedFundamentalCycle:
    """A type-correct comparison of learned and proxy loop transports."""

    chord: tuple[int, int]
    traversals: tuple[tuple[int, int], ...]
    transport_holonomy: np.ndarray
    proxy_holonomy: np.ndarray
    relative_holonomy: np.ndarray
    transport_paper_distance: float
    transport_unit_distance: float
    proxy_paper_distance: float
    proxy_unit_distance: float
    relative_paper_distance: float
    relative_unit_distance: float
    transport_eigenphases: tuple[float, ...]
    proxy_eigenphases: tuple[float, ...]
    relative_eigenphases: tuple[float, ...]


@dataclass(frozen=True)
class AtlasAnalysis:
    edges: dict[tuple[int, int], AtlasEdge]
    cycles: tuple[FundamentalCycle, ...]
    edge_fidelity: float
    state_return_error: float
    cycle_coverage: int


def fit_chart(samples: np.ndarray, dimension: int) -> AtlasChart:
    if samples.ndim != 2 or len(samples) < 2:
        raise ValueError("Chart samples must have shape [samples, features]")
    dimension = min(dimension, samples.shape[0] - 1, samples.shape[1])
    if dimension < 1:
        raise ValueError("Chart dimension must be positive")
    mean = samples.mean(axis=0, keepdims=True)
    _, _, right_t = np.linalg.svd(samples - mean, full_matrices=False)
    return AtlasChart(mean=mean, basis=right_t[:dimension].T)


def random_chart(samples: np.ndarray, dimension: int, rng: np.random.Generator) -> AtlasChart:
    dimension = min(dimension, samples.shape[0] - 1, samples.shape[1])
    basis, _ = np.linalg.qr(rng.normal(size=(samples.shape[1], dimension)))
    return AtlasChart(mean=samples.mean(axis=0, keepdims=True), basis=basis[:, :dimension])


def fit_edge(
    source: int,
    target: int,
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    source_chart: AtlasChart,
    target_chart: AtlasChart,
    *,
    ridge: float,
    target_order: np.ndarray | None = None,
) -> AtlasEdge:
    if source_chart.basis.shape[1] != target_chart.basis.shape[1]:
        raise ValueError("Source and target charts must have the same support dimension")
    if len(source_samples) != len(target_samples):
        raise ValueError("An overlap needs matched source and target samples")
    if target_order is not None:
        target_samples = target_samples[target_order]
    source_coordinates = source_chart.coordinates(source_samples).T
    target_coordinates = target_chart.coordinates(target_samples).T
    width = source_coordinates.shape[0]
    gram = source_coordinates @ source_coordinates.T
    ridge_map = (
        target_coordinates @ source_coordinates.T @ np.linalg.inv(gram + ridge * np.eye(width))
    )
    transport = orthogonal_polar(ridge_map)
    proxy = orthogonal_polar(target_chart.basis.T @ source_chart.basis)
    defect = proxy.T @ transport
    shear = float(np.linalg.norm(transport - proxy, ord="fro") / (2.0 * np.sqrt(width)))
    delta = transport - proxy
    covariance = source_coordinates @ source_coordinates.T / len(source_samples)
    mismatch = float(np.mean(np.sum((delta @ source_coordinates) ** 2, axis=0)))
    lower_bound = float(max(np.linalg.eigvalsh(covariance).min(), 0.0) * np.sum(delta**2))
    singular_values = np.linalg.svd(ridge_map, compute_uv=False)
    sigma_min = float(singular_values.min())
    sigma_max = float(singular_values.max())
    return AtlasEdge(
        source=source,
        target=target,
        source_chart=source_chart,
        target_chart=target_chart,
        ridge_map=ridge_map,
        transport=transport,
        proxy=proxy,
        defect=defect,
        sigma_min=sigma_min,
        shear=shear,
        transfer_mismatch=mismatch,
        transfer_lower_bound=lower_bound,
        sigma_max=sigma_max,
        condition_ratio=sigma_min / max(sigma_max, 1e-12),
    )


def bootstrap_transport_stability(
    edge: AtlasEdge,
    source_samples: np.ndarray,
    target_samples: np.ndarray,
    *,
    ridge: float,
    samples: int,
    seed: int,
) -> float:
    """Mean unit-normalized polar deviation from the full fitted transport."""

    if samples < 1:
        return float("nan")
    rng = np.random.default_rng(seed)
    deviations: list[float] = []
    width = edge.transport.shape[0]
    for _ in range(samples):
        indices = rng.integers(0, len(source_samples), size=len(source_samples))
        source_coordinates = edge.source_chart.coordinates(source_samples[indices]).T
        target_coordinates = edge.target_chart.coordinates(target_samples[indices]).T
        gram = source_coordinates @ source_coordinates.T
        ridge_map = (
            target_coordinates @ source_coordinates.T @ np.linalg.inv(gram + ridge * np.eye(width))
        )
        transport = orthogonal_polar(ridge_map)
        deviations.append(
            float(np.linalg.norm(transport - edge.transport, ord="fro") / (2 * np.sqrt(width)))
        )
    return float(np.mean(deviations))


def canonical_edges(edges: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted({tuple(sorted(edge)) for edge in edges if edge[0] != edge[1]}))


def connection(edges: Mapping[tuple[int, int], AtlasEdge], source: int, target: int) -> np.ndarray:
    key = tuple(sorted((source, target)))
    edge = edges[key]
    return edge.defect if (source, target) == key else edge.defect.T


def path_holonomy(
    edges: Mapping[tuple[int, int], AtlasEdge],
    traversals: Sequence[tuple[int, int]],
) -> np.ndarray:
    width = next(iter(edges.values())).defect.shape[0]
    product = np.eye(width)
    for source, target in traversals:
        product = connection(edges, source, target) @ product
    return product


def typed_connection(
    edges: Mapping[tuple[int, int], AtlasEdge],
    source: int,
    target: int,
    *,
    kind: str,
) -> np.ndarray:
    """Return a genuinely typed source-to-target orthogonal transport."""

    if kind not in {"transport", "proxy"}:
        raise ValueError("kind must be 'transport' or 'proxy'")
    key = tuple(sorted((source, target)))
    edge = edges[key]
    operator = edge.transport if kind == "transport" else edge.proxy
    return operator if (source, target) == key else operator.T


def typed_path_holonomy(
    edges: Mapping[tuple[int, int], AtlasEdge],
    traversals: Sequence[tuple[int, int]],
    *,
    kind: str,
) -> np.ndarray:
    width = next(iter(edges.values())).transport.shape[0]
    product = np.eye(width)
    for source, target in traversals:
        product = typed_connection(edges, source, target, kind=kind) @ product
    return product


def holonomy_distances(holonomy: np.ndarray) -> tuple[float, float]:
    """Return the paper normalization and a true [0, 1] normalization.

    The paper divides by ``sqrt(2k)`` but labels the result as [0, 1].  For a
    general orthogonal matrix its actual range is [0, sqrt(2)].  Dividing by
    ``2 sqrt(k)`` gives the separately reported unit-normalized value.
    """

    width = holonomy.shape[0]
    norm = float(np.linalg.norm(holonomy - np.eye(width), ord="fro"))
    return norm / np.sqrt(2.0 * width), norm / (2.0 * np.sqrt(width))


def spanning_tree(
    vertices: int, edges: Sequence[tuple[int, int]], root: int = 0
) -> tuple[dict[int, int], tuple[tuple[int, int], ...]]:
    adjacency = {vertex: [] for vertex in range(vertices)}
    for source, target in canonical_edges(edges):
        adjacency[source].append(target)
        adjacency[target].append(source)
    parent = {root: root}
    queue = deque([root])
    tree: list[tuple[int, int]] = []
    while queue:
        source = queue.popleft()
        for target in sorted(adjacency[source]):
            if target in parent:
                continue
            parent[target] = source
            tree.append(tuple(sorted((source, target))))
            queue.append(target)
    if len(parent) != vertices:
        raise ValueError("Atlas graph is disconnected")
    return parent, tuple(tree)


def _tree_path(parent: Mapping[int, int], source: int, target: int) -> list[int]:
    source_ancestors: list[int] = []
    current = source
    while True:
        source_ancestors.append(current)
        if parent[current] == current:
            break
        current = parent[current]
    target_ancestors: list[int] = []
    current = target
    while True:
        target_ancestors.append(current)
        if parent[current] == current:
            break
        current = parent[current]
    target_set = set(target_ancestors)
    common = next(vertex for vertex in source_ancestors if vertex in target_set)
    left = source_ancestors[: source_ancestors.index(common) + 1]
    right = target_ancestors[: target_ancestors.index(common)]
    return [*left, *reversed(right)]


def fundamental_cycles(
    vertices: int,
    edges: Mapping[tuple[int, int], AtlasEdge],
    *,
    root: int = 0,
) -> tuple[FundamentalCycle, ...]:
    parent, tree = spanning_tree(vertices, tuple(edges), root)
    tree_set = set(tree)
    cycles: list[FundamentalCycle] = []
    for chord in sorted(set(edges) - tree_set):
        source, target = chord
        path = _tree_path(parent, source, target)
        traversals = [*pairwise(path), (target, source)]
        product = path_holonomy(edges, traversals)
        paper, unit = holonomy_distances(product)
        phases = np.abs(np.angle(np.linalg.eigvals(product)))
        cycles.append(
            FundamentalCycle(
                chord=chord,
                traversals=tuple(traversals),
                holonomy=product,
                paper_distance=paper,
                unit_distance=unit,
                max_abs_eigenphase=float(phases.max(initial=0.0)),
                mean_abs_eigenphase=float(phases.mean()) if len(phases) else 0.0,
            )
        )
    return tuple(cycles)


def typed_fundamental_cycles(
    vertices: int,
    edges: Mapping[tuple[int, int], AtlasEdge],
    *,
    root: int = 0,
) -> tuple[TypedFundamentalCycle, ...]:
    """Compute Q, P, and relative loop holonomies in one base fiber."""

    parent, tree = spanning_tree(vertices, tuple(edges), root)
    tree_set = set(tree)
    cycles: list[TypedFundamentalCycle] = []
    for chord in sorted(set(edges) - tree_set):
        source, target = chord
        path = _tree_path(parent, source, target)
        traversals = (*pairwise(path), (target, source))
        transport = typed_path_holonomy(edges, traversals, kind="transport")
        proxy = typed_path_holonomy(edges, traversals, kind="proxy")
        relative = proxy.T @ transport
        transport_paper, transport_unit = holonomy_distances(transport)
        proxy_paper, proxy_unit = holonomy_distances(proxy)
        relative_paper, relative_unit = holonomy_distances(relative)
        cycles.append(
            TypedFundamentalCycle(
                chord=chord,
                traversals=traversals,
                transport_holonomy=transport,
                proxy_holonomy=proxy,
                relative_holonomy=relative,
                transport_paper_distance=transport_paper,
                transport_unit_distance=transport_unit,
                proxy_paper_distance=proxy_paper,
                proxy_unit_distance=proxy_unit,
                relative_paper_distance=relative_paper,
                relative_unit_distance=relative_unit,
                transport_eigenphases=tuple(np.angle(np.linalg.eigvals(transport)).tolist()),
                proxy_eigenphases=tuple(np.angle(np.linalg.eigvals(proxy)).tolist()),
                relative_eigenphases=tuple(np.angle(np.linalg.eigvals(relative)).tolist()),
            )
        )
    return tuple(cycles)


def component_cycles(
    edges: Mapping[tuple[int, int], AtlasEdge],
) -> tuple[FundamentalCycle, ...]:
    """Compute cycles after compactly relabeling one connected component."""

    vertices = sorted({vertex for key in edges for vertex in key})
    remap = {vertex: index for index, vertex in enumerate(vertices)}
    remapped = {
        (remap[source], remap[target]): replace(edge, source=remap[source], target=remap[target])
        for (source, target), edge in edges.items()
    }
    return fundamental_cycles(len(vertices), remapped) if remapped else ()


def typed_component_cycles(
    edges: Mapping[tuple[int, int], AtlasEdge],
) -> tuple[TypedFundamentalCycle, ...]:
    vertices = sorted({vertex for key in edges for vertex in key})
    remap = {vertex: index for index, vertex in enumerate(vertices)}
    remapped = {
        (remap[source], remap[target]): replace(edge, source=remap[source], target=remap[target])
        for (source, target), edge in edges.items()
    }
    return typed_fundamental_cycles(len(vertices), remapped) if remapped else ()


def reparameterize_atlas(
    edges: Mapping[tuple[int, int], AtlasEdge],
    gauges: Mapping[int, np.ndarray],
) -> dict[tuple[int, int], AtlasEdge]:
    """Change every chart basis and recompute the correctly typed edge objects."""

    output: dict[tuple[int, int], AtlasEdge] = {}
    for key, edge in edges.items():
        source_gauge = gauges[edge.source]
        target_gauge = gauges[edge.target]
        source_chart = replace(
            edge.source_chart,
            basis=edge.source_chart.basis @ source_gauge.T,
        )
        target_chart = replace(
            edge.target_chart,
            basis=edge.target_chart.basis @ target_gauge.T,
        )
        ridge_map = target_gauge @ edge.ridge_map @ source_gauge.T
        transport = target_gauge @ edge.transport @ source_gauge.T
        proxy = orthogonal_polar(target_chart.basis.T @ source_chart.basis)
        defect = proxy.T @ transport
        output[key] = replace(
            edge,
            source_chart=source_chart,
            target_chart=target_chart,
            ridge_map=ridge_map,
            transport=transport,
            proxy=proxy,
            defect=defect,
        )
    return output


def gauge_transform(
    edges: Mapping[tuple[int, int], AtlasEdge], gauges: Mapping[int, np.ndarray]
) -> dict[tuple[int, int], AtlasEdge]:
    """Reparameterize all charts and typed maps under vertexwise orthogonal gauges."""

    return reparameterize_atlas(edges, gauges)


def replace_defects(
    edges: Mapping[tuple[int, int], AtlasEdge],
    defects: Sequence[np.ndarray],
) -> dict[tuple[int, int], AtlasEdge]:
    return {
        key: replace(edge, defect=defect)
        for (key, edge), defect in zip(sorted(edges.items()), defects, strict=True)
    }


def rank_matched_null(
    edges: Mapping[tuple[int, int], AtlasEdge], rng: np.random.Generator
) -> dict[tuple[int, int], AtlasEdge]:
    """Randomly conjugate each defect, preserving spectrum and rank(g-I)."""

    defects: list[np.ndarray] = []
    for _, edge in sorted(edges.items()):
        width = edge.defect.shape[0]
        rotation, _ = np.linalg.qr(rng.normal(size=(width, width)))
        defects.append(rotation @ edge.defect @ rotation.T)
    return replace_defects(edges, defects)


def proxy_connection_control(
    edges: Mapping[tuple[int, int], AtlasEdge],
) -> dict[tuple[int, int], AtlasEdge]:
    """Set learned transport Q equal to the chart proxy P."""

    return {
        key: replace(edge, transport=edge.proxy, defect=np.eye(edge.proxy.shape[0]))
        for key, edge in edges.items()
    }


def spectrum_matched_transport_null(
    edges: Mapping[tuple[int, int], AtlasEdge],
    rng: np.random.Generator,
) -> dict[tuple[int, int], AtlasEdge]:
    """Randomize edge-relative orientations while preserving every defect spectrum."""

    output: dict[tuple[int, int], AtlasEdge] = {}
    for key, edge in edges.items():
        width = edge.defect.shape[0]
        rotation, _ = np.linalg.qr(rng.normal(size=(width, width)))
        source_defect = rotation @ edge.defect @ rotation.T
        transport = edge.proxy @ source_defect
        output[key] = replace(edge, transport=transport, defect=source_defect)
    return output


def topology_shuffled_transport_null(
    vertices: int,
    edges: Mapping[tuple[int, int], AtlasEdge],
) -> dict[tuple[int, int], AtlasEdge]:
    """Shuffle relative defects after comparing them in a proxy-trivialized root frame."""

    parent, _ = spanning_tree(vertices, tuple(edges))
    width = next(iter(edges.values())).proxy.shape[0]
    to_root = {0: np.eye(width)}
    remaining = set(parent) - {0}
    while remaining:
        for vertex in sorted(remaining):
            ancestor = parent[vertex]
            if ancestor not in to_root:
                continue
            vertex_to_parent = typed_connection(edges, vertex, ancestor, kind="proxy")
            to_root[vertex] = to_root[ancestor] @ vertex_to_parent
            remaining.remove(vertex)
    ordered = sorted(edges.items())
    root_defects = [
        to_root[edge.source] @ edge.defect @ to_root[edge.source].T for _, edge in ordered
    ]
    shuffled = root_defects[1:] + root_defects[:1]
    output: dict[tuple[int, int], AtlasEdge] = {}
    for (key, edge), root_defect in zip(ordered, shuffled, strict=True):
        source_defect = to_root[edge.source].T @ root_defect @ to_root[edge.source]
        output[key] = replace(
            edge,
            transport=edge.proxy @ source_defect,
            defect=source_defect,
        )
    return output


def compose_state_path(
    edges: Mapping[tuple[int, int], AtlasEdge],
    samples: np.ndarray,
    traversals: Sequence[tuple[int, int]],
) -> np.ndarray:
    state = samples
    for source, target in traversals:
        edge = edges[tuple(sorted((source, target)))]
        state = edge.apply(state, reverse=(source > target))
    return state


def largest_connected_component(
    edges: Mapping[tuple[int, int], AtlasEdge],
) -> dict[tuple[int, int], AtlasEdge]:
    if not edges:
        return {}
    adjacency = {vertex: set() for edge in edges for vertex in edge}
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    components: list[set[int]] = []
    unseen = set(adjacency)
    while unseen:
        seed = min(unseen)
        reachable = {seed}
        frontier = [seed]
        while frontier:
            source = frontier.pop()
            for target in adjacency[source] - reachable:
                reachable.add(target)
                frontier.append(target)
        unseen -= reachable
        components.append(reachable)
    reachable = max(components, key=lambda component: (len(component), -min(component)))
    return {key: edge for key, edge in edges.items() if key[0] in reachable and key[1] in reachable}


def stable_subgraph(
    edges: Mapping[tuple[int, int], AtlasEdge],
    *,
    min_sigma: float,
    min_condition_ratio: float,
    max_bootstrap_stability: float,
) -> dict[tuple[int, int], AtlasEdge]:
    retained = {
        key: edge
        for key, edge in edges.items()
        if edge.sigma_min >= min_sigma
        and edge.condition_ratio >= min_condition_ratio
        and (
            not np.isfinite(max_bootstrap_stability)
            or (
                np.isfinite(edge.bootstrap_stability)
                and edge.bootstrap_stability <= max_bootstrap_stability
            )
        )
    }
    return largest_connected_component(retained)


def fit_atlas_edges(
    samples: np.ndarray,
    edge_pairs: Sequence[tuple[int, int]],
    fit_indices: np.ndarray,
    *,
    dimension: int,
    ridge: float,
    target_orders: Mapping[tuple[int, int], np.ndarray] | None = None,
    random_basis_seed: int | None = None,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[tuple[int, int], AtlasEdge]:
    if samples.ndim != 3:
        raise ValueError("samples must have shape [instances, vertices, features]")
    rng = np.random.default_rng(random_basis_seed)
    charts: dict[int, AtlasChart] = {}
    for vertex in range(samples.shape[1]):
        chart_samples = samples[fit_indices, vertex]
        charts[vertex] = (
            fit_chart(chart_samples, dimension)
            if random_basis_seed is None
            else random_chart(chart_samples, dimension, rng)
        )
    fitted: dict[tuple[int, int], AtlasEdge] = {}
    for edge_index, (source, target) in enumerate(canonical_edges(edge_pairs)):
        key = (source, target)
        source_samples = samples[fit_indices, source]
        target_samples = samples[fit_indices, target]
        target_order = None if target_orders is None else target_orders.get(key)
        edge = fit_edge(
            source,
            target,
            source_samples,
            target_samples,
            charts[source],
            charts[target],
            ridge=ridge,
            target_order=target_order,
        )
        bootstrap_target = target_samples if target_order is None else target_samples[target_order]
        stability = bootstrap_transport_stability(
            edge,
            source_samples,
            bootstrap_target,
            ridge=ridge,
            samples=bootstrap_samples,
            seed=bootstrap_seed + edge_index,
        )
        fitted[key] = replace(edge, bootstrap_stability=stability)
    return fitted


def heldout_edge_fidelity(
    edges: Mapping[tuple[int, int], AtlasEdge],
    samples: np.ndarray,
    evaluation_indices: np.ndarray,
) -> float:
    errors: list[float] = []
    for (source, target), edge in edges.items():
        predicted = edge.apply(samples[evaluation_indices, source])
        scale = np.concatenate(
            (samples[evaluation_indices, source], samples[evaluation_indices, target]), axis=0
        )
        errors.append(_normalized_mse(predicted, samples[evaluation_indices, target], scale))
    return float(np.mean(errors)) if errors else float("nan")


def heldout_edge_fidelities(
    edges: Mapping[tuple[int, int], AtlasEdge],
    samples: np.ndarray,
    evaluation_indices: np.ndarray,
) -> dict[tuple[int, int], float]:
    output: dict[tuple[int, int], float] = {}
    for (source, target), edge in edges.items():
        predicted = edge.apply(samples[evaluation_indices, source])
        scale = np.concatenate(
            (samples[evaluation_indices, source], samples[evaluation_indices, target]), axis=0
        )
        output[(source, target)] = _normalized_mse(
            predicted, samples[evaluation_indices, target], scale
        )
    return output


def heldout_state_return_error(
    edges: Mapping[tuple[int, int], AtlasEdge],
    samples: np.ndarray,
    evaluation_indices: np.ndarray,
) -> float:
    if not edges:
        return float("nan")
    vertices = sorted({vertex for key in edges for vertex in key})
    inverse_remap = dict(enumerate(vertices))
    errors: list[float] = []
    for cycle in typed_component_cycles(edges):
        traversals = tuple(
            (inverse_remap[source], inverse_remap[target]) for source, target in cycle.traversals
        )
        start = traversals[0][0]
        initial = samples[evaluation_indices, start]
        returned = compose_state_path(edges, initial, traversals)
        errors.append(_normalized_mse(returned, initial, initial))
    return float(np.mean(errors)) if errors else float("nan")


def _normalized_mse(predicted: np.ndarray, target: np.ndarray, scale: np.ndarray) -> float:
    variance = max(float(np.mean((scale - scale.mean(axis=0, keepdims=True)) ** 2)), 1e-12)
    return float(np.mean((predicted - target) ** 2) / variance)


def fit_atlas(
    samples: np.ndarray,
    edge_pairs: Sequence[tuple[int, int]],
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    dimension: int,
    ridge: float,
    persistence: float = 0.0,
    target_orders: Mapping[tuple[int, int], np.ndarray] | None = None,
    random_basis_seed: int | None = None,
) -> AtlasAnalysis:
    """Fit and score a paper-style atlas on matched diagram instances.

    ``samples`` is ``[instances, vertices, ambient_features]``.  Atlas charts
    and transports see only ``fit_indices``; edge fidelity and state-return
    error use only ``evaluation_indices``.
    """

    if samples.ndim != 3:
        raise ValueError("samples must have shape [instances, vertices, features]")
    if set(fit_indices) & set(evaluation_indices):
        raise ValueError("Atlas fit and evaluation indices must be disjoint")
    rng = np.random.default_rng(random_basis_seed)
    charts: dict[int, AtlasChart] = {}
    for vertex in range(samples.shape[1]):
        chart_samples = samples[fit_indices, vertex]
        charts[vertex] = (
            fit_chart(chart_samples, dimension)
            if random_basis_seed is None
            else random_chart(chart_samples, dimension, rng)
        )
    fitted: dict[tuple[int, int], AtlasEdge] = {}
    for source, target in canonical_edges(edge_pairs):
        key = (source, target)
        edge = fit_edge(
            source,
            target,
            samples[fit_indices, source],
            samples[fit_indices, target],
            charts[source],
            charts[target],
            ridge=ridge,
            target_order=None if target_orders is None else target_orders.get(key),
        )
        if edge.sigma_min >= persistence:
            fitted[key] = edge
    if not fitted:
        return AtlasAnalysis({}, (), float("nan"), float("nan"), 0)

    # The persistence filter can disconnect the graph. Analyze its largest
    # connected component, as in the paper.
    adjacency = {vertex: set() for edge in fitted for vertex in edge}
    for source, target in fitted:
        adjacency[source].add(target)
        adjacency[target].add(source)
    components: list[set[int]] = []
    unseen = set(adjacency)
    while unseen:
        seed = min(unseen)
        reachable = {seed}
        frontier = [seed]
        while frontier:
            source = frontier.pop()
            for target in adjacency[source] - reachable:
                reachable.add(target)
                frontier.append(target)
        unseen -= reachable
        components.append(reachable)
    reachable = max(components, key=lambda component: (len(component), -min(component)))
    component_edges = {
        key: edge for key, edge in fitted.items() if key[0] in reachable and key[1] in reachable
    }
    remap = {vertex: index for index, vertex in enumerate(sorted(reachable))}
    cycles = component_cycles(component_edges)

    edge_errors: list[float] = []
    for (source, target), edge in component_edges.items():
        predicted = edge.apply(samples[evaluation_indices, source])
        scale = np.concatenate(
            (samples[evaluation_indices, source], samples[evaluation_indices, target]), axis=0
        )
        edge_errors.append(_normalized_mse(predicted, samples[evaluation_indices, target], scale))

    inverse_remap = {value: key for key, value in remap.items()}
    returns: list[float] = []
    for cycle in cycles:
        original_traversals = tuple(
            (inverse_remap[source], inverse_remap[target]) for source, target in cycle.traversals
        )
        start = original_traversals[0][0]
        initial = samples[evaluation_indices, start]
        returned = compose_state_path(component_edges, initial, original_traversals)
        returns.append(_normalized_mse(returned, initial, initial))
    return AtlasAnalysis(
        edges=component_edges,
        cycles=cycles,
        edge_fidelity=float(np.mean(edge_errors)) if edge_errors else float("nan"),
        state_return_error=float(np.mean(returns)) if returns else float("nan"),
        cycle_coverage=len(cycles),
    )
