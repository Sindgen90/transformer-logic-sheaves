from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import permutations
from random import Random

from .data import AssignedExpression
from .logic import VARIABLES, Expr, binary, fixed, nary, random_symbolic_expr, unary


@dataclass(frozen=True, slots=True)
class RewriteEdge:
    source: int
    target: int
    label: str


@dataclass(frozen=True, slots=True)
class PathPair:
    start: int
    target: int
    left: tuple[int, ...]
    right: tuple[int, ...]
    label: str


@dataclass(frozen=True, slots=True)
class EquivalenceDiagram:
    family: str
    vertices: tuple[AssignedExpression, ...]
    edges: tuple[RewriteEdge, ...]
    loops: tuple[tuple[int, ...], ...]
    path_pairs: tuple[PathPair, ...] = ()

    def validate(self) -> None:
        if not self.vertices or len({vertex.value for vertex in self.vertices}) != 1:
            raise ValueError(f"{self.family} vertices are not semantically equivalent")
        for edge in self.edges:
            if not 0 <= edge.source < len(self.vertices):
                raise ValueError(f"Invalid source vertex in {self.family}")
            if not 0 <= edge.target < len(self.vertices):
                raise ValueError(f"Invalid target vertex in {self.family}")
        for loop in self.loops:
            start, end = self._walk(loop)
            if start != end:
                raise ValueError(f"Open loop in {self.family}: {start} -> {end}")
        for pair in self.path_pairs:
            left_start, left_end = self._walk(pair.left)
            right_start, right_end = self._walk(pair.right)
            if (left_start, left_end) != (pair.start, pair.target):
                raise ValueError(f"Left path {pair.label!r} has wrong endpoints")
            if (right_start, right_end) != (pair.start, pair.target):
                raise ValueError(f"Right path {pair.label!r} has wrong endpoints")

    def _walk(self, path: tuple[int, ...]) -> tuple[int, int]:
        if not path:
            raise ValueError("Paths must contain at least one edge")
        first = self.edges[path[0]]
        current = first.source
        start = current
        for edge_index in path:
            edge = self.edges[edge_index]
            if edge.source != current:
                raise ValueError(f"Disconnected path in {self.family}")
            current = edge.target
        return start, current


def _assignment(rng: Random, variables: tuple[str, ...]) -> tuple[tuple[str, int], ...]:
    return tuple((name, rng.randrange(2)) for name in variables)


def _operands(
    rng: Random,
    count: int,
    operand_depth: int,
    variables: tuple[str, ...],
) -> tuple[Expr, ...]:
    return tuple(
        random_symbolic_expr(rng, operand_depth, variables=variables) for _ in range(count)
    )


def _diagram(
    family: str,
    expressions: tuple[Expr, ...],
    assignment: tuple[tuple[str, int], ...],
    edges: tuple[RewriteEdge, ...],
    loops: tuple[tuple[int, ...], ...],
    path_pairs: tuple[PathPair, ...] = (),
) -> EquivalenceDiagram:
    result = EquivalenceDiagram(
        family=family,
        vertices=tuple(AssignedExpression(expression, assignment) for expression in expressions),
        edges=edges,
        loops=loops,
        path_pairs=path_pairs,
    )
    result.validate()
    return result


def commutativity_involution(
    rng: Random, operand_depth: int, variables: tuple[str, ...]
) -> EquivalenceDiagram:
    a, b = _operands(rng, 2, operand_depth, variables)
    vertices = (binary("AND", a, b), binary("AND", b, a))
    edges = (
        RewriteEdge(0, 1, "commute_and"),
        RewriteEdge(1, 0, "commute_and"),
    )
    return _diagram(
        "commutativity_involution", vertices, _assignment(rng, variables), edges, ((0, 1),)
    )


def double_negation_loop(
    rng: Random, operand_depth: int, variables: tuple[str, ...]
) -> EquivalenceDiagram:
    (a,) = _operands(rng, 1, operand_depth, variables)
    vertices = (a, unary("NOT", unary("NOT", a)))
    edges = (
        RewriteEdge(0, 1, "double_negation_expand"),
        RewriteEdge(1, 0, "double_negation_reduce"),
    )
    return _diagram("double_negation", vertices, _assignment(rng, variables), edges, ((0, 1),))


def demorgan_and_square(
    rng: Random, operand_depth: int, variables: tuple[str, ...]
) -> EquivalenceDiagram:
    a, b = _operands(rng, 2, operand_depth, variables)
    vertices = (
        unary("NOT", binary("AND", a, b)),
        unary("NOT", binary("AND", b, a)),
        binary("OR", unary("NOT", b), unary("NOT", a)),
        binary("OR", unary("NOT", a), unary("NOT", b)),
    )
    edges = (
        RewriteEdge(0, 1, "commute_and_under_not"),
        RewriteEdge(1, 2, "demorgan_and"),
        RewriteEdge(2, 3, "commute_or"),
        RewriteEdge(3, 0, "demorgan_and_inverse"),
    )
    return _diagram("demorgan_and_square", vertices, _assignment(rng, variables), edges, ((0, 1, 2, 3),))


def demorgan_or_square(
    rng: Random, operand_depth: int, variables: tuple[str, ...]
) -> EquivalenceDiagram:
    a, b = _operands(rng, 2, operand_depth, variables)
    vertices = (
        unary("NOT", binary("OR", a, b)),
        unary("NOT", binary("OR", b, a)),
        binary("AND", unary("NOT", b), unary("NOT", a)),
        binary("AND", unary("NOT", a), unary("NOT", b)),
    )
    edges = (
        RewriteEdge(0, 1, "commute_or_under_not"),
        RewriteEdge(1, 2, "demorgan_or"),
        RewriteEdge(2, 3, "commute_and"),
        RewriteEdge(3, 0, "demorgan_or_inverse"),
    )
    return _diagram("demorgan_or_square", vertices, _assignment(rng, variables), edges, ((0, 1, 2, 3),))


def associativity_pentagon(
    rng: Random, operand_depth: int, variables: tuple[str, ...]
) -> EquivalenceDiagram:
    a, b, c, d = _operands(rng, 4, operand_depth, variables)
    vertices = (
        binary("AND", binary("AND", binary("AND", a, b), c), d),
        binary("AND", binary("AND", a, b), binary("AND", c, d)),
        binary("AND", a, binary("AND", b, binary("AND", c, d))),
        binary("AND", binary("AND", a, binary("AND", b, c)), d),
        binary("AND", a, binary("AND", binary("AND", b, c), d)),
    )
    edges = (
        RewriteEdge(0, 1, "associate_and_right"),
        RewriteEdge(1, 2, "associate_and_right"),
        RewriteEdge(0, 3, "associate_and_right"),
        RewriteEdge(3, 4, "associate_and_right"),
        RewriteEdge(4, 2, "associate_and_right"),
        RewriteEdge(2, 4, "associate_and_left"),
        RewriteEdge(4, 3, "associate_and_left"),
        RewriteEdge(3, 0, "associate_and_left"),
    )
    paths = (PathPair(0, 2, (0, 1), (2, 3, 4), "short_vs_long_association"),)
    return _diagram(
        "associativity_pentagon",
        vertices,
        _assignment(rng, variables),
        edges,
        ((0, 1, 5, 6, 7),),
        paths,
    )


def _permutation_hexagon(
    rng: Random,
    operand_depth: int,
    variables: tuple[str, ...],
    *,
    family: str,
    operator: str,
    label_prefix: str,
) -> EquivalenceDiagram:
    operands = _operands(rng, 3, operand_depth, variables)
    order = ((0, 1, 2), (1, 0, 2), (1, 2, 0), (2, 1, 0), (2, 0, 1), (0, 2, 1))
    if operator == "AND":
        vertices = tuple(nary("AND", *(operands[index] for index in item)) for item in order)
    else:
        vertices = tuple(fixed(operator, *(operands[index] for index in item)) for item in order)
    edges = tuple(
        RewriteEdge(index, (index + 1) % 6, f"{label_prefix}_{'12' if index % 2 == 0 else '23'}")
        for index in range(6)
    )
    return _diagram(family, vertices, _assignment(rng, variables), edges, (tuple(range(6)),))


def nary_and_hexagon(
    rng: Random, operand_depth: int, variables: tuple[str, ...]
) -> EquivalenceDiagram:
    return _permutation_hexagon(
        rng,
        operand_depth,
        variables,
        family="nary_and_hexagon",
        operator="AND",
        label_prefix="swap_and3",
    )


def exact1_hexagon(
    rng: Random, operand_depth: int, variables: tuple[str, ...]
) -> EquivalenceDiagram:
    return _permutation_hexagon(
        rng,
        operand_depth,
        variables,
        family="exact1_hexagon",
        operator="EXACT1_3",
        label_prefix="swap_exact1",
    )


def commutativity_cube(
    rng: Random, operand_depth: int, variables: tuple[str, ...]
) -> EquivalenceDiagram:
    a, b, c, d, e, f = _operands(rng, 6, operand_depth, variables)

    def vertex(mask: int) -> Expr:
        left = binary("AND", b, a) if mask & 1 else binary("AND", a, b)
        middle = binary("OR", d, c) if mask & 2 else binary("OR", c, d)
        right = binary("XOR", f, e) if mask & 4 else binary("XOR", e, f)
        return fixed("MAJ3", left, middle, right)

    vertices = tuple(vertex(mask) for mask in range(8))
    labels = ("commute_and_in_context", "commute_or_in_context", "commute_xor_in_context")
    edge_list: list[RewriteEdge] = []
    edge_lookup: dict[tuple[int, int], int] = {}
    for mask in range(8):
        for bit in range(3):
            if mask & (1 << bit):
                continue
            target = mask | (1 << bit)
            edge_lookup[(mask, target)] = len(edge_list)
            edge_list.append(RewriteEdge(mask, target, labels[bit]))
            edge_lookup[(target, mask)] = len(edge_list)
            edge_list.append(RewriteEdge(target, mask, labels[bit]))

    loops: list[tuple[int, ...]] = []
    for first in range(3):
        for second in range(first + 1, 3):
            remaining = ({0, 1, 2} - {first, second}).pop()
            for fixed_bit in (0, 1):
                start = fixed_bit << remaining
                one = start | (1 << first)
                both = one | (1 << second)
                two = start | (1 << second)
                loops.append(
                    (
                        edge_lookup[(start, one)],
                        edge_lookup[(one, both)],
                        edge_lookup[(both, two)],
                        edge_lookup[(two, start)],
                    )
                )

    forward_paths: list[tuple[int, ...]] = []
    for order in permutations(range(3)):
        current = 0
        path: list[int] = []
        for bit in order:
            target = current | (1 << bit)
            path.append(edge_lookup[(current, target)])
            current = target
        forward_paths.append(tuple(path))
    path_pairs = tuple(
        PathPair(0, 7, forward_paths[0], path, f"rewrite_order_{index}")
        for index, path in enumerate(forward_paths[1:], start=1)
    )
    return _diagram(
        "commutativity_cube",
        vertices,
        _assignment(rng, variables),
        tuple(edge_list),
        tuple(loops),
        path_pairs,
    )


def distributivity_diamond(
    rng: Random, operand_depth: int, variables: tuple[str, ...]
) -> EquivalenceDiagram:
    a, b, c = _operands(rng, 3, operand_depth, variables)
    vertices = (
        binary("AND", a, binary("OR", b, c)),
        binary("OR", binary("AND", a, b), binary("AND", a, c)),
        binary("AND", binary("OR", b, c), a),
        binary("OR", binary("AND", b, a), binary("AND", c, a)),
    )
    edges = (
        RewriteEdge(0, 1, "distribute_and_over_or_left"),
        RewriteEdge(0, 2, "commute_and"),
        RewriteEdge(2, 3, "distribute_and_over_or_right"),
        RewriteEdge(3, 1, "parallel_commute_and"),
        RewriteEdge(1, 0, "distribute_and_over_or_left_inverse"),
    )
    paths = (PathPair(0, 1, (0,), (1, 2, 3), "direct_vs_commuted_distribution"),)
    return _diagram(
        "distributivity_diamond",
        vertices,
        _assignment(rng, variables),
        edges,
        ((0, 4), (1, 2, 3, 4)),
        paths,
    )


def ite_expansion_loop(
    rng: Random, operand_depth: int, variables: tuple[str, ...]
) -> EquivalenceDiagram:
    condition, then_branch, else_branch = _operands(rng, 3, operand_depth, variables)
    compact = fixed("ITE3", condition, then_branch, else_branch)
    expanded = binary(
        "OR",
        binary("AND", condition, then_branch),
        binary("AND", unary("NOT", condition), else_branch),
    )
    edges = (
        RewriteEdge(0, 1, "ite_expand"),
        RewriteEdge(1, 0, "ite_reduce"),
    )
    return _diagram(
        "ite_expansion", (compact, expanded), _assignment(rng, variables), edges, ((0, 1),)
    )


def majority_duality_loop(
    rng: Random, operand_depth: int, variables: tuple[str, ...]
) -> EquivalenceDiagram:
    a, b, c = _operands(rng, 3, operand_depth, variables)
    compact = unary("NOT", fixed("MAJ3", a, b, c))
    dual = fixed("MAJ3", unary("NOT", a), unary("NOT", b), unary("NOT", c))
    edges = (
        RewriteEdge(0, 1, "majority_duality"),
        RewriteEdge(1, 0, "majority_duality_inverse"),
    )
    return _diagram(
        "majority_duality", (compact, dual), _assignment(rng, variables), edges, ((0, 1),)
    )


DiagramBuilder = Callable[[Random, int, tuple[str, ...]], EquivalenceDiagram]
DIAGRAM_BUILDERS: tuple[DiagramBuilder, ...] = (
    commutativity_involution,
    double_negation_loop,
    demorgan_and_square,
    demorgan_or_square,
    associativity_pentagon,
    nary_and_hexagon,
    commutativity_cube,
    distributivity_diamond,
    ite_expansion_loop,
    majority_duality_loop,
    exact1_hexagon,
)


def make_equivalence_suite(
    count_per_family: int,
    *,
    operand_depth: int,
    seed: int,
    variables: tuple[str, ...] = VARIABLES[:4],
) -> list[EquivalenceDiagram]:
    """Generate label-balanced held-out instances of every diagram family."""

    if count_per_family < 2:
        raise ValueError("count_per_family must be at least 2")
    rng = Random(seed)
    diagrams: list[EquivalenceDiagram] = []
    for builder in DIAGRAM_BUILDERS:
        family_diagrams: list[EquivalenceDiagram] = []
        seen: set[str] = set()
        attempts = 0
        while len(family_diagrams) < count_per_family and attempts < 10_000 * count_per_family:
            attempts += 1
            diagram = builder(rng, operand_depth, variables)
            target = len(family_diagrams) % 2
            key = "|".join(str(vertex) for vertex in diagram.vertices)
            if diagram.vertices[0].value != target or key in seen:
                continue
            seen.add(key)
            family_diagrams.append(diagram)
        if len(family_diagrams) != count_per_family:
            raise RuntimeError(
                f"Could only generate {len(family_diagrams)}/{count_per_family} "
                f"instances for {builder.__name__}"
            )
        diagrams.extend(family_diagrams)
    return diagrams


def make_rewrite_calibration_pairs(
    count_per_label: int,
    *,
    operand_depth: int,
    seed: int,
    variables: tuple[str, ...] = VARIABLES[:4],
) -> dict[str, list[tuple[AssignedExpression, AssignedExpression]]]:
    """Sample isolated rewrite pairs without exposing complete held-out diagrams."""

    if count_per_label < 2:
        raise ValueError("count_per_label must be at least 2")
    rng = Random(seed)
    examples = [builder(rng, operand_depth, variables) for builder in DIAGRAM_BUILDERS]
    label_sources: dict[str, tuple[DiagramBuilder, str]] = {}
    for builder, diagram in zip(DIAGRAM_BUILDERS, examples):
        for edge in diagram.edges:
            label_sources.setdefault(edge.label, (builder, edge.label))

    result: dict[str, list[tuple[AssignedExpression, AssignedExpression]]] = {}
    for label, (builder, _) in label_sources.items():
        pairs: list[tuple[AssignedExpression, AssignedExpression]] = []
        while len(pairs) < count_per_label:
            diagram = builder(rng, operand_depth, variables)
            matching = [edge for edge in diagram.edges if edge.label == label]
            edge = rng.choice(matching)
            pairs.append((diagram.vertices[edge.source], diagram.vertices[edge.target]))
        result[label] = pairs
    return result
