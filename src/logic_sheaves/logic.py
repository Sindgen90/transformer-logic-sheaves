from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from random import Random

BINARY_OPS = ("AND", "OR", "XOR")
UNARY_OPS = ("NOT",)
OPS = UNARY_OPS + BINARY_OPS


@dataclass(frozen=True, slots=True)
class Expr:
    """An immutable Boolean expression in prefix-tree form."""

    op: str
    children: tuple[Expr, ...] = ()

    def __post_init__(self) -> None:
        if self.op in {"0", "1"} and self.children:
            raise ValueError("Literals cannot have children")
        if self.op in UNARY_OPS and len(self.children) != 1:
            raise ValueError(f"{self.op} expects one child")
        if self.op in BINARY_OPS and len(self.children) != 2:
            raise ValueError(f"{self.op} expects two children")
        if self.op not in {"0", "1", *OPS}:
            raise ValueError(f"Unknown operator {self.op!r}")

    @property
    def is_literal(self) -> bool:
        return self.op in {"0", "1"}

    @property
    def depth(self) -> int:
        if self.is_literal:
            return 0
        return 1 + max(child.depth for child in self.children)

    @property
    def value(self) -> int:
        if self.op == "0":
            return 0
        if self.op == "1":
            return 1
        values = tuple(child.value for child in self.children)
        if self.op == "NOT":
            return 1 - values[0]
        if self.op == "AND":
            return values[0] & values[1]
        if self.op == "OR":
            return values[0] | values[1]
        if self.op == "XOR":
            return values[0] ^ values[1]
        raise AssertionError("unreachable")

    def prefix_tokens(self) -> list[str]:
        tokens: list[str] = []

        def visit(node: Expr) -> None:
            tokens.append(node.op)
            for child in node.children:
                visit(child)

        visit(self)
        return tokens

    def nodes_prefix(self) -> Iterator[Expr]:
        yield self
        for child in self.children:
            yield from child.nodes_prefix()

    def __str__(self) -> str:
        if self.is_literal:
            return self.op
        if self.op == "NOT":
            return f"(NOT {self.children[0]})"
        return f"({self.op} {self.children[0]} {self.children[1]})"


def literal(value: int | bool) -> Expr:
    return Expr("1" if value else "0")


def unary(op: str, child: Expr) -> Expr:
    return Expr(op, (child,))


def binary(op: str, left: Expr, right: Expr) -> Expr:
    return Expr(op, (left, right))


def random_expr(
    rng: Random,
    max_depth: int,
    *,
    exact_depth: bool = False,
    stop_probability: float = 0.22,
) -> Expr:
    """Generate an expression, optionally requiring that its depth is exact."""

    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if max_depth == 0:
        return literal(rng.randrange(2))

    if not exact_depth and rng.random() < stop_probability:
        return literal(rng.randrange(2))

    op = rng.choice(OPS)
    if op == "NOT":
        return unary(
            op,
            random_expr(
                rng,
                max_depth - 1,
                exact_depth=exact_depth,
                stop_probability=stop_probability,
            ),
        )

    if exact_depth:
        deep_side = rng.randrange(2)
        child_depths = [rng.randrange(max_depth), rng.randrange(max_depth)]
        child_depths[deep_side] = max_depth - 1
        children = tuple(
            random_expr(
                rng,
                child_depth,
                exact_depth=(child_depth == max_depth - 1 and index == deep_side),
                stop_probability=stop_probability,
            )
            for index, child_depth in enumerate(child_depths)
        )
    else:
        children = tuple(
            random_expr(
                rng,
                max_depth - 1,
                exact_depth=False,
                stop_probability=stop_probability,
            )
            for _ in range(2)
        )
    return binary(op, children[0], children[1])


def random_expr_with_value(
    rng: Random,
    max_depth: int,
    target_value: int,
    *,
    exact_depth: bool = False,
    max_attempts: int = 1_000,
) -> Expr:
    for _ in range(max_attempts):
        expression = random_expr(rng, max_depth, exact_depth=exact_depth)
        if expression.value == target_value:
            return expression
    raise RuntimeError(f"Could not sample value {target_value} at depth {max_depth}")


def demorgan_commutativity_cycle(left: Expr, right: Expr) -> tuple[Expr, Expr, Expr, Expr]:
    """A semantic square made from commutativity and De Morgan's law.

    v0 --commute under NOT--> v1 --De Morgan--> v2
     ^                                           |
     |------------- inverse De Morgan --- v3 <--|
                                      root commute
    """

    v0 = unary("NOT", binary("AND", left, right))
    v1 = unary("NOT", binary("AND", right, left))
    v2 = binary("OR", unary("NOT", right), unary("NOT", left))
    v3 = binary("OR", unary("NOT", left), unary("NOT", right))
    assert len({node.value for node in (v0, v1, v2, v3)}) == 1
    return v0, v1, v2, v3
