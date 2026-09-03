from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from random import Random

BINARY_OPS = ("AND", "OR", "XOR")
UNARY_OPS = ("NOT",)
OPS = UNARY_OPS + BINARY_OPS
VARIABLES = tuple(f"x{index}" for index in range(6))
FIXED_ARITY_OPS = {
    "MAJ3": 3,
    "ITE3": 3,
    "EXACT1_3": 3,
    "ATLEAST2_4": 4,
}
MAX_VARIADIC_ARITY = 4
EXTENDED_OPERATOR_TOKENS = (
    "AND3",
    "AND4",
    "OR3",
    "OR4",
    "XOR3",
    "XOR4",
    *FIXED_ARITY_OPS,
)


@dataclass(frozen=True, slots=True)
class Expr:
    """An immutable Boolean expression in prefix-tree form."""

    op: str
    children: tuple[Expr, ...] = ()

    def __post_init__(self) -> None:
        if self.op in {"0", "1", *VARIABLES} and self.children:
            raise ValueError("Literals and variables cannot have children")
        if self.op in UNARY_OPS and len(self.children) != 1:
            raise ValueError(f"{self.op} expects one child")
        if self.op in BINARY_OPS and not 2 <= len(self.children) <= MAX_VARIADIC_ARITY:
            raise ValueError(
                f"{self.op} expects between 2 and {MAX_VARIADIC_ARITY} children"
            )
        if self.op in FIXED_ARITY_OPS and len(self.children) != FIXED_ARITY_OPS[self.op]:
            raise ValueError(f"{self.op} expects {FIXED_ARITY_OPS[self.op]} children")
        if self.op not in {"0", "1", *VARIABLES, *OPS, *FIXED_ARITY_OPS}:
            raise ValueError(f"Unknown operator {self.op!r}")

    @property
    def is_literal(self) -> bool:
        return self.op in {"0", "1"}

    @property
    def is_variable(self) -> bool:
        return self.op in VARIABLES

    @property
    def is_leaf(self) -> bool:
        return self.is_literal or self.is_variable

    @property
    def depth(self) -> int:
        if self.is_leaf:
            return 0
        return 1 + max(child.depth for child in self.children)

    def evaluate(self, assignment: dict[str, int] | None = None) -> int:
        if self.op == "0":
            return 0
        if self.op == "1":
            return 1
        if self.is_variable:
            if assignment is None or self.op not in assignment:
                raise ValueError(f"Variable {self.op!r} has no assigned value")
            value = int(assignment[self.op])
            if value not in (0, 1):
                raise ValueError(f"Variable {self.op!r} must be assigned 0 or 1")
            return value
        values = tuple(child.evaluate(assignment) for child in self.children)
        if self.op == "NOT":
            return 1 - values[0]
        if self.op == "AND":
            return int(all(values))
        if self.op == "OR":
            return int(any(values))
        if self.op == "XOR":
            return sum(values) % 2
        if self.op == "MAJ3":
            return int(sum(values) >= 2)
        if self.op == "ITE3":
            return values[1] if values[0] else values[2]
        if self.op == "EXACT1_3":
            return int(sum(values) == 1)
        if self.op == "ATLEAST2_4":
            return int(sum(values) >= 2)
        raise AssertionError("unreachable")

    @property
    def value(self) -> int:
        return self.evaluate()

    def prefix_tokens(self) -> list[str]:
        tokens: list[str] = []

        def visit(node: Expr) -> None:
            if node.op in BINARY_OPS and len(node.children) > 2:
                tokens.append(f"{node.op}{len(node.children)}")
            else:
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
        if self.is_leaf:
            return self.op
        if self.op == "NOT":
            return f"(NOT {self.children[0]})"
        return f"({self.op} {' '.join(map(str, self.children))})"


def literal(value: int | bool) -> Expr:
    return Expr("1" if value else "0")


def variable(name: str) -> Expr:
    if name not in VARIABLES:
        raise ValueError(f"Unknown variable {name!r}; choose one of {VARIABLES}")
    return Expr(name)


def unary(op: str, child: Expr) -> Expr:
    return Expr(op, (child,))


def binary(op: str, left: Expr, right: Expr) -> Expr:
    return Expr(op, (left, right))


def nary(op: str, *children: Expr) -> Expr:
    if op not in BINARY_OPS:
        raise ValueError(f"{op!r} is not a variadic Boolean operator")
    return Expr(op, tuple(children))


def fixed(op: str, *children: Expr) -> Expr:
    if op not in FIXED_ARITY_OPS:
        raise ValueError(f"{op!r} is not a fixed-arity extended operator")
    return Expr(op, tuple(children))


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


SYMBOLIC_OPERATOR_ARITIES = (
    ("NOT", 1),
    ("AND", 2),
    ("AND", 3),
    ("AND", 4),
    ("OR", 2),
    ("OR", 3),
    ("OR", 4),
    ("XOR", 2),
    ("XOR", 3),
    ("XOR", 4),
    *FIXED_ARITY_OPS.items(),
)


def random_symbolic_expr(
    rng: Random,
    max_depth: int,
    *,
    exact_depth: bool = False,
    variables: tuple[str, ...] = VARIABLES[:4],
    stop_probability: float = 0.18,
) -> Expr:
    """Generate expressions with variables, n-ary, and fixed-arity operators."""

    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if not variables or any(name not in VARIABLES for name in variables):
        raise ValueError(f"variables must be a non-empty subset of {VARIABLES}")

    def leaf() -> Expr:
        if rng.random() < 0.75:
            return variable(rng.choice(variables))
        return literal(rng.randrange(2))

    if max_depth == 0:
        return leaf()
    if not exact_depth and rng.random() < stop_probability:
        return leaf()

    op, arity = rng.choice(SYMBOLIC_OPERATOR_ARITIES)
    deep_child = rng.randrange(arity) if exact_depth else None
    children: list[Expr] = []
    for index in range(arity):
        if index == deep_child:
            child_depth = max_depth - 1
            child_exact = True
        else:
            child_depth = rng.randrange(max_depth)
            child_exact = False
        children.append(
            random_symbolic_expr(
                rng,
                child_depth,
                exact_depth=child_exact,
                variables=variables,
                stop_probability=stop_probability,
            )
        )
    if op in FIXED_ARITY_OPS:
        return fixed(op, *children)
    if op == "NOT":
        return unary(op, children[0])
    return nary(op, *children)


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
