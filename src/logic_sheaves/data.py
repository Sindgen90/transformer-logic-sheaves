from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from random import Random

import torch
from torch.utils.data import Dataset

from .logic import (
    EXTENDED_OPERATOR_TOKENS,
    OPS,
    VARIABLES,
    Expr,
    random_expr,
    random_symbolic_expr,
)

PAD = "<PAD>"
CLS = "<CLS>"
VOCAB = (PAD, CLS, "0", "1", *OPS)
ENV = "<ENV>"
SEP = "<SEP>"
ASSIGNMENT_TOKENS = tuple(f"{name}={value}" for name in VARIABLES for value in (0, 1))
VOCAB = (
    *VOCAB,
    ENV,
    SEP,
    *VARIABLES,
    *ASSIGNMENT_TOKENS,
    *EXTENDED_OPERATOR_TOKENS,
)
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}
PAD_ID = TOKEN_TO_ID[PAD]
CLS_ID = TOKEN_TO_ID[CLS]


@dataclass(frozen=True)
class EncodedBatch:
    tokens: torch.Tensor
    padding_mask: torch.Tensor
    labels: torch.Tensor


@dataclass(frozen=True, slots=True)
class AssignedExpression:
    """A symbolic expression paired with an explicit Boolean environment."""

    expression: Expr
    assignment: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        names = [name for name, _ in self.assignment]
        if len(names) != len(set(names)):
            raise ValueError("Assignment contains a variable more than once")
        if any(name not in VARIABLES for name in names):
            raise ValueError(f"Assignment names must come from {VARIABLES}")
        if any(value not in (0, 1) for _, value in self.assignment):
            raise ValueError("Assignment values must be 0 or 1")

    @property
    def value(self) -> int:
        return self.expression.evaluate(dict(self.assignment))

    @property
    def depth(self) -> int:
        return self.expression.depth

    @property
    def op(self) -> str:
        return self.expression.op

    def prefix_tokens(self) -> list[str]:
        values = dict(self.assignment)
        ordered = [(name, values[name]) for name in VARIABLES if name in values]
        return [ENV, *(f"{name}={value}" for name, value in ordered), SEP, *self.expression.prefix_tokens()]

    def nodes_prefix(self):
        return self.expression.nodes_prefix()

    def __str__(self) -> str:
        environment = ",".join(f"{name}={value}" for name, value in self.assignment)
        return f"[{environment}] {self.expression}"


LogicExample = Expr | AssignedExpression


def encode(expression: LogicExample) -> list[int]:
    return [CLS_ID, *(TOKEN_TO_ID[token] for token in expression.prefix_tokens())]


def collate_expressions(expressions: Sequence[LogicExample]) -> EncodedBatch:
    encoded = [encode(expression) for expression in expressions]
    max_length = max(map(len, encoded))
    tokens = torch.full((len(encoded), max_length), PAD_ID, dtype=torch.long)
    for row, sequence in enumerate(encoded):
        tokens[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return EncodedBatch(
        tokens=tokens,
        padding_mask=tokens.eq(PAD_ID),
        labels=torch.tensor([expression.value for expression in expressions], dtype=torch.long),
    )


class ExpressionDataset(Dataset[LogicExample]):
    def __init__(self, expressions: Sequence[LogicExample]) -> None:
        self.expressions = tuple(expressions)

    def __len__(self) -> int:
        return len(self.expressions)

    def __getitem__(self, index: int) -> LogicExample:
        return self.expressions[index]


def unique_expressions(
    count: int,
    *,
    max_depth: int,
    seed: int,
    exact_depth: bool = False,
    excluded: Iterable[Expr] = (),
    balance: bool = True,
) -> list[Expr]:
    """Sample a deterministic, unique, approximately label-balanced set."""

    rng = Random(seed)
    seen = {str(expression) for expression in excluded}
    result: list[Expr] = []
    label_counts = [0, 0]
    attempts = 0
    max_attempts = max(20_000, 1_000 * count)
    while len(result) < count and attempts < max_attempts:
        attempts += 1
        expression = random_expr(rng, max_depth, exact_depth=exact_depth)
        key = str(expression)
        if key in seen:
            continue
        label = expression.value
        if balance and label_counts[label] > (count + 1) // 2:
            continue
        seen.add(key)
        result.append(expression)
        label_counts[label] += 1
    if len(result) != count:
        raise RuntimeError(
            f"Could only generate {len(result)}/{count} unique expressions at depth "
            f"{max_depth}; reduce the requested count or increase depth"
        )
    return result


def unique_symbolic_examples(
    count: int,
    *,
    max_depth: int,
    seed: int,
    variables: tuple[str, ...] = VARIABLES[:4],
    exact_depth: bool = False,
    excluded: Iterable[AssignedExpression] = (),
    balance: bool = True,
) -> list[AssignedExpression]:
    """Sample deterministic symbolic expressions with explicit assignments."""

    rng = Random(seed)
    seen = {str(example) for example in excluded}
    result: list[AssignedExpression] = []
    label_counts = [0, 0]
    attempts = 0
    max_attempts = max(30_000, 2_000 * count)
    while len(result) < count and attempts < max_attempts:
        attempts += 1
        expression = random_symbolic_expr(
            rng,
            max_depth,
            exact_depth=exact_depth,
            variables=variables,
        )
        assignment = tuple((name, rng.randrange(2)) for name in variables)
        example = AssignedExpression(expression, assignment)
        key = str(example)
        if key in seen:
            continue
        label = example.value
        if balance and label_counts[label] >= (count + 1) // 2:
            continue
        seen.add(key)
        result.append(example)
        label_counts[label] += 1
    if len(result) != count:
        raise RuntimeError(
            f"Could only generate {len(result)}/{count} unique symbolic examples at depth "
            f"{max_depth}; reduce the requested count or increase depth"
        )
    return result


@dataclass(frozen=True)
class DataSplits:
    train: tuple[Expr, ...]
    validation: tuple[Expr, ...]
    test_id: tuple[Expr, ...]
    test_ood: dict[int, tuple[Expr, ...]]


def make_splits(
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    train_depth: int,
    ood_depths: Sequence[int],
    seed: int,
) -> DataSplits:
    train = unique_expressions(train_size, max_depth=train_depth, seed=seed)
    validation = unique_expressions(
        validation_size,
        max_depth=train_depth,
        seed=seed + 1,
        excluded=train,
    )
    test_id = unique_expressions(
        test_size,
        max_depth=train_depth,
        seed=seed + 2,
        excluded=(*train, *validation),
    )
    test_ood = {
        depth: tuple(
            unique_expressions(
                test_size,
                max_depth=depth,
                exact_depth=True,
                seed=seed + 100 + depth,
            )
        )
        for depth in ood_depths
    }
    return DataSplits(
        train=tuple(train),
        validation=tuple(validation),
        test_id=tuple(test_id),
        test_ood=test_ood,
    )


@dataclass(frozen=True)
class SymbolicDataSplits:
    train: tuple[AssignedExpression, ...]
    validation: tuple[AssignedExpression, ...]
    test_id: tuple[AssignedExpression, ...]
    test_ood: dict[int, tuple[AssignedExpression, ...]]


def make_symbolic_splits(
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    train_depth: int,
    ood_depths: Sequence[int],
    seed: int,
    variables: tuple[str, ...] = VARIABLES[:4],
) -> SymbolicDataSplits:
    train = unique_symbolic_examples(
        train_size,
        max_depth=train_depth,
        seed=seed,
        variables=variables,
    )
    validation = unique_symbolic_examples(
        validation_size,
        max_depth=train_depth,
        seed=seed + 1,
        variables=variables,
        excluded=train,
    )
    test_id = unique_symbolic_examples(
        test_size,
        max_depth=train_depth,
        seed=seed + 2,
        variables=variables,
        excluded=(*train, *validation),
    )
    test_ood = {
        depth: tuple(
            unique_symbolic_examples(
                test_size,
                max_depth=depth,
                exact_depth=True,
                seed=seed + 100 + depth,
                variables=variables,
            )
        )
        for depth in ood_depths
    }
    return SymbolicDataSplits(
        train=tuple(train),
        validation=tuple(validation),
        test_id=tuple(test_id),
        test_ood=test_ood,
    )
