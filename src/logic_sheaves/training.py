from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ExpressionDataset, collate_expressions
from .logic import Expr
from .model import ModelConfig, TinyLogicTransformer


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1_500
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    eval_every: int = 100
    seed: int = 0
    device: str = "auto"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


@torch.inference_mode()
def evaluate(
    model: TinyLogicTransformer,
    expressions: Sequence[Expr],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total_loss = 0.0
    total = 0
    loader = DataLoader(
        ExpressionDataset(expressions),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_expressions,
    )
    for batch in loader:
        tokens = batch.tokens.to(device)
        padding_mask = batch.padding_mask.to(device)
        labels = batch.labels.to(device)
        logits = model(tokens, padding_mask)
        total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        correct += logits.argmax(dim=-1).eq(labels).sum().item()
        total += labels.numel()
    return {"loss": total_loss / total, "accuracy": correct / total}


def train_model(
    train_expressions: Sequence[Expr],
    validation_expressions: Sequence[Expr],
    *,
    model_config: ModelConfig,
    train_config: TrainConfig,
    checkpoint_path: Path | None = None,
    on_evaluation: Callable[[TinyLogicTransformer, int, dict[str, float | int]], None]
    | None = None,
) -> tuple[TinyLogicTransformer, list[dict[str, float | int]]]:
    seed_everything(train_config.seed)
    device = resolve_device(train_config.device)
    model = TinyLogicTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    loader_generator = torch.Generator().manual_seed(train_config.seed)
    loader = DataLoader(
        ExpressionDataset(train_expressions),
        batch_size=min(train_config.batch_size, len(train_expressions)),
        shuffle=True,
        generator=loader_generator,
        collate_fn=collate_expressions,
        drop_last=False,
    )
    history: list[dict[str, float | int]] = []
    iterator = iter(loader)
    running_loss = 0.0
    model.train()
    for step in range(1, train_config.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        tokens = batch.tokens.to(device)
        padding_mask = batch.padding_mask.to(device)
        labels = batch.labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, padding_mask)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

        if step == 1 or step % train_config.eval_every == 0 or step == train_config.steps:
            validation = evaluate(
                model,
                validation_expressions,
                batch_size=train_config.batch_size,
                device=device,
            )
            interval = 1 if step == 1 else min(train_config.eval_every, step)
            record: dict[str, float | int] = {
                "step": step,
                "train_loss": running_loss / interval,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
            }
            history.append(record)
            running_loss = 0.0
            if on_evaluation is not None:
                on_evaluation(model, step, record)
            model.train()

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": model_config.to_dict(),
                "train_config": asdict(train_config),
            },
            checkpoint_path,
        )
        checkpoint_path.with_suffix(".history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    return model, history
