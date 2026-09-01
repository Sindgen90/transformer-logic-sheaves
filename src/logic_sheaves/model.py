from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .data import PAD_ID, VOCAB


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.0
    max_length: int = 256

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class TinyLogicTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(len(VOCAB), config.d_model, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(config.d_model, 2)

    def _apply_patch(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        if positions.shape != (hidden.shape[0],):
            raise ValueError("patch_positions must contain one position per batch item")
        if values.shape != (hidden.shape[0], hidden.shape[2]):
            raise ValueError("patch_values must have shape [batch, d_model]")
        if positions.min().item() < 0 or positions.max().item() >= hidden.shape[1]:
            raise ValueError("A patch position is outside the sequence")
        patched = hidden.clone()
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        patched[rows, positions] = values
        return patched

    def encode(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        capture_stages: bool = False,
        patch_stage: int | None = None,
        patch_positions: torch.Tensor | None = None,
        patch_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Encode tokens, optionally capturing or patching intermediate stages.

        Stage 0 is the token-plus-position embedding. Stage k is the output of
        Transformer layer k, before the encoder's final normalization. A patch
        is applied immediately after its selected stage.
        """

        if tokens.shape[1] > self.config.max_length:
            raise ValueError(
                f"Sequence length {tokens.shape[1]} exceeds max_length={self.config.max_length}"
            )
        if patch_stage is not None:
            if patch_stage < 0 or patch_stage > self.config.n_layers:
                raise ValueError(f"patch_stage must be between 0 and {self.config.n_layers}")
            if patch_positions is None or patch_values is None:
                raise ValueError("patch_positions and patch_values are required when patching")

        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        stages: list[torch.Tensor] = []
        if capture_stages:
            stages.append(hidden)
        if patch_stage == 0:
            hidden = self._apply_patch(hidden, patch_positions, patch_values)

        for stage, layer in enumerate(self.encoder.layers, start=1):
            hidden = layer(hidden, src_key_padding_mask=padding_mask)
            if capture_stages:
                stages.append(hidden)
            if patch_stage == stage:
                hidden = self._apply_patch(hidden, patch_positions, patch_values)

        if self.encoder.norm is not None:
            hidden = self.encoder.norm(hidden)
        return hidden, tuple(stages)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.encode(tokens, padding_mask)
        logits = self.classifier(hidden[:, 0])
        if return_hidden:
            return logits, hidden
        return logits

    def stage_representations(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Return embeddings and every pre-final-norm layer output."""

        _, stages = self.encode(tokens, padding_mask, capture_stages=True)
        return stages

    def forward_patched(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        patch_stage: int,
        patch_positions: torch.Tensor,
        patch_values: torch.Tensor,
    ) -> torch.Tensor:
        hidden, _ = self.encode(
            tokens,
            padding_mask,
            patch_stage=patch_stage,
            patch_positions=patch_positions,
            patch_values=patch_values,
        )
        return self.classifier(hidden[:, 0])
