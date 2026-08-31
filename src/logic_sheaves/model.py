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

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.shape[1] > self.config.max_length:
            raise ValueError(
                f"Sequence length {tokens.shape[1]} exceeds max_length={self.config.max_length}"
            )
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        logits = self.classifier(hidden[:, 0])
        if return_hidden:
            return logits, hidden
        return logits
