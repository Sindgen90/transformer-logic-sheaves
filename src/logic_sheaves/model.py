from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

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

    def qkv_projections(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """Return the exact pre-attention Q, K, and V projections at every layer.

        Each tensor has shape ``[batch, sequence, heads, head_dimension]``. Because
        this encoder uses pre-norm layers, projections are taken from ``norm1`` of
        the residual stream entering each layer.
        """

        stages = self.stage_representations(tokens, padding_mask)
        outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for index, layer in enumerate(self.encoder.layers):
            attention_input = layer.norm1(stages[index]) if layer.norm_first else stages[index]
            if layer.self_attn.in_proj_weight is None:
                raise RuntimeError("Separate Q/K/V projection weights are not supported")
            projected = F.linear(
                attention_input,
                layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias,
            )
            heads = layer.self_attn.num_heads
            head_dimension = self.config.d_model // heads
            q, k, v = (
                item.reshape(item.shape[0], item.shape[1], heads, head_dimension)
                for item in projected.chunk(3, dim=-1)
            )
            outputs.append((q, k, v))
        return tuple(outputs)

    @staticmethod
    def _attention_from_qkv(
        layer: nn.TransformerEncoderLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Evaluate a layer's attention from explicit per-head Q/K/V tensors."""

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attention_mask = (
            None if padding_mask is None else (~padding_mask)[:, None, None, :]
        )
        mixed = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=float(layer.self_attn.dropout) if layer.training else 0.0,
        )
        mixed = mixed.transpose(1, 2).contiguous().reshape(mixed.shape[0], mixed.shape[2], -1)
        return layer.self_attn.out_proj(mixed)

    @staticmethod
    def _replace_qkv_at_positions(
        tensor: torch.Tensor,
        positions: torch.Tensor,
        values: torch.Tensor,
        head: int | None,
    ) -> torch.Tensor:
        if positions.shape != (tensor.shape[0],):
            raise ValueError("patch_positions must contain one position per batch item")
        if values.shape != (tensor.shape[0], tensor.shape[2], tensor.shape[3]):
            raise ValueError("Q/K/V patch values must have shape [batch, heads, head_dimension]")
        if positions.min().item() < 0 or positions.max().item() >= tensor.shape[1]:
            raise ValueError("A Q/K/V patch position is outside the sequence")
        if head is not None and not 0 <= head < tensor.shape[2]:
            raise ValueError(f"patch_head must be between 0 and {tensor.shape[2] - 1}")
        patched = tensor.clone()
        rows = torch.arange(tensor.shape[0], device=tensor.device)
        if head is None:
            patched[rows, positions] = values
        else:
            patched[rows, positions, head] = values[:, head]
        return patched

    def forward_qkv_patched(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor | None,
        *,
        patch_layer: int,
        patch_positions: torch.Tensor,
        patch_values: Mapping[str, torch.Tensor],
        patch_head: int | None = None,
    ) -> torch.Tensor:
        """Run a causal intervention on pre-attention Q, K, and/or V activations.

        ``patch_layer`` is one-based. Values are the donor projections at one
        position per batch item, with shape ``[batch, heads, head_dimension]``.
        Supplying a subset of ``query``, ``key``, and ``value`` changes only those
        components. ``patch_head=None`` replaces every head; otherwise only the
        selected head is changed.
        """

        if not 1 <= patch_layer <= self.config.n_layers:
            raise ValueError(f"patch_layer must be between 1 and {self.config.n_layers}")
        invalid = set(patch_values) - {"query", "key", "value"}
        if invalid or not patch_values:
            raise ValueError(f"Invalid Q/K/V patch components: {sorted(invalid)}")
        if tokens.shape[1] > self.config.max_length:
            raise ValueError(
                f"Sequence length {tokens.shape[1]} exceeds max_length={self.config.max_length}"
            )

        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        names = ("query", "key", "value")
        for layer_number, layer in enumerate(self.encoder.layers, start=1):
            if not layer.norm_first:
                raise RuntimeError("Q/K/V patching currently requires a pre-norm encoder")
            attention_input = layer.norm1(hidden)
            if layer.self_attn.in_proj_weight is None:
                raise RuntimeError("Separate Q/K/V projection weights are not supported")
            projected = F.linear(
                attention_input,
                layer.self_attn.in_proj_weight,
                layer.self_attn.in_proj_bias,
            )
            heads = layer.self_attn.num_heads
            head_dimension = self.config.d_model // heads
            components = [
                item.reshape(item.shape[0], item.shape[1], heads, head_dimension)
                for item in projected.chunk(3, dim=-1)
            ]
            if layer_number == patch_layer:
                for index, name in enumerate(names):
                    if name in patch_values:
                        components[index] = self._replace_qkv_at_positions(
                            components[index],
                            patch_positions,
                            patch_values[name],
                            patch_head,
                        )
            attention_output = self._attention_from_qkv(layer, *components, padding_mask)
            hidden = hidden + layer.dropout1(attention_output)
            hidden = hidden + layer._ff_block(layer.norm2(hidden))

        if self.encoder.norm is not None:
            hidden = self.encoder.norm(hidden)
        return self.classifier(hidden[:, 0])

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
