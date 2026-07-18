"""Define the config-driven decoder-only Helix transformer architecture."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as functional
from torch import nn

from config import TransformerConfig


class CausalSelfAttention(nn.Module):
    """Apply fused causal multi-head self-attention to a token sequence."""

    def __init__(self, config: TransformerConfig) -> None:
        """Create combined query/key/value and output projections."""

        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.qkv_projection = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.output_projection = nn.Linear(config.n_embd, config.n_embd)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return causally attended hidden states with the input shape preserved."""

        batch_size, sequence_length, embedding_dim = hidden_states.shape
        query, key, value = self.qkv_projection(hidden_states).split(self.n_embd, dim=2)

        query = query.view(
            batch_size, sequence_length, self.n_head, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, sequence_length, self.n_head, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, sequence_length, self.n_head, self.head_dim
        ).transpose(1, 2)

        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, embedding_dim
        )
        return cast(torch.Tensor, self.residual_dropout(self.output_projection(attended)))


class MLP(nn.Module):
    """Apply the GPT-2 feed-forward transformation independently per position."""

    def __init__(self, config: TransformerConfig) -> None:
        """Create the four-times-expanded GELU feed-forward network."""

        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.input_projection = nn.Linear(config.n_embd, hidden_dim)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(hidden_dim, config.n_embd)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Transform hidden states while preserving batch and sequence dimensions."""

        hidden_states = self.input_projection(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.output_projection(hidden_states)
        return cast(torch.Tensor, self.residual_dropout(hidden_states))


class TransformerBlock(nn.Module):
    """Compose pre-norm attention and MLP sublayers with residual connections."""

    def __init__(self, config: TransformerConfig) -> None:
        """Create one pre-normalized decoder-only transformer block."""

        super().__init__()
        self.attention_norm = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply attention and MLP residual updates to hidden states."""

        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class HelixTransformer(nn.Module):
    """Map token IDs to next-token logits with a decoder-only transformer."""

    def __init__(self, config: TransformerConfig) -> None:
        """Build and initialize the complete Helix language-model architecture."""

        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )
        self.final_layer_norm = nn.LayerNorm(config.n_embd)
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._initialize_weights)
        self._scale_residual_projections()
        self.output_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        """Apply GPT-2 initialization to linear, embedding, and layer-norm modules."""

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _scale_residual_projections(self) -> None:
        """Scale residual-path projection initialization according to model depth."""

        residual_std = 0.02 / math.sqrt(2 * self.config.n_layer)
        for module in self.blocks:
            if not isinstance(module, TransformerBlock):
                raise TypeError("HelixTransformer blocks must be TransformerBlock instances")
            nn.init.normal_(module.attention.output_projection.weight, mean=0.0, std=residual_std)
            nn.init.normal_(module.mlp.output_projection.weight, mean=0.0, std=residual_std)

    @property
    def parameter_count(self) -> int:
        """Return the number of unique trainable and non-trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Return logits for a rank-two LongTensor of token IDs."""

        if idx.ndim != 2:
            raise ValueError("HelixTransformer input must have shape (batch, sequence)")
        if idx.dtype != torch.long:
            raise TypeError("HelixTransformer input must be a torch.long tensor")

        _, sequence_length = idx.shape
        if sequence_length == 0:
            raise ValueError("HelixTransformer input sequence must not be empty")
        if sequence_length > self.config.block_size:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds block_size "
                f"{self.config.block_size}"
            )

        positions = torch.arange(sequence_length, device=idx.device)
        hidden_states = self.token_embedding(idx) + self.position_embedding(positions)
        hidden_states = self.embedding_dropout(hidden_states)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        return cast(torch.Tensor, self.output_head(hidden_states))
