"""
Positional Encoding
====================
Add positional information to token embeddings in the Transformer.

Since Transformers have no inherent notion of sequence order (unlike RNNs),
we must inject positional information explicitly.

Two common approaches:
1. Sinusoidal (fixed) — from "Attention Is All You Need"
2. Learned — nn.Embedding trained with the model

This file replaces the old attention.py (BahdanauAttention) because
the Transformer has built-in Multi-Head Attention — no separate
attention module needed.

Reference:
    Vaswani et al., "Attention Is All You Need" (2017), Section 3.5
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding.

    Adds fixed sinusoidal positional embeddings to the input tensor.
    The positional encoding has the same dimension (d_model) as the
    token embeddings so they can be summed.

    Args:
        d_model: Dimension of the model.
        max_seq_len: Maximum sequence length to pre-compute.
        dropout: Dropout probability applied after adding PE.
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
    ):
        """Initialize positional encoding.

        Steps:
            1. Create dropout layer
            2. Compute sinusoidal positional encoding matrix [max_seq_len, d_model]
               - Even indices (2i): sin(pos / 10000^(2i/d_model))
               - Odd indices (2i+1): cos(pos / 10000^(2i/d_model))
            3. Register as buffer (not a trainable parameter)
               - self.register_buffer('pe', pe.unsqueeze(0))
               - Shape: [1, max_seq_len, d_model]

        Hint:
            position = torch.arange(0, max_seq_len).unsqueeze(1)           # [max_seq_len, 1]
            div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        """
        super(PositionalEncoding, self).__init__()
        # TODO: Create dropout layer
        # TODO: Compute the sinusoidal PE matrix
        # TODO: Register as buffer
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model] with PE added.

        Steps:
            1. Slice PE to match input sequence length: pe[:, :seq_len, :]
            2. Add PE to input: x = x + pe
            3. Apply dropout
        """
        # TODO: Add positional encoding and apply dropout
        raise NotImplementedError("Implement forward")
