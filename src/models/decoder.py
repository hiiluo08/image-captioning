"""
Transformer Decoder
====================
Generate captions using a Transformer Decoder with:
- Self-attention over previously generated tokens (causal/masked)
- Cross-attention over encoder spatial features (image regions)
- Position-wise feed-forward networks

Key concepts:
- Positional Encoding (sinusoidal or learned)
- Causal mask (prevent attending to future tokens)
- Multi-Head Self-Attention + Cross-Attention
- Teacher forcing via causal mask during training
- Autoregressive generation during inference

References:
- Vaswani et al., "Attention Is All You Need" (2017)
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class DecoderTransformer(nn.Module):
    """Transformer-based caption decoder.

    Uses PyTorch's nn.TransformerDecoder with:
    - Word embeddings + positional encoding for caption tokens
    - Cross-attention to attend to encoder's spatial image features
    - Causal mask for autoregressive generation

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Dimension of the model (must match encoder output).
        nhead: Number of attention heads.
        num_layers: Number of Transformer decoder layers.
        dim_feedforward: Dimension of feed-forward network.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 128,
        pad_idx: int = 0,
    ):
        """Initialize the Transformer decoder.

        Steps:
            1. Create word embedding: nn.Embedding(vocab_size, d_model)
            2. Create positional encoding (from positional_encoding.py)
            3. Create Transformer decoder:
               - nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
               - nn.TransformerDecoder(decoder_layer, num_layers)
            4. Create output projection: nn.Linear(d_model, vocab_size)
            5. Store pad_idx for mask generation
        """
        super(DecoderTransformer, self).__init__()
        # TODO: nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # TODO: PositionalEncoding(d_model, max_seq_len, dropout)
        # TODO: nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                  dropout, batch_first=True)
        # TODO: nn.TransformerDecoder(decoder_layer, num_layers)
        # TODO: nn.Linear(d_model, vocab_size)
        # TODO: Initialize weights (optional but recommended)
        raise NotImplementedError("Implement __init__")

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal (upper-triangular) mask for self-attention.

        Prevents the decoder from attending to future tokens.

        Args:
            seq_len: Length of the target sequence.
            device: Device to create tensor on.

        Returns:
            mask: [seq_len, seq_len] with -inf above diagonal, 0 on/below.

        Hint: Use torch.triu(torch.ones(...), diagonal=1).bool()
              or nn.Transformer.generate_square_subsequent_mask(seq_len)
        """
        # TODO: Generate causal mask
        raise NotImplementedError("Implement _generate_causal_mask")

    def _generate_padding_mask(
        self, captions: torch.Tensor
    ) -> torch.Tensor:
        """Generate padding mask for the target sequence.

        Args:
            captions: Caption token indices [batch_size, seq_len]

        Returns:
            mask: [batch_size, seq_len] with True where padding.
        """
        # TODO: Return (captions == self.pad_idx)
        raise NotImplementedError("Implement _generate_padding_mask")

    def forward(
        self,
        encoder_features: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            encoder_features: Spatial features from CNN encoder
                              [batch_size, num_patches, d_model]
            captions: Ground truth caption tokens [batch_size, seq_len]
                      (should include <start> but not <end> for input)

        Returns:
            output: Predicted logits [batch_size, seq_len, vocab_size]

        Steps:
            1. Embed caption tokens → [batch, seq_len, d_model]
            2. Add positional encoding
            3. Generate causal mask (seq_len × seq_len)
            4. Generate padding mask for captions
            5. Pass through Transformer Decoder:
               - tgt = caption embeddings (with positional encoding)
               - memory = encoder_features
               - tgt_mask = causal mask
               - tgt_key_padding_mask = padding mask
            6. Project to vocabulary: [batch, seq_len, vocab_size]

        Note: Unlike RNN, no teacher_forcing_ratio needed!
              The causal mask naturally handles this.
        """
        # TODO: Implement Transformer decoder forward pass
        raise NotImplementedError("Implement forward")

    def generate(
        self,
        encoder_features: torch.Tensor,
        start_token_idx: int,
        end_token_idx: int,
        max_length: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate a caption autoregressively (greedy decoding).

        Args:
            encoder_features: Spatial features [1, num_patches, d_model]
            start_token_idx: Index of the <start> token.
            end_token_idx: Index of the <end> token.
            max_length: Maximum caption length.
            temperature: Sampling temperature (1.0 = no change).

        Returns:
            generated_ids: Tensor of word indices [1, seq_len]

        Steps:
            1. Start with [<start>] token
            2. Repeat:
               a. Embed current sequence + positional encoding
               b. Generate causal mask for current length
               c. Pass through Transformer Decoder with encoder memory
               d. Take logits of the LAST position
               e. Apply temperature and argmax (greedy) or sample
               f. Append predicted token
               g. Stop if <end> token or max_length reached
        """
        # TODO: Implement autoregressive greedy decoding
        raise NotImplementedError("Implement generate")
