"""
RNN Decoder
============
Generate captions word-by-word given image features.

The decoder receives the encoded image features and generates a sequence
of words using an LSTM/GRU recurrent network.

Key concepts:
- Word embeddings (nn.Embedding)
- LSTM/GRU recurrent layers
- Teacher forcing during training
- Greedy / Beam search during inference
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class DecoderRNN(nn.Module):
    """RNN-based caption decoder.

    Args:
        embed_size: Dimension of word embeddings (must match encoder output).
        hidden_size: Number of hidden units in the LSTM/GRU.
        vocab_size: Size of the vocabulary.
        num_layers: Number of recurrent layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_size: int = 256,
        hidden_size: int = 512,
        vocab_size: int = 10000,
        num_layers: int = 1,
        dropout: float = 0.5,
    ):
        """Initialize the decoder.

        Steps:
            1. Create word embedding layer
            2. Create LSTM/GRU layer
            3. Create output linear layer (hidden_size -> vocab_size)
            4. Create dropout layer
        """
        super(DecoderRNN, self).__init__()
        # TODO: nn.Embedding(vocab_size, embed_size)
        # TODO: nn.LSTM(embed_size, hidden_size, num_layers,
        #               batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # TODO: nn.Linear(hidden_size, vocab_size)
        # TODO: nn.Dropout(dropout)
        raise NotImplementedError("Implement __init__")

    def forward(
        self,
        features: torch.Tensor,
        captions: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training with teacher forcing.

        Args:
            features: Encoded image features [batch_size, embed_size]
            captions: Ground truth captions [batch_size, max_seq_len]
            lengths: Original caption lengths [batch_size]

        Returns:
            outputs: Predicted word scores [batch_size, max_seq_len, vocab_size]

        Steps:
            1. Embed the captions (remove <end> token)
            2. Concatenate image features as the first "word"
            3. Pack padded sequences (if using lengths)
            4. Pass through LSTM
            5. Unpack and pass through output linear layer
        """
        # TODO: Implement forward pass with teacher forcing
        raise NotImplementedError("Implement forward")

    def generate(
        self,
        features: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate a caption using greedy decoding.

        Args:
            features: Encoded image features [1, embed_size]
            max_length: Maximum caption length.
            temperature: Sampling temperature.

        Returns:
            generated_ids: Tensor of generated word indices [1, seq_len]

        Steps:
            1. Start with image features as input
            2. At each step, predict next word
            3. Use predicted word as input for next step
            4. Stop when <end> token is generated or max_length reached
        """
        # TODO: Implement greedy decoding
        raise NotImplementedError("Implement generate")
