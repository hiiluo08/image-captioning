"""
Image Captioning Model
=======================
Wrapper that combines the Encoder and Decoder into a single model.

This module provides a unified interface for:
- Training: forward pass with teacher forcing
- Inference: generate captions for new images
"""

import torch
import torch.nn as nn

from .encoder import EncoderCNN
from .decoder import DecoderRNN


class ImageCaptioningModel(nn.Module):
    """Full image captioning model combining encoder and decoder.

    Args:
        embed_size: Embedding dimension (shared between encoder output and decoder input).
        hidden_size: Decoder LSTM hidden size.
        vocab_size: Number of words in the vocabulary.
        num_layers: Number of decoder LSTM layers.
        encoder_model: CNN backbone name.
        pretrained: Whether to use pretrained CNN weights.
    """

    def __init__(
        self,
        embed_size: int = 256,
        hidden_size: int = 512,
        vocab_size: int = 10000,
        num_layers: int = 1,
        encoder_model: str = "resnet50",
        pretrained: bool = True,
    ):
        """Initialize encoder and decoder."""
        super(ImageCaptioningModel, self).__init__()
        # TODO: Create EncoderCNN instance
        # TODO: Create DecoderRNN instance
        raise NotImplementedError("Implement __init__")

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            images: Input images [batch, 3, H, W]
            captions: Ground truth captions [batch, max_seq_len]
            lengths: Caption lengths [batch]

        Returns:
            outputs: Predicted word scores [batch, max_seq_len, vocab_size]
        """
        # TODO: 1. Encode images -> features
        # TODO: 2. Decode features + captions -> outputs
        raise NotImplementedError("Implement forward")

    def generate_caption(
        self,
        image: torch.Tensor,
        max_length: int = 50,
        beam_size: int = 1,
    ) -> torch.Tensor:
        """Generate a caption for a single image.

        Args:
            image: Single image tensor [1, 3, H, W]
            max_length: Maximum caption length.
            beam_size: Beam search width (1 = greedy).

        Returns:
            generated_ids: Tensor of word indices.
        """
        # TODO: 1. Encode the image
        # TODO: 2. Use decoder.generate() for greedy decoding
        # TODO: 3. (Optional) Implement beam search if beam_size > 1
        raise NotImplementedError("Implement generate_caption")
