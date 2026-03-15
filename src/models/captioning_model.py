"""
Image Captioning Model (CNN + Transformer)
===========================================
Wrapper that combines the CNN Encoder and Transformer Decoder.

Architecture:
    Image → CNN Encoder → Spatial Features → Transformer Decoder → Caption

This module provides a unified interface for:
- Training: forward pass with causal masking
- Inference: autoregressive caption generation
"""

import torch
import torch.nn as nn

from .encoder import EncoderCNN
from .decoder import DecoderTransformer


class ImageCaptioningModel(nn.Module):
    """Full image captioning model: CNN Encoder + Transformer Decoder.

    Args:
        vocab_size: Number of words in the vocabulary.
        d_model: Transformer model dimension (shared between encoder projection and decoder).
        nhead: Number of attention heads in the Transformer.
        num_decoder_layers: Number of Transformer decoder layers.
        dim_feedforward: Dimension of Transformer feed-forward network.
        dropout: Dropout probability.
        encoder_model: CNN backbone name (e.g., "resnet50").
        pretrained: Whether to use pretrained CNN weights.
        max_seq_len: Maximum caption length.
        pad_idx: Padding token index.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        encoder_model: str = "resnet50",
        pretrained: bool = True,
        max_seq_len: int = 128,
        pad_idx: int = 0,
    ):
        """Initialize encoder and decoder."""
        super(ImageCaptioningModel, self).__init__()
        # TODO: Create EncoderCNN(d_model, encoder_model, pretrained)
        # TODO: Create DecoderTransformer(vocab_size, d_model, nhead,
        #           num_decoder_layers, dim_feedforward, dropout, max_seq_len, pad_idx)
        raise NotImplementedError("Implement __init__")

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            images: Input images [batch, 3, H, W]
            captions: Ground truth captions [batch, seq_len]

        Returns:
            output: Predicted logits [batch, seq_len, vocab_size]

        Steps:
            1. Encode images → spatial features [batch, num_patches, d_model]
            2. Decode: Transformer(encoder_features, captions) → logits
        """
        # TODO: 1. Pass images through encoder
        # TODO: 2. Pass encoder features + captions through decoder
        raise NotImplementedError("Implement forward")

    def generate_caption(
        self,
        image: torch.Tensor,
        start_token_idx: int,
        end_token_idx: int,
        max_length: int = 50,
        beam_size: int = 1,
    ) -> torch.Tensor:
        """Generate a caption for a single image.

        Args:
            image: Single image tensor [1, 3, H, W]
            start_token_idx: Index of <start> token.
            end_token_idx: Index of <end> token.
            max_length: Maximum caption length.
            beam_size: Beam search width (1 = greedy).

        Returns:
            generated_ids: Tensor of word indices.
        """
        # TODO: 1. Encode the image → features [1, num_patches, d_model]
        # TODO: 2. Use decoder.generate() for greedy decoding
        # TODO: 3. (Optional) Implement beam search if beam_size > 1
        raise NotImplementedError("Implement generate_caption")
