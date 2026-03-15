"""
Attention Mechanism (Optional / Advanced)
==========================================
Implement attention to allow the decoder to focus on different
parts of the image when generating each word.

References:
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate"
- Xu et al., "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"

Note: To use attention, the encoder needs to output spatial features
      (e.g., [batch, num_regions, feature_dim]) instead of a single vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism.

    Computes attention weights over encoder spatial features
    based on the current decoder hidden state.

    Args:
        encoder_dim: Dimension of encoder feature vectors.
        decoder_dim: Dimension of decoder hidden state.
        attention_dim: Dimension of the attention intermediate layer.
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        attention_dim: int = 256,
    ):
        """Initialize attention layers.

        Layers needed:
            - Linear layer for encoder features: encoder_dim -> attention_dim
            - Linear layer for decoder hidden:   decoder_dim -> attention_dim
            - Linear layer for computing energy:  attention_dim -> 1
        """
        super(BahdanauAttention, self).__init__()
        # TODO: Create the three linear layers described above
        # TODO: Create a ReLU activation
        raise NotImplementedError("Implement __init__")

    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_hidden: torch.Tensor,
    ) -> tuple:
        """Compute attention weights and context vector.

        Args:
            encoder_out: Encoder spatial features [batch, num_pixels, encoder_dim]
            decoder_hidden: Decoder hidden state [batch, decoder_dim]

        Returns:
            context: Weighted feature vector [batch, encoder_dim]
            alpha: Attention weights [batch, num_pixels]

        Steps:
            1. Project encoder features:  W_e * encoder_out
            2. Project decoder hidden:    W_d * decoder_hidden
            3. Compute energy:            tanh(projected_encoder + projected_decoder)
            4. Compute attention scores:  softmax(energy)
            5. Compute context vector:    weighted sum of encoder features
        """
        # TODO: Implement attention computation
        raise NotImplementedError("Implement forward")
