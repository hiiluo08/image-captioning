"""
CNN Encoder (Spatial Features for Transformer)
================================================
Extract spatial image features using a pretrained CNN.

Unlike the RNN approach which outputs a single feature vector,
the Transformer approach requires a SEQUENCE of feature vectors
(one per spatial region/patch of the image).

Output shape: [batch_size, num_patches, d_model]
    - For ResNet-50 with 224x224 input: num_patches = 7*7 = 49
    - Each patch gets projected to d_model dimensions

Architecture options:
- ResNet-50 (recommended for starting)
- ResNet-101
- EfficientNet-B0
"""

import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    """CNN-based image encoder that outputs spatial features for Transformer.

    Instead of producing a single vector, this encoder preserves spatial
    information by outputting a grid of feature vectors — one per image region.
    These become the "memory" (key/value) for the Transformer Decoder's
    cross-attention layers.

    Args:
        d_model: Dimension of model (Transformer hidden size).
        model_name: Name of the pretrained CNN backbone.
        pretrained: Whether to use pretrained weights.
        fine_tune: Whether to allow fine-tuning of CNN layers.
    """

    def __init__(
        self,
        d_model: int = 512,
        model_name: str = "resnet50",
        pretrained: bool = True,
        fine_tune: bool = False,
    ):
        """Initialize the encoder.

        Steps:
            1. Load pretrained CNN model
            2. Remove the final pooling + classification layers
               (keep everything up to the last conv block)
            3. Add a projection layer to map CNN feature dim → d_model
            4. Freeze/unfreeze CNN layers based on fine_tune flag

        Hint for ResNet-50:
            - Use nn.Sequential(*list(resnet.children())[:-2]) to remove
              AdaptiveAvgPool2d and Linear layers
            - Last conv block output: [batch, 2048, 7, 7] for 224x224 input
            - Project 2048 → d_model with nn.Conv2d(2048, d_model, kernel_size=1)
              or flatten spatial dims and use nn.Linear(2048, d_model)
        """
        super(EncoderCNN, self).__init__()
        # TODO: Load pretrained CNN
        # TODO: Remove pooling + FC layers (keep conv layers only)
        # TODO: Add projection: nn.Linear(cnn_feature_dim, d_model)
        # TODO: Freeze CNN parameters if fine_tune is False
        raise NotImplementedError("Implement __init__")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from images.

        Args:
            images: Input images [batch_size, 3, H, W]

        Returns:
            features: Spatial feature sequence [batch_size, num_patches, d_model]
                      For ResNet-50 + 224x224: [batch_size, 49, d_model]

        Steps:
            1. Pass images through CNN backbone → [batch, cnn_dim, h, w]
            2. Reshape: [batch, cnn_dim, h, w] → [batch, h*w, cnn_dim]
               (flatten spatial dims into a sequence)
            3. Project: [batch, h*w, cnn_dim] → [batch, h*w, d_model]
        """
        # TODO: Extract spatial features and reshape for Transformer
        raise NotImplementedError("Implement forward")

    def fine_tune(self, enable: bool = True, num_layers: int = 3) -> None:
        """Enable/disable fine-tuning of the last N layers of the CNN.

        Args:
            enable: Whether to enable fine-tuning.
            num_layers: Number of layers from the end to fine-tune.
        """
        # TODO: Freeze/unfreeze parameters of the last num_layers
        raise NotImplementedError("Implement fine_tune")
