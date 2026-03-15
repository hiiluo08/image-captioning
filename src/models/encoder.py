"""
CNN Encoder
===========
Extract image features using a pretrained CNN.

The encoder takes an image and produces a fixed-size feature vector
that will be fed into the decoder for caption generation.

Architecture options:
- ResNet-50 (recommended for starting)
- ResNet-101
- EfficientNet-B0
"""

import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    """CNN-based image encoder using a pretrained backbone.

    Takes an image tensor and outputs a feature vector of size `embed_size`.

    Args:
        embed_size: Dimension of the output embedding vector.
        model_name: Name of the pretrained model to use.
        pretrained: Whether to use pretrained weights.
        fine_tune: Whether to allow fine-tuning of CNN layers.
    """

    def __init__(
        self,
        embed_size: int = 256,
        model_name: str = "resnet50",
        pretrained: bool = True,
        fine_tune: bool = False,
    ):
        """Initialize the encoder.

        Steps:
            1. Load pretrained CNN model
            2. Remove the final classification layer
            3. Add a linear layer to project features to embed_size
            4. Add batch normalization
            5. Freeze/unfreeze CNN layers based on fine_tune flag
        """
        super(EncoderCNN, self).__init__()
        # TODO: Load pretrained CNN (e.g., models.resnet50(pretrained=True))
        # TODO: Remove the last FC layer (replace with nn.Identity() or extract features)
        # TODO: Add a projection layer: nn.Linear(cnn_feature_size, embed_size)
        # TODO: Add nn.BatchNorm1d(embed_size)
        # TODO: Freeze CNN parameters if fine_tune is False
        raise NotImplementedError("Implement __init__")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features.

        Args:
            images: Input images [batch_size, 3, H, W]

        Returns:
            Feature vectors [batch_size, embed_size]
        """
        # TODO: 1. Pass images through CNN backbone
        # TODO: 2. Flatten features
        # TODO: 3. Project through linear layer
        # TODO: 4. Apply batch normalization
        raise NotImplementedError("Implement forward")

    def fine_tune(self, enable: bool = True, num_layers: int = 3) -> None:
        """Enable/disable fine-tuning of the last N layers of the CNN.

        Args:
            enable: Whether to enable fine-tuning.
            num_layers: Number of layers from the end to fine-tune.
        """
        # TODO: Freeze/unfreeze parameters of the last num_layers
        raise NotImplementedError("Implement fine_tune")
