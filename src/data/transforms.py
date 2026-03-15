"""
Image Transforms
=================
Define image preprocessing pipelines for training and evaluation.

Transforms should include:
- Resize to target size (e.g., 224x224 for ResNet)
- Data augmentation for training (random crop, horizontal flip, etc.)
- Normalization with ImageNet statistics
- Convert to tensor
"""

from typing import Dict

from torchvision import transforms


def get_transforms(
    image_size: int = 224,
    mean: list = None,
    std: list = None,
    is_training: bool = True,
) -> transforms.Compose:
    """Get image transform pipeline.

    Args:
        image_size: Target image size (height = width).
        mean: Normalization mean (default: ImageNet stats).
        std: Normalization std (default: ImageNet stats).
        is_training: If True, include data augmentation.

    Returns:
        torchvision.transforms.Compose pipeline.

    Hint:
        Training transforms might include:
            - RandomResizedCrop
            - RandomHorizontalFlip
            - ColorJitter
            - ToTensor
            - Normalize

        Validation/Test transforms:
            - Resize
            - CenterCrop
            - ToTensor
            - Normalize
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]  # ImageNet
    if std is None:
        std = [0.229, 0.224, 0.225]  # ImageNet

    # TODO: Build and return the transform pipeline
    # TODO: Use different transforms for training vs evaluation
    raise NotImplementedError("Implement get_transforms")
