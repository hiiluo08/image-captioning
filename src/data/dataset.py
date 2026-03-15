"""
Caption Dataset
================
Custom PyTorch Dataset for loading image-caption pairs.

This dataset should:
- Load images from disk and apply transforms
- Load corresponding captions
- Return (image_tensor, caption_tensor) pairs
"""

import os
from typing import Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset
from PIL import Image

from .vocabulary import Vocabulary


class CaptionDataset(Dataset):
    """PyTorch Dataset for Image Captioning.

    Args:
        images_dir: Path to directory containing images.
        captions_file: Path to file with image-caption mappings.
        vocabulary: Vocabulary instance for text numericalization.
        transform: Optional image transform pipeline.
    """

    def __init__(
        self,
        images_dir: str,
        captions_file: str,
        vocabulary: Vocabulary,
        transform: Optional[Callable] = None,
    ):
        """Initialize the dataset.

        Steps:
            1. Store parameters
            2. Load and parse the captions file
            3. Create list of (image_filename, caption) pairs
        """
        # TODO: Store images_dir, vocabulary, transform
        # TODO: Load captions file and parse it
        # TODO: Create a list of (image_filename, caption_string) pairs
        raise NotImplementedError("Implement __init__")

    def __len__(self) -> int:
        """Return the total number of samples."""
        # TODO: Return dataset size
        raise NotImplementedError("Implement __len__")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image_tensor, caption_tensor):
                - image_tensor: Transformed image tensor [C, H, W]
                - caption_tensor: Numericalized caption tensor [seq_len]
        """
        # TODO: 1. Get image filename and caption for this index
        # TODO: 2. Load image with PIL and apply transforms
        # TODO: 3. Numericalize the caption using vocabulary
        # TODO: 4. Return (image_tensor, caption_tensor)
        raise NotImplementedError("Implement __getitem__")
