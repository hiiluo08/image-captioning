"""
DataLoader
==========
Create DataLoaders with custom collate function for padding captions.

Since captions have different lengths, you need a custom collate_fn to:
- Pad captions to the same length within each batch
- Return batched images and padded captions
"""

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .dataset import CaptionDataset


class CaptionCollate:
    """Custom collate function for padding captions in a batch.

    Args:
        pad_idx: Index of the <pad> token in the vocabulary.
        batch_first: If True, output tensors have shape [batch, seq_len].
    """

    def __init__(self, pad_idx: int, batch_first: bool = True):
        """Initialize collate function.

        Args:
            pad_idx: Padding token index from vocabulary.
            batch_first: Whether batch dimension comes first.
        """
        # TODO: Store pad_idx and batch_first
        raise NotImplementedError("Implement __init__")

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collate a batch of (image, caption) pairs.

        Args:
            batch: List of (image_tensor, caption_tensor) tuples.

        Returns:
            Tuple of:
                - images: Batched image tensors [batch, C, H, W]
                - captions: Padded caption tensors [batch, max_seq_len]
                - lengths: Original caption lengths [batch]

        Hint: Use torch.nn.utils.rnn.pad_sequence for padding.
        """
        # TODO: 1. Separate images and captions from the batch
        # TODO: 2. Stack images into a single tensor
        # TODO: 3. Pad captions to the maximum length in the batch
        # TODO: 4. Record original caption lengths
        # TODO: 5. Return (images, padded_captions, lengths)
        raise NotImplementedError("Implement __call__")


def get_dataloader(
    dataset: CaptionDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pad_idx: int = 0,
) -> DataLoader:
    """Create a DataLoader with custom collate function.

    Args:
        dataset: CaptionDataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes for data loading.
        pad_idx: Padding token index.

    Returns:
        PyTorch DataLoader.
    """
    # TODO: Create CaptionCollate instance
    # TODO: Create and return DataLoader with the custom collate_fn
    raise NotImplementedError("Implement get_dataloader")
