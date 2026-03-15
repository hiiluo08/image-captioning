"""
Model Evaluation
=================
Run evaluation on a dataset and compute metrics.

This module ties together:
- Loading a trained model
- Running inference on test/validation data
- Computing metrics (BLEU, etc.)
- Visualization of results
"""

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from src.models.captioning_model import ImageCaptioningModel
from src.data.vocabulary import Vocabulary
from .metrics import bleu_score


def evaluate_model(
    model: ImageCaptioningModel,
    dataloader: DataLoader,
    vocabulary: Vocabulary,
    device: torch.device,
    max_length: int = 50,
) -> Dict[str, float]:
    """Evaluate model on a dataset and compute metrics.

    Args:
        model: Trained ImageCaptioningModel.
        dataloader: DataLoader for evaluation data.
        vocabulary: Vocabulary for decoding predictions.
        device: Device to run on.
        max_length: Maximum caption length for generation.

    Returns:
        Dictionary with evaluation metrics (BLEU scores, etc.)

    Steps:
        1. Set model to eval mode
        2. For each batch, generate captions
        3. Decode predicted indices to text
        4. Collect all predictions and references
        5. Compute BLEU scores
        6. Return metrics dictionary
    """
    # TODO: Implement evaluation loop
    raise NotImplementedError("Implement evaluate_model")


def visualize_predictions(
    model: ImageCaptioningModel,
    dataloader: DataLoader,
    vocabulary: Vocabulary,
    device: torch.device,
    num_samples: int = 10,
) -> List[Tuple]:
    """Generate and visualize predictions for sample images.

    Args:
        model: Trained model.
        dataloader: DataLoader.
        vocabulary: Vocabulary.
        device: Device.
        num_samples: Number of samples to visualize.

    Returns:
        List of (image, predicted_caption, reference_captions) tuples.
    """
    # TODO: Generate captions for num_samples images
    # TODO: Return list of (image, prediction, references) for visualization
    raise NotImplementedError("Implement visualize_predictions")
