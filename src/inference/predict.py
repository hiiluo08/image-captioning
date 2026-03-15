"""
Inference / Prediction
=======================
Generate captions for new images using a trained CNN + Transformer model.

Features:
- Load trained model from checkpoint
- Preprocess input image
- Generate caption using greedy or beam search decoding
- Command-line interface for quick testing
"""

import argparse
from typing import Optional

import torch
from PIL import Image

from src.models.captioning_model import ImageCaptioningModel
from src.data.vocabulary import Vocabulary
from src.data.transforms import get_transforms


def load_model(
    checkpoint_path: str,
    vocab_size: int,
    d_model: int = 512,
    nhead: int = 8,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    device: torch.device = None,
) -> ImageCaptioningModel:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        vocab_size: Vocabulary size.
        d_model: Transformer model dimension.
        nhead: Number of attention heads.
        num_decoder_layers: Number of decoder layers.
        dim_feedforward: Feed-forward network dimension.
        device: Device to load model on.

    Returns:
        Loaded ImageCaptioningModel in eval mode.
    """
    # TODO: Create model with Transformer params, load state_dict, set to eval mode
    raise NotImplementedError("Implement load_model")


def generate_caption(
    image_path: str,
    model: ImageCaptioningModel,
    vocabulary: Vocabulary,
    transform=None,
    device: torch.device = None,
    max_length: int = 50,
    beam_size: int = 1,
) -> str:
    """Generate a caption for a single image.

    Args:
        image_path: Path to the input image.
        model: Trained model.
        vocabulary: Vocabulary for decoding.
        transform: Image transform pipeline.
        device: Device.
        max_length: Maximum caption length.
        beam_size: Beam search width (1 = greedy decoding).

    Returns:
        Generated caption string.

    Steps:
        1. Load and preprocess the image
        2. Pass through encoder → spatial features [1, num_patches, d_model]
        3. Generate caption indices with Transformer decoder (autoregressive)
        4. Convert indices to words using vocabulary
        5. Return the caption string
    """
    # TODO: Implement caption generation pipeline
    raise NotImplementedError("Implement generate_caption")


def beam_search(
    model: ImageCaptioningModel,
    features: torch.Tensor,
    vocabulary: Vocabulary,
    beam_size: int = 5,
    max_length: int = 50,
) -> str:
    """Generate caption using beam search decoding.

    Args:
        model: Trained model.
        features: Encoder spatial features [1, num_patches, d_model].
        vocabulary: Vocabulary.
        beam_size: Number of beams.
        max_length: Maximum caption length.

    Returns:
        Best caption string found by beam search.

    Note: Beam search with Transformer — at each step, expand each
          beam with top-k predictions, re-score, and keep top beam_size
          candidates. The encoder features are computed once and reused.
    """
    # TODO: Implement beam search for Transformer
    # This is more complex than greedy - consider implementing after
    # greedy decoding works correctly
    raise NotImplementedError("Implement beam_search")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image caption")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam search width")
    parser.add_argument("--max_length", type=int, default=50, help="Max caption length")

    args = parser.parse_args()

    # TODO: Load vocabulary, model, and generate caption
    # TODO: Print the generated caption
    print("TODO: Implement CLI inference")
