"""
Vocabulary
==========
Handles the mapping between words and numerical indices for the caption text.

Your Vocabulary class should support:
- Building vocab from a list of captions
- Converting text to tensor of indices
- Converting tensor of indices back to text
- Special tokens: <pad>, <start>, <end>, <unk>
"""

from typing import List, Dict, Optional


class Vocabulary:
    """Maps words to indices and vice versa.

    Attributes:
        word2idx (dict): Word to index mapping.
        idx2word (dict): Index to word mapping.
        freq_threshold (int): Minimum frequency for a word to be included.
    """

    # Special tokens
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<start>"
    END_TOKEN = "<end>"
    UNK_TOKEN = "<unk>"

    def __init__(self, freq_threshold: int = 5):
        """Initialize Vocabulary.

        Args:
            freq_threshold: Minimum word frequency to include in vocabulary.
        """
        # TODO: Initialize word2idx and idx2word dictionaries
        # TODO: Add special tokens (PAD, START, END, UNK) with fixed indices
        raise NotImplementedError("Implement __init__")

    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        # TODO: Return vocabulary size
        raise NotImplementedError("Implement __len__")

    def build_vocabulary(self, captions: List[str]) -> None:
        """Build vocabulary from a list of captions.

        Args:
            captions: List of caption strings.

        Steps:
            1. Tokenize each caption
            2. Count word frequencies
            3. Add words with frequency >= freq_threshold to vocab
        """
        # TODO: Implement vocabulary building
        raise NotImplementedError("Implement build_vocabulary")

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize a text string into a list of words.

        Args:
            text: Input text string.

        Returns:
            List of lowercase tokens.

        Hint: You can use a simple approach like str.lower().split()
              or use spacy/nltk for better tokenization.
        """
        # TODO: Implement tokenization
        raise NotImplementedError("Implement tokenize")

    def numericalize(self, text: str) -> List[int]:
        """Convert a text string to a list of indices.

        Args:
            text: Input text string.

        Returns:
            List of indices: [<start>, word1_idx, word2_idx, ..., <end>]
        """
        # TODO: Tokenize text and convert each token to its index
        # TODO: Wrap with START and END tokens
        # TODO: Use UNK index for unknown words
        raise NotImplementedError("Implement numericalize")

    def denumericalize(self, indices: List[int]) -> str:
        """Convert a list of indices back to a text string.

        Args:
            indices: List of word indices.

        Returns:
            Decoded caption string.
        """
        # TODO: Convert indices back to words and join them
        raise NotImplementedError("Implement denumericalize")
