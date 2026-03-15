"""
Evaluation Metrics
===================
Compute standard image captioning metrics.

Metrics to implement:
- BLEU (1, 2, 3, 4) — using nltk.translate.bleu_score
- METEOR (optional) — using nltk.translate.meteor_score
- CIDEr (optional) — more complex, may use external library
"""

from typing import List, Dict

import nltk


def bleu_score(
    predictions: List[str],
    references: List[List[str]],
    weights: tuple = None,
) -> Dict[str, float]:
    """Compute BLEU scores for predicted captions.

    Args:
        predictions: List of predicted caption strings.
        references: List of lists of reference caption strings.
                   Each prediction can have multiple references.
        weights: N-gram weights for BLEU score.
                Default computes BLEU-1 through BLEU-4.

    Returns:
        Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores.

    Hint:
        Use nltk.translate.bleu_score.corpus_bleu or sentence_bleu.
        Remember to tokenize predictions and references first!
    """
    # TODO: Tokenize predictions and references
    # TODO: Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4
    # TODO: Return as dictionary
    raise NotImplementedError("Implement bleu_score")


def meteor_score(
    predictions: List[str],
    references: List[List[str]],
) -> float:
    """Compute METEOR score. (Optional)

    Args:
        predictions: List of predicted caption strings.
        references: List of lists of reference caption strings.

    Returns:
        Average METEOR score.
    """
    # TODO: (Optional) Implement METEOR using nltk.translate.meteor_score
    raise NotImplementedError("Implement meteor_score")
