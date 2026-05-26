"""
Canonical inference module for the pronunciation scoring API.

Implementation lives in infer.py; this module re-exports PronunciationScorer
so that api.py and any other callers can use the cleaner `inference` import
without duplicating code.

Key guarantees (enforced in infer.py):
  - Only "wav2vec2_pronunciation_scorer_v2" checkpoints activate the learned head.
  - Older / incompatible checkpoints fall back to wav2vec2 cosine scoring.
  - score_with_details() always returns {"final_score", "cosine_similarity", …}.
  - All outputs are clamped; no NaN or out-of-range values are propagated.
"""

from infer import PronunciationScorer, CHECKPOINT_ARCH

__all__ = ["PronunciationScorer", "CHECKPOINT_ARCH"]
