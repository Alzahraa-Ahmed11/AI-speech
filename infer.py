"""
infer.py — Pronunciation Scorer inference wrapper.

Checkpoint policy:
  Only checkpoints tagged with CHECKPOINT_ARCH = "wav2vec2_pronunciation_scorer_v2"
  activate the learned scoring head. Any other arch tag (or a missing tag) causes a
  graceful fallback to wav2vec2 cosine-similarity scoring.

Score pipeline:
  raw logit → temperature + cosine-margin penalty → sigmoid → [0, 1] probability
  → × 100 → clamped calibrated score → × confidence (cosine-derived) → final score

Public API (unchanged):
  score()             → float [0, 100]
  score_with_details()→ dict  {final_score, label, cosine_similarity, …}
  embed()             → np.ndarray [embed_dim]
"""

import logging
import os
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from dataset import load_audio, SAMPLE_RATE, MAX_AUDIO_LEN
from model import PronunciationScoringModel

logger = logging.getLogger(__name__)

CHECKPOINT_ARCH = "wav2vec2_pronunciation_scorer_v2"


def _score_to_label(score: float) -> str:
    """
    Map a 0-100 score to a pronunciation quality label.

    NOTE on band tightening: the production API in api.py applies a
    similarity-aware label policy (cosine cap + bounded disagreement
    penalty). This CLI helper only sees the score, so it uses the same
    tightened band cutoffs to stay consistent — but it cannot apply the
    similarity safeguards. For accurate live scoring use the /score
    endpoint, which does both.
    """
    def _score_to_label(self, score: float) -> str:
        return "Good" if score >= 50 else "Try Again"


class PronunciationScorer:
    """Load a trained model and score child pronunciation against a reference."""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pt",
        embed_dim: int = 256,
        device: str = None,
        debug: bool = False,
    ):
        self.debug = debug
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device is None else torch.device(device)

        self.model = PronunciationScoringModel(
            embed_dim=embed_dim,
            freeze_feature_extractor=True,
        )
        # After loading a checkpoint the transformer may be unfrozen.
        # Unfreeze here unconditionally so state_dict keys always match.
        self.model.unfreeze_encoder()

        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            checkpoint_state = ckpt["model_state_dict"]
            model_state = self.model.state_dict()
            compatible_state = {
                key: value
                for key, value in checkpoint_state.items()
                if key in model_state and value.shape == model_state[key].shape
            }
            skipped_keys = sorted(set(checkpoint_state) - set(compatible_state))

            arch = ckpt.get("arch", "")
            arch_ok = (not arch) or (arch == CHECKPOINT_ARCH)
            if arch and not arch_ok:
                logger.warning(
                    "Checkpoint arch '%s' differs from expected '%s'; "
                    "scoring-head weights will be skipped (cosine fallback).",
                    arch, CHECKPOINT_ARCH,
                )

            scorer_keys = [key for key in model_state if key.startswith("scorer.net.")]
            has_scoring_head = arch_ok and all(key in compatible_state for key in scorer_keys)
            self.model.use_scoring_head = has_scoring_head
            result = self.model.load_state_dict(compatible_state, strict=False)
            if skipped_keys:
                logger.warning(
                    "Skipped checkpoint keys incompatible with wav2vec2 model: %s",
                    skipped_keys,
                )
            if not has_scoring_head:
                logger.warning(
                    "No compatible scoring-head weights found; using wav2vec2 cosine fallback."
                )
            if result.unexpected_keys:
                logger.warning(
                    "Checkpoint keys not present in model (ignored): %s",
                    result.unexpected_keys,
                )
            if result.missing_keys:
                logger.warning(
                    "Model keys not found in checkpoint (random init): %s",
                    result.missing_keys,
                )
            meta = ckpt.get("metrics", {})
            logger.info(
                "Loaded checkpoint  path=%s  epoch=%s  val_auc=%s",
                checkpoint_path,
                ckpt.get("epoch", "?"),
                meta.get("val_auc", meta.get("loss", "?")),
            )
        else:
            logger.warning(
                "No checkpoint at %s — model uses random weights.", checkpoint_path
            )

            self.model.use_scoring_head = False

        self.model.to(self.device)
        self.model.eval()

    # ── Preprocessing ──────────────────────────────────────────────────────────

    def _preprocess(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        source_sr: int = SAMPLE_RATE,
    ) -> torch.Tensor:
        if isinstance(audio, str):
            waveform = load_audio(audio)

        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio.copy()).float()
            if waveform.dim() == 2:
                waveform = waveform.mean(dim=0)
            waveform = self._resample_if_needed(waveform, source_sr)
            waveform = waveform[:MAX_AUDIO_LEN]

        elif isinstance(audio, torch.Tensor):
            waveform = audio.clone().float()
            if waveform.dim() == 2:
                waveform = waveform.mean(dim=0)
            waveform = self._resample_if_needed(waveform, source_sr)
            waveform = waveform[:MAX_AUDIO_LEN]

        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        peak = waveform.abs().max()
        if peak > 1e-8:
            waveform = waveform / peak

        if self.debug:
            logger.debug(
                "_preprocess: shape=%s  peak=%.4f  duration=%.2fs",
                tuple(waveform.shape),
                waveform.abs().max().item(),
                waveform.shape[0] / SAMPLE_RATE,
            )

        return waveform  # [samples], float32, ∈ [-1, 1], at SAMPLE_RATE

    def _resample_if_needed(
        self, waveform: torch.Tensor, source_sr: int
    ) -> torch.Tensor:
        if source_sr != SAMPLE_RATE:
            waveform = T.Resample(orig_freq=source_sr, new_freq=SAMPLE_RATE)(
                waveform.unsqueeze(0)
            ).squeeze(0)
        return waveform

    # ── Internal encode helper ─────────────────────────────────────────────────

    def _encode(self, waveform: torch.Tensor) -> torch.Tensor:
        x   = waveform.unsqueeze(0).to(self.device)
        emb = self.model.encode(x)  # [1, embed_dim], L2-normalised

        if self.debug:
            logger.debug(
                "_encode: norm=%.4f  mean=%.4f  std=%.4f",
                emb.norm().item(),
                emb.mean().item(),
                emb.std().item(),
            )
        return emb  # [1, embed_dim]

    # ── Public API ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score(
        self,
        reference: Union[str, np.ndarray, torch.Tensor],
        child:     Union[str, np.ndarray, torch.Tensor],
    ) -> float:
        """Return a pronunciation score in [0, 100].  Higher = better match."""
        ref_wav   = self._preprocess(reference)
        child_wav = self._preprocess(child)

        ref_emb = self._encode(ref_wav)
        child_emb = self._encode(child_wav)
        prob = torch.clamp(self.model.score_from_embeddings(ref_emb, child_emb), 0.0, 1.0)
        calibrated_score = float(prob.mul(100.0).clamp(0.0, 100.0).item())
        cosine_sim = float(torch.clamp(F.cosine_similarity(ref_emb, child_emb), -1.0, 1.0).item())
        confidence = min(1.0, max(0.25, (cosine_sim - 0.35) / 0.37))
        final_score = round(max(0.0, min(100.0, calibrated_score * confidence)), 2)
        if self.debug:
            logger.debug(
                "score  cosine=%.4f  calibrated=%.2f  confidence=%.3f  final=%.2f",
                cosine_sim, calibrated_score, confidence, final_score,
            )
        return final_score

    @torch.no_grad()
    def score_with_details(
        self,
        reference: Union[str, np.ndarray, torch.Tensor],
        child:     Union[str, np.ndarray, torch.Tensor],
    ) -> dict:
        """
        Return score plus diagnostics.

        Keys
        ────
        final_score            : float 0-100
        label                  : str
        cosine_similarity      : float -1 to 1
        reference_duration_sec : float
        child_duration_sec     : float
        """
        ref_wav   = self._preprocess(reference)
        child_wav = self._preprocess(child)

        ref_emb   = self._encode(ref_wav)
        child_emb = self._encode(child_wav)

        if self.debug:
            cos  = F.cosine_similarity(ref_emb, child_emb).item()
            dist = (ref_emb - child_emb).norm().item()
            logger.debug("cos_sim=%.4f  L2_dist=%.4f", cos, dist)

        cosine_sim = float(
            torch.clamp(
                F.cosine_similarity(ref_emb, child_emb), -1.0, 1.0
            ).item()
        )
        prob = torch.clamp(self.model.score_from_embeddings(ref_emb, child_emb), 0.0, 1.0)
        calibrated_score = float(prob.mul(100.0).clamp(0.0, 100.0).item())
        confidence = min(1.0, max(0.25, (cosine_sim - 0.35) / 0.37))
        final_score = max(0.0, min(100.0, calibrated_score * confidence))

        logger.debug(
            "score_with_details  cosine=%.4f  calibrated=%.2f  confidence=%.3f  final=%.2f  ref_norm=%.4f  child_norm=%.4f",
            cosine_sim,
            calibrated_score,
            confidence,
            final_score,
            ref_emb.norm(dim=-1).item(),
            child_emb.norm(dim=-1).item(),
        )

        return {
            "final_score":            round(final_score, 2),
            "label":                  _score_to_label(final_score),
            "cosine_similarity":      round(cosine_sim, 4),
            "reference_duration_sec": round(ref_wav.shape[0] / SAMPLE_RATE, 3),
            "child_duration_sec":     round(child_wav.shape[0] / SAMPLE_RATE, 3),
        }

    @torch.no_grad()
    def embed(self, audio: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Return the L2-normalised embedding as [embed_dim] numpy array."""
        wav = self._preprocess(audio)
        emb = self._encode(wav)
        return emb.cpu().numpy().squeeze(0)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score a child pronunciation against a reference")
    parser.add_argument("--ref",        required=True,  help="Reference audio file path")
    parser.add_argument("--child",      required=True,  help="Child audio file path")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--embed_dim",  type=int, default=256)
    parser.add_argument("--debug",      action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    scorer = PronunciationScorer(
        checkpoint_path=args.checkpoint,
        embed_dim=args.embed_dim,
        debug=args.debug,
    )
    result = scorer.score_with_details(args.ref, args.child)

    print(f"Score              : {result['final_score']:.2f}")
    print(f"Label              : {result['label']}")
    print(f"Cosine Similarity  : {result['cosine_similarity']:.4f}")
    print(f"Reference Duration : {result['reference_duration_sec']}s")
    print(f"Child Duration     : {result['child_duration_sec']}s")
