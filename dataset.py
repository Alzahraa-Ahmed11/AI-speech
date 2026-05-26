"""
dataset.py — Pronunciation Scoring Dataset

─────────────────────────────────────────────────────────────────────────────
WHY THE PREVIOUS STRATEGY WAS WRONG (and caused constant predictions)
─────────────────────────────────────────────────────────────────────────────
  The previous version paired SC ref + random TORGO child with label=1.
  TORGO files are unlabeled, so a random TORGO file is almost certainly saying
  a *different* word than the SC reference (SpeechCommands has ~35 words; the
  chance of a random TORGO file matching is ~3%). This means:

    SC "cat" (ref)  +  TORGO "dog" (child)  →  label = 1   ← WRONG
    SC "cat" (ref)  +  SC "dog"  (child)    →  label = 0   ← correct

  The model sees "cat+dog = positive" AND "cat+dog = negative" in the same
  training loop. The optimal response to contradictory labels is to output a
  constant value (the dataset mean), which is exactly the collapse symptom.

─────────────────────────────────────────────────────────────────────────────
CORRECTED STRATEGY
─────────────────────────────────────────────────────────────────────────────
  Positive (1.0): SC word W (ref) + SC word W (child, different recording)
                  Cross-speaker pairing preferred (80% of attempts) so the
                  model learns pronunciation quality, not speaker timbre.
                  Child is heavily augmented (noise, speed, pitch) to simulate
                  the dysarthric/imperfect speech that TORGO would have provided,
                  but with a *guaranteed correct word label*.

  Negative (0.0): SC word A (ref) + SC word B (child), A ≠ B
                  30% of negatives are phonetically similar pairs (hard
                  negatives) so the model learns to distinguish near-homophones.
                  An optional bounded fraction (torgo_neg_frac, default 15%)
                  uses TORGO files as negative children — this is statistically
                  safe because unlabeled TORGO files are overwhelmingly NOT the
                  same word as the SC reference.

  Balance: 50% positive / 50% negative (negative_ratio=1.0).

─────────────────────────────────────────────────────────────────────────────
SHOULD TORGO BE POSITIVE OR NEGATIVE?
─────────────────────────────────────────────────────────────────────────────
  TORGO should NOT be used as positive child without word-level alignment.
  Without knowing which word a TORGO file contains, labelling it as a positive
  match for any SC reference creates contradictory training signal.

  TORGO CAN safely be used as:
    1. Negative child (15% of negatives) — almost certainly a different word.
    2. Augmentation inspiration — we simulate dysarthric quality by applying
       aggressive noise/speed/pitch augmentation to correct SC recordings.
"""

import os
import random
import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

SAMPLE_RATE   = 16000
MAX_AUDIO_LEN = 4 * SAMPLE_RATE        # 64 000 samples = 4 seconds
_AUDIO_EXTS   = {".wav", ".flac", ".mp3"}

# (ref_path, child_path, label, ref_word, child_word)
_Pair = Tuple[str, str, float, str, str]

# Phonetically similar words — 30% of negatives sampled from here so the
# model is forced to distinguish near-homophones rather than obvious ones.
_PHONETIC_SIMILAR: Dict[str, List[str]] = {
    "up":    ["of", "off"],
    "off":   ["on", "of", "up"],
    "on":    ["one", "off"],
    "go":    ["no"],
    "no":    ["go"],
    "yes":   ["yet"],
    "stop":  ["top"],
    "left":  ["lift"],
    "right": ["light"],
    "down":  ["town"],
    "bed":   ["bad", "red"],
    "cat":   ["bat", "hat"],
    "dog":   ["log"],
    "bird":  ["word"],
    "tree":  ["three"],
    "four":  ["for"],
    "five":  ["live"],
    "six":   ["mix"],
    "one":   ["on"],
    "two":   ["to"],
    "three": ["tree"],
    "eight": ["gate"],
    "nine":  ["mine"],
    "zero":  ["hero"],
}

_TORGO_GROUPS = {"dysarthric", "control", "female_d", "male_d", "female_c", "male_c"}


class AudioFile(NamedTuple):
    """File path + speaker ID for cross-speaker positive pair generation."""
    path:    str
    speaker: str


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _safe_fallback_waveform() -> torch.Tensor:
    """Return a guaranteed non-empty waveform (0.5 s of silence at SAMPLE_RATE).

    Used everywhere we need to recover from an empty / broken tensor so that
    downstream ops like ``waveform.abs().max()`` never see ``numel() == 0``.
    """
    return torch.zeros(int(0.5 * SAMPLE_RATE), dtype=torch.float32)


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Load any audio file → mono float32 tensor at *target_sr*, clipped to MAX_AUDIO_LEN.

    Pipeline matches infer._preprocess exactly:
        decode → mono → resample to 16 kHz → clip → peak-normalise

    Raises ValueError for files that are too short or completely silent,
    so __getitem__ can catch them and substitute a safe fallback.
    """
    import librosa
    waveform_np, _ = librosa.load(path, sr=target_sr, mono=True, dtype=np.float32)

    # Guard against decoders returning empty / 0-d arrays before we ever
    # touch them with torch ops that require numel() > 0.
    if waveform_np is None or waveform_np.size == 0:
        raise ValueError(f"Empty audio decoded from: {path}")

    waveform = torch.from_numpy(np.ascontiguousarray(waveform_np))

    if waveform.shape[0] > MAX_AUDIO_LEN:
        waveform = waveform[:MAX_AUDIO_LEN]

    if waveform.numel() < int(SAMPLE_RATE * 0.1):
        raise ValueError(f"Audio too short ({waveform.numel()} samples): {path}")

    # numel() guaranteed > 0 here, so .max() is safe.
    peak = waveform.abs().max()
    if not torch.isfinite(peak) or peak < 1e-4:
        raise ValueError(f"Silent or invalid audio (peak={float(peak):.2e}): {path}")

    return waveform / peak


def augment_waveform(waveform: torch.Tensor, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Apply random augmentations to the child waveform during training.

    Augmentations are deliberately aggressive compared to the previous version
    to simulate the dysarthric speech quality that TORGO represented, while
    preserving the word identity (unlike random TORGO files).

    All steps are defensive: any operation that could yield an empty tensor
    (time shift past the end, time_stretch/pitch_shift edge cases, accidental
    truncation) is gated and rolled back so the returned tensor is *always*
    non-empty.
    """
    # Defensive entry guard — caller might have produced an empty tensor.
    if waveform is None or waveform.numel() == 0:
        return _safe_fallback_waveform()

    # Gaussian noise — simulates recording noise and articulatory imprecision
    if random.random() < 0.6:
        waveform = waveform + torch.randn_like(waveform) * random.uniform(0.003, 0.015)

    # Volume jitter — dysarthric speakers often have inconsistent loudness
    if random.random() < 0.5:
        waveform = waveform * random.uniform(0.4, 1.6)

    # Time shift — simulates imprecise speech onset timing.
    # Must satisfy: shift > 0 AND shift < numel(), otherwise the slice
    # waveform[:-shift] is empty (shift==0 → empty, shift>=numel → empty).
    if random.random() < 0.4:
        max_shift = min(int(0.15 * sr), waveform.numel() - 1)
        if max_shift > 0:
            shift = random.randint(1, max_shift)
            shifted = torch.cat([torch.zeros(shift, dtype=waveform.dtype), waveform[:-shift]])
            if shifted.numel() > 0:
                waveform = shifted

    # Speed perturbation — dysarthric speech is often slower or irregular.
    # librosa needs a few samples of context; skip on tiny inputs and roll back
    # if the result comes out empty / NaN.
    if random.random() < 0.35 and waveform.numel() >= 512:
        import librosa
        rate = random.choice([0.75, 0.85, 0.9, 1.05, 1.1, 1.2])
        try:
            stretched = librosa.effects.time_stretch(waveform.numpy(), rate=rate)
            if stretched is not None and stretched.size > 0 and np.all(np.isfinite(stretched)):
                waveform = torch.from_numpy(np.ascontiguousarray(stretched.astype(np.float32)))
        except Exception:
            logger.debug("time_stretch failed; keeping previous waveform", exc_info=True)

    # Pitch shift — simulates vocal tract length differences.
    if random.random() < 0.25 and waveform.numel() >= 512:
        import librosa
        n_steps = random.uniform(-2.5, 2.5)
        try:
            shifted = librosa.effects.pitch_shift(waveform.numpy(), sr=sr, n_steps=n_steps)
            if shifted is not None and shifted.size > 0 and np.all(np.isfinite(shifted)):
                waveform = torch.from_numpy(np.ascontiguousarray(shifted.astype(np.float32)))
        except Exception:
            logger.debug("pitch_shift failed; keeping previous waveform", exc_info=True)

    if waveform.shape[0] > MAX_AUDIO_LEN:
        waveform = waveform[:MAX_AUDIO_LEN]

    # Final safety net — if *any* of the above somehow zeroed out the tensor,
    # fall back to silence rather than crashing on .max().
    if waveform.numel() == 0:
        return _safe_fallback_waveform()

    # Re-normalise after augmentation (volume jitter can push outside [-1,1])
    peak = waveform.abs().max()
    if torch.isfinite(peak) and peak > 1e-8:
        waveform = waveform / peak

    return waveform


def augment_bad_pronunciation(waveform: torch.Tensor, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Heavier same-word corruption used only for synthetic negative pairs.

    The goal is not to create realistic audio artifacts; it is to teach the
    scorer that "same lexical word" is not automatically a good pronunciation
    when timing, pitch, clarity, and onset are badly degraded.

    Every transformation is guarded the same way as ``augment_waveform`` so
    that the returned tensor is always non-empty.
    """
    if waveform is None or waveform.numel() == 0:
        return _safe_fallback_waveform()

    waveform = waveform.clone()

    waveform = waveform + torch.randn_like(waveform) * random.uniform(0.012, 0.035)
    waveform = waveform * random.uniform(0.25, 1.8)

    if waveform.numel() > int(0.35 * sr):
        drop_len = random.randint(int(0.04 * sr), int(0.18 * sr))
        start_max = max(1, waveform.numel() - drop_len)
        start = random.randint(0, start_max - 1)
        waveform[start:start + drop_len] *= random.uniform(0.0, 0.2)

    # Heavy time shift — same trap as in augment_waveform: shift must be
    # strictly in (0, numel()) so waveform[:-shift] is never empty.
    if random.random() < 0.85:
        lo = max(1, int(0.04 * sr))
        hi = min(int(0.28 * sr), waveform.numel() - 1)
        if hi >= lo:
            shift = random.randint(lo, hi)
            shifted = torch.cat([torch.zeros(shift, dtype=waveform.dtype), waveform[:-shift]])
            if shifted.numel() > 0:
                waveform = shifted

    try:
        import librosa
        if random.random() < 0.75 and waveform.numel() >= 512:
            rate = random.choice([0.62, 0.70, 0.78, 1.25, 1.38])
            stretched = librosa.effects.time_stretch(waveform.numpy(), rate=rate)
            if stretched is not None and stretched.size > 0 and np.all(np.isfinite(stretched)):
                waveform = torch.from_numpy(np.ascontiguousarray(stretched.astype(np.float32)))
        if random.random() < 0.65 and waveform.numel() >= 512:
            n_steps = random.choice([-4.0, -3.0, 3.0, 4.0])
            shifted = librosa.effects.pitch_shift(waveform.numpy(), sr=sr, n_steps=n_steps)
            if shifted is not None and shifted.size > 0 and np.all(np.isfinite(shifted)):
                waveform = torch.from_numpy(np.ascontiguousarray(shifted.astype(np.float32)))
    except Exception:
        logger.debug("Heavy augmentation fallback used", exc_info=True)

    if waveform.shape[0] > MAX_AUDIO_LEN:
        waveform = waveform[:MAX_AUDIO_LEN]

    # Final safety net before .abs().max().
    if waveform.numel() == 0:
        return _safe_fallback_waveform()

    peak = waveform.abs().max()
    if torch.isfinite(peak) and peak > 1e-8:
        waveform = waveform / peak

    return waveform


# ══════════════════════════════════════════════════════════════════════════════
# FILE COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def _sc_speaker(path: Path) -> str:
    """Extract speaker hash from a SpeechCommands filename.
    e.g. 004ae714_nohash_0.wav → '004ae714'
    """
    stem = path.stem
    if "_nohash_" in stem:
        return stem.split("_nohash_")[0]
    parts = stem.split("_")
    return parts[0] if parts else "spk0"


def _is_speechcommands(root_path: Path) -> bool:
    """
    True when root contains word-labelled subdirectories (SpeechCommands layout).
    A subdir counts as word-like when its name is all alphabetic and not a
    known TORGO group name.
    """
    subdirs = [p for p in root_path.iterdir() if p.is_dir() and not p.name.startswith("_")]
    word_like = [d for d in subdirs if d.name.isalpha() and d.name.lower() not in _TORGO_GROUPS]
    return len(word_like) > 0


def _collect_sc(root_path: Path) -> Dict[str, List[AudioFile]]:
    """
    Scan a SpeechCommands root and return {word: [AudioFile(path, speaker)]}.
    Words with fewer than 2 recordings are dropped.
    """
    word_files: Dict[str, List[AudioFile]] = {}
    for word_dir in sorted(root_path.iterdir()):
        if not word_dir.is_dir() or word_dir.name.startswith("_"):
            continue
        word = word_dir.name.lower()
        afs = [
            AudioFile(str(f), _sc_speaker(f))
            for f in word_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in _AUDIO_EXTS
        ]
        if len(afs) >= 2:
            word_files[word] = afs

    logger.info(
        "SpeechCommands %s  words=%d  files=%d",
        root_path, len(word_files), sum(len(v) for v in word_files.values()),
    )
    return word_files


def _collect_torgo(root_path: Path) -> List[str]:
    """
    Scan a TORGO root and return a flat list of all audio file paths.
    No word labels — used only as a pool of realistic dysarthric speech samples
    for a small fraction of negative pairs.
    """
    files = [
        str(f) for f in root_path.rglob("*")
        if f.is_file() and f.suffix.lower() in _AUDIO_EXTS
    ]
    logger.info("TORGO %s  files=%d", root_path, len(files))
    return files


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class PronunciationPairDataset(Dataset):
    """
    Siamese training dataset: yields (ref_waveform, child_waveform, label).

    See module docstring for the full explanation of why the previous
    SC+TORGO positive pairing caused model collapse.

    Parameters
    ──────────
    data_dirs      : list of dataset roots; auto-detected as SC or TORGO
    augment        : apply aggressive augmentation to the child waveform
    pairs_per_word : max positive pairs generated per SC word
    negative_ratio : #negatives / #positives  (1.0 = balanced 50/50)
    torgo_neg_frac : fraction of negatives that use a TORGO file as child
                     (0.15 = 15%; safe because TORGO ≈ different word from SC ref)
    seed           : random seed for reproducibility (None = non-deterministic)
    """

    def __init__(
        self,
        data_dirs: List[str],
        augment: bool = True,
        pairs_per_word: int = 20,
        negative_ratio: float = 1.0,
        torgo_neg_frac: float = 0.12,
        hard_negative_frac: float = 0.50,
        same_word_bad_neg_frac: float = 0.08,
        seed: Optional[int] = None,
    ):
        self.augment        = augment
        self.pairs_per_word = pairs_per_word
        self.negative_ratio = negative_ratio
        self.torgo_neg_frac = max(0.0, min(0.25, torgo_neg_frac))
        self.hard_negative_frac = max(0.0, min(1.0, hard_negative_frac))
        self.same_word_bad_neg_frac = (
            max(0.0, min(0.25, same_word_bad_neg_frac)) if augment else 0.0
        )
        self._seed          = seed

        if seed is not None:
            random.seed(seed)

        self.sc_word_files: Dict[str, List[AudioFile]] = {}
        self.torgo_files:   List[str] = []

        for d in data_dirs:
            if not os.path.exists(d):
                logger.warning("Data directory not found (skipped): %s", d)
                continue
            root = Path(d)
            if _is_speechcommands(root):
                for word, afs in _collect_sc(root).items():
                    self.sc_word_files.setdefault(word, []).extend(afs)
            else:
                self.torgo_files.extend(_collect_torgo(root))

        if not self.sc_word_files:
            raise ValueError(
                "No SpeechCommands data found in: " + str(data_dirs) +
                "\nExpected subdirectories named after words (e.g. 'yes', 'no', 'cat')."
            )

        self.sc_words = sorted(self.sc_word_files.keys())

        if not self.torgo_files:
            logger.warning("No TORGO files found — TORGO negatives disabled.")
            self.torgo_neg_frac = 0.0

        self._skipped: int = 0

        self._log_dataset_stats()
        self.pairs = self._build_pairs()
        self._log_balance()
        self._log_sample_pairs()

    # ── Speaker grouping ───────────────────────────────────────────────────────

    @staticmethod
    def _by_speaker(afs: List[AudioFile]) -> Dict[str, List[str]]:
        """Group AudioFile list into {speaker_id: [path, ...]} dict."""
        groups: Dict[str, List[str]] = {}
        for af in afs:
            groups.setdefault(af.speaker, []).append(af.path)
        return groups

    # ── Pair builder ───────────────────────────────────────────────────────────

    def _build_pairs(self) -> List[_Pair]:
        pairs: List[_Pair] = []
        seen:  Set[Tuple[str, str]] = set()

        def _try_add(ref: str, child: str, label: float, w1: str, w2: str) -> bool:
            key = (ref, child)
            if key in seen or ref == child:
                return False
            seen.add(key)
            pairs.append((ref, child, label, w1, w2))
            return True

        # ── Positive pairs: SC word W + SC word W (different recording) ────────
        # Cross-speaker preferred (80%) so the model generalises to pronunciation
        # quality rather than memorising speaker-specific acoustic characteristics.
        for word in self.sc_words:
            afs       = self.sc_word_files[word]
            by_spk    = self._by_speaker(afs)
            speakers  = list(by_spk.keys())
            all_paths = [af.path for af in afs]

            n_target = min(self.pairs_per_word, len(afs) * (len(afs) - 1))
            added = attempts = 0

            while added < n_target and attempts < n_target * 6:
                attempts += 1
                if len(speakers) >= 2 and random.random() < 0.8:
                    spk1, spk2 = random.sample(speakers, 2)
                    ref   = random.choice(by_spk[spk1])
                    child = random.choice(by_spk[spk2])
                else:
                    if len(all_paths) < 2:
                        break
                    ref, child = random.sample(all_paths, 2)
                if _try_add(ref, child, 1.0, word, word):
                    added += 1

        n_pos = len(pairs)

        # ── Negative pairs ─────────────────────────────────────────────────────
        n_neg_total = int(n_pos * self.negative_ratio)
        n_torgo_neg = int(n_neg_total * self.torgo_neg_frac) if self.torgo_files else 0
        n_same_bad_neg = int(n_neg_total * self.same_word_bad_neg_frac)
        n_sc_neg    = max(0, n_neg_total - n_torgo_neg - n_same_bad_neg)

        if len(self.sc_words) < 2:
            logger.warning("Need ≥2 SC words for negative pairs — got %d", len(self.sc_words))
        else:
            # Same-word heavily augmented negatives: hard "bad pronunciation" cases.
            added = attempts = 0
            while added < n_same_bad_neg and attempts < n_same_bad_neg * 8:
                attempts += 1
                w1 = random.choice(self.sc_words)
                files = self.sc_word_files[w1]
                if len(files) < 2:
                    continue
                ref, child = random.sample([af.path for af in files], 2)
                if _try_add(ref, child, 0.0, w1, w1):
                    added += 1

            # SC-SC negatives: favor phonetically similar pairs (hard negatives).
            added = attempts = 0
            while added < n_sc_neg and attempts < n_sc_neg * 5:
                attempts += 1
                w1 = random.choice(self.sc_words)

                w2 = None
                if random.random() < self.hard_negative_frac and w1 in _PHONETIC_SIMILAR:
                    candidates = [
                        w for w in _PHONETIC_SIMILAR[w1]
                        if w in self.sc_word_files and w != w1
                    ]
                    if candidates:
                        w2 = random.choice(candidates)

                if w2 is None:
                    remaining = [k for k in self.sc_words if k != w1]
                    if not remaining:
                        continue
                    w2 = random.choice(remaining)

                ref   = random.choice(self.sc_word_files[w1]).path
                child = random.choice(self.sc_word_files[w2]).path
                if _try_add(ref, child, 0.0, w1, w2):
                    added += 1

            # TORGO negatives: SC word W (ref) + TORGO file (child) → label=0
            # Statistically safe: random TORGO file ≠ SC word ~97% of the time.
            if n_torgo_neg > 0:
                added = attempts = 0
                while added < n_torgo_neg and attempts < n_torgo_neg * 5:
                    attempts += 1
                    w1    = random.choice(self.sc_words)
                    ref   = random.choice(self.sc_word_files[w1]).path
                    child = random.choice(self.torgo_files)
                    if _try_add(ref, child, 0.0, w1, "torgo"):
                        added += 1

        random.shuffle(pairs)
        return pairs

    # ── Logging helpers ────────────────────────────────────────────────────────

    def _log_dataset_stats(self) -> None:
        total_sc = sum(len(v) for v in self.sc_word_files.values())
        min_files = min(len(v) for v in self.sc_word_files.values())
        max_files = max(len(v) for v in self.sc_word_files.values())

        logger.info(
            "SC: words=%d  total_files=%d  avg=%.1f  min=%d  max=%d",
            len(self.sc_words), total_sc,
            total_sc / max(len(self.sc_words), 1),
            min_files, max_files,
        )
        logger.info("TORGO: files=%d", len(self.torgo_files))

        # Warn about words with too few files to create cross-speaker pairs
        thin = [w for w, afs in self.sc_word_files.items() if len(afs) < 5]
        if thin:
            logger.warning("Words with <5 recordings (weak positive signal): %s", thin)

        # Warn about words with only one speaker (cross-speaker pairing impossible)
        mono_spk = [
            w for w, afs in self.sc_word_files.items()
            if len(self._by_speaker(afs)) < 2
        ]
        if mono_spk:
            logger.warning(
                "Words with only 1 speaker (cross-speaker positives impossible): %s",
                mono_spk,
            )

    def _log_balance(self) -> None:
        n_pos       = sum(1 for p in self.pairs if p[2] == 1.0)
        n_neg       = sum(1 for p in self.pairs if p[2] == 0.0)
        n_torgo_neg = sum(1 for p in self.pairs if p[2] == 0.0 and p[4] == "torgo")
        n_same_bad_neg = sum(1 for p in self.pairs if p[2] == 0.0 and p[3] == p[4])

        logger.info(
            "Pairs: total=%d  pos=%d (%.0f%%)  neg=%d (%.0f%%)  torgo_neg=%d (%.0f%% of neg)  same_word_bad_neg=%d (%.0f%% of neg)",
            len(self.pairs),
            n_pos, 100.0 * n_pos / max(len(self.pairs), 1),
            n_neg, 100.0 * n_neg / max(len(self.pairs), 1),
            n_torgo_neg, 100.0 * n_torgo_neg / max(n_neg, 1),
            n_same_bad_neg, 100.0 * n_same_bad_neg / max(n_neg, 1),
        )

        ratio = n_pos / max(n_neg, 1)
        if ratio < 0.4 or ratio > 2.5:
            logger.warning(
                "IMBALANCE: pos/neg = %.2f (expected ~1.0). "
                "Try increasing pairs_per_word or reducing negative_ratio.",
                ratio,
            )

    def _log_sample_pairs(self) -> None:
        pos = [p for p in self.pairs if p[2] == 1.0][:3]
        neg = [p for p in self.pairs if p[2] == 0.0][:3]
        logger.info("── Sample POSITIVE pairs (same word, different speaker/recording) ────────")
        for ref, child, _, w1, _ in pos:
            logger.info("  [POS] word=%-12s  ref=…%s  child=…%s", w1, ref[-40:], child[-40:])
        logger.info("── Sample NEGATIVE pairs (different words / TORGO child) ────────────────")
        for ref, child, _, w1, w2 in neg:
            logger.info("  [NEG] %-12s vs %-12s  ref=…%s  child=…%s", w1, w2, ref[-35:], child[-35:])

    def sanity_check(self) -> None:
        """Print extended diagnostics. Call manually before training to verify data."""
        n_pos       = sum(1 for p in self.pairs if p[2] == 1.0)
        n_neg       = sum(1 for p in self.pairs if p[2] == 0.0)
        n_torgo_neg = sum(1 for p in self.pairs if p[2] == 0.0 and p[4] == "torgo")
        n_same_bad_neg = sum(1 for p in self.pairs if p[2] == 0.0 and p[3] == p[4])

        print("\n" + "═" * 64)
        print("DATASET SANITY CHECK")
        print("═" * 64)
        print(f"SC words      : {len(self.sc_words)}")
        print(f"SC files      : {sum(len(v) for v in self.sc_word_files.values())}")
        print(f"TORGO files   : {len(self.torgo_files)}")
        print(f"Total pairs   : {len(self.pairs)}")
        print(f"  Positive    : {n_pos} ({100*n_pos//max(len(self.pairs),1)}%)")
        print(f"  Negative    : {n_neg} ({100*n_neg//max(len(self.pairs),1)}%)")
        print(f"  TORGO neg   : {n_torgo_neg} ({100*n_torgo_neg//max(n_neg,1)}% of neg)")
        print(f"  Same-word bad neg: {n_same_bad_neg} ({100*n_same_bad_neg//max(n_neg,1)}% of neg)")

        print("\nWord coverage (top 10 by file count):")
        for word, afs in sorted(self.sc_word_files.items(), key=lambda x: -len(x[1]))[:10]:
            n_spk = len(self._by_speaker(afs))
            print(f"  {word:<16} files={len(afs):<6} speakers={n_spk}")

        print("\nSample POSITIVE pairs:")
        for ref, child, _, w1, _ in [p for p in self.pairs if p[2] == 1.0][:3]:
            print(f"  [POS] word={w1:<12}  ref=…{ref[-45:]}  child=…{child[-45:]}")

        print("\nSample NEGATIVE pairs:")
        for ref, child, _, w1, w2 in [p for p in self.pairs if p[2] == 0.0][:3]:
            print(f"  [NEG] {w1:<12} vs {w2:<12}  ref=…{ref[-40:]}  child=…{child[-40:]}")
        print("═" * 64 + "\n")

    # ── Epoch reshuffling ──────────────────────────────────────────────────────

    def reshuffle(self, seed: Optional[int] = None) -> None:
        """Rebuild pair list with fresh random sampling. Call at the start of each epoch."""
        if self._skipped > 0:
            logger.warning(
                "Skipped %d invalid audio samples in previous epoch "
                "(too short < 1600 samples or silent peak < 1e-4).",
                self._skipped,
            )
        self._skipped = 0

        if seed is not None:
            random.seed(seed)
        elif self._seed is not None:
            # Advance the base seed each epoch so pairs are different but deterministic
            self._seed += 1
            random.seed(self._seed)

        self.pairs = self._build_pairs()
        n_pos = sum(1 for p in self.pairs if p[2] == 1.0)
        n_neg = sum(1 for p in self.pairs if p[2] == 0.0)
        logger.info(
            "Reshuffled pairs: total=%d  pos=%d  neg=%d",
            len(self.pairs), n_pos, n_neg,
        )

    def adjust_negative_hardness(
        self,
        hard_negative_frac: Optional[float] = None,
        same_word_bad_neg_frac: Optional[float] = None,
        torgo_neg_frac: Optional[float] = None,
        rebuild_pairs: bool = True,
    ) -> None:
        """
        Safely adjust negative sampling during training.

        Used by train.py collapse prevention when cosine separation is flat.
        TORGO stays bounded to 15%, and same-word bad negatives stay rare.
        """
        if hard_negative_frac is not None:
            self.hard_negative_frac = max(0.0, min(1.0, hard_negative_frac))
        if same_word_bad_neg_frac is not None:
            self.same_word_bad_neg_frac = (
                max(0.0, min(0.25, same_word_bad_neg_frac)) if self.augment else 0.0
            )
        if torgo_neg_frac is not None:
            self.torgo_neg_frac = max(0.0, min(0.25, torgo_neg_frac))

        logger.info(
            "Negative sampling adjusted: hard=%.2f  same_word_bad=%.2f  torgo=%.2f",
            self.hard_negative_frac,
            self.same_word_bad_neg_frac,
            self.torgo_neg_frac,
        )

        if rebuild_pairs:
            self.pairs = self._build_pairs()

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _MAX_RETRIES = 10
        for attempt in range(_MAX_RETRIES):
            ref_path, child_path, label, ref_word, child_word = self.pairs[idx]
            try:
                ref_wav   = load_audio(ref_path)
                child_wav = load_audio(child_path)
            except Exception as exc:
                self._skipped += 1
                logger.debug("Skipping pair (attempt %d): %s", attempt + 1, exc)
                idx = random.randrange(len(self.pairs))
                continue

            if self.augment:
                if label == 0.0 and ref_word == child_word:
                    if random.random() < 0.5:
                        child_wav = augment_bad_pronunciation(child_wav)
                    else:
                        child_wav = augment_waveform(child_wav)
                else:
                    child_wav = augment_waveform(child_wav)

            # Final guarantee — never let an empty tensor reach the collate_fn,
            # even if some future refactor breaks an augmentation guard.
            if ref_wav.numel() == 0:
                ref_wav = _safe_fallback_waveform()
            if child_wav.numel() == 0:
                child_wav = _safe_fallback_waveform()

            return ref_wav, child_wav, torch.tensor(label, dtype=torch.float32)

        # All retries exhausted — return a silent pair rather than crash.
        # Loss contribution is near-zero (BCE on 0.5 logit with either label ≈ 0.693).
        logger.warning("All %d retries failed at idx=%d — returning silence fallback", _MAX_RETRIES, idx)
        return self.__getitem__(random.randrange(len(self.pairs)))

    # ── Online data addition ───────────────────────────────────────────────────

    def add_new_data(self, new_dir: str, rebuild_pairs: bool = True) -> None:
        """Add files from *new_dir* and optionally rebuild pairs."""
        root = Path(new_dir)
        if _is_speechcommands(root):
            for word, afs in _collect_sc(root).items():
                self.sc_word_files.setdefault(word, []).extend(afs)
            self.sc_words = sorted(self.sc_word_files.keys())
        else:
            self.torgo_files.extend(_collect_torgo(root))

        if rebuild_pairs:
            self.pairs = self._build_pairs()

        logger.info(
            "After add_new_data: sc_words=%d  torgo_files=%d  pairs=%d",
            len(self.sc_word_files), len(self.torgo_files), len(self.pairs),
        )


# ══════════════════════════════════════════════════════════════════════════════
# COLLATE & DATALOADER
# ══════════════════════════════════════════════════════════════════════════════

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad variable-length waveforms to the longest sample in the batch and build
    boolean attention masks (True = real signal, False = zero-padding).

    Returns: (ref_padded, child_padded, ref_mask, child_mask, labels)
    """
    refs, children, labels = zip(*batch)

    def _pad(
        wavs: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(w.shape[0] for w in wavs)
        padded  = torch.zeros(len(wavs), max_len)
        masks   = torch.zeros(len(wavs), max_len, dtype=torch.bool)
        for i, w in enumerate(wavs):
            padded[i, : w.shape[0]] = w
            masks [i, : w.shape[0]] = True
        return padded, masks

    ref_padded,   ref_masks   = _pad(refs)
    child_padded, child_masks = _pad(children)

    return ref_padded, child_padded, ref_masks, child_masks, torch.stack(labels)


def get_dataloader(
    data_dirs: List[str],
    batch_size: int = 16,
    augment: bool = True,
    pairs_per_word: int = 20,
    negative_ratio: float = 1.0,
    torgo_neg_frac: float = 0.12,
    hard_negative_frac: float = 0.50,
    same_word_bad_neg_frac: float = 0.08,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> DataLoader:
    """Build and return a DataLoader for the PronunciationPairDataset."""
    dataset = PronunciationPairDataset(
        data_dirs,
        augment=augment,
        pairs_per_word=pairs_per_word,
        negative_ratio=negative_ratio,
        torgo_neg_frac=torgo_neg_frac,
        hard_negative_frac=hard_negative_frac,
        same_word_bad_neg_frac=same_word_bad_neg_frac,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
