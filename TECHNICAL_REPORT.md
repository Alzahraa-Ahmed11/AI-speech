# AI Pronunciation Scoring System — Complete Technical Documentation

**Project:** Pronunciation Scoring System for Speech Therapy Applications  
**Architecture:** Siamese Wav2Vec2 Neural Network with Learned Comparison Head  
**Stack:** PyTorch · HuggingFace Transformers · FastAPI · librosa · torchaudio  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [System Architecture](#3-system-architecture)
4. [Dataset Strategy](#4-dataset-strategy)
5. [Model Evolution](#5-model-evolution)
6. [Training Pipeline](#6-training-pipeline)
7. [Inference Pipeline](#7-inference-pipeline)
8. [API Design and Workflow](#8-api-design-and-workflow)
9. [Evaluation Metrics and Results](#9-evaluation-metrics-and-results)
10. [File-by-File Summary](#10-file-by-file-summary)
11. [Challenges and Fixes](#11-challenges-and-fixes)
12. [Limitations](#12-limitations)
13. [Future Improvements](#13-future-improvements)
14. [Conclusion](#14-conclusion)

---

## 1. Project Overview

This system is an **end-to-end automated pronunciation scoring engine** designed to assess how closely a child's spoken utterance matches a native-speaker reference recording. It assigns a continuous score from 0 to 100 and a binary quality label ("Good" / "Try Again"), enabling real-time feedback in speech therapy and language learning applications.

The system processes raw audio waveforms from two sources — a native reference speaker and a child learner — and produces a calibrated pronunciation quality score. The entire pipeline runs on commodity hardware (CPU or single GPU) and is deployed as an HTTP API, making it accessible to mobile and web applications.

**Core capabilities:**
- Score any word-level pronunciation pair against a native reference
- Handle multiple audio formats (WAV, FLAC, MP3, WebM, OGG)
- Operate at variable audio lengths up to 4 seconds
- Provide embedding extraction for downstream analysis
- Support both CLI and HTTP API access modes

---

## 2. Problem Statement

### The core challenge

Automated pronunciation scoring is fundamentally a **similarity learning problem**: given two audio recordings of the same word (one native, one by a learner), determine how phonetically similar they are. The difficulty lies in learning a similarity metric that:

1. Is **speaker-independent** — recognizes good pronunciation regardless of who the reference speaker is
2. Is **noise-robust** — accurately scores despite microphone differences, background noise, and recording quality variation
3. Captures **phonemic quality**, not just acoustic proximity — two recordings by the same speaker of the same word may sound identical in raw cosine space but differ in articulation quality

### The target population

The system was designed with two user groups in mind:
- **Native reference speakers** — providing high-quality recordings that define correct pronunciation
- **Child learners with possible dysarthric traits** — producing speech that may be slower, imprecise in timing, inconsistent in pitch and loudness, or phonemically imprecise

This population contrast motivates several design decisions throughout the pipeline: the augmentation strategy that simulates dysarthric speech quality, the cross-speaker positive pair preference, and the conservative label policy that penalizes false positives (passing incorrect pronunciation as correct).

### Why existing approaches were insufficient

A naive baseline — computing cosine similarity between raw Wav2Vec2 embeddings — fails because:
- Wav2Vec2's pretrained embedding space was optimized for speech recognition, not pronunciation quality assessment
- Cosine similarity treats all dimensions equally; pronunciation quality may manifest in a subspace the pretrained model does not emphasize
- A fixed cosine threshold cannot capture the non-linear relationship between acoustic similarity and perceived pronunciation quality

The system addresses these limitations with a learned `ScoringHead` that combines geometric embedding features with a trainable MLP.

---

## 3. System Architecture

### End-to-end component diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE (Training)                              │
│                                                                         │
│  SpeechCommands Dataset        TORGO Dataset (optional)                 │
│  {word → [AudioFile]}          [flat audio paths]                       │
│         │                              │                                │
│         └──────────┬───────────────────┘                                │
│                    ▼                                                     │
│          PronunciationPairDataset  (dataset.py)                         │
│          ┌─────────────────────────────────────────┐                    │
│          │  Positive: SC word W + SC word W        │                    │
│          │  Negative: SC word A + SC word B (A≠B)  │                    │
│          │  Negative: SC ref + TORGO child (15%)   │                    │
│          │  Augmentation: noise/speed/pitch/shift  │                    │
│          └─────────────────────────────────────────┘                    │
│                    │                                                     │
│                    ▼                                                     │
│             collate_fn → (ref_padded, child_padded, ref_mask,           │
│                           child_mask, labels)                           │
│                    │                                                     │
│                    ▼                                                     │
│         ┌─────────────────────────────────────────┐                    │
│         │    PronunciationScoringModel (model.py)  │                    │
│         │                                          │                    │
│         │  Wav2Vec2PronunciationEncoder (encoder.py)│                   │
│         │  ├── CNN feature extractor [FROZEN]      │                    │
│         │  ├── Transformer [FROZEN Phase 1]        │                    │
│         │  │              [PARTIAL Phase 2]        │                    │
│         │  ├── Masked mean pooling                 │                    │
│         │  └── Projection MLP → L2-norm            │                    │
│         │            ↓ ref_emb    ↓ child_emb      │                    │
│         │       ScoringHead (model.py)             │                    │
│         │  [ref, child, |ref-child|, ref*child]    │                    │
│         │  → MLP → raw logit                       │                    │
│         └─────────────────────────────────────────┘                    │
│                    │                                                     │
│              BCEWithLogitsLoss → Optimizer                              │
│              + CosineEmbeddingLoss (encoder signal)                     │
│                    │                                                     │
│              checkpoints/best_model.pt                                  │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        ONLINE (Inference)                               │
│                                                                         │
│  HTTP client (browser / mobile app)                                     │
│         │  POST /score  (multipart: ref audio + child audio)            │
│         ▼                                                               │
│      api.py  (FastAPI)                                                  │
│  ├── Audio decode: soundfile → librosa → pydub → torchaudio             │
│  ├── 15-check validation pipeline                                       │
│  ├── _scorer._preprocess() → 16 kHz mono float32 tensor                │
│  └── with _score_lock, torch.inference_mode():                         │
│         ▼                                                               │
│      infer.py  (PronunciationScorer)                                    │
│  ├── _encode(ref_wav)   → ref_emb   [1, 256] L2-norm                   │
│  ├── _encode(child_wav) → child_emb [1, 256] L2-norm                   │
│  ├── score_from_embeddings()                                            │
│  │   sigmoid((logit / 1.35) - ReLU(0.72 - cos) × 3.0)                 │
│  ├── confidence = clamp((cos - 0.35) / 0.37, 0.25, 1.0)               │
│  └── final_score = calibrated × confidence × 100                       │
│         ▼                                                               │
│  ScoreResponse { score, label, similarity, ref_duration, child_duration}│
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                       OFFLINE (Evaluation)                              │
│                                                                         │
│      eval.py                                                            │
│  ├── PronunciationPairDataset (augment=False, shuffle=False)            │
│  ├── model.logit_from_embeddings() + sigmoid  [raw, uncalibrated]       │
│  ├── Threshold sweep: [0.50, 0.55, ..., 0.90]                          │
│  ├── Metrics: accuracy, F0.5, F1, precision, recall, AUC               │
│  └── Best threshold → recommended THRESHOLD for api.py                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Shared data contracts between components

| Contract | Producer | Consumer | Format |
|---|---|---|---|
| Waveform tensor | `dataset.py` / `infer.py` | `encoder.py` | `[batch, samples]` float32 at 16 kHz |
| Attention mask | `collate_fn` | `encoder.py` | `[batch, samples]` bool, True = real signal |
| L2-normalized embedding | `encoder.py` | `model.py` / `infer.py` | `[batch, 256]` float32, unit norm |
| Raw logit | `ScoringHead` | `BCEWithLogitsLoss` / `eval.py` | `[batch]` float32, unbounded |
| Calibrated score | `score_from_embeddings()` | `infer.py` / `api.py` | `[batch]` float32 ∈ [0, 1] |
| Checkpoint | `train.py` | `infer.py` / `api.py` / `eval.py` | `{model_state_dict, arch, epoch, metrics}` |

---

## 4. Dataset Strategy

### Datasets used

**Google SpeechCommands:** ~35 word classes, thousands of recordings per word, labeled at the word level, multiple speakers per word. Used as the primary source of labeled positive and negative pairs.

**TORGO:** Recordings of speakers with dysarthria and matched controls. **No word-level labels.** Used only as a pool of realistic dysarthric-style speech for a bounded fraction of negative pairs.

### The label-noise collapse problem

The original dataset strategy paired a SpeechCommands reference with a random TORGO child recording and assigned label = 1 (correct pronunciation). This was fundamentally incorrect.

SpeechCommands contains ~35 words. A random TORGO file matches the reference word approximately **3% of the time**. This means 97% of TORGO "positive" pairs were actually different-word pairs — contradicting the SC-SC negative pairs where the same word combination (e.g., "cat" vs. "dog") was labeled 0. The model encountered the same pair type with both labels simultaneously. The optimal response to contradictory labels is to output the dataset mean — a constant value — which is precisely the collapse symptom (pred_std ≈ 0, AUC ≈ 0.5) that was observed.

### Corrected pair generation strategy

The fixed strategy enforces **guaranteed label correctness**:

```
Positive (label = 1.0):
  SpeechCommands word W (ref) + SpeechCommands word W (child, different recording)
  Cross-speaker preferred: 80% of positives use two different speaker IDs
  Child waveform is augmented to simulate dysarthric-like variability

Negative (label = 0.0):  [three sub-types]
  ① SC word A + SC word B, A ≠ B  (standard negatives)
     50% drawn from _PHONETIC_SIMILAR dictionary (hard negatives)
  ② SC word W + TORGO file  (12% of negatives; statistically safe)
  ③ SC word W + SC word W + heavy augmentation  (8%; same-word bad pronunciation)

Balance: 50% positive / 50% negative (negative_ratio = 1.0)
```

### Augmentation as dysarthric simulation

Since TORGO cannot be used as positive examples, the system simulates dysarthric speech quality on SpeechCommands children through stochastic augmentation:

| Augmentation | Probability | Effect |
|---|---|---|
| Gaussian noise | 60% | Articulatory imprecision |
| Volume jitter | 50% | Inconsistent breath support |
| Time shift | 40% | Imprecise speech onset |
| Speed perturbation | 35% | Slower/faster dysarthric rate (0.75×–1.2×) |
| Pitch shift | 25% | Vocal tract length differences (±2.5 semitones) |

A heavier "bad pronunciation" augmentation variant is applied to same-word negative pairs, using more extreme parameters (speed rates of 0.62× or 1.38×, pitch shifts of ±3–4 semitones, segment suppression) to teach the model that lexical correctness does not guarantee phonemic quality.

### Hard negatives and curriculum learning

The `_PHONETIC_SIMILAR` dictionary maps 24 SpeechCommands words to their phonetically similar neighbors (e.g., "right" → ["light"], "go" → ["no"]). Half of all SC-SC negatives are drawn from this dictionary, forcing the model to distinguish near-homophones rather than relying on gross acoustic dissimilarity. The `adjust_negative_hardness()` method in `PronunciationPairDataset` allows the training loop to dynamically increase negative difficulty if collapse indicators are detected.

---

## 5. Model Evolution

### Stage 1 — Cosine similarity baseline

The initial approach computed cosine similarity directly between Wav2Vec2 last-hidden-state mean-pooled embeddings:

```
score = cosine(ref_emb, child_emb)  → fixed threshold → pass/fail
```

**Limitations:** Wav2Vec2 was pretrained for speech recognition. Its embedding space is not optimized for pronunciation quality. Cosine similarity in this space conflates speaker identity, word identity, and pronunciation quality — all three dimensions contribute equally.

### Stage 2 — Legacy CNN encoder

A CNN-based spectrogram encoder (`_LegacyCnnPronunciationEncoder`) was developed as an alternative:
- Log-magnitude STFT → 4-layer Conv2d stack with BatchNorm + ELU
- Average + max pooling → 256-dim projection MLP
- L2-normalized output

This eliminated Wav2Vec2's pretrained bias but sacrificed the rich contextual representations that transformer architectures provide. The encoder is retained in `model.py` as a documented fallback but is no longer the active encoder.

### Stage 3 — Wav2Vec2 Siamese model (current system)

The current architecture addresses both limitations: it uses Wav2Vec2's rich pretrained representations and adds a **learned comparison head** that can capture pronunciation-specific similarity patterns.

**Architecture:**

```
Shared encoder (Wav2Vec2PronunciationEncoder):
  Raw PCM [batch, samples]
    → CNN feature extractor [FROZEN, ~15M params]
    → Transformer (12 layers) [FROZEN → partially unfrozen in Phase 2]
    → last_hidden_state [batch, time, 768]
    → Masked mean pooling [batch, 768]
    → Projection MLP: 768 → LayerNorm → GELU → Dropout → 256
    → L2 normalize → embedding [batch, 256]

ScoringHead (learned comparison):
  Features: [ref_emb, child_emb, |ref-child|, ref*child] → [batch, 1024]
  MLP: 1024 → ReLU → Dropout → 512 → ReLU → Dropout → 256 → 1
  Output: raw logit [batch], unbounded
  Init: Kaiming He (ReLU-appropriate)

Training objective:
  BCEWithLogitsLoss(logit, label)         ← primary
  + CosineEmbeddingLoss(ref_emb, child_emb, ±1) × cos_weight  ← encoder signal
```

**Weight sharing:** Both reference and child audio pass through the **same encoder weights** (Siamese architecture). This enforces a symmetric, unified pronunciation embedding space and halves the parameter count compared to two separate encoders.

---

## 6. Training Pipeline

### Two-phase training protocol

The training is divided into two phases that address the core risk of destroying pretrained Wav2Vec2 features:

**Phase 1 — Warmup (transformer frozen):**
- All Wav2Vec2 transformer weights: `requires_grad = False`
- Only the projection MLP and `ScoringHead` are trained
- Flat learning rate (no cosine decay)
- Purpose: Let the head establish a useful signal gradient before the heavy backbone can be perturbed
- Wav2Vec2 (94M params) vs. trainable head (~460K params) — without this phase, the head's tiny gradient signal cannot overcome the backbone's momentum

**Phase 2 — Fine-tuning (transformer selectively unfrozen):**
- Top N transformer layers unfrozen (configurable)
- `feature_projection` and `encoder.layer_norm` also unfrozen
- CNN feature extractor remains permanently frozen
- Differential learning rates via `get_param_groups()`:

```
Head + projection:   full_lr
Transformer layers:  full_lr × 0.05   (20× smaller)
```

The 5% LR scale prevents catastrophic forgetting — the transformer's pretrained features are nudged toward pronunciation-relevant representations rather than overwritten.

**Learning rate schedule:**
- Phase 1: Flat (constant) LR
- Phase 2: Cosine annealing after warmup

Cosine annealing that begins immediately (before warmup) was identified as a collapse contributor — the LR drops during the critical period when the head is establishing its gradient signal.

### Loss function

The training objective combines two losses:

```
L_total = L_BCE + λ_cos × L_cosine

L_BCE    = BCEWithLogitsLoss(logit, label)
L_cosine = CosineEmbeddingLoss(ref_emb, child_emb, y ∈ {+1, -1})

λ_cos = 1.5  (cos_weight — increased from original 0.2)
cos_margin = 0.0  (margin removed from CosineEmbeddingLoss)
```

The cosine loss provides a direct gradient signal to the encoder's embedding space, encouraging it to place same-word embeddings close together and different-word embeddings apart on the unit hypersphere. The original values (`λ=0.2`, `margin=0.3`) caused near-zero gradient for negative pairs from the start of training — a critical collapse contributor.

### Epoch reshuffling and curriculum

`PronunciationPairDataset.reshuffle()` rebuilds the pair list at each epoch start with a different random seed (base seed + epoch offset). This provides implicit data augmentation at the pair level. The `adjust_negative_hardness()` method allows the training loop to increase hard-negative fraction dynamically if the model's cosine similarity separation remains flat — a curriculum learning mechanism responding to live training diagnostics.

---

## 7. Inference Pipeline

### Preprocessing contract

All audio inputs are normalized to a canonical form before any model computation:

```
1. Decode audio to mono float32 (format-agnostic)
2. Resample to 16,000 Hz
3. Clip to MAX_AUDIO_LEN = 4 seconds (64,000 samples)
4. Peak-normalize to [-1, 1]  (divide by abs().max())
```

This contract is explicitly shared between `dataset.py` (`load_audio()`) and `infer.py` (`_preprocess()`). The module-level note in `dataset.py` states: *"Pipeline matches infer._preprocess exactly."* Divergence between training and inference preprocessing is a common source of train-serve skew; this contract prevents it.

### Embedding computation

```python
ref_emb   = encoder(ref_wav,   ref_mask)    # [1, 256] L2-normalized
child_emb = encoder(child_wav, child_mask)  # [1, 256] L2-normalized
```

The encoder runs with `torch.no_grad()` when the transformer is fully frozen (Phase 1) and with full gradient tracking when fine-tuning (Phase 2). At inference time, `@torch.no_grad()` is applied at the scoring method level.

### Scoring formula

The final score is computed through three sequential steps:

**Step 1 — MLP calibrated score:**
```
raw_logit = ScoringHead(ref_emb, child_emb)
cosine_penalty = ReLU(0.72 - cosine_similarity) × 3.0
calibrated_score = sigmoid(raw_logit / 1.35 - cosine_penalty) × 100
```

**Step 2 — Geometric confidence:**
```
cosine_sim = F.cosine_similarity(ref_emb, child_emb)
confidence = clamp((cosine_sim - 0.35) / 0.37, min=0.25, max=1.0)
```

| Cosine range | Confidence | Meaning |
|---|---|---|
| ≤ 0.35 | 0.25 (floor) | Embeddings near-orthogonal; heavy penalty |
| 0.35 → 0.72 | 0.25 → 1.0 (linear) | Partial geometric agreement |
| ≥ 0.72 | 1.0 (ceiling) | Embeddings highly similar; MLP fully trusted |

**Step 3 — Final score:**
```
final_score = calibrated_score × confidence  ∈ [0, 100]
```

This two-stage design applies **two independent geometric safeguards**: the cosine margin penalty inside `score_from_embeddings()` prevents the MLP from overriding a clear geometric rejection, while the confidence multiplier in `infer.py` and `api.py` provides a second proportional discount.

### Cosine-only fallback

When no compatible scoring head is found in a checkpoint (wrong architecture tag, shape mismatch, or no checkpoint at all), the system falls back to:

```
score = sigmoid((cosine_similarity - 0.94) / 0.04)
```

This steep sigmoid at cosine = 0.94 provides a usable score even from an untrained model, ensuring the system degrades gracefully rather than crashing.

---

## 8. API Design and Workflow

### Server architecture

The API is a **FastAPI** application with a single scoring endpoint. Key design decisions:

- **Model as singleton:** `PronunciationScorer` is instantiated once at startup via the `lifespan` context manager and shared across all requests
- **Threading lock:** `_score_lock = threading.Lock()` serializes all inference calls — critical because PyTorch's CUDA operations are not thread-safe for simultaneous forward passes
- **`torch.inference_mode()`:** Applied inside the lock for maximum inference speed (disables gradient tracking and view versioning)

### Multi-format audio decoding

The API accepts any audio format the client submits. A four-strategy fallback chain decodes raw bytes to a mono float32 numpy array:

```
Strategy 1: soundfile  (BytesIO — no disk I/O; handles WAV/FLAC/OGG)
     ↓ on failure
Strategy 2: librosa    (temp file; handles MP3 via audioread + ffmpeg)
     ↓ on failure
Strategy 3: pydub      (temp file; handles WebM/OGG via ffmpeg directly)
     ↓ on failure
Strategy 4: torchaudio (temp file; handles WAV/FLAC/MP3 with backend)
     ↓ all fail
HTTPException(400) with concatenated error diagnostics
```

A single temp file is created (reused across strategies 2–4) and always deleted in a `finally` block regardless of success or failure.

### Validation pipeline

Every uploaded file passes 15 sequential checks before reaching the model:

```
Raw bytes:    empty check → size limit (10 MB)
Decoded:      sample rate validity → empty array → 1D shape → NaN/Inf → silence (peak < 1e-4)
Preprocessed: invalid shape → NaN/Inf → minimum duration (0.3 s) → post-normalize silence
Embedding:    shape [1, 256] → all-finite values
Model output: scalar → finite
```

HTTP status codes are semantically accurate: `400` for bad input, `413` for oversized uploads, `422` for unprocessable audio content, `500` for internal model failures, `503` for model not loaded.

### Response schema

```json
{
  "score":          72.45,
  "label":          "Good",
  "similarity":     0.8312,
  "ref_duration":   0.875,
  "child_duration": 0.950
}
```

The `label` field uses a binary policy: **"Good"** (score ≥ 65.0) or **"Try Again"** (score < 65.0). This is intentionally simpler than the five-tier label in the CLI inference tool, appropriate for real-time child-facing feedback where a clear pass/fail signal is more actionable than a nuanced gradient.

### CORS and environment configuration

All operational parameters are environment-variable driven — checkpoint path, device, upload size limit, minimum duration, CORS origins, host, and port. This allows the same codebase to serve development, staging, and production environments without code changes.

---

## 9. Evaluation Metrics and Results

### Evaluation methodology

The evaluation script (`eval.py`) measures model performance on a held-out validation dataset constructed with `augment=False` and `shuffle=False` for deterministic results. The evaluation uses **raw sigmoid probabilities** (not production-calibrated scores) to measure intrinsic model discriminability independent of calibration choices.

### Threshold sweep

Nine decision thresholds from 0.50 to 0.90 (step 0.05) are evaluated:

```
## Threshold | Accuracy | F0.5 | F1 | Precision | Recall | AUC
```

For each threshold, the following metrics are computed:

| Metric | Formula | Interpretation |
|---|---|---|
| Accuracy | `(TP+TN)/N` | Overall correctness |
| F1 | `2PR/(P+R)` | Balanced precision-recall harmonic mean |
| **F0.5** | `1.25PR/(0.25P+R)` | **Primary metric** — precision-weighted |
| Precision | `TP/(TP+FP)` | Of predicted positives, fraction correct |
| Recall | `TP/(TP+FN)` | Of actual positives, fraction recovered |
| AUC | Mann-Whitney U | Threshold-independent discrimination |

### Why F0.5 is the primary metric

F0.5 (β = 0.5) weights precision twice as heavily as recall. The asymmetry is domain-motivated:

- **False positive** (scoring incorrect pronunciation as "Good"): the child receives no corrective feedback and reinforces the wrong articulation — **directly harmful to learning**
- **False negative** (scoring correct pronunciation as "Try Again"): the child repeats unnecessarily — mildly frustrating but not harmful

The best threshold is selected by lexicographic maximization of **(F0.5, precision, accuracy)**, encoding this value judgment formally in the evaluation criterion.

### AUC implementation

AUC is computed from scratch using the Wilcoxon-Mann-Whitney rank-sum formulation, eliminating the scikit-learn dependency:

```
AUC = (Σ rank(positive_i) - n_pos × (n_pos + 1) / 2) / (n_pos × n_neg)
```

Ties in predicted probabilities are handled by assigning each tied group the average of the ranks it would occupy, producing the same result as sklearn's `roc_auc_score`.

### Collapse detection

The original model collapsed to AUC ≈ 0.5 and constant predictions (pred_std ≈ 0). AUC serves as the primary collapse indicator: an AUC near 0.5 means the model performs at chance. The training loop monitors this during training; `eval.py` confirms it on the final checkpoint.

---

## 10. File-by-File Summary

| File | Role | Key responsibility |
|---|---|---|
| `encoder.py` | Acoustic backbone | Wraps HuggingFace Wav2Vec2; manages freeze/unfreeze protocol; masked mean pooling; projection MLP; L2-normalized embedding |
| `model.py` | Model hub | Defines `AttentivePooling`, `ScoringHead`, and `PronunciationScoringModel`; documents and fixes four collapse root causes; implements two-phase training protocol; calibrated inference scoring |
| `dataset.py` | Data pipeline | Generates semantically correct Siamese pairs; implements augmentation; auto-detects SC/TORGO formats; epoch reshuffling; collation with padding and masks |
| `infer.py` | Inference wrapper | Loads checkpoint with compatibility filtering; multi-format preprocessing; calibrated scoring formula; confidence multiplier; CLI entry point |
| `api.py` | HTTP server | FastAPI application; four-strategy audio decoder; 15-check validation pipeline; threading-safe model singleton; binary label policy; structured logging |
| `eval.py` | Offline evaluation | Threshold sweep; custom AUC; F0.5-optimized threshold selection; raw logit evaluation (bypass calibration); confusion matrix reporting |

---

## 11. Challenges and Fixes

### Challenge 1 — Label noise causing model collapse

**Problem:** Pairing SpeechCommands references with random TORGO children and assigning label=1 produced contradictory training signal. The same word pair (e.g., "cat" + "dog") appeared as both positive and negative in the same training loop. The model converged to constant output (the dataset mean), a well-known failure mode for contradictory labels.

**Indicators:** pred_std ≈ 0, training AUC ≈ 0.5, loss converges to binary cross-entropy of a constant.

**Fix:** Redesigned the entire pair generation strategy. TORGO files are used exclusively as negative children (statistically safe — ~97% chance of word mismatch). Positive pairs are SC-only, guaranteed same word, different recording.

---

### Challenge 2 — Dead neurons in the ScoringHead

**Problem:** The original `ScoringHead` used `LayerNorm + GELU` activations on L2-normalized inputs. At random initialization, all hidden neurons are near zero after LayerNorm maps them to N(0,1), and GELU's smooth non-linearity kills approximately 50% of activations per layer. With two such layers, less than 25% of neurons contribute any gradient to the output. The head collapses to near-constant output within 1–2 batches.

**Fix:** Replaced `LayerNorm + GELU` with `ReLU`. Applied Kaiming He initialization (`kaiming_normal_`, mode=`fan_in`, nonlinearity=`relu`) to all linear layers in the head, setting initial weight variance to `2/fan_in` — appropriate for ReLU and ensuring gradients flow from step 1.

---

### Challenge 3 — Near-zero encoder gradient from CosineEmbeddingLoss

**Problem:** With `cos_margin = 0.3` and `cos_weight = 0.2`, a randomly initialized encoder produces cosine similarity ≈ 0 between any two embeddings. For negative pairs, `CosineEmbeddingLoss = max(0, cos - margin) = max(0, 0 - 0.3) = 0`. Negative pairs provided **zero gradient** to the encoder from the very first training step. Only positive pairs provided signal, and the small `cos_weight = 0.2` made even that signal negligible.

**Fix:** Set `cos_margin = 0.0` (loss is non-zero from step 1 for all pairs) and `cos_weight = 1.5` (strong cosine signal from the beginning of training).

---

### Challenge 4 — Catastrophic forgetting of pretrained features

**Problem:** Training all 94M Wav2Vec2 transformer parameters simultaneously with the 460K-parameter head at the same learning rate caused the transformer to overfit rapidly to the small dataset, destroying the general phonetic representations acquired during self-supervised pretraining.

**Fix:** Phase 1 freezes the entire transformer. Phase 2 unfreezes only the top N layers with a 20× reduced learning rate (5% of head LR). This allows fine-tuning of high-level phonetic representations without disturbing the low-level acoustic features encoded in lower layers.

---

### Challenge 5 — LR decay during critical warmup

**Problem:** Cosine annealing applied from epoch 1 reduced the learning rate during the warmup phase, when the head most needs a strong signal to establish useful gradients. Low LR early slows the head's convergence, delaying the point at which the encoder can safely begin receiving gradient updates.

**Fix:** Flat learning rate during `warmup_epochs`, then cosine decay applied only from Phase 2 onward.

---

### Challenge 6 — Variable-length audio in batches

**Problem:** Audio waveforms in a training batch have different lengths (due to natural duration variation and speed augmentation). PyTorch's default collation cannot stack variable-length tensors.

**Fix:** `collate_fn` zero-pads all waveforms to the batch maximum length and constructs boolean attention masks (`True` = real signal, `False` = padding). The encoder's `_feature_attention_mask()` downsamples this mask to match the Wav2Vec2 hidden state time dimension, preventing padding frames from contributing to the mean-pooled embedding.

---

### Challenge 7 — Inference vs. evaluation scoring formula discrepancy

**Problem:** The calibrated `score_from_embeddings()` (with temperature and cosine margin penalty) was initially used in evaluation, conflating calibration choices with model quality. Changing calibration parameters would change evaluation metrics even if the underlying model had not changed.

**Fix:** `eval.py` uses `logit_from_embeddings()` + raw `sigmoid()`, measuring the model's intrinsic discriminative ability. Calibration is evaluated separately.

---

## 12. Limitations

### Data limitations

- **No true word-level TORGO labels:** TORGO cannot be used as positive pairs without word-level alignment. The dysarthric speech quality it represents is approximated through augmentation, which may not capture all real dysarthric acoustic patterns.
- **SpeechCommands vocabulary:** The system is trained on ~35 words. Generalization to arbitrary vocabulary requires either a much larger vocabulary in training or a different positive pair strategy.
- **No real child speech in training:** SpeechCommands consists of adult speakers. The positive pairs do not include genuine child pronunciation variability; this is partially compensated by augmentation but remains a distribution mismatch.

### Model limitations

- **Utterance-level scoring only:** The model compares complete utterances. Phoneme-level or syllable-level localization of pronunciation errors is not supported.
- **No confidence estimate:** Scoring produces a point estimate with no uncertainty interval. Short or noisy audio produces a score without any indication of reliability.
- **Hardcoded calibration constants:** `score_temperature = 1.35`, `cosine_margin = 0.72`, `cosine_penalty_weight = 3.0`, `fallback_margin = 0.94` are fixed attributes. These may need re-tuning for different languages, age groups, or acoustic conditions.

### System limitations

- **Single-threaded inference:** The threading lock in `api.py` serializes all requests. Under high concurrency, requests queue at the lock. The server processes one scoring call at a time.
- **Evaluation threshold does not transfer directly to production:** `eval.py` measures on raw sigmoid [0,1]; `api.py` uses a calibrated [0,100] scale. The recommended raw threshold and the production `THRESHOLD = 65.0` are related but not equivalent.
- **No authentication:** The `/score` endpoint is open. There is no API key validation, rate limiting, or per-user access control.
- **No phoneme-level feedback:** The system tells a child "Good" or "Try Again" but cannot indicate which phoneme was mispronounced.

---

## 13. Future Improvements

### Short-term

- **Phoneme-level scoring:** Align reference and child audio at the phoneme level using a forced aligner (e.g., Montreal Forced Aligner) and score each phoneme segment independently, producing actionable per-phoneme feedback.

- **Learned calibration:** Replace hardcoded temperature and margin constants with a learned post-hoc calibration step (Platt scaling or isotonic regression on a validation set), enabling automatic re-calibration when the checkpoint changes.

- **Waveform cache:** Pre-compute and cache decoded waveforms for the training set to eliminate repeated librosa decoding from disk, reducing Phase 1 training time significantly.

- **Batch scoring API:** Add a `score_batch()` endpoint that stacks multiple pairs into a single model forward pass, enabling higher throughput for bulk evaluation use cases.

### Medium-term

- **Real child speech data:** Collect and incorporate word-level labeled child speech recordings to close the adult-to-child distribution gap in the positive training pairs.

- **Attentive pooling integration:** Replace masked mean pooling in the encoder with the `AttentivePooling` module (already defined in `model.py`) to allow the model to focus on phonetically informative frames rather than treating silence and noise frames equally.

- **Layer-wise progressive unfreezing:** Instead of unfreezing the top N transformer layers simultaneously in Phase 2, unfreeze one layer at a time from the top down across sub-epochs, providing finer control over catastrophic forgetting.

- **Calibration curve evaluation:** Add per-probability-decile calibration reporting to `eval.py` to assess whether predicted probabilities are well-calibrated — a prerequisite for interpreting the score as a probability.

### Long-term

- **Multi-language support:** Replace `facebook/wav2vec2-base` (English-focused) with a multilingual model (`facebook/wav2vec2-large-xlsr-53`) and expand the training vocabulary beyond SpeechCommands.

- **Sentence-level scoring:** Extend the system from single-word to full-sentence pronunciation assessment using segmentation and alignment to produce both overall and per-word scores.

- **Online fine-tuning:** Implement a feedback loop where therapist corrections on scored pairs are used to fine-tune the model incrementally, personalizing the scoring system to a specific child's developmental trajectory.

- **Uncertainty quantification:** Add Monte Carlo dropout inference to produce confidence intervals alongside point estimates, allowing the system to flag low-confidence scores for human review.

---

## 14. Conclusion

This system demonstrates that automated pronunciation scoring for child speech is a tractable machine learning problem, but one that demands careful engineering at every layer of the stack.

The most significant technical contribution is the **systematic identification and resolution of four compounding training collapse causes** — dead neurons, near-zero encoder gradients, catastrophic forgetting, and premature LR decay — each of which individually produces a failure mode, and all of which were present simultaneously in the original system. The corrected design (ReLU + Kaiming init in the scoring head, aggressive cosine loss weights, two-phase freeze/fine-tune protocol, flat warmup LR) converts a model that outputs constant predictions into one with measurable discriminative ability.

The **dataset redesign** from contradictory SC+TORGO positive pairs to guaranteed-correct SC+SC pairs is arguably the most impactful single change. No amount of model architecture improvement can overcome training data where the same acoustic pair is labeled both 0 and 1.

The **scoring formula** — combining a learned MLP comparison head with two independent geometric confidence safeguards (cosine margin penalty and confidence multiplier) — addresses the fundamental limitation of pure cosine scoring while maintaining interpretability. The score reflects both the model's learned pronunciation judgment and a geometric sanity check on the embedding space.

The **production API** wraps this pipeline in a robust, format-agnostic HTTP service with a 15-check validation pipeline, four-strategy audio decoder, and structured diagnostics logging — enabling deployment in real speech therapy applications without requiring clients to manage audio format details.

The system's primary remaining limitation is the absence of real child speech in training. Bridging this gap — through data collection, forced-alignment of TORGO recordings, or transfer from real-world therapy session recordings — represents the highest-value next step toward a clinically deployable pronunciation scoring system.

---

*Documentation compiled from individual file analysis of: `encoder.py`, `model.py`, `dataset.py`, `infer.py`, `api.py`, `eval.py`*
