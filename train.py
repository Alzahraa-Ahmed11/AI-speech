"""
train.py — wav2vec2 Pronunciation Scoring — training script

Two-phase anti-collapse protocol
─────────────────────────────────
  Phase 1 (warmup_epochs): wav2vec2 transformer frozen; only the projection
    head and MLP scoring head are trained at full LR.
  Phase 2 (remaining epochs): transformer unfrozen with differential LR
    (head = lr, transformer = lr × 0.05) + cosine decay to end of training.

Pair strategy
─────────────
  Positive (1.0): same-word, cross-speaker SC pairs (+ light augmentation)
  Negative (0.0): different-word SC pairs; hard negatives are phonetically
    similar words (hard_negative_frac) + same-word heavy-corruption pairs
    (same_word_bad_neg_frac)

Primary metric: ROC-AUC
  0.50 = random  |  0.65 = weak  |  0.75 = decent  |  0.85+ = strong

Anti-collapse signals (checked every epoch):
  pred_std < 0.05  → output collapse
  sep      < 0.05  → model not discriminating positive/negative pairs
  cos_gap  < 0.10  → encoder embeddings not separating
"""

import os
import json
import logging
import argparse
import math
import random
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    import warnings
    warnings.warn(
        "scikit-learn not found — using built-in AUC fallback. "
        "Install with: pip install scikit-learn",
        RuntimeWarning,
        stacklevel=2,
    )

    def roc_auc_score(y_true, y_score):  # type: ignore[misc]
        y_true  = np.asarray(y_true,  dtype=np.float32)
        y_score = np.asarray(y_score, dtype=np.float32)
        n_pos = int(y_true.sum())
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        idx = np.argsort(-y_score)
        yt  = y_true[idx]
        tps = np.cumsum(yt)
        fps = np.cumsum(1 - yt)
        tpr = np.concatenate([[0.0], tps / n_pos])
        fpr = np.concatenate([[0.0], fps / n_neg])
        return float(np.trapz(tpr, fpr))

from model import PronunciationScoringModel
from dataset import PronunciationPairDataset, collate_fn

CHECKPOINT_ARCH = "wav2vec2_pronunciation_scorer_v2"
REQUIRED_CHECKPOINT_PREFIXES = (
    "encoder.speech_model.",
    "encoder.proj.",
    "scorer.net.",
)


def default_data_dirs() -> List[str]:
    candidates = [
        ["DATA_SET/SpeechCommands", "DATA_SET/torgo"],
        ["data/SpeechCommands", "data/torgo"],
        ["data/mini", "data/raw"],
    ]
    for group in candidates:
        if any(os.path.exists(p) for p in group):
            return [p for p in group if os.path.exists(p)]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("logs",        exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class StableLoss(nn.Module):
    """
    BCE + cosine contrastive loss.

    cos_weight=1.5 makes the cosine term the dominant signal so the encoder
    receives strong gradient from the start. cos_margin=0.4 pushes negative
    pair cosines below that value without zeroing their gradient at init.
    negative_bce_weight upweights false positives to reduce over-prediction.
    """

    def __init__(
        self,
        cos_margin: float = 0.4,
        cos_weight: float = 1.5,
        negative_bce_weight: float = 1.35,
    ):
        super().__init__()
        self.cos_margin = cos_margin
        self.cos_weight = cos_weight
        self.negative_bce_weight = negative_bce_weight

    def forward(
        self,
        logits:    torch.Tensor,
        ref_emb:   torch.Tensor,
        child_emb: torch.Tensor,
        target:    torch.Tensor,
    ) -> torch.Tensor:
        bce_each = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce_weights = torch.where(
            target < 0.5,
            torch.full_like(target, self.negative_bce_weight),
            torch.ones_like(target),
        )
        bce_loss = (bce_each * bce_weights).mean()

        ref_emb   = F.normalize(ref_emb,   p=2, dim=-1, eps=1e-8)
        child_emb = F.normalize(child_emb, p=2, dim=-1, eps=1e-8)
        cos = F.cosine_similarity(ref_emb, child_emb, dim=-1, eps=1e-8)

        pos_mask = target >= 0.5
        neg_mask = ~pos_mask
        pos_loss = (1.0 - cos[pos_mask]).mean() if pos_mask.any() else cos.new_tensor(0.0)
        neg_loss = (
            F.relu(cos[neg_mask] - self.cos_margin).mean()
            if neg_mask.any()
            else cos.new_tensor(0.0)
        )
        return bce_loss + self.cos_weight * (pos_loss + neg_loss)


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class WarmupFlatCosineScheduler:
    """Flat LR during warmup, then cosine decay for fine-tuning."""

    def __init__(
        self,
        optimizer:     optim.Optimizer,
        warmup_epochs: int,
        total_epochs:  int,
        eta_min:       float = 1e-6,
        warmup_decay:  float = 0.95,
    ):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.eta_min       = eta_min
        self.warmup_decay  = warmup_decay
        self._epoch        = 0
        self._base_lrs     = [pg["lr"] for pg in optimizer.param_groups]

    def step(self):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            decay = self.warmup_decay ** max(self._epoch - 1, 0)
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = max(self.eta_min, base_lr * decay)
        else:
            progress = (self._epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
                pg["lr"] = self.eta_min + (base_lr - self.eta_min) * cos_factor

    def get_last_lr(self) -> List[float]:
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:     PronunciationScoringModel,
    optimizer: optim.Optimizer,
    epoch:     int,
    metrics:   Dict,
    path:      str,
    scaler=None,
) -> None:
    payload = {
        "architecture":         CHECKPOINT_ARCH,
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":              metrics,
        "config": {
            "embed_dim": model.embed_dim,
            "encoder_model": getattr(model.encoder, "model_name", "unknown"),
        },
        "scaler_state_dict": (
            scaler.state_dict()
            if scaler is not None and scaler.is_enabled()
            else None
        ),
    }
    dir_name = os.path.dirname(os.path.abspath(path))
    if os.path.exists(path):
        archive_dir = os.path.join(dir_name, "archive")
        os.makedirs(archive_dir, exist_ok=True)
        stem, ext = os.path.splitext(os.path.basename(path))
        archive_path = os.path.join(archive_dir, f"{stem}_preserved_before_ep{epoch}{ext}")
        suffix = 1
        while os.path.exists(archive_path):
            archive_path = os.path.join(
                archive_dir, f"{stem}_preserved_before_ep{epoch}_{suffix}{ext}"
            )
            suffix += 1
        shutil.copy2(path, archive_path)
        logger.info("Preserved existing checkpoint -> %s", archive_path)

    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise
    logger.info("Saved checkpoint → %s", path)

def _is_new_arch_checkpoint(checkpoint_state: Dict[str, torch.Tensor]) -> bool:
    return all(
        any(key.startswith(prefix) for key in checkpoint_state)
        for prefix in REQUIRED_CHECKPOINT_PREFIXES
    )


def _has_complete_compatible_state(
    model_state: Dict[str, torch.Tensor],
    compatible_state: Dict[str, torch.Tensor],
) -> bool:
    required_keys = [
        key
        for key in model_state
        if key.startswith(REQUIRED_CHECKPOINT_PREFIXES)
    ]
    return all(key in compatible_state for key in required_keys)


def load_checkpoint(
    model:     PronunciationScoringModel,
    optimizer: Optional[optim.Optimizer],
    path:      str,
    scaler=None,
) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_state = ckpt["model_state_dict"]
    if ckpt.get("architecture") != CHECKPOINT_ARCH:
        logger.warning(
            "Ignoring incompatible legacy checkpoint %s; starting a fresh wav2vec2 run.",
            path,
        )
        return 0

    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in checkpoint_state.items()
        if key in model_state and value.shape == model_state[key].shape
    }
    if not _has_complete_compatible_state(model_state, compatible_state):
        logger.warning(
            "Checkpoint %s does not contain a complete wav2vec2 encoder/scoring head; starting fresh.",
            path,
        )
        return 0

    skipped_keys = sorted(set(checkpoint_state) - set(compatible_state))
    result = model.load_state_dict(compatible_state, strict=False)
    if skipped_keys:
        logger.warning("Checkpoint keys incompatible with current model (ignored): %s", skipped_keys)
    if result.unexpected_keys:
        logger.warning("Checkpoint keys not in model (ignored): %s", result.unexpected_keys)
    if result.missing_keys:
        logger.warning("Model keys not in checkpoint (random init): %s", result.missing_keys)

    logger.info(
        "Loaded %d/%d compatible model tensors from %s",
        len(compatible_state),
        len(model_state),
        path,
    )

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            logger.info("Restored optimizer state from %s", path)
        except (ValueError, KeyError) as exc:
            logger.warning(
                "Could not restore optimizer state (param groups likely changed): %s", exc,
            )
    elif optimizer is not None:
        logger.info("Checkpoint has no optimizer state — keeping freshly built optimizer.")

    if scaler is not None:
        sd = ckpt.get("scaler_state_dict")
        if sd is not None:
            try:
                scaler.load_state_dict(sd)
                logger.info("Restored GradScaler state from %s", path)
            except Exception as exc:
                logger.warning("Could not restore GradScaler state: %s", exc)

    epoch = ckpt.get("epoch", 0)
    logger.info("Loaded checkpoint from epoch %d", epoch)
    return epoch


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate(
    model:     PronunciationScoringModel,
    loader:    DataLoader,
    device:    torch.device,
    criterion: nn.Module,
    threshold: float = 0.5,
    use_amp:   bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    Run the model over *loader* and return (mean_loss, metrics).

    Metrics keys: auc, acc, sep, pos_mean, neg_mean, pred_std,
                  cos_pos, cos_neg, _preds, _labels
    """
    model.eval()

    total_loss: float       = 0.0
    all_logits: List[float] = []
    all_labels: List[float] = []
    all_cos:    List[float] = []

    for ref, child, ref_mask, child_mask, labels in loader:
        ref        = ref.to(device)
        child      = child.to(device)
        ref_mask   = ref_mask.to(device)
        child_mask = child_mask.to(device)
        labels_dev = labels.to(device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            ref_emb   = model.encode(ref,   ref_mask)
            child_emb = model.encode(child, child_mask)
            logits    = model.logit_from_embeddings(ref_emb, child_emb)
            cos       = F.cosine_similarity(ref_emb, child_emb, dim=-1, eps=1e-8)
            loss      = criterion(logits, ref_emb, child_emb, labels_dev)

        total_loss += loss.item()
        all_logits.extend(logits.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_cos.extend(cos.cpu().tolist())

    avg_loss  = total_loss / max(len(loader), 1)
    preds_np  = torch.sigmoid(
        torch.tensor(all_logits, dtype=torch.float32)
    ).numpy().astype(np.float32)
    labels_np = np.array(all_labels, dtype=np.float32)
    cos_np    = np.array(all_cos,    dtype=np.float32)

    pos_mask = labels_np == 1.0
    neg_mask = labels_np == 0.0

    pos_mean = float(preds_np[pos_mask].mean()) if pos_mask.any() else float("nan")
    neg_mean = float(preds_np[neg_mask].mean()) if neg_mask.any() else float("nan")
    sep = (
        pos_mean - neg_mean
        if not (np.isnan(pos_mean) or np.isnan(neg_mean))
        else float("nan")
    )

    try:
        auc = float(roc_auc_score(labels_np, preds_np))
    except ValueError:
        auc = float("nan")

    binary = (preds_np >= threshold).astype(int)
    acc    = float((binary == labels_np.astype(int)).mean())

    return avg_loss, {
        "auc":      auc,
        "acc":      acc,
        "sep":      sep,
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "pred_std": float(preds_np.std()),
        "cos_pos":  float(cos_np[pos_mask].mean()) if pos_mask.any() else float("nan"),
        "cos_neg":  float(cos_np[neg_mask].mean()) if neg_mask.any() else float("nan"),
        "_preds":   preds_np,
        "_labels":  labels_np,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch log helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cos_gap(metrics: Dict[str, Any]) -> float:
    if np.isnan(metrics["cos_pos"]) or np.isnan(metrics["cos_neg"]):
        return float("nan")
    return float(metrics["cos_pos"] - metrics["cos_neg"])


def _log_metrics(
    phase:        str,
    epoch:        int,
    total_epochs: int,
    loss:         float,
    metrics:      Dict[str, Any],
    lr:           float,
    *,
    is_best:   bool = False,
    phase_tag: str  = "",
) -> None:
    best_tag = "  ★ NEW BEST" if is_best else ""
    logger.info(
        "Ep %d/%d [%s%s]  loss=%.4f  lr=%.2e | "
        "AUC=%.3f  acc=%.3f  sep=%.3f  cos_gap=%.3f  pred_std=%.4f%s",
        epoch, total_epochs, phase.upper(), phase_tag,
        loss, lr,
        metrics["auc"], metrics["acc"], metrics["sep"],
        _cos_gap(metrics), metrics["pred_std"],
        best_tag,
    )


def _check_collapse(phase: str, epoch: int, metrics: Dict[str, Any]) -> None:
    if metrics["pred_std"] < 0.05:
        logger.warning(
            "COLLAPSE [%s ep%d]: pred_std=%.4f — outputs near-identical.",
            phase, epoch, metrics["pred_std"],
        )
    sep = metrics["sep"]
    if not np.isnan(sep) and sep < 0.05 and epoch >= 3:
        logger.warning(
            "NO SEPARATION [%s ep%d]: sep=%.4f — model not discriminating.",
            phase, epoch, sep,
        )
    cg = _cos_gap(metrics)
    if not np.isnan(cg) and cg < 0.1 and epoch >= 3:
        logger.warning(
            "LOW COS_GAP [%s ep%d]: cos_gap=%.4f — encoder embeddings not separating.",
            phase, epoch, cg,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def train(
    data_dirs:             List[str],
    epochs:                int                 = 30,
    batch_size:            int                 = 16,
    lr:                    float               = 2.5e-4,
    warmup_epochs:         int                 = 5,
    resume_from:           Optional[str]       = None,
    val_dirs:              Optional[List[str]] = None,
    val_split:             float               = 0.15,
    pairs_per_word:        int                 = 60,
    embed_dim:             int                 = 256,
    seed:                  Optional[int]       = 42,
    cos_weight:            float               = 1.6,
    cos_margin:            float               = 0.25,
    negative_bce_weight:   float               = 1.75,
    hard_negative_frac:    float               = 0.75,
    same_word_bad_neg_frac: float              = 0.20,
    finetune_layers:       int                 = 4,
    mixed_precision:       bool                = True,
    num_workers:           int                 = 0,
) -> None:
    """
    Train the pronunciation scoring model.

    Phase 1 (warmup_epochs): transformer frozen, head only.
    Phase 2 (remaining):     transformer unfrozen, differential LR.
    Best checkpoint selected by validation AUC.
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(mixed_precision and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logger.info("Device: %s  mixed_precision=%s", device, use_amp)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Datasets ──────────────────────────────────────────────────────────────
    logger.info("Building training dataset …")
    train_ds = PronunciationPairDataset(
        data_dirs,
        augment=True,
        pairs_per_word=pairs_per_word,
        hard_negative_frac=hard_negative_frac,
        same_word_bad_neg_frac=same_word_bad_neg_frac,
        seed=seed,
    )

    if not val_dirs:
        logger.warning(
            "--val_dirs not provided; validation draws from the same audio files "
            "as training. Pass --val_dirs with held-out speakers for reliable metrics."
        )
    logger.info("Building validation dataset …")
    _val_ppw = max(5, pairs_per_word // 4)
    val_ds = PronunciationPairDataset(
        val_dirs if val_dirs else data_dirs,
        augment=False,
        pairs_per_word=_val_ppw,
        hard_negative_frac=hard_negative_frac,
        same_word_bad_neg_frac=0.0,
        seed=0,
    )
    logger.info("Sizes: train=%d pairs  val=%d pairs", len(train_ds), len(val_ds))

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PronunciationScoringModel(
        embed_dim=embed_dim, freeze_feature_extractor=True,
    ).to(device)
    trainable_params, total_params = _count_trainable_params(model)
    logger.info(
        "Model params: trainable=%d / total=%d during warmup",
        trainable_params,
        total_params,
    )

    # ── Optimizer (Phase 1: head only) ────────────────────────────────────────
    head_params = (
        list(model.encoder.head_parameters()) + list(model.scorer.parameters())
    )
    optimizer = optim.AdamW(head_params, lr=lr, betas=(0.9, 0.98), weight_decay=5e-5)

    scheduler = WarmupFlatCosineScheduler(
        optimizer, warmup_epochs=warmup_epochs, total_epochs=epochs, eta_min=1e-6,
    )

    criterion = StableLoss(
        cos_margin=cos_margin,
        cos_weight=cos_weight,
        negative_bce_weight=negative_bce_weight,
    )

    start_epoch        = 0
    best_val_auc       = 0.0
    best_summary: Optional[Dict[str, Any]] = None
    phase2_started     = False
    low_cos_gap_epochs = 0

    logger.info(
        "Hyperparameters: lr=%.2e  warmup=%d  cos_weight=%.1f  cos_margin=%.1f  "
        "neg_bce_weight=%.2f  embed_dim=%d  batch=%d",
        lr, warmup_epochs, cos_weight, cos_margin, negative_bce_weight, embed_dim, batch_size,
    )
    logger.info(
        "Fine-tuning: trainable wav2vec2 layers after warmup=%s  num_workers=%d",
        "all" if finetune_layers < 0 else finetune_layers,
        num_workers,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    checkpoint_path = resume_from or "checkpoints/last_model.pt"
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, scaler=scaler)
        best_ckpt = "checkpoints/best_model.pt"
        if start_epoch > 0 and os.path.exists(best_ckpt):
            try:
                bckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
                m = bckpt.get("metrics", {})
                best_val_auc = float(m.get("val_auc") or 0.0)
                logger.info("Restored best val_auc=%.4f from %s", best_val_auc, best_ckpt)
            except Exception:
                pass
        if start_epoch >= warmup_epochs:
            logger.info("Resume: past warmup — entering phase 2 immediately.")
            model.unfreeze_encoder(
                trainable_layers=None if finetune_layers < 0 else finetune_layers
            )
            trainable_params, total_params = _count_trainable_params(model)
            logger.info("Model params after resume unfreeze: trainable=%d / total=%d", trainable_params, total_params)
            resumed_lr = min(lr, 1e-4 if start_epoch >= 20 else lr * 0.5)
            logger.info("Resume fine-tune LR: requested=%.2e  effective=%.2e", lr, resumed_lr)
            optimizer = optim.AdamW(
                model.get_param_groups(resumed_lr, encoder_lr_scale=0.05),
                betas=(0.9, 0.98), weight_decay=5e-5,
            )
            scheduler = WarmupFlatCosineScheduler(
                optimizer, warmup_epochs=0, total_epochs=max(1, epochs), eta_min=1e-6,
            )
            phase2_started = True

    history: List[Dict] = []

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, start_epoch + epochs):
        ep_num = epoch + 1
        total  = start_epoch + epochs

        # ── Phase transition: unfreeze transformer ─────────────────────────────
        if ep_num > warmup_epochs and not phase2_started:
            logger.info(
                "═══ Phase 2 (ep %d): unfreezing transformer  "
                "head=%.2e  transformer=%.2e ═══",
                ep_num, lr, lr * 0.05,
            )
            model.unfreeze_encoder(
                trainable_layers=None if finetune_layers < 0 else finetune_layers
            )
            trainable_params, total_params = _count_trainable_params(model)
            logger.info("Model params after unfreeze: trainable=%d / total=%d", trainable_params, total_params)
            optimizer = optim.AdamW(
                model.get_param_groups(lr, encoder_lr_scale=0.05),
                betas=(0.9, 0.98), weight_decay=5e-5,
            )
            remaining = (start_epoch + epochs) - epoch
            scheduler = WarmupFlatCosineScheduler(
                optimizer, warmup_epochs=0, total_epochs=max(1, remaining), eta_min=1e-6,
            )
            phase2_started = True

        # ── Training pass ─────────────────────────────────────────────────────
        train_ds.reshuffle()
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model.train()
        total_loss     = 0.0
        train_preds_:  List[float] = []
        train_labels_: List[float] = []
        train_cos_:    List[float] = []
        grad_norms_:   List[float] = []
        phase_tag = " WARMUP" if ep_num <= warmup_epochs else " FINETUNE"

        for ref, child, ref_mask, child_mask, labels in train_loader:
            ref        = ref.to(device)
            child      = child.to(device)
            ref_mask   = ref_mask.to(device)
            child_mask = child_mask.to(device)
            labels     = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                ref_emb   = model.encode(ref,   ref_mask)
                child_emb = model.encode(child, child_mask)
                logits    = model.logit_from_embeddings(ref_emb, child_emb)
                loss      = criterion(logits, ref_emb, child_emb, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            _trainable = [p for p in model.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(_trainable, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            grad_norms_.append(float(grad_norm))

            with torch.no_grad():
                preds = torch.sigmoid(logits)
                cos   = F.cosine_similarity(ref_emb, child_emb, dim=-1, eps=1e-8)
                train_preds_.extend(preds.detach().cpu().tolist())
                train_labels_.extend(labels.cpu().tolist())
                train_cos_.extend(cos.detach().cpu().tolist())

        scheduler.step()
        avg_train_loss = total_loss / max(len(train_loader), 1)
        current_lr     = optimizer.param_groups[0]["lr"]
        max_gnorm      = max(grad_norms_) if grad_norms_ else 0.0
        mean_gnorm     = sum(grad_norms_) / len(grad_norms_) if grad_norms_ else 0.0

        # ── Train metrics ─────────────────────────────────────────────────────
        tp_np = np.array(train_preds_,  dtype=np.float32)
        tl_np = np.array(train_labels_, dtype=np.float32)
        tc_np = np.array(train_cos_,    dtype=np.float32)
        pos_mask_t = tl_np == 1.0
        neg_mask_t = tl_np == 0.0

        pos_m = float(tp_np[pos_mask_t].mean()) if pos_mask_t.any() else float("nan")
        neg_m = float(tp_np[neg_mask_t].mean()) if neg_mask_t.any() else float("nan")
        t_sep = (pos_m - neg_m) if not (np.isnan(pos_m) or np.isnan(neg_m)) else float("nan")

        try:
            t_auc = float(roc_auc_score(tl_np, tp_np))
        except ValueError:
            t_auc = float("nan")

        train_metrics: Dict[str, Any] = {
            "auc":      t_auc,
            "acc":      float(((tp_np >= 0.5).astype(int) == tl_np.astype(int)).mean()),
            "sep":      t_sep,
            "pos_mean": pos_m,
            "neg_mean": neg_m,
            "pred_std": float(tp_np.std()),
            "cos_pos":  float(tc_np[pos_mask_t].mean()) if pos_mask_t.any() else float("nan"),
            "cos_neg":  float(tc_np[neg_mask_t].mean()) if neg_mask_t.any() else float("nan"),
            "_preds":   tp_np,
            "_labels":  tl_np,
        }

        _log_metrics("TRAIN", ep_num, total, avg_train_loss, train_metrics, current_lr,
                     phase_tag=phase_tag)
        logger.info("  grad_norm  max=%.4f  mean=%.4f  (clipped at 1.0)", max_gnorm, mean_gnorm)
        _check_collapse("TRAIN", ep_num, train_metrics)

        # ── Validation pass ───────────────────────────────────────────────────
        val_loss, val_metrics = evaluate(model, val_loader, device, criterion, use_amp=use_amp)
        model.train()

        val_auc = val_metrics["auc"]
        is_best = (not np.isnan(val_auc)) and (val_auc > best_val_auc)
        _log_metrics("VAL", ep_num, total, val_loss, val_metrics, current_lr,
                     is_best=is_best, phase_tag=phase_tag)
        _check_collapse("VAL", ep_num, val_metrics)

        # ── Anti-collapse: boost cos_weight if cos_gap stays low ──────────────
        val_cg = _cos_gap(val_metrics)
        if not np.isnan(val_cg) and val_cg < 0.05:
            low_cos_gap_epochs += 1
        else:
            low_cos_gap_epochs = 0

        if low_cos_gap_epochs >= 2:
            old_weight = criterion.cos_weight
            criterion.cos_weight = min(4.0, criterion.cos_weight * 1.25)
            train_ds.adjust_negative_hardness(
                hard_negative_frac=min(0.90, train_ds.hard_negative_frac + 0.10),
                same_word_bad_neg_frac=min(0.12, train_ds.same_word_bad_neg_frac + 0.02),
                rebuild_pairs=False,
            )
            logger.warning(
                "AUTO ANTI-COLLAPSE ep%d: cos_gap=%.4f for %d epochs; "
                "cos_weight %.2f → %.2f  hard_neg_frac=%.2f",
                ep_num, val_cg, low_cos_gap_epochs,
                old_weight, criterion.cos_weight, train_ds.hard_negative_frac,
            )
            low_cos_gap_epochs = 0

        # ── History record ────────────────────────────────────────────────────
        def _safe(v: float) -> Optional[float]:
            return round(v, 4) if not np.isnan(v) else None

        record: Dict = {
            "epoch":                  ep_num,
            "phase":                  "warmup" if ep_num <= warmup_epochs else "finetune",
            "train_loss":             round(avg_train_loss, 4),
            "train_auc":              _safe(t_auc),
            "train_sep":              _safe(t_sep),
            "train_std":              round(float(tp_np.std()), 4),
            "train_cos_gap":          _safe(_cos_gap(train_metrics)),
            "val_loss":               round(val_loss, 4),
            "val_auc":                _safe(val_metrics["auc"]),
            "val_acc":                round(val_metrics["acc"], 4),
            "val_sep":                _safe(val_metrics["sep"]),
            "val_std":                round(val_metrics["pred_std"], 4),
            "val_cos_gap":            _safe(val_cg),
            "cos_weight":             round(float(criterion.cos_weight), 4),
            "hard_negative_frac":     round(float(train_ds.hard_negative_frac), 4),
            "same_word_bad_neg_frac": round(float(train_ds.same_word_bad_neg_frac), 4),
            "lr":                     round(current_lr, 8),
            "max_grad_norm":          round(max_gnorm, 4),
        }
        history.append(record)

        # ── Checkpoints ───────────────────────────────────────────────────────
        save_checkpoint(model, optimizer, ep_num, record, "checkpoints/last_model.pt", scaler=scaler)

        if is_best:
            best_val_auc = val_auc
            best_summary = record
            save_checkpoint(model, optimizer, ep_num, record, "checkpoints/best_model.pt", scaler=scaler)
            logger.info(
                "★ New best  val_auc=%.4f  acc=%.4f  sep=%.4f  ep=%d",
                val_auc, val_metrics["acc"], val_metrics["sep"], ep_num,
            )

        if ep_num % 5 == 0:
            save_checkpoint(model, optimizer, ep_num, record, f"checkpoints/epoch_{ep_num}.pt", scaler=scaler)

    # ── Final summary ──────────────────────────────────────────────────────────
    with open("logs/history.json", "w") as f:
        json.dump(history, f, indent=2)

    if best_summary is None and history:
        best_summary = max(history, key=lambda r: r.get("val_auc") or 0.0)

    if best_summary:
        logger.info(
            "Best epoch %d: AUC=%.4f  acc=%.4f  sep=%.4f  cos_gap=%.4f  pred_std=%.4f",
            best_summary.get("epoch", 0),
            best_summary.get("val_auc") or float("nan"),
            best_summary.get("val_acc") or float("nan"),
            best_summary.get("val_sep") or float("nan"),
            best_summary.get("val_cos_gap") or float("nan"),
            best_summary.get("val_std") or float("nan"),
        )

    logger.info("Training complete.  Best val_auc=%.4f  (see logs/history.json)", best_val_auc)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train pronunciation scoring model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dirs",      nargs="+", default=None,
                        help="Training data roots (SpeechCommands + TORGO)")
    parser.add_argument("--val_dirs",       nargs="+", default=None,
                        help="Separate validation roots")
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=2.5e-4)
    parser.add_argument("--warmup_epochs",  type=int,   default=5,
                        help="Epochs to keep transformer frozen (phase 1)")
    parser.add_argument("--cos_weight",     type=float, default=1.6,
                        help="Weight of cosine contrastive loss")
    parser.add_argument("--cos_margin",     type=float, default=0.25,
                        help="Negative cosine margin")
    parser.add_argument("--negative_bce_weight", type=float, default=1.75,
                        help="BCE weight for negative labels")
    parser.add_argument("--hard_negative_frac", type=float, default=0.75,
                        help="Fraction of SC negatives from phonetic near misses")
    parser.add_argument("--same_word_bad_neg_frac", type=float, default=0.20,
                        help="Same-word heavy-augmentation negatives")
    parser.add_argument("--finetune_layers", type=int, default=4,
                        help="Number of final wav2vec2 transformer layers to fine-tune after warmup; -1 = all")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable CUDA mixed precision training")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader worker processes")
    parser.add_argument("--resume",         default=None)
    parser.add_argument("--val_split",      type=float, default=0.15)
    parser.add_argument("--pairs_per_word", type=int,   default=60)
    parser.add_argument("--embed_dim",      type=int,   default=256)
    parser.add_argument("--seed",           type=int,   default=42)

    args = parser.parse_args()
    data_dirs = args.data_dirs or default_data_dirs()
    if not data_dirs:
        parser.error(
            "No data roots found. Pass --data_dirs, or create DATA_SET/SpeechCommands."
        )
    logger.info("Using data_dirs=%s", data_dirs)

    train(
        data_dirs              = data_dirs,
        epochs                 = args.epochs,
        batch_size             = args.batch_size,
        lr                     = args.lr,
        warmup_epochs          = args.warmup_epochs,
        cos_weight             = args.cos_weight,
        cos_margin             = args.cos_margin,
        negative_bce_weight    = args.negative_bce_weight,
        hard_negative_frac     = args.hard_negative_frac,
        same_word_bad_neg_frac = args.same_word_bad_neg_frac,
        resume_from            = args.resume,
        val_dirs               = args.val_dirs,
        val_split              = args.val_split,
        pairs_per_word         = args.pairs_per_word,
        embed_dim              = args.embed_dim,
        seed                   = args.seed,
        finetune_layers        = args.finetune_layers,
        mixed_precision        = not args.no_amp,
        num_workers            = args.num_workers,
    )

# TEST GITHUB UPDATE
