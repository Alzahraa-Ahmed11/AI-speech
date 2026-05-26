import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import PronunciationPairDataset, collate_fn
from infer import PronunciationScorer


DEFAULT_THRESHOLDS = [round(float(x), 2) for x in np.arange(0.50, 0.91, 0.05)]


def default_data_dirs() -> List[str]:
    candidates = [
        ["DATA_SET/SpeechCommands", "DATA_SET/torgo"],
        ["data/SpeechCommands", "data/torgo"],
        ["data/mini", "data/raw"],
    ]
    for group in candidates:
        if any(os.path.exists(path) for path in group):
            return [path for path in group if os.path.exists(path)]
    return []


def roc_auc_score_safe(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(probs)
    sorted_probs = probs[order]
    sorted_ranks = np.empty(len(probs), dtype=np.float64)

    start = 0
    while start < len(probs):
        end = start + 1
        while end < len(probs) and sorted_probs[end] == sorted_probs[start]:
            end += 1
        sorted_ranks[start:end] = (start + 1 + end) / 2.0
        start = end

    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks

    pos_ranks = ranks[labels == 1].sum()
    auc = (pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def classification_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = float((preds == labels).mean()) if len(labels) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    beta2 = 0.25
    f0_5 = (
        (1.0 + beta2) * precision * recall / ((beta2 * precision) + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "f1": float(f1),
        "f0_5": float(f0_5),
        "precision": float(precision),
        "recall": float(recall),
        "auc": roc_auc_score_safe(labels, probs),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def print_threshold_report(results: List[Tuple[float, Dict[str, float]]]) -> None:
    print("## Threshold | Accuracy | F0.5 | F1 | Precision | Recall | AUC")
    print("----------------------------------------------------------------")
    for threshold, metrics in results:
        print(
            f"{threshold:>9.2f} | "
            f"{metrics['accuracy']:.2f}     | "
            f"{metrics['f0_5']:.2f}  | "
            f"{metrics['f1']:.2f} | "
            f"{metrics['precision']:.2f}      | "
            f"{metrics['recall']:.2f}   | "
            f"{metrics['auc']:.2f}"
        )


def print_confusion_matrices(results: List[Tuple[float, Dict[str, float]]]) -> None:
    print()
    print("Confusion Matrices")
    print("------------------")
    for threshold, metrics in results:
        print(
            f"Threshold {threshold:.2f}: "
            f"TP={metrics['tp']}, TN={metrics['tn']}, "
            f"FP={metrics['fp']}, FN={metrics['fn']}"
        )


@torch.no_grad()
def run_inference(
    scorer: PronunciationScorer,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    model = scorer.model
    device = scorer.device
    model.eval()

    all_probs: List[float] = []
    all_labels: List[float] = []

    for ref, child, ref_mask, child_mask, labels in loader:
        ref = ref.to(device)
        child = child.to(device)
        ref_mask = ref_mask.to(device)
        child_mask = child_mask.to(device)

        ref_emb = model.encode(ref, ref_mask)
        child_emb = model.encode(child, child_mask)
        logits = model.logit_from_embeddings(ref_emb, child_emb)
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return (
        np.asarray(all_labels, dtype=np.float32),
        np.asarray(all_probs, dtype=np.float32),
    )


def build_validation_loader(args: argparse.Namespace) -> DataLoader:
    data_dirs = args.val_dirs or args.data_dirs or default_data_dirs()
    if not data_dirs:
        raise SystemExit(
            "No data roots found. Pass --data_dirs or --val_dirs, "
            "or create DATA_SET/SpeechCommands."
        )

    pairs_per_word = max(5, args.pairs_per_word // 4)
    dataset = PronunciationPairDataset(
        data_dirs,
        augment=False,
        pairs_per_word=pairs_per_word,
        seed=args.seed,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained pronunciation scoring checkpoint."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--data_dirs", nargs="+", default=None)
    parser.add_argument("--val_dirs", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pairs_per_word", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    scorer = PronunciationScorer(
        checkpoint_path=args.checkpoint,
        embed_dim=args.embed_dim,
        device=args.device,
    )
    loader = build_validation_loader(args)

    labels, probs = run_inference(scorer, loader)
    valid = (labels == 0.0) | (labels == 1.0)
    if not valid.all():
        labels = labels[valid]
        probs = probs[valid]

    results = [
        (threshold, classification_metrics(labels, probs, threshold))
        for threshold in DEFAULT_THRESHOLDS
    ]
    best_threshold, best_metrics = max(
        results,
        key=lambda item: (item[1]["f0_5"], item[1]["precision"], item[1]["accuracy"]),
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {len(labels)}")
    print()
    print_threshold_report(results)
    print_confusion_matrices(results)
    print()
    print("Best Threshold")
    print("--------------")
    print(f"Recommended threshold = {best_threshold:.2f}")
    print(f"Accuracy: {best_metrics['accuracy']:.2f}")
    print(f"F1: {best_metrics['f1']:.2f}")
    print(f"F0.5: {best_metrics['f0_5']:.2f}")
    print(f"Precision: {best_metrics['precision']:.2f}")
    print(f"Recall: {best_metrics['recall']:.2f}")
    print(f"AUC: {best_metrics['auc']:.2f}")
    print(
        "Confusion Matrix: "
        f"TP={best_metrics['tp']}, TN={best_metrics['tn']}, "
        f"FP={best_metrics['fp']}, FN={best_metrics['fn']}"
    )
    print(
        f"Recommended threshold = {best_threshold:.2f} because it gives best "
        "F0.5, favoring precision to reduce false positives."
    )


if __name__ == "__main__":
    main()
