"""
Train a lightweight calibration model from feedback audio samples.

This script builds a simple logistic regression calibration layer that
maps physics and deep learning scores to a calibrated probability.
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np

from detector import HybridEnsembleDetector


LABEL_MAP = {
    "AI_GENERATED": 1,
    "AI": 1,
    "FAKE": 1,
    "SYNTHETIC": 1,
    "HUMAN": 0,
    "REAL": 0
}


def sigmoid(z):
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def train_logreg(X, y, lr=0.5, epochs=300, l2=0.001):
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    n = float(X.shape[0])

    for _ in range(epochs):
        z = X.dot(w) + b
        p = sigmoid(z)
        error = p - y
        grad_w = (X.T.dot(error) / n) + (l2 * w)
        grad_b = error.mean()
        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t = 0.5
    best_f1 = -1.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tp = float(((preds == 1) & (y_true == 1)).sum())
        fp = float(((preds == 1) & (y_true == 0)).sum())
        fn = float(((preds == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1


def iter_audio_files(data_dir, max_per_class=0):
    samples = []
    counts = {0: 0, 1: 0}

    for label_name, label_value in LABEL_MAP.items():
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            continue

        for root, _, files in os.walk(label_dir):
            for name in files:
                if not name.lower().endswith((".mp3", ".wav")):
                    continue
                if max_per_class and counts[label_value] >= max_per_class:
                    continue

                file_path = os.path.join(root, name)
                meta_path = os.path.splitext(file_path)[0] + ".json"
                sample = {
                    "path": file_path,
                    "label": label_value
                }

                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as handle:
                            meta = json.load(handle)
                        if "physics_score" in meta and "dl_score" in meta:
                            sample["physics_score"] = float(meta["physics_score"])
                            sample["dl_score"] = float(meta["dl_score"])
                    except Exception:
                        pass

                samples.append(sample)
                counts[label_value] += 1

    return samples


def main():
    parser = argparse.ArgumentParser(description="Train calibration layer from feedback samples")
    parser.add_argument("--data-dir", default="data/feedback", help="Feedback dataset directory")
    parser.add_argument("--output", default="data/calibration.json", help="Output calibration JSON file")
    parser.add_argument("--history-dir", default=os.environ.get(
        "CALIBRATION_HISTORY_DIR",
        "data/calibration_history"
    ), help="Directory to store calibration history backups")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
    parser.add_argument("--l2", type=float, default=0.001, help="L2 regularization")
    parser.add_argument("--min-samples", type=int, default=20, help="Minimum samples required")
    parser.add_argument("--max-per-class", type=int, default=0, help="Max samples per class (0 = all)")
    parser.add_argument("--deepfake-model-path", default=os.environ.get(
        "DEEPFAKE_MODEL_PATH",
        "garystafford/wav2vec2-deepfake-voice-detector"
    ))
    parser.add_argument("--whisper-model-path", default=os.environ.get(
        "WHISPER_MODEL_PATH",
        "openai/whisper-base"
    ))
    parser.add_argument("--use-local-deepfake-model", action="store_true", default=False)
    parser.add_argument("--use-local-whisper-model", action="store_true", default=False)
    parser.add_argument("--max-audio-duration", type=int, default=30)

    args = parser.parse_args()

    if args.history_dir:
        os.makedirs(args.history_dir, exist_ok=True)

    if not os.path.isdir(args.data_dir):
        print(f"Data directory not found: {args.data_dir}")
        return 1

    samples = iter_audio_files(args.data_dir, max_per_class=args.max_per_class)
    if not samples:
        print("No audio samples found.")
        return 1

    needs_scoring = any("physics_score" not in sample for sample in samples)
    detector = None
    if needs_scoring:
        detector = HybridEnsembleDetector(
            deepfake_model_path=args.deepfake_model_path,
            whisper_model_path=args.whisper_model_path,
            use_local_deepfake_model=args.use_local_deepfake_model,
            use_local_whisper_model=args.use_local_whisper_model,
            max_audio_duration=args.max_audio_duration
        )

    features = []
    labels = []
    skipped = 0

    for sample in samples:
        if "physics_score" in sample and "dl_score" in sample:
            phys_score = sample["physics_score"]
            dl_score = sample["dl_score"]
        else:
            if detector is None:
                skipped += 1
                continue
            scores = detector.extract_scores(sample["path"], input_type="file")
            if scores.get("status") != "success":
                skipped += 1
                continue
            phys_score = scores["physics_score"]
            dl_score = scores["dl_score"]

        features.append([phys_score, dl_score])
        labels.append(sample["label"])

    if skipped:
        print(f"Skipped {skipped} samples due to scoring errors.")

    if len(features) < args.min_samples:
        print(f"Not enough samples to train. Found {len(features)}.")
        return 1

    X = np.array(features, dtype=np.float64)
    y = np.array(labels, dtype=np.float64)

    w, b = train_logreg(X, y, lr=args.lr, epochs=args.epochs, l2=args.l2)
    probs = sigmoid(X.dot(w) + b)
    threshold, f1 = best_threshold(y, probs)
    predictions = (probs >= threshold).astype(int)
    accuracy = float((predictions == y).mean())

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(args.output):
        version_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + os.urandom(4).hex()
        history_name = f"calibration_{version_id}.json"
        history_path = os.path.join(args.history_dir, history_name)
        shutil.copy2(args.output, history_path)
        meta_path = os.path.join(args.history_dir, f"calibration_{version_id}.meta.json")
        meta = {
            "versionId": version_id,
            "source": args.output,
            "archivedAt": datetime.utcnow().isoformat() + "Z",
            "reason": "self_learning_train"
        }
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    calibration = {
        "version": 1,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "weights": [float(w[0]), float(w[1])],
        "bias": float(b),
        "threshold": float(threshold),
        "feature_order": ["physics_score", "dl_score"],
        "metrics": {
            "accuracy": accuracy,
            "f1": float(f1)
        },
        "samples": {
            "count": int(len(features)),
            "ai": int((y == 1).sum()),
            "human": int((y == 0).sum())
        }
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)

    print(f"Calibration saved to {args.output}")
    print(f"Accuracy: {accuracy:.3f} | F1: {f1:.3f} | Threshold: {threshold:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
