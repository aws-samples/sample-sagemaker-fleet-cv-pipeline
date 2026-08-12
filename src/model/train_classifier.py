# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Cell classification training script for SageMaker SDK v3.
Trains a ResNet18 binary classifier (good vs defect) on the same dataset
used by the anomaly detector -- demonstrating two approaches to the same
quality inspection task within one pipeline.
Exports to ONNX for edge deployment via SageMaker Neo.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from utils.tarball import extract_tarballs
from utils.dataset import resolve_dataset_root, load_images, list_images

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CLASSES = 2  # good vs defect
CLASS_NAMES = ["good", "defect"]


def build_model(num_classes: int, pretrained_path: str = None) -> nn.Module:
    """Build ResNet18 classifier with custom head.

    Starts from ImageNet weights (cached in the training image at build
    time): from-scratch ResNet18 on the small per-camera datasets memorizes
    the minority class instead of generalizing (observed: train loss ~0.003
    with held-out defect recall 0.0). Warm-start weights from a prior
    pipeline run, when present, override the ImageNet initialization.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if pretrained_path and Path(pretrained_path).exists():
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        logger.info("Loaded pretrained weights from %s", pretrained_path)
    return model


def load_labeled_data(data_dir: Path, image_size: int):
    """Load images from good/ and defect/ subdirectories. Returns (images, labels, files)."""
    all_images = []
    all_labels = []
    all_files = []
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            logger.warning("Class directory not found: %s", class_dir)
            continue
        imgs = load_images(class_dir, image_size)
        all_images.append(imgs)
        all_labels.extend([idx] * len(imgs))
        all_files.extend(f"{class_name}/{p.name}" for p in list_images(class_dir))
        logger.info("Loaded %d images for class %s", len(imgs), class_name)
    if not all_images:
        raise ValueError(f"No class directories found in {data_dir}. Expected: {CLASS_NAMES}")
    images = np.concatenate(all_images, axis=0)
    labels = np.array(all_labels, dtype=np.int64)
    return images, labels, all_files


def stratified_split(labels: np.ndarray, train_fraction: float, seed: int):
    """Return (train_idx, holdout_idx), seeded and stratified per class.

    Stratification matters on the ~90% 'good' distribution: an unstratified
    draw can leave one side with zero defects.
    """
    rng = np.random.default_rng(seed)
    train_parts, holdout_parts = [], []
    for class_idx in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_idx)
        rng.shuffle(class_indices)
        split = int(train_fraction * len(class_indices))
        train_parts.append(class_indices[:split])
        holdout_parts.append(class_indices[split:])
    train_idx = np.concatenate(train_parts)
    holdout_idx = np.concatenate(holdout_parts)
    rng.shuffle(train_idx)
    rng.shuffle(holdout_idx)
    return train_idx, holdout_idx


def prepare_classifier_data(train_root: Path, val_root: Path, image_size: int, seed: int):
    """Build classifier training/holdout arrays from the train and validation
    channels ONLY.

    The test channel is deliberately not an input to this script (it is not
    even mounted into the training job): it is reserved, complete and
    untouched, for the downstream ONNX quality gate. Training data is the
    train channel's 'good' images plus a stratified 70% of the validation
    channel (the only training-side source of defect examples); the
    remaining 30% of validation is held out for the in-training metric.
    """
    val_images, val_labels, _ = load_labeled_data(val_root, image_size)
    if not (val_labels == CLASS_NAMES.index("defect")).any():
        raise ValueError(
            f"Validation channel at {val_root} contains no defect images; "
            "the classifier cannot learn the defect class from it."
        )
    train_idx, holdout_idx = stratified_split(val_labels, train_fraction=0.7, seed=seed)

    good_images = load_images(train_root / "good", image_size)
    good_labels = np.zeros(len(good_images), dtype=np.int64)

    train_images = np.concatenate([good_images, val_images[train_idx]], axis=0)
    train_labels = np.concatenate([good_labels, val_labels[train_idx]], axis=0)
    holdout_images = val_images[holdout_idx]
    holdout_labels = val_labels[holdout_idx]
    logger.info(
        "Classifier data: train=%d (defects=%d, %d from train-good, %d from validation) "
        "holdout=%d (defects=%d), seed=%d",
        len(train_labels), int(train_labels.sum()), len(good_labels), len(train_idx),
        len(holdout_labels), int(holdout_labels.sum()), seed,
    )
    return train_images, train_labels, holdout_images, holdout_labels


def compute_metrics(model: nn.Module, images: np.ndarray, labels: np.ndarray, device: torch.device) -> dict:
    """Compute accuracy plus defect-sensitive metrics.

    Plain accuracy is misleading on this ~90% 'good' dataset: a
    majority-class predictor scores ~0.9 while detecting nothing. Balanced
    accuracy and defect recall expose that failure mode.
    """
    model.eval()
    with torch.no_grad():
        x = torch.tensor(images, dtype=torch.float32).to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
    return classification_metrics(labels, preds)


def classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Shared metric computation for numpy label/prediction arrays."""
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    accuracy = float(np.mean(preds == labels))
    per_class_recall = []
    for class_idx in range(NUM_CLASSES):
        mask = labels == class_idx
        recall = float(np.mean(preds[mask] == class_idx)) if mask.any() else 0.0
        per_class_recall.append(recall)
    defect_idx = CLASS_NAMES.index("defect")
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(per_class_recall)),
        "defect_recall": per_class_recall[defect_idx],
    }


def main():
    parser = argparse.ArgumentParser(description="Train cell classifier")
    parser.add_argument("--model_name", type=str, default="resnet18_classifier")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=1337,
                        help="Seed for the train/holdout split and torch init")
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    parser.add_argument("--pretrained_model", type=str, default=os.environ.get("SM_CHANNEL_PRETRAINED_MODEL", "/opt/ml/input/data/pretrained_model"))
    args = parser.parse_args()

    train_dir = extract_tarballs(Path(args.train), "train")
    val_dir = extract_tarballs(Path(args.validation), "validation")

    # Look for pretrained model
    pretrained_path = None
    if args.pretrained_model and Path(args.pretrained_model).exists():
        pretrained_dir = extract_tarballs(Path(args.pretrained_model), "pretrained")
        pt_files = list(pretrained_dir.rglob("*.pt"))
        if pt_files:
            pretrained_path = str(pt_files[0])

    train_root = resolve_dataset_root(train_dir, "train", require_defect=False)
    val_root = resolve_dataset_root(val_dir, "validation", require_defect=True)

    # Classifier data comes from the train and validation channels only; the
    # test channel stays untouched for the ONNX quality gate downstream.
    train_images, train_labels, test_images, test_labels = prepare_classifier_data(
        train_root, val_root, args.image_size, args.seed
    )

    torch.manual_seed(args.seed)
    train_x = torch.tensor(train_images, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.int64)
    # Oversample the minority class so every batch is roughly balanced.
    # The dataset is ~90% 'good'; with plain shuffling most batches carry
    # zero defect examples and the model converges to a majority-class
    # predictor (observed live: accuracy 0.88, defect recall 0.0).
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(np.float64)
    sample_weights = torch.tensor(
        [1.0 / max(class_counts[label], 1.0) for label in train_labels],
        dtype=torch.float64,
    )
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_labels),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info("Class counts=%s; balancing batches via WeightedRandomSampler", class_counts.tolist())
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, sampler=sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device=%s", device)

    model = build_model(NUM_CLASSES, pretrained_path).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Balanced batches (WeightedRandomSampler above) make an unweighted
    # loss appropriate here; stacking loss weights on top would over-
    # correct toward the minority class.
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, len(train_loader))
        logger.info("Epoch %d/%d loss=%.4f", epoch + 1, args.num_epochs, epoch_loss)

    # Evaluate
    eval_metrics = compute_metrics(model, test_images, test_labels, device)
    test_accuracy = eval_metrics["accuracy"]
    logger.info(
        "Held-out eval: accuracy=%.4f balanced_accuracy=%.4f defect_recall=%.4f",
        eval_metrics["accuracy"],
        eval_metrics["balanced_accuracy"],
        eval_metrics["defect_recall"],
    )

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_cpu = model.to("cpu").eval()

    # ONNX export
    dummy = torch.randn(1, 3, args.image_size, args.image_size, dtype=torch.float32)
    onnx_path = os.path.join(args.model_dir, "model.onnx")
    torch.onnx.export(
        model_cpu,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=11,
    )
    logger.info("Exported ONNX model to %s", onnx_path)

    # PyTorch state dict
    pt_path = os.path.join(args.model_dir, "model.pt")
    torch.save(model_cpu.state_dict(), pt_path)

    # Metrics
    metrics = {
        **eval_metrics,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "model_name": args.model_name,
        "seed": args.seed,
    }
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"validation:accuracy={test_accuracy:.4f}")
    print(f"validation:balanced_accuracy={eval_metrics['balanced_accuracy']:.4f}")
    print(f"validation:defect_recall={eval_metrics['defect_recall']:.4f}")
    logger.info("Training completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)
