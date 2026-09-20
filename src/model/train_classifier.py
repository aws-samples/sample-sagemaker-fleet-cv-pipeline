# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Cell classification training script for SageMaker SDK v3. Trains a ResNet18 binary classifier (good vs defect) on the same dataset used by the anomaly detector -- demonstrating two approaches to the same quality inspection task within one pipeline. Exports to ONNX for edge deployment via SageMaker Neo.'
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
from utils.onnx_export import export_onnx
from utils.dataset import resolve_dataset_root, load_images, list_images
from utils.lineage import (
    load_dataset_manifest,
    resolve_warm_start,
    write_training_manifest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CLASSES = 2  # good vs defect
CLASS_NAMES = ["good", "defect"]


def build_model(num_classes: int, pretrained_path: str = None) -> nn.Module:
    'Build ResNet18 classifier with custom head.'
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


def prepare_classifier_data(train_root: Path, val_root: Path, image_size: int):
    'Train on the TRAIN split; evaluate on the untouched VALIDATION split.'
    train_images, train_labels, _ = load_labeled_data(train_root, image_size)
    val_images, val_labels, _ = load_labeled_data(val_root, image_size)

    for split_name, labels, root in (
        ("train", train_labels, train_root),
        ("validation", val_labels, val_root),
    ):
        missing = [
            CLASS_NAMES[idx] for idx in range(NUM_CLASSES) if not (labels == idx).any()
        ]
        if missing:
            raise ValueError(
                f"The {split_name} split at {root} has no {', '.join(missing)} "
                f"images. The classifier needs both classes in train (to learn) "
                f"and in validation (to measure per-class recall). Regenerate the "
                f"dataset with a larger --num-samples or a higher --anomaly-ratio."
            )

    logger.info(
        "Classifier data: train=%d (defects=%d) validation=%d (defects=%d)",
        len(train_labels), int(train_labels.sum()),
        len(val_labels), int(val_labels.sum()),
    )
    return train_images, train_labels, val_images, val_labels


def compute_metrics(model: nn.Module, images: np.ndarray, labels: np.ndarray, device: torch.device) -> dict:
    'Compute accuracy plus defect-sensitive metrics.'
    model.eval()
    with torch.no_grad():
        x = torch.tensor(images, dtype=torch.float32).to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
    return classification_metrics(labels, preds)


def classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    'Shared metric computation for numpy label/prediction arrays.'
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    if labels.shape != preds.shape:
        raise ValueError(
            f"labels and predictions must align, got {labels.shape} and {preds.shape}"
        )
    accuracy = float(np.mean(preds == labels))
    per_class_recall = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_idx
        if not mask.any():
            raise ValueError(
                f"No '{class_name}' examples in the evaluation set; per-class "
                f"recall is undefined."
            )
        per_class_recall.append(float(np.mean(preds[mask] == class_idx)))
    defect_idx = CLASS_NAMES.index("defect")
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(per_class_recall)),
        "defect_recall": per_class_recall[defect_idx],
        "good_recall": per_class_recall[CLASS_NAMES.index("good")],
    }


def main():
    parser = argparse.ArgumentParser(description="Train cell classifier")
    parser.add_argument("--model_name", type=str, default="resnet18_classifier")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=1337,
                        help="Seed for torch init and batch sampling")
    parser.add_argument("--allow_incompatible_warm_start", type=str, default="false",
                        help="Warm-start even from a parent whose split scheme or "
                             "preprocessing schema differs. Off by default: such a "
                             "parent may have been fitted on images this dataset holds "
                             "out, which makes the gate score not a held-out score.")
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    parser.add_argument("--pretrained_model", type=str, default=os.environ.get("SM_CHANNEL_PRETRAINED_MODEL", "/opt/ml/input/data/pretrained_model"))
    args = parser.parse_args()

    train_dir = extract_tarballs(Path(args.train), "train")
    val_dir = extract_tarballs(Path(args.validation), "validation")

    dataset_manifest = load_dataset_manifest(train_dir, val_dir)

    # Look for a pretrained model, and check it is a valid starting point BEFORE
    # loading its weights -- see utils/lineage.py for why within-version split
    # disjointness is not sufficient.
    pretrained_path = None
    parent_manifest = None
    forced_warm_start = False
    use_parent = False
    if args.pretrained_model and Path(args.pretrained_model).exists():
        pretrained_dir = extract_tarballs(Path(args.pretrained_model), "pretrained")
        use_parent, parent_manifest, _, forced_warm_start = resolve_warm_start(
            dataset_manifest, pretrained_dir, args.image_size,
            allow_incompatible=str(args.allow_incompatible_warm_start).lower()
            in ("1", "true", "yes", "on"),
        )
        if use_parent:
            # The checkpoint the compatibility check was performed against, not an
            # independently chosen one. Selecting the manifest and the weights
            # separately let a compatible manifest validate incompatible weights.
            from utils.lineage import find_parent_checkpoint  # noqa: PLC0415
            bound_checkpoint, _ = find_parent_checkpoint(pretrained_dir)
            pt_files = [bound_checkpoint] if bound_checkpoint else []
            if not pt_files:
                logger.warning(
                    "Parent artifact has no .pt checkpoint; training from initial "
                    "weights and recording NO ancestry, even though its manifest "
                    "was compatible."
                )
                use_parent = False
                forced_warm_start = False
            if pt_files:
                pretrained_path = str(pt_files[0])

    train_root = resolve_dataset_root(train_dir, "train", require_defect=True)
    val_root = resolve_dataset_root(val_dir, "validation", require_defect=True)

    # Training data is the train channel; validation is spent only on measuring.
    # The test channel stays untouched for the ONNX quality gate downstream.
    train_images, train_labels, val_images, val_labels = prepare_classifier_data(
        train_root, val_root, args.image_size
    )

    torch.manual_seed(args.seed)
    train_x = torch.tensor(train_images, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.int64)
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

    # Evaluate on the untouched validation split
    eval_metrics = compute_metrics(model, val_images, val_labels, device)
    test_accuracy = eval_metrics["accuracy"]
    logger.info(
        "Validation eval: accuracy=%.4f balanced_accuracy=%.4f defect_recall=%.4f",
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
    export_onnx(model_cpu, dummy, onnx_path, "logits")
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

    write_training_manifest(
        args.model_dir, dataset_manifest, args.image_size, args.model_name,
        # Only weights that were ACTUALLY loaded. Recording a rejected
        # parent described weights this model never saw, and stopped a
        # genuine cold start from clearing inherited contamination.
        parent=parent_manifest if use_parent else None,
        unsafe_ancestry=forced_warm_start,
    )

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
