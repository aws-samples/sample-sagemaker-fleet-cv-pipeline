# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Anomaly training script for SageMaker SDK v3.'
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.tarball import extract_tarballs
from utils.onnx_export import export_onnx
from utils.dataset import resolve_dataset_root, load_images
from utils.metrics import binary_auc
from utils.lineage import (
    load_dataset_manifest,
    resolve_warm_start,
    write_training_manifest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TinyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _align_spatial(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] == ref.shape[-2:]:
        return pred
    return F.interpolate(pred, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def usable_state_dict(payload) -> bool:
    'Whether a deserialised checkpoint payload can actually seed weights.'
    return isinstance(payload, dict) and bool(payload)


def main():
    parser = argparse.ArgumentParser(description="Train lightweight anomaly detector")
    parser.add_argument("--model_name", type=str, default="autoencoder")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=256,
                        help="Must be divisible by 4. The autoencoder halves the "
                             "spatial dimensions twice, so any other size cannot be "
                             "reconstructed exactly -- see the check in main().")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Seed for torch init and batch shuffling")
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

    if args.image_size % 4 != 0:
        raise SystemExit(
            f"--image_size {args.image_size} is not divisible by 4. This "
            f"autoencoder halves the spatial dimensions twice, so the exported "
            f"graph would return {((args.image_size + 3) // 4) * 4}x"
            f"{((args.image_size + 3) // 4) * 4} rather than {args.image_size}x"
            f"{args.image_size}, and evaluation would reject it after training had "
            f"already completed. Use a multiple of 4 (the default is 256)."
        )
    if args.image_size < 32:
        raise SystemExit(
            f"--image_size {args.image_size} is too small; two halvings leave "
            f"fewer than 8 pixels per side. Use at least 32."
        )

    train_dir = extract_tarballs(Path(args.train), "train")
    val_dir = extract_tarballs(Path(args.validation), "validation")

    # The dataset's identity travels inside the split tarballs, so it is readable
    # from either channel.
    dataset_manifest = load_dataset_manifest(train_dir, val_dir)

    pretrained_state_dict = None
    parent_manifest = None
    forced_warm_start = False
    use_parent = False
    if args.pretrained_model and Path(args.pretrained_model).exists():
        logger.info("Pretrained model path exists, extracting...")
        pretrained_dir = extract_tarballs(Path(args.pretrained_model), "pretrained")
        # Compatibility is checked BEFORE the weights are loaded. Within-version
        # split disjointness does not establish that this dataset's holdout
        # excludes the parent's training images -- see utils/lineage.py.
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
                pretrained_state_dict = torch.load(pt_files[0], map_location="cpu", weights_only=True)
                if not usable_state_dict(pretrained_state_dict):
                    raise ValueError(
                        f"Parent checkpoint {pt_files[0]} did not deserialize to a "
                        f"non-empty state dict; got "
                        f"{type(pretrained_state_dict).__name__} "
                        f"({pretrained_state_dict!r:.60}). Warm-start cannot proceed "
                        f"from it, and recording this parent would claim ancestry the "
                        f"model does not have."
                    )
                logger.info("Found pretrained model: %s", pt_files[0])

    train_root = resolve_dataset_root(train_dir, "train", require_defect=False)
    val_root = resolve_dataset_root(val_dir, "validation", require_defect=True)

    train_good = load_images(train_root / "good", args.image_size)
    val_good = load_images(val_root / "good", args.image_size)
    val_defect = load_images(val_root / "defect", args.image_size)

    torch.manual_seed(args.seed)
    train_tensor = torch.tensor(train_good, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device=%s", device)
    model = TinyAutoencoder().to(device)
    if use_parent:
        model.load_state_dict(pretrained_state_dict)
        logger.info("Loaded pretrained weights")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            recon = _align_spatial(recon, batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        epoch_loss /= max(1, len(train_loader))
        logger.info("Epoch %d/%d loss=%.6f", epoch + 1, args.num_epochs, epoch_loss)

    model.eval()
    with torch.no_grad():
        good_tensor = torch.tensor(val_good, dtype=torch.float32).to(device)
        defect_tensor = torch.tensor(val_defect, dtype=torch.float32).to(device)
        good_recon = model(good_tensor)
        defect_recon = model(defect_tensor)
        good_recon = _align_spatial(good_recon, good_tensor)
        defect_recon = _align_spatial(defect_recon, defect_tensor)

        good_err = torch.mean((good_recon - good_tensor) ** 2, dim=(1, 2, 3)).cpu().numpy()
        defect_err = torch.mean((defect_recon - defect_tensor) ** 2, dim=(1, 2, 3)).cpu().numpy()

    labels = np.concatenate([np.zeros_like(good_err), np.ones_like(defect_err)])
    scores = np.concatenate([good_err, defect_err])
    image_auroc = binary_auc(labels=labels, scores=scores)

    os.makedirs(args.model_dir, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    dummy = torch.randn(1, 3, args.image_size, args.image_size, dtype=torch.float32)
    onnx_path = os.path.join(args.model_dir, "model.onnx")
    export_onnx(model_cpu, dummy, onnx_path, "reconstruction")
    logger.info("Exported ONNX model to %s", onnx_path)

    pt_path = os.path.join(args.model_dir, "model.pt")
    torch.save(model_cpu.state_dict(), pt_path)
    logger.info("Saved PyTorch state_dict to %s", pt_path)

    metrics = {
        "image_auroc": float(image_auroc),
        "model_name": args.model_name,
        "hyperparameters": vars(args),
    }
    with open(os.path.join(args.model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(args.model_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": "tiny_autoencoder", "image_size": args.image_size}, f, indent=2)

    # Recorded in the artifact so a future warm-start can decide whether THESE
    # weights are a valid starting point.
    write_training_manifest(
        args.model_dir, dataset_manifest, args.image_size, "tiny_autoencoder",
        # Only weights that were ACTUALLY loaded. Recording a rejected
        # parent described weights this model never saw, and stopped a
        # genuine cold start from clearing inherited contamination.
        parent=parent_manifest if use_parent else None,
        unsafe_ancestry=forced_warm_start,
    )

    print(f"validation:image_auroc={image_auroc:.4f}")
    logger.info("Training completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)
