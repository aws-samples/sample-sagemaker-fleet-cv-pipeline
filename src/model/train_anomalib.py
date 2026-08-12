# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Anomaly training script for SageMaker SDK v3.
This script trains a lightweight convolutional autoencoder on normal images,
then reports anomaly-detection AUROC using reconstruction error.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils.tarball import extract_tarballs
from utils.dataset import resolve_dataset_root, load_images
from utils.metrics import binary_auc

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


def main():
    parser = argparse.ArgumentParser(description="Train lightweight anomaly detector")
    parser.add_argument("--model_name", type=str, default="autoencoder")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1337,
                        help="Seed for torch init and batch shuffling")
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"))
    parser.add_argument("--pretrained_model", type=str, default=os.environ.get("SM_CHANNEL_PRETRAINED_MODEL", "/opt/ml/input/data/pretrained_model"))
    args = parser.parse_args()

    train_dir = extract_tarballs(Path(args.train), "train")
    _ = extract_tarballs(Path(args.validation), "validation")
    test_dir = extract_tarballs(Path(args.test), "test")

    pretrained_state_dict = None
    if args.pretrained_model and Path(args.pretrained_model).exists():
        logger.info("Pretrained model path exists, extracting...")
        pretrained_dir = extract_tarballs(Path(args.pretrained_model), "pretrained")
        pt_files = list(pretrained_dir.rglob("*.pt"))
        if pt_files:
            pretrained_state_dict = torch.load(pt_files[0], map_location="cpu", weights_only=True)
            logger.info("Found pretrained model: %s", pt_files[0])

    train_root = resolve_dataset_root(train_dir, "train", require_defect=False)
    test_root = resolve_dataset_root(test_dir, "test", require_defect=True)

    train_good = load_images(train_root / "good", args.image_size)
    test_good = load_images(test_root / "good", args.image_size)
    test_defect = load_images(test_root / "defect", args.image_size)

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
    if pretrained_state_dict:
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
        good_tensor = torch.tensor(test_good, dtype=torch.float32).to(device)
        defect_tensor = torch.tensor(test_defect, dtype=torch.float32).to(device)
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
    torch.onnx.export(
        model_cpu,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["reconstruction"],
        dynamic_axes={"input": {0: "batch"}, "reconstruction": {0: "batch"}},
        opset_version=11,
    )
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

    print(f"validation:image_auroc={image_auroc:.4f}")
    logger.info("Training completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
        sys.exit(1)