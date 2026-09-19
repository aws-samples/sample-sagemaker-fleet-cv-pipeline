#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Run the whole model lifecycle locally on a tiny dataset, without AWS.'
import argparse
import json
import subprocess
import os
import sys
from datetime import datetime, timezone
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The gate threshold comes from config so the smoke cannot drift from what the pipeline ships.
sys.path.insert(0, str(REPO_ROOT))
from config import DEFAULT_MIN_AUROC  # noqa: E402

# Defaults sized for speed, not for model quality. The anomaly ratio is well
# above the pipeline default so a tiny split still carries enough of both
# classes for a per-class metric.
DEFAULT_SAMPLES = 40
DEFAULT_IMAGE_SIZE = 64
DEFAULT_EPOCHS = 2
DEFAULT_BATCH = 8
DEFAULT_ANOMALY_RATIO = 0.25


class SmokeFailure(RuntimeError):
    """A stage failed, or produced something the next stage could not use."""


def run(step: str, argv: list, env_extra: dict = None) -> str:
    """Run a stage, streaming nothing but reporting its tail on failure."""
    print(f"[{step}] {' '.join(str(a) for a in argv[1:3])} ...", flush=True)
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(
        [str(a) for a in argv], capture_output=True, text=True, env=env, cwd=REPO_ROOT
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:])
        raise SmokeFailure(f"{step} exited {proc.returncode}:\n{tail}")
    return proc.stdout + proc.stderr


def require(path: Path, step: str) -> Path:
    if not path.exists():
        raise SmokeFailure(f"{step} did not produce {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--num-samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--anomaly-ratio", type=float, default=DEFAULT_ANOMALY_RATIO)
    parser.add_argument("--work-dir", default=None,
                        help="Keep artifacts here instead of a temp directory")
    # Read from config, not restated. Its help claimed it matched MinAUROC while defaulting to a
    # different number, so the smoke applied a gate the pipeline does not ship.
    parser.add_argument("--min-auroc", type=float, default=DEFAULT_MIN_AUROC,
                        help=f"Gate threshold to apply. Defaults to the shipped MinAUROC "
                             f"({DEFAULT_MIN_AUROC}); pass a value to override")
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.6)
    parser.add_argument("--min-defect-recall", type=float, default=0.5)
    parser.add_argument("--expect-gate-pass", action="store_true",
                        help="Fail the smoke run unless BOTH quality gates pass. Off by "
                             "default: the tiny default run is not meant to train a good "
                             "model, only to prove every stage runs and hands off.")
    args = parser.parse_args()

    python = sys.executable
    tmp = None
    if args.work_dir:
        parent = Path(args.work_dir)
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        work = parent / f"run-{stamp}-{os.getpid()}"
        if work.exists():
            raise SystemExit(f"Refusing to reuse an existing directory: {work}")
        work.mkdir()
        print(f"Work directory: {work}")
    else:
        tmp = tempfile.TemporaryDirectory(prefix="fleetcv-smoke-")
        work = Path(tmp.name)

    try:
        data = work / "data"

        # --- 1. Preprocess -------------------------------------------------
        run("preprocess", [
            python, "src/data/preprocess_cv.py",
            "--num-samples", args.num_samples,
            "--image-size", args.image_size,
            "--anomaly-ratio", args.anomaly_ratio,
            "--output-dir", data,
            "--temp-dir", work / "staging",
        ])
        manifest = json.loads(
            require(data / "dataset_manifest.json", "preprocess").read_text()
        )
        for split in ("train", "validation", "test"):
            require(data / split / f"{split}.tar.gz", "preprocess")
            counts = manifest["splits"][split]["counts"]
            if counts["good"] < 1 or counts["defect"] < 1:
                raise SmokeFailure(
                    f"split {split} lacks both classes: {counts}"
                )
        print(f"       dataset_id={manifest['dataset_id']}")
        for split in ("train", "validation", "test"):
            print(f"       {split}: {manifest['splits'][split]['counts']}")

        # --- 2. Train ------------------------------------------------------
        anom_model = work / "model_anomalib"
        run("train-anomalib", [
            python, "src/model/train_anomalib.py",
            "--train", data / "train", "--validation", data / "validation",
            "--pretrained_model", work / "absent",
            "--model_dir", anom_model,
            "--num_epochs", args.epochs, "--image_size", args.image_size,
            "--batch_size", args.batch_size,
        ])
        require(anom_model / "model.onnx", "train-anomalib")

        clf_model = work / "model_classifier"
        run("train-classifier", [
            python, "src/model/train_classifier.py",
            "--train", data / "train", "--validation", data / "validation",
            "--pretrained_model", work / "absent",
            "--model_dir", clf_model,
            "--num_epochs", args.epochs, "--image_size", args.image_size,
            "--batch_size", args.batch_size,
        ])
        require(clf_model / "model.onnx", "train-classifier")

        # --- 3. Evaluate the exported ONNX models on the test split --------
        anom_eval = work / "eval_anomalib"
        run("evaluate-anomalib", [
            python, "src/model/evaluate_anomalib_onnx.py",
            "--image-size", args.image_size,
            "--model-dir", anom_model,
            "--test-dir", data / "test",
            "--output-dir", anom_eval,
        ])
        anom_metrics = json.loads(
            require(anom_eval / "model_quality_statistics.json", "evaluate-anomalib").read_text()
        )

        clf_eval = work / "eval_classifier"
        run("evaluate-classifier", [
            python, "src/model/evaluate_classifier_onnx.py",
            "--image-size", args.image_size,
            "--model-dir", clf_model,
            "--test-dir", data / "test",
            "--output-dir", clf_eval,
        ])
        clf_metrics = json.loads(
            require(clf_eval / "model_quality_statistics.json", "evaluate-classifier").read_text()
        )

        # --- 4. Apply the same arithmetic the pipeline quality gates use ---
        auroc = anom_metrics["binary_classification_metrics"]["auc"]["value"]
        balanced = clf_metrics["balanced_accuracy"]
        recall = clf_metrics["defect_recall"]

        anomaly_pass = auroc >= args.min_auroc
        classifier_pass = (
            balanced >= args.min_balanced_accuracy and recall >= args.min_defect_recall
        )

        print("\n--- quality gates -------------------------------------------")
        print(f"  anomaly    AUROC              {auroc:.4f}  >= {args.min_auroc}"
              f"   -> {'PASS' if anomaly_pass else 'FAIL'}")
        print(f"  classifier balanced_accuracy  {balanced:.4f}  >= "
              f"{args.min_balanced_accuracy}   -> "
              f"{'PASS' if balanced >= args.min_balanced_accuracy else 'FAIL'}")
        print(f"  classifier defect_recall      {recall:.4f}  >= "
              f"{args.min_defect_recall}   -> "
              f"{'PASS' if recall >= args.min_defect_recall else 'FAIL'}")
        print(f"  classifier accuracy           {clf_metrics['accuracy']:.4f}  "
              f"(reported, not gated)")

        print("\nAll stages ran and handed off correctly.")
        if args.expect_gate_pass and not (anomaly_pass and classifier_pass):
            raise SmokeFailure(
                "--expect-gate-pass was set but a gate failed. A tiny run is not "
                "expected to train a good model; raise --num-samples and --epochs "
                "if you want to assert the happy path."
            )
        if not (anomaly_pass and classifier_pass):
            print("NOTE: a gate did not pass. That is expected for the tiny default "
                  "run -- it proves the gate reaches a verdict, not that the model "
                  "is good. Use --expect-gate-pass with a larger run to assert "
                  "the happy path.")
        return 0
    except SmokeFailure as exc:
        print(f"\nSMOKE FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
