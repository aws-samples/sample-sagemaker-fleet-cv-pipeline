# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"Warm-start compatibility: decide whether a parent model's weights are usable."
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TRAINING_MANIFEST_NAME = "training_manifest.json"
DATASET_MANIFEST_NAME = "dataset_manifest.json"

COMPATIBILITY_FIELDS = ("split_scheme", "split_scheme_version", "image_size",
                        "label_mapping", "split_ratios")

STABLE_BY_CONSTRUCTION = frozenset({("content-addressed-sha256", 2)})

UNTRUSTED_LEGACY_CONTRACTS = frozenset({("content-addressed-sha256", 1)})

# Kept in sync with utils.dataset by value rather than by import: lineage.py is imported by
# the training container, whose module layout differs from the repository's.
SPLIT_SCHEME_V2_DECODED_RGB = 2


def parse_split_scheme_version(value):
    'A declared split scheme version as an EXACT integer, or None if unusable.'
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def version_or_none(manifest: dict):
    """The manifest's declared split scheme version, strictly parsed."""
    return parse_split_scheme_version((manifest or {}).get("split_scheme_version"))

DATASET_IDENTITY_FIELDS = (
    "dataset_id",
    "split_scheme",
    "split_scheme_version",
    "label_mapping",
    "image_size",
    "split_ratios",
)


def _requires_identical_dataset(scheme: str, scheme_version=None) -> bool:
    'True if this scheme only permits warm-start WITHIN one dataset version.'
    version = parse_split_scheme_version(scheme_version)
    if version is None:
        # Unparseable means no privilege, never the benefit of the doubt.
        return True
    return (scheme, version) not in STABLE_BY_CONSTRUCTION


def load_dataset_manifest(*search_dirs) -> dict:
    'Return the dataset manifest found in any of *search_dirs*, or {}.'
    found = []  # (path, manifest)
    for directory in search_dirs:
        if directory is None:
            continue
        directory = Path(directory)
        if not directory.exists():
            continue
        for candidate in [directory / DATASET_MANIFEST_NAME,
                          *sorted(directory.rglob(DATASET_MANIFEST_NAME))]:
            if not candidate.is_file():
                continue
            try:
                found.append((candidate, json.loads(candidate.read_text(encoding="utf-8"))))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Dataset manifest {candidate} is not valid JSON ({exc}). The "
                    f"channel is corrupt; refusing to guess the dataset identity."
                ) from exc

    if not found:
        return {}

    # Every channel must agree about WHICH dataset this is. Returning the first
    # match meant conflicting channels resolved to whichever the walk reached
    # first, and that value became the training and warm-start identity.
    identity = DATASET_IDENTITY_FIELDS
    first_path, first = found[0]
    for path, manifest in found[1:]:
        disagreements = [
            f"{field}: {first.get(field)!r} at {first_path} vs "
            f"{manifest.get(field)!r} at {path}"
            for field in identity
            if first.get(field) != manifest.get(field)
        ]
        if disagreements:
            raise ValueError(
                "The training channels carry manifests that disagree about the "
                "dataset identity, so there is no single answer to what this model "
                "was fitted on:\n  " + "\n  ".join(disagreements)
            )
    return first


def write_training_manifest(model_dir, dataset_manifest: dict, image_size: int,
                            model_name: str, parent: dict = None,
                            unsafe_ancestry: bool = False) -> dict:
    'Record, in the model artifact, what this model was fitted on.'
    manifest = {
        "schema_version": 1,
        "model_name": model_name,
        "image_size": image_size,
        "dataset_id": dataset_manifest.get("dataset_id"),
        "split_scheme": dataset_manifest.get("split_scheme"),
        "split_scheme_version": dataset_manifest.get("split_scheme_version"),
        "label_mapping": dataset_manifest.get("label_mapping"),
        "dataset_source": dataset_manifest.get("source"),
        "split_ratios": dataset_manifest.get("split_ratios"),
        "parent": _parent_reference(parent),
        # Sticky: true if these weights were inherited despite a failed
        # compatibility check, or inherited from a model for which that held.
        "unsafe_ancestry": bool(
            unsafe_ancestry
            or (parent or {}).get("unsafe_ancestry")
            or ((parent or {}).get("parent") or {}).get("unsafe_ancestry")
        ),
    }
    path = Path(model_dir) / TRAINING_MANIFEST_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("Training manifest: dataset_id=%s split_scheme=%s",
                manifest["dataset_id"], manifest["split_scheme"])
    return manifest


# Fields kept from a parent. Everything needed to judge a later warm-start, and
# nothing that grows with generation count.
_PARENT_REFERENCE_FIELDS = (
    "model_name", "dataset_id", "split_scheme", "split_scheme_version",
    "image_size", "label_mapping", "split_ratios", "trained_at",
)


def _parent_reference(parent: dict) -> dict:
    'A fixed-size summary of *parent*, plus its transitive safety state.'
    if not parent:
        return None
    reference = {field: parent.get(field) for field in _PARENT_REFERENCE_FIELDS}
    grandparent = parent.get("parent") or {}
    reference["generation"] = int(grandparent.get("generation", 0)) + 1
    # Inherited if the parent itself was unsafe, or inherited unsafety.
    reference["unsafe_ancestry"] = bool(
        parent.get("unsafe_ancestry") or grandparent.get("unsafe_ancestry")
    )
    reference["artifact_digest"] = parent.get("artifact_digest")
    return reference


def find_parent_checkpoint(pretrained_dir):
    'Return (checkpoint_path, manifest) for the artifact, or (None, {}).'
    if pretrained_dir is None:
        return None, {}
    directory = Path(pretrained_dir)
    if not directory.exists():
        return None, {}

    checkpoints = sorted(directory.rglob("*.pt"))
    if not checkpoints:
        # No weights. A manifest alone establishes nothing to warm-start FROM, and
        # the caller must not record ancestry for weights it never loaded.
        return None, {}
    if len(checkpoints) > 1:
        raise ValueError(
            f"Parent artifact {directory} contains {len(checkpoints)} checkpoints "
            f"({', '.join(str(c.relative_to(directory)) for c in checkpoints)}). "
            f"Which weights to inherit is ambiguous, and picking by filesystem "
            f"order could load weights the compatibility check did not validate."
        )

    checkpoint = checkpoints[0]
    manifest_path = checkpoint.parent / TRAINING_MANIFEST_NAME
    if not manifest_path.is_file():
        # Fall back to the artifact root, which is where a single-model artifact
        # puts it, but nowhere else -- a manifest from an unrelated subdirectory
        # must not stand in for this checkpoint's provenance.
        manifest_path = directory / TRAINING_MANIFEST_NAME
    if not manifest_path.is_file():
        logger.warning(
            "Checkpoint %s has no %s beside it or at the artifact root; treating "
            "its provenance as unknown", checkpoint, TRAINING_MANIFEST_NAME,
        )
        return checkpoint, {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Unreadable %s: %s", manifest_path, exc)
        return checkpoint, {}

    if isinstance(manifest, dict) and manifest:
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        manifest["artifact_digest"] = f"sha256:{digest.hexdigest()}"
    return checkpoint, manifest


def find_parent_manifest(pretrained_dir) -> dict:
    'Read a training manifest from an artifact directory, or {}.'
    if pretrained_dir is None:
        return {}
    directory = Path(pretrained_dir)
    if not directory.exists():
        return {}
    for candidate in sorted(directory.rglob(TRAINING_MANIFEST_NAME)):
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Ignoring unreadable %s: %s", candidate, exc)
    return {}


def warm_start_allowed(current: dict, parent: dict, image_size: int,
                       allow_incompatible: bool = False) -> tuple:
    'Return ``(allowed, reason)`` for warm-starting from *parent*.'
    if parent and parent.get("unsafe_ancestry") and not allow_incompatible:
        return False, (
            "the parent's weights carry UNSAFE ANCESTRY: somewhere in their history "
            "an incompatibility was overridden, so they may have been fitted on "
            "images this dataset holds out. The flag propagates precisely so that a "
            "single forced warm-start does not become invisible to later "
            "generations. Train cold, or set allow_incompatible to inherit it "
            "knowingly -- which marks this model unsafe too"
        )

    if not parent:
        return False, (
            f"the parent artifact carries no {TRAINING_MANIFEST_NAME}, so there is "
            f"no way to tell which images it was trained on. A parent produced "
            f"before split assignment became content-addressed may have trained on "
            f"images this dataset holds out, which would make the quality gate "
            f"score data the model has already seen"
        )

    if not current:
        return False, (
            f"this run has no {DATASET_MANIFEST_NAME}, so the parent's "
            f"compatibility cannot be established"
        )

    if not current.get("split_scheme"):
        return False, (
            "this dataset records no split_scheme, so it cannot be shown that its "
            "holdout excludes the parent's training images"
        )

    for side, metadata in (("current", current), ("parent", parent)):
        if metadata.get("split_scheme_version") is None:
            continue
        if parse_split_scheme_version(metadata.get("split_scheme_version")) is None:
            return (False,
                    f"the {side} dataset records split_scheme_version "
                    f"{metadata.get('split_scheme_version')!r}, which is not a form this project "
                    f"writes; warm start is refused rather than assuming what it meant")

    mismatches = []
    missing = []
    for field in COMPATIBILITY_FIELDS:
        expected = image_size if field == "image_size" else current.get(field)
        actual = parent.get(field)
        if expected in (None, {}, ()) or actual in (None, {}, ()):
            missing.append(
                f"{field}: parent={actual!r} current={expected!r}"
            )
            continue
        if expected != actual:
            mismatches.append(f"{field}: parent={actual!r} current={expected!r}")

    if missing:
        return False, (
            "the compatibility contract cannot be evaluated because required "
            "field(s) are absent (" + "; ".join(missing) + "). An absent field is "
            "not a match: a parent that never recorded its assignment parameters "
            "cannot be shown to have held out this dataset's test images. "
            "Republish the dataset, or train cold"
        )

    if mismatches:
        return False, (
            "the parent was produced under a different contract ("
            + "; ".join(mismatches)
            + "), so its weights may have been fitted on images this dataset "
              "holds out"
        )

    scheme = current.get("split_scheme")
    scheme_version = current.get("split_scheme_version")
    if _requires_identical_dataset(scheme, scheme_version):
        current_id = current.get("dataset_id")
        parent_id = parent.get("dataset_id")
        if not current_id or not parent_id:
            return False, (
                f"the {scheme!r} scheme is not stable across dataset versions, so "
                f"warm-start requires both datasets to be identified, but "
                f"parent dataset_id={parent_id!r} current={current_id!r}"
            )
        if current_id != parent_id:
            return False, (
                f"the {scheme!r} scheme does not keep an image in the same split "
                f"as the dataset changes -- it derives splits from a single "
                f"sequential random stream, so altering the sample count moves "
                f"images between splits. Parent dataset_id={parent_id} differs "
                f"from current {current_id}, so the parent may have trained on "
                f"images this dataset holds out. Warm-start is only permitted "
                f"within one dataset version for this scheme"
            )
        if (scheme, version_or_none(current)) in UNTRUSTED_LEGACY_CONTRACTS:
            return False, (
                f"the parent was produced under {scheme} "
                f"v{current.get('split_scheme_version')}, whose recorded ancestry cannot "
                f"be trusted: that version permitted cross-dataset inheritance without "
                f"marking it unsafe, so a clean-looking parent may carry weights fitted "
                f"on images this dataset holds out, and the bounded ancestry reference no "
                f"longer names the dataset it came from. Republish under v"
                f"{SPLIT_SCHEME_V2_DECODED_RGB} and train cold"
            )
        return True, (
            f"parent and current dataset are the SAME version "
            f"(dataset_id={current_id}) under scheme {scheme} "
            f"v{current.get('split_scheme_version')}, so no image can have moved "
            f"between splits"
        )

    return True, (
        f"parent and current dataset share the same split scheme "
        f"({scheme} v{current.get('split_scheme_version')}), preprocessing schema "
        f"and label mapping, and that scheme assigns each image's split from its "
        f"own content, so an image cannot have moved between splits between them"
    )


def resolve_warm_start(current: dict, pretrained_dir, image_size: int,
                       allow_incompatible: bool = False) -> tuple:
    'Return ``(use_parent, parent_manifest, reason, forced)`` and log the decision.'
    checkpoint, parent = find_parent_checkpoint(pretrained_dir)
    allowed, reason = warm_start_allowed(current, parent, image_size)
    if allowed and checkpoint is None:
        # Cannot happen via find_parent_checkpoint (no checkpoint yields no
        # manifest, which warm_start_allowed refuses), but asserted so a future
        # change cannot reintroduce ancestry without weights.
        allowed, reason = False, "the parent artifact contains no checkpoint"

    if allowed:
        logger.info("Warm start ACCEPTED: %s", reason)
        return True, parent, reason, False

    if allow_incompatible:
        logger.warning(
            "Warm start FORCED despite an incompatible parent: %s. The reported "
            "quality-gate score may not be a held-out score. This model will be "
            "marked with unsafe ancestry, and so will every model warm-started "
            "from it.", reason,
        )
        return True, parent, reason, True

    logger.warning(
        "COLD START: %s. Training from initial weights instead. Pass "
        "--allow_incompatible_warm_start to override, accepting that the gate "
        "score may not be held out.", reason,
    )
    return False, parent, reason, False
