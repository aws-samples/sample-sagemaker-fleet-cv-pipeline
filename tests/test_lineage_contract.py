# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'What a training manifest must record, and what it must refuse to forget.'
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.lineage import warm_start_allowed, write_training_manifest

CONTENT_ADDRESSED = {
    "dataset_id": "D1",
    "split_scheme": "content-addressed-sha256",
    "split_scheme_version": 2,
    "label_mapping": "platelets",
    "split_ratios": {"train": 0.7, "validation": 0.15, "test": 0.15},
}


def test_different_ratios_are_incompatible(tmp_path):
    'The scheme name alone does not pin the assignment.'
    parent = write_training_manifest(
        tmp_path / "p", CONTENT_ADDRESSED, 256, "classifier"
    )
    changed = dict(
        CONTENT_ADDRESSED,
        split_ratios={"train": 0.5, "validation": 0.15, "test": 0.35},
    )
    allowed, reason = warm_start_allowed(changed, parent, 256)
    assert not allowed
    assert "split_ratios" in reason


def test_unsafe_ancestry_is_refused_by_default(tmp_path):
    """The next generation must not accept forced weights as clean."""
    forced = write_training_manifest(
        tmp_path / "b", CONTENT_ADDRESSED, 256, "classifier",
        parent={"dataset_id": "a"}, unsafe_ancestry=True,
    )
    allowed, reason = warm_start_allowed(CONTENT_ADDRESSED, forced, 256)
    assert not allowed
    assert "UNSAFE ANCESTRY" in reason
    assert warm_start_allowed(
        CONTENT_ADDRESSED, forced, 256, allow_incompatible=True
    )[0]


def test_a_real_cold_start_clears_inherited_contamination(tmp_path):
    clean = write_training_manifest(
        tmp_path / "c", CONTENT_ADDRESSED, 256, "classifier", parent=None
    )
    assert clean["unsafe_ancestry"] is False
    assert clean["parent"] is None
    assert warm_start_allowed(CONTENT_ADDRESSED, clean, 256)[0]


def test_manifest_size_is_bounded_over_many_generations(tmp_path):
    'Embedding the parent whole grew to ~10 MB and then failed to serialize.'
    parent = None
    sizes = []
    for generation in range(1, 1001):
        parent = write_training_manifest(
            tmp_path / f"g{generation}", CONTENT_ADDRESSED, 256, "classifier",
            parent=parent,
        )
        sizes.append(len(json.dumps(parent)))

    assert sizes[-1] < 2000, f"manifest grew to {sizes[-1]} bytes"
    assert sizes[-1] - sizes[10] < 100, (
        f"still growing with generation: {sizes[10]} -> {sizes[-1]}"
    )
    json.loads(json.dumps(parent))  # must not RecursionError


@pytest.fixture(scope="module")
def download_bccd():
    spec = importlib.util.spec_from_file_location(
        "download_bccd_lineage", REPO_ROOT / "scripts/download_bccd.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_bccd_lineage"] = module
    spec.loader.exec_module(module)
    return module


def _annotation(tmp_path, name, objects):
    body = "".join(
        f"<object>{'<name>' + o + '</name>' if o is not None else ''}</object>"
        for o in objects
    )
    path = tmp_path / f"{name}.xml"
    path.write_text(f"<annotation>{body}</annotation>")
    return path


def test_an_unknown_class_name_is_refused(tmp_path, download_bccd):
    'A typo must not silently become a `good` label.'
    path = _annotation(tmp_path, "e", ["RBC", "Platelet"])
    with pytest.raises(ValueError, match="not in the known BCCD vocabulary"):
        download_bccd.parse_voc_annotation(path)


def test_platelets_are_detected(tmp_path, download_bccd):
    """The parser returns every recognised object; label_images checks platelets."""
    path = _annotation(tmp_path, "d", ["RBC", "Platelets"])
    objects = download_bccd.parse_voc_annotation(path)
    assert [o["class"] for o in objects] == ["RBC", "Platelets"]
    assert any(o["class"] == "Platelets" for o in objects), "would be labelled defect"


def test_the_manifest_beside_the_checkpoint_is_used(tmp_path):
    from utils.lineage import find_parent_checkpoint

    artifact = tmp_path / "artifact"
    (artifact / "m").mkdir(parents=True)
    write_training_manifest(artifact / "m", CONTENT_ADDRESSED, 256, "classifier")
    (artifact / "m" / "model.pt").write_bytes(b"weights")

    checkpoint, manifest = find_parent_checkpoint(artifact)
    assert checkpoint.parent.name == "m"
    assert manifest["dataset_id"] == CONTENT_ADDRESSED["dataset_id"]


def test_two_checkpoints_are_ambiguous(tmp_path):
    """Picking by filesystem order could load unvalidated weights."""
    from utils.lineage import find_parent_checkpoint

    artifact = tmp_path / "artifact"
    for name in ("a", "b"):
        (artifact / name).mkdir(parents=True)
        (artifact / name / "model.pt").write_bytes(name.encode())
    with pytest.raises(ValueError, match="checkpoints"):
        find_parent_checkpoint(artifact)


def test_the_warm_start_load_is_gated_on_the_parent_decision():
    'The load must follow use_parent, not the payload.'
    import ast

    source = (REPO_ROOT / "src" / "model" / "train_anomalib.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    guards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "load_state_dict"
            for inner in ast.walk(node)
        ):
            guards.append(node.test)

    assert len(guards) == 1, (
        f"expected exactly one guarded load_state_dict in train_anomalib, found "
        f"{len(guards)}"
    )
    test = guards[0]
    assert isinstance(test, ast.Name) and test.id == "use_parent", (
        "the warm-start load is not gated on use_parent, so the weights loaded and "
        "the ancestry recorded can disagree"
    )

    assert "did not deserialize to a" in source, (
        "no payload validation at the torch.load site; a None or empty checkpoint "
        "would reach the gate above and crash instead of reporting which file"
    )


def test_the_parent_digest_survives_into_the_child_manifest(tmp_path):
    "The digest is only useful if it reaches the CHILD's parent reference."
    import hashlib

    from utils.lineage import (TRAINING_MANIFEST_NAME, find_parent_checkpoint,
                               write_training_manifest)

    payload = b"parent-weights-bytes-that-are-distinctive"
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    (parent_dir / "model.pt").write_bytes(payload)
    (parent_dir / TRAINING_MANIFEST_NAME).write_text(json.dumps({
        "camera_id": "cam-001", "model_type": "anomalib", "run_id": "parent-run",
        "split_contract_version": 1,
    }))

    _, parent_manifest = find_parent_checkpoint(parent_dir)
    expected = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    assert parent_manifest.get("artifact_digest") == expected, "the parent digest is wrong"

    child_dir = tmp_path / "child"
    child_dir.mkdir()
    child = write_training_manifest(
        child_dir,
        dataset_manifest={"camera_id": "cam-001", "split_contract_version": 1},
        image_size=256,
        model_name="anomalib",
        parent=parent_manifest,
    )

    recorded = (child.get("parent") or {}).get("artifact_digest")
    assert recorded == expected, (
        f"the child's parent reference records {recorded!r} rather than the digest of the "
        f"checkpoint it inherited ({expected!r}); the child cannot identify its weights"
    )
