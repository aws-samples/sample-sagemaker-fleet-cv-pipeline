# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'Tests on the EXPORTED ONNX graphs, not on the code that submits them.'
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.onnx_export import export_onnx

torch = pytest.importorskip(
    "torch", reason="torch not installed -- pip install -r requirements-dev.txt"
)
onnx = pytest.importorskip(
    "onnx", reason="onnx not installed -- pip install -r requirements-dev.txt"
)
ort = pytest.importorskip("onnxruntime")
import numpy as np  # noqa: E402

EXPECTED_INPUT_NAME = "input"
EXPECTED_OPSET = 11
EXPECTED_IR_VERSION = 6


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def anomaly_onnx(tmp_path_factory):
    """Export the real TinyAutoencoder exactly as train_anomalib.py does."""
    train_anomalib = _load("src/model/train_anomalib.py", "train_anomalib_export")
    model = train_anomalib.TinyAutoencoder().eval()
    path = tmp_path_factory.mktemp("anom") / "model.onnx"
    dummy = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    export_onnx(model, dummy, path, "reconstruction")
    return path, 256


@pytest.fixture(scope="module")
def classifier_onnx(tmp_path_factory):
    """Export the real classifier head exactly as train_classifier.py does."""
    pytest.importorskip("torchvision")
    train_classifier = _load("src/model/train_classifier.py", "train_classifier_export")
    model = train_classifier.build_model(2).eval()
    path = tmp_path_factory.mktemp("clf") / "model.onnx"
    dummy = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    export_onnx(model, dummy, path, "logits")
    return path, 224


@pytest.fixture(params=["anomaly", "classifier"])
def exported(request, anomaly_onnx, classifier_onnx):
    return anomaly_onnx if request.param == "anomaly" else classifier_onnx


def test_input_is_named_as_the_compile_step_expects(exported):
    'The compile step hardcodes "input" in DataInputConfig.'
    path, _ = exported
    model = onnx.load(str(path))
    names = [i.name for i in model.graph.input]
    assert names == [EXPECTED_INPUT_NAME], (
        f"exported input is {names}, but src/model/compile_neo_step.py builds "
        f"DataInputConfig keyed on {EXPECTED_INPUT_NAME!r}"
    )


def test_spatial_dimensions_match_the_compile_shape(exported):
    """Neo needs concrete input shapes; the compile step pins batch 1 at the
    image size the model was exported for."""
    path, image_size = exported
    model = onnx.load(str(path))
    dims = [d.dim_value or d.dim_param
            for d in model.graph.input[0].type.tensor_type.shape.dim]
    assert len(dims) == 4, f"expected NCHW, got {dims}"
    assert dims[1] == 3
    assert dims[2] == image_size and dims[3] == image_size


def test_batch_axis_is_dynamic(exported):
    """Evaluation runs whole splits in one call, so batch must not be fixed."""
    path, _ = exported
    model = onnx.load(str(path))
    batch = model.graph.input[0].type.tensor_type.shape.dim[0]
    assert batch.dim_param, (
        "batch axis is fixed; evaluation batches more than one image at a time"
    )


def test_opset_and_ir_version_are_the_verified_pair(exported):
    'Pinned because these were the values a real Neo job accepted.'
    path, _ = exported
    model = onnx.load(str(path))
    opsets = {(i.domain or "ai.onnx"): i.version for i in model.opset_import}
    assert opsets == {"ai.onnx": EXPECTED_OPSET}, f"unexpected opsets: {opsets}"
    assert model.ir_version == EXPECTED_IR_VERSION


def test_graph_passes_the_onnx_checker(exported):
    path, _ = exported
    onnx.checker.check_model(onnx.load(str(path)))


def test_model_is_self_contained(exported):
    """No external data files: Neo receives a single-file tarball."""
    path, _ = exported
    model = onnx.load(str(path))
    for initializer in model.graph.initializer:
        assert initializer.data_location != onnx.TensorProto.EXTERNAL, (
            f"{initializer.name} stores data externally; the ONNX-only tarball the "
            f"compile step builds would not include it"
        )
    siblings = [p.name for p in path.parent.iterdir() if p != path]
    assert siblings == [], f"export produced side files: {siblings}"


def test_runs_under_onnx_runtime_with_finite_output(exported):
    path, image_size = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    batch = np.random.rand(2, 3, image_size, image_size).astype(np.float32)
    output = session.run(None, {EXPECTED_INPUT_NAME: batch})[0]
    assert output.shape[0] == 2, "dynamic batch did not take effect"
    assert np.isfinite(output).all()
