# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

'The ONE ONNX export used by both trainers and by the export-contract tests.'
import logging

logger = logging.getLogger(__name__)

# Verified against real SageMaker Neo compilation jobs in us-east-2: the graphs
# these settings produce compile, and they carry IR version 6 / opset 11.
OPSET_VERSION = 11
INPUT_NAME = "input"
BATCH_AXIS = {0: "batch"}


def export_onnx(model, dummy_input, onnx_path, output_name: str):
    'Export *model* to *onnx_path* under the settings Neo was verified against.'
    import torch  # imported here so this module stays importable without torch

    model_cpu = model.cpu().eval()
    torch.onnx.export(
        model_cpu,
        dummy_input,
        onnx_path,
        input_names=[INPUT_NAME],
        output_names=[output_name],
        dynamic_axes={INPUT_NAME: BATCH_AXIS, output_name: BATCH_AXIS},
        opset_version=OPSET_VERSION,
        dynamo=False,
    )
    logger.info("Exported ONNX model to %s (opset %d, output %r)",
                onnx_path, OPSET_VERSION, output_name)
    return onnx_path
