# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import runpy
import sys
from pathlib import Path


def _read_hyperparameters() -> dict:
    hp_path = Path("/opt/ml/input/config/hyperparameters.json")
    if not hp_path.exists():
        return {}
    raw = json.loads(hp_path.read_text(encoding="utf-8"))
    parsed = {}
    for key, value in raw.items():
        if isinstance(value, str) and len(value) >= 2 and value.startswith('"') and value.endswith('"'):
            parsed[key] = value[1:-1]
        else:
            parsed[key] = value
    return parsed


# The only training programs this image ships. Restricting the dispatch to
# this allowlist means a crafted 'program' hyperparameter cannot execute
# arbitrary files (e.g. via path traversal out of /opt/ml/code).
ALLOWED_PROGRAMS = {
    "train_anomalib.py",
    "train_classifier.py",
}


def main() -> int:
    hp = _read_hyperparameters()
    program = hp.pop("program", "train_anomalib.py")

    if program not in ALLOWED_PROGRAMS:
        print(
            f"Training program not allowed: {program!r}. "
            f"Allowed: {sorted(ALLOWED_PROGRAMS)}",
            file=sys.stderr,
        )
        return 2

    script_path = Path("/opt/ml/code") / program
    if not script_path.exists():
        print(f"Training program not found: {script_path}", file=sys.stderr)
        return 2

    argv = [str(script_path)]
    for key, value in hp.items():
        # Skip SageMaker-injected/internal keys that are not user script arguments.
        if key.startswith("_"):
            continue
        argv.extend([f"--{key}", str(value)])

    print("Launching training program:", " ".join(argv))
    # Run the allowlisted script in this interpreter instead of a subprocess:
    # there is exactly one training program per job, so a child process adds
    # nothing, and in-process execution means no external command is ever
    # constructed from hyperparameter data.
    sys.argv = argv
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return exc.code
        print(exc.code, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
