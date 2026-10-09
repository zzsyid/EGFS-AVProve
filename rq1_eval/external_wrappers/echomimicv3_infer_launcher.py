#!/usr/bin/env python3
"""Launch EchoMimicV3 infer_flash with local compatibility patches loaded first."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--echomimicv3-root", type=Path, required=True)
    parser.add_argument("infer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    infer_args = args.infer_args
    if infer_args and infer_args[0] == "--":
        infer_args = infer_args[1:]

    patch_dir = Path(__file__).with_name("echomimicv3_sitecustomize")
    sys.path.insert(0, str(patch_dir))
    __import__("sitecustomize")

    def _allow_legacy_torch_load() -> None:
        return None

    import transformers.modeling_utils as _modeling_utils
    from transformers.utils import import_utils as _import_utils

    _import_utils.check_torch_load_is_safe = _allow_legacy_torch_load
    _modeling_utils.check_torch_load_is_safe = _allow_legacy_torch_load

    infer_path = args.echomimicv3_root / "infer_flash.py"
    if not infer_path.exists():
        raise FileNotFoundError(infer_path)

    sys.path.insert(0, str(args.echomimicv3_root))
    os.chdir(args.echomimicv3_root)
    sys.argv = [str(infer_path), *infer_args]
    runpy.run_path(str(infer_path), run_name="__main__")


if __name__ == "__main__":
    main()
