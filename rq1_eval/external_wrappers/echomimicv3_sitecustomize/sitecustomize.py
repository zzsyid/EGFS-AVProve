"""EchoMimicV3 startup patches for server-side inference."""

from __future__ import annotations

try:
    from transformers.utils import import_utils as _import_utils
except Exception:  # pragma: no cover
    _import_utils = None


def _allow_legacy_torch_load() -> None:
    return None


if _import_utils is not None and hasattr(_import_utils, "check_torch_load_is_safe"):
    _import_utils.check_torch_load_is_safe = _allow_legacy_torch_load
