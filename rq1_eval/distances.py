#!/usr/bin/env python3
"""Distances for RQ1 (RQ1_eval_spec.md sec.6).

Report both continuous and quantized distances:

    d_cos(z, z')              continuous cosine distance, diagnostic only
    d_ham_bits(w, w')         Hamming distance in bits  (FE pass/fail basis)
    d_ham(w, w') = d_ham_bits / n   normalized

All separability / epsilon metrics are decided on Hamming; d_cos only explains
*why* a channel failed. Inputs z are assumed L2-normalized (channels.py); inputs
w are uint8 {0,1} codes (quantize.py).
"""

from __future__ import annotations

import numpy as np


def d_cos(z: np.ndarray, z2: np.ndarray) -> float:
    """Cosine distance 1 - <z, z2> / (|z||z2|). Robust to non-unit inputs."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    z2 = np.asarray(z2, dtype=np.float64).reshape(-1)
    nz = np.linalg.norm(z)
    nz2 = np.linalg.norm(z2)
    if nz == 0 or nz2 == 0:
        return float("nan")
    return float(1.0 - np.dot(z, z2) / (nz * nz2))


def d_ham_bits(w: np.ndarray, w2: np.ndarray) -> int:
    w = np.asarray(w, dtype=np.uint8).reshape(-1)
    w2 = np.asarray(w2, dtype=np.uint8).reshape(-1)
    if w.shape != w2.shape:
        raise ValueError(f"code length mismatch {w.shape} vs {w2.shape}")
    return int(np.count_nonzero(w != w2))


def d_ham(w: np.ndarray, w2: np.ndarray) -> float:
    w = np.asarray(w).reshape(-1)
    n = w.shape[0]
    return d_ham_bits(w, w2) / n if n else float("nan")


def _selftest() -> None:
    rng = np.random.default_rng(0)
    z = rng.standard_normal(256); z /= np.linalg.norm(z)
    assert abs(d_cos(z, z)) < 1e-9
    assert abs(d_cos(z, -z) - 2.0) < 1e-9
    w = rng.integers(0, 2, 255, dtype=np.uint8)
    w2 = w.copy(); w2[:10] ^= 1
    assert d_ham_bits(w, w2) == 10
    assert abs(d_ham(w, w2) - 10 / 255) < 1e-9
    print("distances self-test OK")


if __name__ == "__main__":
    _selftest()
