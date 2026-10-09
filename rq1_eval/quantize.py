#!/usr/bin/env python3
"""Quantization Q: R^D -> {0,1}^n  (RQ1_eval_spec.md sec.3).

RQ1 is fundamentally the question: can the Hamming geometry of Q(z) separate
same-source from tamper? Two backends:

  Q=lsh    : n keyed random hyperplanes, sign(z . w_i).  Zero-config, Stage-0
             default. Hyperplanes are derived from a key via a PRF (sha256-seeded
             RNG) -> "keyed" per the keyed-embedding design, NOT security by
             obscurity (security rests on the signature; keying just binds the
             projection to a registration key and is reproducible).
  Q=ithash : ITQ learned binary hash (fit on data), for Stage-1. Also helps the
             sec.4.5 CE-predictability concern. Implemented numpy-only.

n in {128, 256, 512} scannable (use 255 to line up with BCH(2^8-1) lengths).
One quantizer per channel (channels have different input dims); per-channel keys
are derived from a master key + channel name so each channel's projection differs
but is reproducible.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

import numpy as np


def _seed_from_key(key: bytes, *tags: str) -> int:
    h = hashlib.sha256()
    h.update(key)
    for t in tags:
        h.update(b"|")
        h.update(t.encode())
    return int.from_bytes(h.digest()[:8], "big")


class LSHQuantizer:
    """Sign-of-random-projection LSH with keyed hyperplanes."""

    kind = "lsh"

    def __init__(self, in_dim: int, n_bits: int, key: bytes = b"rq1", tag: str = ""):
        self.in_dim = in_dim
        self.n_bits = n_bits
        self.key = key
        self.tag = tag
        rng = np.random.default_rng(_seed_from_key(key, "lsh", tag, str(in_dim), str(n_bits)))
        # [n_bits, in_dim] gaussian hyperplanes
        self.W = rng.standard_normal((n_bits, in_dim)).astype(np.float32)

    def encode(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        if z.shape[0] != self.in_dim:
            raise ValueError(f"[lsh:{self.tag}] dim {z.shape[0]} != {self.in_dim}")
        proj = self.W @ z
        return (proj >= 0).astype(np.uint8)

    def encode_batch(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=np.float32)
        return (Z @ self.W.T >= 0).astype(np.uint8)


class ITQQuantizer:
    """Iterative Quantization (Gong & Lazebnik). PCA -> learned rotation -> sign.

    Stage-1 upgrade. fit(X) learns mean, PCA components (to n_bits), and rotation
    R minimizing quantization loss ||sign(V R) - V R||. Before fit, falls back to
    keyed LSH on the raw input so the pipeline still runs.
    """

    kind = "ithash"

    def __init__(self, in_dim: int, n_bits: int, key: bytes = b"rq1", tag: str = "",
                 n_iter: int = 50):
        self.in_dim = in_dim
        self.n_bits = n_bits
        self.key = key
        self.tag = tag
        self.n_iter = n_iter
        self.mean_: Optional[np.ndarray] = None
        self.pca_: Optional[np.ndarray] = None   # [n_bits, in_dim]
        self.R_: Optional[np.ndarray] = None      # [n_bits, n_bits]
        self._fallback = LSHQuantizer(in_dim, n_bits, key, tag=tag + ":itq-fallback")

    @property
    def fitted(self) -> bool:
        return self.R_ is not None

    def fit(self, X: np.ndarray) -> "ITQQuantizer":
        X = np.asarray(X, dtype=np.float32)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n_bits, Vt.shape[0])
        pca = Vt[:k]
        if k < self.n_bits:
            pca = np.vstack([pca, np.zeros((self.n_bits - k, self.in_dim), np.float32)])
        self.pca_ = pca.astype(np.float32)
        V = Xc @ self.pca_.T                       # [N, n_bits]
        rng = np.random.default_rng(_seed_from_key(self.key, "itq", self.tag))
        R, _ = np.linalg.qr(rng.standard_normal((self.n_bits, self.n_bits)))
        for _ in range(self.n_iter):
            Z = V @ R
            B = np.where(Z >= 0, 1.0, -1.0)
            U, _, Wt = np.linalg.svd(B.T @ V)
            R = (Wt.T @ U.T)
        self.R_ = R.astype(np.float32)
        return self

    def encode(self, z: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return self._fallback.encode(z)
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        v = (z - self.mean_) @ self.pca_.T
        return (v @ self.R_ >= 0).astype(np.uint8)


def build_quantizer(kind: str, in_dim: int, n_bits: int,
                    key: bytes = b"rq1", tag: str = ""):
    if kind == "lsh":
        return LSHQuantizer(in_dim, n_bits, key, tag)
    if kind == "ithash":
        return ITQQuantizer(in_dim, n_bits, key, tag)
    raise ValueError(f"unknown quantizer kind {kind!r}")


class QuantizerBank:
    """Per-channel quantizers built from channel dims + a master key."""

    def __init__(self, dims: Dict[str, int], n_bits: int, kind: str = "lsh",
                 master_key: bytes = b"rq1-master"):
        self.n_bits = n_bits
        self.kind = kind
        self.q = {
            ch: build_quantizer(kind, d, n_bits,
                                 key=master_key, tag=ch)
            for ch, d in dims.items()
        }

    def encode(self, channel: str, z: np.ndarray) -> np.ndarray:
        return self.q[channel].encode(z)

    def fit(self, samples_by_channel: Dict[str, np.ndarray]) -> None:
        """Fit ITQ quantizers (no-op for LSH)."""
        if self.kind != "ithash":
            return
        for ch, X in samples_by_channel.items():
            self.q[ch].fit(X)


# --------------------------------------------------------------------------- #
def _selftest() -> None:
    rng = np.random.default_rng(0)
    print("== LSH determinism + dim handling ==")
    for d, n in [(192, 255), (512, 255), (256, 255), (768, 511)]:
        q1 = LSHQuantizer(d, n, key=b"k", tag="t")
        q2 = LSHQuantizer(d, n, key=b"k", tag="t")
        z = rng.standard_normal(d).astype(np.float32)
        z /= np.linalg.norm(z)
        w1, w2 = q1.encode(z), q2.encode(z)
        assert w1.shape == (n,) and w1.dtype == np.uint8
        assert np.array_equal(w1, w2), "LSH not deterministic for same key"
    print("  shapes + determinism OK")

    print("== different keys -> different codes ==")
    qa = LSHQuantizer(256, 255, key=b"kA", tag="t")
    qb = LSHQuantizer(256, 255, key=b"kB", tag="t")
    z = rng.standard_normal(256).astype(np.float32); z /= np.linalg.norm(z)
    ham = int(np.count_nonzero(qa.encode(z) != qb.encode(z)))
    print(f"  cross-key Hamming = {ham}/255 (expect ~half)")
    assert 80 < ham < 175

    print("== close vectors -> small Hamming, orthogonal -> ~half ==")
    q = LSHQuantizer(256, 512, key=b"k", tag="t")
    z = rng.standard_normal(256); z /= np.linalg.norm(z)
    delta = rng.standard_normal(256); delta /= np.linalg.norm(delta)
    zc = z + 0.10 * delta; zc /= np.linalg.norm(zc)   # small-angle perturbation
    zo = rng.standard_normal(256); zo /= np.linalg.norm(zo)
    d_close = np.count_nonzero(q.encode(z) != q.encode(zc)) / 512
    d_orth = np.count_nonzero(q.encode(z) != q.encode(zo)) / 512
    print(f"  d_ham(close)={d_close:.3f}  d_ham(orthogonal)={d_orth:.3f}")
    assert d_close < 0.15 < d_orth

    print("== ITQ fit/encode ==")
    X = rng.standard_normal((400, 256)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    itq = ITQQuantizer(256, 255, key=b"k", tag="t").fit(X)
    w = itq.encode(X[0])
    assert w.shape == (255,) and w.dtype == np.uint8
    # pre-fit fallback works
    itq2 = ITQQuantizer(256, 255, key=b"k", tag="t")
    assert not itq2.fitted and itq2.encode(X[0]).shape == (255,)
    print("  ITQ fit + fallback OK")

    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _selftest()
