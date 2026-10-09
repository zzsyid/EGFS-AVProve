#!/usr/bin/env python3
"""Fuzzy extractor / fuzzy commitment for RQ1 (code-offset construction).

Implements the secure-sketch bridge from continuous neural features (after
quantization Q -> n-bit code w) to discrete cryptographic verification, exactly
as specified in RQ1_eval_spec.md sec.4 and proposal_v5.1.md sec.6.

Construction (Juels-Wattenberg fuzzy commitment / code-offset secure sketch):

    gen(w):                          # w in {0,1}^n
        s  = ecc.random_codeword()   # s in C  (uniformly random codeword)
        p  = w XOR s                 # public helper data (the "sketch")
        c  = w                       # canonical code = the quantized code itself
        D  = H(c || nonce)           # commitment digest registered publicly
        return c, p, D

    rep(w', p):                      # returns recovered code w_rec or None
        s_hat = ecc.decode(p XOR w') # decode(s XOR e), e = w XOR w'
        if s_hat is None: return None # outside decoding ball -> bottom
        return p XOR s_hat            # == w  iff  Hamming(w, w') <= t

Single-channel content-accept:
        H(rep(w', p) || nonce) == D   <==>(under H collision resistance)  Hamming(w, w') <= t

tau == ECC correction radius t (in bits). The ECC backend exposes (n, k, t).

Design notes / why pure-numpy BCH:
  * bchlib / galois are NOT installed in the target env, so a self-contained
    binary BCH codec is implemented here (GF(2^m) tables, generator polynomial
    from cyclotomic cosets, syndrome + Berlekamp-Massey + Chien-search decode).
  * For the RQ1 separability metrics (outputs (1)(2)(3)) the operative predicate
    is content_accept(w, w', t) == [Hamming(w, w') <= t]; metrics_sep.py uses
    that predicate directly and never needs the heavy codec. The full FE here is
    used to (a) demonstrate the real construction is sound, and (b) drive the
    Stage-2 five-state verifier. A self-test (`python fuzzy.py`) checks that the
    real BCH FE agrees with the Hamming predicate inside the guaranteed radius.

Everything is bit-vectors as numpy uint8 arrays of 0/1, low index = bit 0.
Polynomials over GF(2) and GF(2^m) are python int lists, low index = low degree.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

BitVec = np.ndarray  # uint8 array of {0,1}

# Default primitive polynomials phi(x) for GF(2^m), as integer bitmasks.
# Bit i set => x^i present. These give n = 2^m - 1.
PRIMITIVE_POLY = {
    4: 0b10011,        # x^4 + x + 1                -> n = 15
    5: 0b100101,       # x^5 + x^2 + 1              -> n = 31
    6: 0b1000011,      # x^6 + x + 1                -> n = 63
    7: 0b10001001,     # x^7 + x^3 + 1              -> n = 127
    8: 0b100011101,    # x^8 + x^4 + x^3 + x^2 + 1  -> n = 255
    9: 0b1000010001,   # x^9 + x^4 + 1              -> n = 511
    10: 0b10000001001,  # x^10 + x^3 + 1            -> n = 1023
}


# --------------------------------------------------------------------------- #
# GF(2^m) arithmetic
# --------------------------------------------------------------------------- #
class GF2m:
    """Galois field GF(2^m) with exp/log tables built from a primitive poly."""

    def __init__(self, m: int, prim: Optional[int] = None):
        if prim is None:
            if m not in PRIMITIVE_POLY:
                raise ValueError(f"No default primitive polynomial for m={m}")
            prim = PRIMITIVE_POLY[m]
        self.m = m
        self.n = (1 << m) - 1  # field has 2^m elements; multiplicative order n
        self.prim = prim
        size = 1 << m
        self.exp = [0] * (2 * size)
        self.log = [0] * size
        x = 1
        for i in range(self.n):
            self.exp[i] = x
            self.log[x] = i
            x <<= 1
            if x & size:
                x ^= prim
        # extend exp table for convenient modular indexing
        for i in range(self.n, 2 * size):
            self.exp[i] = self.exp[i - self.n]
        self.log[0] = -1  # undefined; sentinel

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[(self.log[a] + self.log[b]) % self.n]

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("inverse of 0 in GF(2^m)")
        return self.exp[(self.n - self.log[a]) % self.n]

    def pow_alpha(self, e: int) -> int:
        """Return alpha^e (alpha = primitive element), e may be negative."""
        return self.exp[e % self.n]


# --------------------------------------------------------------------------- #
# Binary BCH code
# --------------------------------------------------------------------------- #
class BCHCode:
    """Narrow-sense binary BCH code over GF(2^m), designed distance 2t+1.

    Exposes (n, k, t). Provides:
      random_codeword() -> uint8[n]
      decode(received uint8[n]) -> corrected uint8[n] or None
    Decoding is bounded-distance: guaranteed to correct any pattern of <= t
    bit errors; for > t errors it returns either None (decode failure) or a
    (possibly wrong) codeword. The latter case is *measured* in RQ1, not assumed
    away -- metrics rely on the clean Hamming predicate, see module docstring.
    """

    def __init__(self, m: int, t: int, prim: Optional[int] = None):
        self.gf = GF2m(m, prim)
        self.m = m
        self.n = self.gf.n
        self.t = t
        self.g = self._generator_poly(t)        # GF(2) poly, low->high
        self.deg_g = len(self.g) - 1
        self.k = self.n - self.deg_g
        if self.k <= 0:
            raise ValueError(
                f"BCH(m={m}, t={t}) gives k={self.k} <= 0; reduce t or raise m"
            )

    # ---- construction --------------------------------------------------- #
    def _minimal_poly(self, i: int) -> List[int]:
        """Minimal polynomial of alpha^i over GF(2), coeffs low->high (0/1)."""
        gf = self.gf
        n = gf.n
        # cyclotomic coset of i under x -> 2x mod n
        coset = []
        x = i % n
        while x not in coset:
            coset.append(x)
            x = (x * 2) % n
        # poly = prod_{j in coset} (z + alpha^j), coeffs in GF(2^m)
        poly = [1]  # low->high, GF(2^m) ints
        for j in coset:
            root = gf.pow_alpha(j)
            new = [0] * (len(poly) + 1)
            for d, c in enumerate(poly):
                new[d + 1] ^= c                  # c * z^{d+1}
                new[d] ^= gf.mul(c, root)        # c * root * z^{d}
            poly = new
        # result must have binary coefficients (element of base field GF(2))
        out = []
        for c in poly:
            if c not in (0, 1):
                raise RuntimeError(
                    f"minimal poly of alpha^{i} not binary (got {c}); "
                    "primitive polynomial likely wrong"
                )
            out.append(c)
        return out

    @staticmethod
    def _poly_mul_gf2(a: List[int], b: List[int]) -> List[int]:
        res = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            if ai:
                for j, bj in enumerate(b):
                    if bj:
                        res[i + j] ^= 1
        return res

    def _generator_poly(self, t: int) -> List[int]:
        """g(x) = lcm of minimal polynomials of alpha^1..alpha^{2t}."""
        g = [1]
        seen_cosets = set()
        for i in range(1, 2 * t + 1):
            # identify coset leader to avoid multiplying same minpoly twice
            coset = []
            x = i % self.gf.n
            while x not in coset:
                coset.append(x)
                x = (x * 2) % self.gf.n
            key = min(coset)
            if key in seen_cosets:
                continue
            seen_cosets.add(key)
            g = self._poly_mul_gf2(g, self._minimal_poly(i))
        return g

    # ---- encode / random codeword --------------------------------------- #
    def _encode_poly(self, msg_bits: np.ndarray) -> np.ndarray:
        """Non-systematic encode c(x) = m(x) * g(x), returned as uint8[n]."""
        m_poly = msg_bits.astype(int).tolist()
        c_poly = self._poly_mul_gf2(m_poly, self.g)  # degree <= n-1
        out = np.zeros(self.n, dtype=np.uint8)
        for d, c in enumerate(c_poly):
            if c:
                out[d] ^= 1
        return out

    def random_codeword(self, rng: Optional[np.random.Generator] = None) -> BitVec:
        rng = rng or np.random.default_rng()
        msg = rng.integers(0, 2, size=self.k, dtype=np.uint8)
        return self._encode_poly(msg)

    # ---- decode --------------------------------------------------------- #
    def _syndromes(self, r: np.ndarray) -> List[int]:
        """S_j = r(alpha^j) for j=1..2t, in GF(2^m)."""
        gf = self.gf
        pos = np.nonzero(r)[0]
        syn = []
        for j in range(1, 2 * self.t + 1):
            acc = 0
            for i in pos:
                acc ^= gf.pow_alpha(int(i) * j)
            syn.append(acc)
        return syn

    def _berlekamp_massey(self, S: List[int]) -> Tuple[List[int], int]:
        """Return error-locator polynomial sigma(x) (low->high) and its degree L."""
        gf = self.gf
        C = [1]      # current locator
        B = [1]      # last locator before length change
        L = 0
        m_ = 1
        b = 1
        for r in range(len(S)):
            # discrepancy d = S[r] + sum_{i=1..L} C[i] * S[r-i]
            d = S[r]
            for i in range(1, L + 1):
                if i < len(C) and C[i] and (r - i) >= 0:
                    d ^= gf.mul(C[i], S[r - i])
            if d == 0:
                m_ += 1
            else:
                coef = gf.mul(d, gf.inv(b))
                # T = C - coef * x^m_ * B
                shifted = [0] * m_ + B
                newC = C[:]
                if len(shifted) > len(newC):
                    newC = newC + [0] * (len(shifted) - len(newC))
                for i in range(len(shifted)):
                    if shifted[i]:
                        newC[i] ^= gf.mul(coef, shifted[i])
                if 2 * L <= r:
                    T = C[:]
                    C = newC
                    L = r + 1 - L
                    B = T
                    b = d
                    m_ = 1
                else:
                    C = newC
                    m_ += 1
        # trim trailing zeros
        while len(C) > 1 and C[-1] == 0:
            C.pop()
        return C, L

    def _chien_search(self, sigma: List[int]) -> List[int]:
        """Return error positions i in [0, n) such that sigma(alpha^{-i}) == 0."""
        gf = self.gf
        positions = []
        for i in range(self.n):
            # evaluate sigma at alpha^{-i}: sum sigma_d * alpha^{-i*d}
            acc = 0
            for d, coef in enumerate(sigma):
                if coef:
                    acc ^= gf.mul(coef, gf.pow_alpha((-i * d) % gf.n))
            if acc == 0:
                positions.append(i)
        return positions

    def decode(self, received: BitVec) -> Optional[BitVec]:
        r = np.asarray(received, dtype=np.uint8).copy()
        if r.shape[0] != self.n:
            raise ValueError(f"received length {r.shape[0]} != n={self.n}")
        S = self._syndromes(r)
        if all(s == 0 for s in S):
            return r  # no error
        sigma, L = self._berlekamp_massey(S)
        if L == 0 or L > self.t:
            return None  # uncorrectable
        positions = self._chien_search(sigma)
        if len(positions) != L:
            return None  # locator degree mismatch -> decode failure
        for p in positions:
            r[p] ^= 1
        # verify corrected word is a codeword
        if any(s != 0 for s in self._syndromes(r)):
            return None
        return r


# --------------------------------------------------------------------------- #
# Idealized bounded-distance ECC (fast fallback for pipeline validation)
# --------------------------------------------------------------------------- #
class RadiusECC:
    """Toy ECC for arbitrary n used only to exercise the FE plumbing quickly.

    It does NOT implement a real algebraic code. `random_codeword` returns the
    all-zero word, and `decode` is only meaningful when called as part of the FE
    on (p XOR w') where the structure guarantees the input is e = w XOR w':
    it returns 0 (the zero codeword) iff weight(e) <= t, else None. This makes
    FuzzyCommitment.rep reproduce w exactly when Hamming(w, w') <= t, matching
    the content-accept predicate. NOT cryptographically meaningful; flagged
    insecure and only for n that has no convenient BCH length in Stage-0 smoke
    tests. Real experiments use BCHCode.
    """

    secure = False

    def __init__(self, n: int, t: int):
        self.n = n
        self.t = t
        self.k = n  # placeholder

    def random_codeword(self, rng: Optional[np.random.Generator] = None) -> BitVec:
        return np.zeros(self.n, dtype=np.uint8)

    def decode(self, received: BitVec) -> Optional[BitVec]:
        r = np.asarray(received, dtype=np.uint8)
        if int(r.sum()) <= self.t:
            return np.zeros(self.n, dtype=np.uint8)
        return None


# --------------------------------------------------------------------------- #
# Hashing / helpers
# --------------------------------------------------------------------------- #
def _bits_to_bytes(bits: BitVec) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def commit(code: BitVec, nonce: bytes, domain: bytes = b"rq1-fe") -> bytes:
    """Commitment digest D = H(domain || nonce || code)."""
    h = hashlib.sha256()
    h.update(domain)
    h.update(len(nonce).to_bytes(2, "big"))
    h.update(nonce)
    h.update(_bits_to_bytes(code))
    return h.digest()


def hamming(a: BitVec, b: BitVec) -> int:
    a = np.asarray(a, dtype=np.uint8)
    b = np.asarray(b, dtype=np.uint8)
    return int(np.count_nonzero(a != b))


def content_accept(w: BitVec, w_prime: BitVec, t: int) -> bool:
    """Operative predicate used by metrics_sep.py: Hamming(w, w') <= t."""
    return hamming(w, w_prime) <= t


# --------------------------------------------------------------------------- #
# Code-offset fuzzy commitment
# --------------------------------------------------------------------------- #
@dataclass
class Enrollment:
    """Public artifacts produced at registration (helper data + commitment)."""
    p: BitVec          # public helper data  (w XOR s)
    D: bytes           # commitment digest   H(domain||nonce||c)
    nonce: bytes       # commitment nonce
    n: int
    t: int

    def to_public(self) -> dict:
        return {
            "p_hex": _bits_to_bytes(self.p).hex(),
            "D_hex": self.D.hex(),
            "nonce_hex": self.nonce.hex(),
            "n": self.n,
            "t": self.t,
        }


class FuzzyCommitment:
    """Code-offset fuzzy commitment around any ECC exposing (n, t)."""

    def __init__(self, ecc, domain: bytes = b"rq1-fe"):
        self.ecc = ecc
        self.n = ecc.n
        self.t = ecc.t
        self.domain = domain

    # registration ------------------------------------------------------- #
    def gen(
        self,
        w: BitVec,
        nonce: Optional[bytes] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[BitVec, Enrollment]:
        """Enroll code w. Returns (canonical c == w, Enrollment(p, D, nonce))."""
        w = np.asarray(w, dtype=np.uint8)
        if w.shape[0] != self.n:
            raise ValueError(f"code length {w.shape[0]} != n={self.n}")
        nonce = nonce if nonce is not None else os.urandom(16)
        s = self.ecc.random_codeword(rng)
        p = np.bitwise_xor(w, s).astype(np.uint8)
        c = w  # canonical code
        D = commit(c, nonce, self.domain)
        return c, Enrollment(p=p, D=D, nonce=nonce, n=self.n, t=self.t)

    # reproduction ------------------------------------------------------- #
    def rep(self, w_prime: BitVec, enr: Enrollment) -> Optional[BitVec]:
        """Reproduce w from a noisy reading w'. Returns w_rec or None (bottom)."""
        w_prime = np.asarray(w_prime, dtype=np.uint8)
        if w_prime.shape[0] != self.n:
            raise ValueError(f"reading length {w_prime.shape[0]} != n={self.n}")
        z = np.bitwise_xor(enr.p, w_prime).astype(np.uint8)  # s XOR e
        s_hat = self.ecc.decode(z)
        if s_hat is None:
            return None
        return np.bitwise_xor(enr.p, np.asarray(s_hat, dtype=np.uint8)).astype(np.uint8)

    def verify(self, w_prime: BitVec, enr: Enrollment) -> bool:
        """Full content-accept: rep != bottom AND commitment matches."""
        w_rec = self.rep(w_prime, enr)
        if w_rec is None:
            return False
        return commit(w_rec, enr.nonce, self.domain) == enr.D


# --------------------------------------------------------------------------- #
# Convenience constructor
# --------------------------------------------------------------------------- #
def make_bch_fe(m: int, t: int, domain: bytes = b"rq1-fe") -> FuzzyCommitment:
    return FuzzyCommitment(BCHCode(m=m, t=t), domain=domain)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    rng = np.random.default_rng(0)
    print("== BCH construction sanity ==")
    for m, t in [(7, 5), (8, 8), (8, 10), (9, 6)]:
        bch = BCHCode(m=m, t=t)
        print(f"  BCH(m={m}, t={t}): n={bch.n}, k={bch.k}, deg(g)={bch.deg_g}")
        assert bch.k > 0

    print("== code-offset FE round trip (real BCH) ==")
    m, t = 8, 10
    fe = make_bch_fe(m, t)
    n = fe.n
    fails_within = 0
    false_within = 0   # rep succeeded but recovered wrong code
    trials = 200
    for _ in range(trials):
        w = rng.integers(0, 2, size=n, dtype=np.uint8)
        c, enr = fe.gen(w, rng=rng)
        # inject exactly t errors -> must recover and verify
        err_pos = rng.choice(n, size=t, replace=False)
        w_prime = w.copy()
        w_prime[err_pos] ^= 1
        assert hamming(w, w_prime) == t
        w_rec = fe.rep(w_prime, enr)
        if w_rec is None or not np.array_equal(w_rec, w):
            fails_within += 1
        if not fe.verify(w_prime, enr):
            false_within += 1
    print(f"  errors==t: recover failures={fails_within}/{trials}, "
          f"verify failures={false_within}/{trials}")
    assert fails_within == 0, "BCH failed to correct <= t errors"
    assert false_within == 0, "verify() rejected a within-radius reading"

    print("== beyond radius should (almost always) reject ==")
    accepted_beyond = 0
    beyond = t + 3
    for _ in range(trials):
        w = rng.integers(0, 2, size=n, dtype=np.uint8)
        c, enr = fe.gen(w, rng=rng)
        err_pos = rng.choice(n, size=beyond, replace=False)
        w_prime = w.copy()
        w_prime[err_pos] ^= 1
        if fe.verify(w_prime, enr):
            accepted_beyond += 1
    print(f"  errors={beyond}: false content-accepts={accepted_beyond}/{trials} "
          "(bounded-distance: should be ~0)")

    print("== content_accept predicate matches verify within radius ==")
    w = rng.integers(0, 2, size=n, dtype=np.uint8)
    c, enr = fe.gen(w, rng=rng)
    for d in range(0, t + 1):
        w_prime = w.copy()
        if d > 0:
            pos = rng.choice(n, size=d, replace=False)
            w_prime[pos] ^= 1
        assert content_accept(w, w_prime, t) == fe.verify(w_prime, enr), \
            f"mismatch at distance {d}"
    print("  predicate == FE.verify for all d in [0, t]  OK")

    print("== RadiusECC fallback ==")
    rfe = FuzzyCommitment(RadiusECC(n=256, t=12))
    w = rng.integers(0, 2, size=256, dtype=np.uint8)
    c, enr = rfe.gen(w, rng=rng)
    wp = w.copy(); wp[rng.choice(256, size=12, replace=False)] ^= 1
    assert rfe.verify(wp, enr)
    wp2 = w.copy(); wp2[rng.choice(256, size=20, replace=False)] ^= 1
    assert not rfe.verify(wp2, enr)
    print("  RadiusECC gen/rep/verify OK")

    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    _selftest()
