#!/usr/bin/env python3
"""M7 provenance-aware carrier payload codecs.

M7.1 audio v2 replaces the previous raw payload layout:

    224 data/check bits + 32 keyed checksum

with a global BCH payload:

    BCH(255, k, t) + 1 version/padding bit

For the main audio policy, BCH(m=8, t=16) gives n=255 and k=131.  The 131 data
bits are exactly the minimum provenance lookup tuple:

    source_pointer fragment: 32 bits
    nonce:                   64 bits
    registry hash fragment:  32 bits
    policy/version:           3 bits

The encoded 256-bit carrier is PRF-interleaved so low-rate random bit flips are
spread across the BCH codeword before decoding.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from fuzzy import BCHCode


AUDIO_BCH_V2_POLICY = "tdsc-m7-audio-bch-t16-v2"
AUDIO_BCH_V2_VERSION_BITS = [0, 0, 1]
AUDIO_BCH_V2_TOTAL_BITS = 256
AUDIO_BCH_V2_PAD_BIT = 0


@dataclass(frozen=True)
class AudioBCHV2Decode:
    ok: bool
    data_bits: List[int]
    data_hex: str
    corrected_payload_bits: List[int]
    corrected_payload_hex: str
    corrected_codeword_hex: str
    hamming_corrected: int | None
    reason: str
    policy_version_bits_ok: bool
    pad_bit_ok: bool
    bch_n: int
    bch_k: int
    bch_t: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "audio_bch_v2_ok": self.ok,
            "audio_bch_v2_data_hex": self.data_hex,
            "audio_bch_v2_corrected_payload_hex": self.corrected_payload_hex,
            "audio_bch_v2_corrected_codeword_hex": self.corrected_codeword_hex,
            "audio_bch_v2_hamming_corrected": self.hamming_corrected,
            "audio_bch_v2_reason": self.reason,
            "audio_bch_v2_policy_version_bits_ok": self.policy_version_bits_ok,
            "audio_bch_v2_pad_bit_ok": self.pad_bit_ok,
            "audio_bch_v2_bch_n": self.bch_n,
            "audio_bch_v2_bch_k": self.bch_k,
            "audio_bch_v2_bch_t": self.bch_t,
        }


@lru_cache(maxsize=None)
def get_bch(t: int) -> BCHCode:
    return BCHCode(m=8, t=int(t))


def prf(master_key: str, *parts: object) -> str:
    h = hashlib.sha256()
    h.update(master_key.encode("utf-8"))
    for part in parts:
        h.update(b"|")
        h.update(str(part).encode("utf-8"))
    return h.hexdigest()


def hex_fragment(value: str, n_bits: int) -> str:
    if n_bits <= 0:
        return ""
    n_hex = math.ceil(n_bits / 4)
    raw = value.lower()
    if not raw or any(c not in "0123456789abcdef" for c in raw):
        raw = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return raw[:n_hex]


def hex_to_bits(payload_hex: str, n_bits: int) -> List[int]:
    if n_bits <= 0:
        return []
    raw = bin(int(payload_hex or "0", 16))[2:].zfill(len(payload_hex) * 4)
    return [int(bit) for bit in raw[:n_bits]]


def bits_to_hex(bits: Iterable[int]) -> str:
    bit_list = [int(bit) for bit in bits]
    if not bit_list:
        return ""
    bit_text = "".join(str(bit) for bit in bit_list)
    n_hex = math.ceil(len(bit_list) / 4)
    return hex(int(bit_text, 2))[2:].zfill(n_hex)


def bit_hamming(a: Iterable[int], b: Iterable[int]) -> int:
    aa = [int(x) for x in a]
    bb = [int(x) for x in b]
    n = max(len(aa), len(bb))
    aa = (aa + [0] * n)[:n]
    bb = (bb + [0] * n)[:n]
    return sum(int(x != y) for x, y in zip(aa, bb))


def _permutation(n_bits: int, seed: str, master_key: str) -> List[int]:
    keyed = []
    for index in range(n_bits):
        digest = prf(master_key, "m7-audio-bch-v2-interleave", seed, index)
        keyed.append((digest, index))
    keyed.sort()
    return [index for _, index in keyed]


def interleave_bits(bits: List[int], seed: str, master_key: str) -> List[int]:
    perm = _permutation(len(bits), seed, master_key)
    out = [0] * len(bits)
    for source_index, target_index in enumerate(perm):
        out[target_index] = int(bits[source_index])
    return out


def deinterleave_bits(bits: List[int], seed: str, master_key: str) -> List[int]:
    perm = _permutation(len(bits), seed, master_key)
    out = [0] * len(bits)
    for source_index, target_index in enumerate(perm):
        out[source_index] = int(bits[target_index])
    return out


def _poly_div_exact_low_first(codeword: np.ndarray, divisor: List[int], k: int) -> List[int] | None:
    """Recover message m from non-systematic c(x)=m(x)g(x), low-degree first."""

    work = np.asarray(codeword, dtype=np.uint8).astype(int).tolist()
    divisor_degree = len(divisor) - 1
    quotient = [0] * k
    for degree in range(len(work) - 1, divisor_degree - 1, -1):
        if work[degree]:
            q_index = degree - divisor_degree
            if q_index >= k:
                return None
            quotient[q_index] ^= 1
            for offset, coeff in enumerate(divisor):
                if coeff:
                    work[q_index + offset] ^= 1
    if any(work):
        return None
    return [int(bit) for bit in quotient]


def audio_bch_v2_data_bits(
    source_fragment_hex: str,
    nonce_hex: str,
    registry_fragment_hex: str,
    t: int = 16,
) -> Tuple[List[int], Dict[str, object]]:
    bch = get_bch(t)
    source_bits = hex_to_bits(source_fragment_hex, 32)
    nonce_bits = hex_to_bits(nonce_hex, 64)
    registry_bits = hex_to_bits(registry_fragment_hex, 32)
    data = source_bits + nonce_bits + registry_bits + list(AUDIO_BCH_V2_VERSION_BITS)
    if len(data) > bch.k:
        raise ValueError(f"audio v2 data length {len(data)} exceeds BCH k={bch.k}")
    data = data + [0] * (bch.k - len(data))
    meta = {
        "audio_payload_codec": AUDIO_BCH_V2_POLICY,
        "audio_payload_data_bits": bch.k,
        "audio_payload_payload_bits": AUDIO_BCH_V2_TOTAL_BITS,
        "audio_bch_m": 8,
        "audio_bch_t": t,
        "audio_bch_n": bch.n,
        "audio_bch_k": bch.k,
        "audio_bch_version_bits": "".join(str(bit) for bit in AUDIO_BCH_V2_VERSION_BITS),
        "audio_bch_source_pointer_bits": 32,
        "audio_bch_nonce_bits": 64,
        "audio_bch_registry_record_bits": 32,
    }
    return data, meta


def encode_audio_v2_bits(
    data_bits: List[int],
    seed: str,
    master_key: str,
    t: int = 16,
) -> Tuple[List[int], Dict[str, object]]:
    bch = get_bch(t)
    if len(data_bits) > bch.k:
        raise ValueError(f"data length {len(data_bits)} exceeds BCH k={bch.k}")
    msg = np.asarray(data_bits + [0] * (bch.k - len(data_bits)), dtype=np.uint8)
    codeword = bch._encode_poly(msg)
    carrier = [int(bit) for bit in codeword.tolist()] + [AUDIO_BCH_V2_PAD_BIT]
    encoded = interleave_bits(carrier, seed, master_key)
    meta = {
        "audio_payload_codec": AUDIO_BCH_V2_POLICY,
        "audio_payload_hex": bits_to_hex(encoded),
        "audio_payload_n_bits": len(encoded),
        "audio_bch_codeword_hex": bits_to_hex(codeword.tolist()),
        "audio_bch_data_hex": bits_to_hex(msg.tolist()),
        "audio_bch_m": 8,
        "audio_bch_t": t,
        "audio_bch_n": bch.n,
        "audio_bch_k": bch.k,
        "audio_bch_interleaver": "prf-sort-v1",
        "audio_bch_pad_bit": AUDIO_BCH_V2_PAD_BIT,
    }
    return encoded, meta


def encode_audio_v2_from_fragments(
    source_fragment_hex: str,
    nonce_hex: str,
    registry_fragment_hex: str,
    seed: str,
    master_key: str,
    t: int = 16,
) -> Tuple[List[int], Dict[str, object]]:
    data, data_meta = audio_bch_v2_data_bits(source_fragment_hex, nonce_hex, registry_fragment_hex, t=t)
    encoded, enc_meta = encode_audio_v2_bits(data, seed, master_key, t=t)
    enc_meta.update(data_meta)
    return encoded, enc_meta


def decode_audio_v2_bits(
    recovered_payload_bits: List[int],
    seed: str,
    master_key: str,
    t: int = 16,
) -> AudioBCHV2Decode:
    bch = get_bch(t)
    payload = (recovered_payload_bits[:AUDIO_BCH_V2_TOTAL_BITS]
               + [0] * max(0, AUDIO_BCH_V2_TOTAL_BITS - len(recovered_payload_bits)))
    deinterleaved = deinterleave_bits(payload, seed, master_key)
    codeword_bits = deinterleaved[:bch.n]
    pad_bit = int(deinterleaved[bch.n])
    decoded = bch.decode(np.asarray(codeword_bits, dtype=np.uint8))
    if decoded is None:
        return AudioBCHV2Decode(
            ok=False,
            data_bits=[],
            data_hex="",
            corrected_payload_bits=payload,
            corrected_payload_hex=bits_to_hex(payload),
            corrected_codeword_hex="",
            hamming_corrected=None,
            reason="bch_decode_failed",
            policy_version_bits_ok=False,
            pad_bit_ok=(pad_bit == AUDIO_BCH_V2_PAD_BIT),
            bch_n=bch.n,
            bch_k=bch.k,
            bch_t=bch.t,
        )
    message = _poly_div_exact_low_first(decoded, bch.g, bch.k)
    if message is None:
        return AudioBCHV2Decode(
            ok=False,
            data_bits=[],
            data_hex="",
            corrected_payload_bits=payload,
            corrected_payload_hex=bits_to_hex(payload),
            corrected_codeword_hex=bits_to_hex(decoded.tolist()),
            hamming_corrected=None,
            reason="bch_message_recovery_failed",
            policy_version_bits_ok=False,
            pad_bit_ok=(pad_bit == AUDIO_BCH_V2_PAD_BIT),
            bch_n=bch.n,
            bch_k=bch.k,
            bch_t=bch.t,
        )
    corrected_carrier = [int(bit) for bit in decoded.tolist()] + [AUDIO_BCH_V2_PAD_BIT]
    corrected_payload = interleave_bits(corrected_carrier, seed, master_key)
    version_bits = message[32 + 64 + 32: 32 + 64 + 32 + 3]
    version_ok = version_bits == AUDIO_BCH_V2_VERSION_BITS
    pad_ok = pad_bit == AUDIO_BCH_V2_PAD_BIT
    return AudioBCHV2Decode(
        ok=bool(version_ok and pad_ok),
        data_bits=message,
        data_hex=bits_to_hex(message),
        corrected_payload_bits=corrected_payload,
        corrected_payload_hex=bits_to_hex(corrected_payload),
        corrected_codeword_hex=bits_to_hex(decoded.tolist()),
        hamming_corrected=bit_hamming(codeword_bits, decoded.tolist()),
        reason="ok" if version_ok and pad_ok else "version_or_pad_check_failed",
        policy_version_bits_ok=version_ok,
        pad_bit_ok=pad_ok,
        bch_n=bch.n,
        bch_k=bch.k,
        bch_t=bch.t,
    )


def decode_audio_v2_hex(
    recovered_payload_hex: str,
    seed: str,
    master_key: str,
    t: int = 16,
) -> AudioBCHV2Decode:
    return decode_audio_v2_bits(hex_to_bits(recovered_payload_hex, AUDIO_BCH_V2_TOTAL_BITS), seed, master_key, t=t)


def selftest() -> None:
    master_key = "tdsc-m7-cross-modal-payload-v1"
    seed = "seed1234"
    source = "aabbccdd"
    nonce = "0011223344556677"
    registry = "11223344"
    payload_bits, meta = encode_audio_v2_from_fragments(source, nonce, registry, seed, master_key, t=16)
    dec = decode_audio_v2_bits(payload_bits, seed, master_key, t=16)
    assert dec.ok, dec
    assert dec.data_bits[:32] == hex_to_bits(source, 32)
    assert dec.data_bits[32:96] == hex_to_bits(nonce, 64)
    assert dec.data_bits[96:128] == hex_to_bits(registry, 32)
    noisy = list(payload_bits)
    for index in [0, 7, 19, 23, 41, 63, 88, 103, 127, 151, 170, 199, 211, 230, 244, 255]:
        noisy[index] ^= 1
    dec_noisy = decode_audio_v2_bits(noisy, seed, master_key, t=16)
    assert dec_noisy.ok, dec_noisy
    assert dec_noisy.data_hex == dec.data_hex
    print({
        "policy": AUDIO_BCH_V2_POLICY,
        "payload_bits": len(payload_bits),
        "bch_k": meta["audio_bch_k"],
        "noisy_hamming_corrected": dec_noisy.hamming_corrected,
    })


if __name__ == "__main__":
    selftest()
