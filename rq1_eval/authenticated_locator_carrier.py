#!/usr/bin/env python3
"""Carrier runtime for authenticated Audio32/Video24 provenance locators."""

from __future__ import annotations

import base64
import csv
import hashlib
import math
import subprocess
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import torch
import torchaudio
import torchvision

from registry_signatures import canonical, load_public, read_jsonl, unsigned


PACKET_BITS = 16
INDEX_BITS = 2
DATA_BITS_PER_PACKET = 8
CRC_BITS = 6
MAX_PACKETS = 4


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def run(command: List[str], input_bytes: bytes | None = None) -> None:
    subprocess.run(command, input=input_bytes, check=True)


def safe_name(value: str) -> str:
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in value
    )
    return cleaned.strip("_") or "item"


def bits_to_int(bits: List[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def int_to_bits(value: int, n_bits: int) -> List[int]:
    return [int(bit) for bit in bin(int(value))[2:].zfill(n_bits)[-n_bits:]]


def hex_to_bits(payload_hex: str, n_bits: int) -> List[int]:
    if n_bits <= 0:
        return []
    n_hex = math.ceil(n_bits / 4)
    raw = str(payload_hex or "").lower()[:n_hex] or ("0" * n_hex)
    text = bin(int(raw, 16))[2:].zfill(n_hex * 4)
    return [int(bit) for bit in text[:n_bits]]


def bits_to_hex(bits: List[int]) -> str:
    if not bits:
        return ""
    text = "".join(str(int(bit)) for bit in bits)
    return hex(int(text, 2))[2:].zfill(math.ceil(len(bits) / 4))


def tensor_from_bits(bits: List[int], n_bits: int, device: str) -> torch.Tensor:
    padded = bits[:n_bits] + [0] * max(0, n_bits - len(bits))
    return torch.tensor(padded, dtype=torch.float32, device=device).view(1, n_bits)


def verify_envelope(public: Any, envelope: Mapping[str, Any]) -> None:
    payload = canonical(unsigned(dict(envelope)))
    if hashlib.sha256(payload).hexdigest() != str(envelope.get("record_sha256", "")):
        raise ValueError("signed registry envelope digest mismatch")
    signature = base64.b64decode(
        str(envelope.get("signature_ed25519_b64", "")), validate=True
    )
    public.verify(signature, payload)


def load_signed_manifests(signed_registry: Path, public_key: Path) -> List[Dict[str, Any]]:
    public = load_public(public_key)
    manifests: List[Dict[str, Any]] = []
    for envelope in read_jsonl(signed_registry):
        verify_envelope(public, envelope)
        manifest = dict(envelope.get("manifest", {}))
        if manifest.get("video_path"):
            manifests.append(manifest)
    return manifests


def frame_cache_key(path: Path, resize: int, max_frames: int) -> str:
    resolved = path.resolve()
    try:
        stat = resolved.stat()
        stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
    except FileNotFoundError:
        stamp = "missing"
    material = f"{resolved}|{stamp}|resize={resize}|max_frames={max_frames}|tchw-u8-v1"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def load_video_frames(
    path: Path,
    device: str,
    resize: int,
    max_frames: int,
    cache_root: Path | None = None,
    cache_mode: str = "off",
) -> Tuple[torch.Tensor, int, bool, str]:
    cache_hit = False
    cache_file = ""
    frames_u8 = None
    fps = 25
    cache_path = None
    if cache_root is not None and cache_mode != "off":
        key = frame_cache_key(path, resize, max_frames)
        cache_path = cache_root / key[:2] / f"{key}.pt"
        cache_file = str(cache_path)
        if cache_path.exists():
            payload = torch.load(str(cache_path), map_location="cpu")
            frames_u8 = payload["frames_u8"].contiguous()
            fps = int(payload.get("fps", 25) or 25)
            cache_hit = True
        elif cache_mode == "readonly":
            raise FileNotFoundError(f"frame cache miss: {cache_path}")
    if frames_u8 is None:
        video, _, info = torchvision.io.read_video(
            str(path), output_format="TCHW", pts_unit="sec"
        )
        fps = int(round(float(info.get("video_fps", 25.0)))) or 25
        if max_frames > 0:
            video = video[:max_frames]
        if video.numel() == 0:
            raise ValueError(f"empty decoded video: {path}")
        frames_u8 = video.clamp(0, 255).byte()
        if resize > 0 and tuple(frames_u8.shape[-2:]) != (resize, resize):
            resized = torch.nn.functional.interpolate(
                frames_u8.float().div(255.0),
                size=(resize, resize),
                mode="bilinear",
                align_corners=False,
            )
            frames_u8 = resized.mul(255).round().clamp(0, 255).byte()
        frames_u8 = frames_u8.contiguous()
        if cache_path is not None and cache_mode == "readwrite":
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp")
            torch.save({"frames_u8": frames_u8.cpu(), "fps": fps}, temporary)
            temporary.replace(cache_path)
    return frames_u8.float().div(255.0).to(device), fps, cache_hit, cache_file


def frames_to_rgb24_bytes(frames: torch.Tensor) -> Tuple[bytes, int, int]:
    frames_u8 = (
        frames.detach()
        .cpu()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    if frames_u8.numel() == 0:
        raise ValueError("cannot write empty video frames")
    return frames_u8.numpy().tobytes(), int(frames_u8.shape[2]), int(frames_u8.shape[1])


def write_h264_video(
    frames: torch.Tensor, fps: int, output: Path, ffmpeg: Path, crf: int
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    raw, width, height = frames_to_rgb24_bytes(frames)
    run(
        [
            str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            str(output),
        ],
        raw,
    )


def mux_video_audio(
    video_path: Path,
    wav_path: Path,
    output: Path,
    ffmpeg: Path,
    audio_bitrate: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path), "-i", str(wav_path), "-map", "0:v:0",
            "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a",
            audio_bitrate, "-shortest", str(output),
        ]
    )


def extract_audio(
    input_media: Path, wav_path: Path, ffmpeg: Path, sample_rate: int = 16000
) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_media), "-vn", "-ar", str(sample_rate), "-ac", "1",
            "-c:a", "pcm_s16le", str(wav_path),
        ]
    )


def load_wav(path: Path, device: str, sample_rate: int = 16000) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path), backend="soundfile")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
        sr = sample_rate
    return wav.to(device), sr


def save_wav(path: Path, wav: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), wav.detach().cpu(), sample_rate, backend="soundfile")


def window_starts(
    n_samples: int, sr: int, win_sec: float, hop_sec: float, edge_guard_sec: float
) -> List[int]:
    win = max(1, int(round(win_sec * sr)))
    hop = max(1, int(round(hop_sec * sr)))
    guard = max(0, int(round(edge_guard_sec * sr)))
    end_limit = max(guard, n_samples - guard - win)
    starts = list(range(guard, end_limit + 1, hop))
    if not starts and n_samples >= win:
        starts.append(max(0, min(n_samples - win, guard)))
    return starts


def keyed_crc6(index: int, data_bits: List[int], master_key: str) -> List[int]:
    payload = f"apb-v1|{index}|{''.join(str(int(bit)) for bit in data_bits)}"
    digest = hashlib.sha256((master_key + "|" + payload).encode("utf-8")).digest()
    return int_to_bits(digest[0] >> 2, CRC_BITS)


def make_packet_bits(index: int, data_bits: List[int], master_key: str) -> List[int]:
    if len(data_bits) != DATA_BITS_PER_PACKET:
        raise ValueError(f"packet data must be {DATA_BITS_PER_PACKET} bits")
    return int_to_bits(index, INDEX_BITS) + data_bits + keyed_crc6(index, data_bits, master_key)


def parse_packet_bits(bits: List[int], master_key: str) -> Dict[str, Any]:
    bits = (list(bits) + [0] * PACKET_BITS)[:PACKET_BITS]
    index = bits_to_int(bits[:INDEX_BITS])
    data = bits[INDEX_BITS:INDEX_BITS + DATA_BITS_PER_PACKET]
    observed_crc = bits[INDEX_BITS + DATA_BITS_PER_PACKET:PACKET_BITS]
    expected_crc = keyed_crc6(index, data, master_key)
    return {
        "ok": observed_crc == expected_crc and index < MAX_PACKETS,
        "index": index,
        "data_bits": data,
        "data_value": bits_to_int(data),
    }


def packetize_tag(tag_bits: List[int], master_key: str) -> List[List[int]]:
    packet_count = math.ceil(len(tag_bits) / DATA_BITS_PER_PACKET)
    if packet_count <= 0 or packet_count > MAX_PACKETS:
        raise ValueError("Audio32 locator must contain 1 to 32 bits")
    padded = (tag_bits + [0] * (packet_count * DATA_BITS_PER_PACKET))[
        :packet_count * DATA_BITS_PER_PACKET
    ]
    return [
        make_packet_bits(
            index,
            padded[index * DATA_BITS_PER_PACKET:(index + 1) * DATA_BITS_PER_PACKET],
            master_key,
        )
        for index in range(packet_count)
    ]


def embed_packet_beacon(
    generator: Any,
    detector: Any,
    input_video: Path,
    tag_bits: List[int],
    embed_window_sec: float,
    embed_hop_sec: float,
    scale: float,
    edge_guard_sec: float,
    device: str,
    ffmpeg: Path,
    work_dir: Path,
    master_key: str,
) -> Dict[str, Any]:
    input_wav = work_dir / "input.wav"
    extract_audio(input_video, input_wav, ffmpeg)
    wav, sr = load_wav(input_wav, device)
    packets = packetize_tag(tag_bits, master_key)
    win = max(1, int(round(embed_window_sec * sr)))
    starts = window_starts(
        int(wav.shape[-1]), sr, embed_window_sec, embed_hop_sec, edge_guard_sec
    )
    watermark_accum = torch.zeros_like(wav)
    scores: List[float] = []
    for window_index, start in enumerate(starts):
        end = min(start + win, int(wav.shape[-1]))
        packet = packets[window_index % len(packets)]
        chunk = wav[:, start:end]
        message = tensor_from_bits(packet, PACKET_BITS, device)
        with torch.no_grad():
            watermark = generator.get_watermark(
                chunk.unsqueeze(0), sr, message=message
            ) * float(scale)
            score, _ = detector.detect_watermark(chunk.unsqueeze(0) + watermark, sr)
        watermark_accum[:, start:end] += watermark.squeeze(0)
        scores.append(float(score.reshape(-1)[0].item()))
    watermarked_wav = work_dir / "authenticated_audio32.wav"
    save_wav(watermarked_wav, wav + watermark_accum, sr)
    return {
        "watermarked_wav": watermarked_wav,
        "duration_sec": float(wav.shape[-1]) / float(sr),
        "embed_windows": len(starts),
        "memory_score": sum(scores) / len(scores) if scores else 0.0,
    }


def scan_packets(
    detector: Any,
    wav: torch.Tensor,
    sr: int,
    decode_window_sec: float,
    decode_hop_sec: float,
    edge_guard_sec: float,
    master_key: str,
) -> Dict[str, Any]:
    win = max(1, int(round(decode_window_sec * sr)))
    starts = window_starts(
        int(wav.shape[-1]), sr, decode_window_sec, decode_hop_sec, edge_guard_sec
    )
    valid = []
    for start in starts:
        chunk = wav[:, start:min(start + win, int(wav.shape[-1]))]
        with torch.no_grad():
            _, decoded = detector.detect_watermark(chunk.unsqueeze(0), sr)
        bits = (decoded > 0.5).int().view(-1).detach().cpu().tolist()[:PACKET_BITS]
        parsed = parse_packet_bits(bits, master_key)
        if parsed["ok"]:
            valid.append(parsed)
    return {"total_windows": len(starts), "valid_packets": valid}


def recover_tag(valid_packets: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_index: Dict[int, Counter[int]] = defaultdict(Counter)
    for packet in valid_packets:
        by_index[int(packet["index"])][int(packet["data_value"])] += 1
    recovered: List[int] = []
    for index in range(MAX_PACKETS):
        votes = by_index.get(index, Counter())
        if not votes:
            return {"complete": False, "recovered_bits": []}
        recovered.extend(int_to_bits(votes.most_common(1)[0][0], DATA_BITS_PER_PACKET))
    return {"complete": True, "recovered_bits": recovered}


def repeated_bits(data_bits: List[int], capacity: int) -> List[int]:
    if not data_bits:
        return [0] * capacity
    return [data_bits[index % len(data_bits)] for index in range(capacity)]


def majority_vote_with_margins(
    raw_bits: List[int], data_bits: int
) -> Tuple[List[int], List[float], List[int]]:
    voted: List[int] = []
    margins: List[float] = []
    signed_margins: List[int] = []
    for index in range(data_bits):
        values = [int(value) for value in raw_bits[index::data_bits]]
        ones = sum(values)
        zeros = len(values) - ones
        signed = ones - zeros
        voted.append(int(ones >= zeros))
        signed_margins.append(signed)
        margins.append(abs(signed) / len(values) if values else 0.0)
    return voted, margins, signed_margins


def decode_frames_bits(
    model: Any, frames: torch.Tensor, n_bits: int
) -> Tuple[List[int], Dict[str, Any]]:
    with torch.no_grad():
        if hasattr(model, "extract_message"):
            extracted = model.extract_message(frames)
            decoded = (extracted.reshape(-1)[:n_bits] > 0).int().cpu().tolist()
            return decoded, {"decoder": "extract_message", "pred_shape": list(extracted.shape)}
        detected = model.detect(frames, is_video=True)
        predictions = detected["preds"]
        if predictions.ndim >= 2 and predictions.shape[-1] > n_bits:
            flat = predictions[..., 1:].float().mean(dim=0).reshape(-1)
        else:
            flat = predictions.reshape(-1)
        return (
            (flat[:n_bits] > 0).int().cpu().tolist(),
            {"decoder": "detect", "pred_shape": list(predictions.shape)},
        )


def embed_videoseal_full_duration(
    model: Any,
    input_video: Path,
    locator: str,
    data_bits: int,
    device: str,
    resize: int,
    batch_frames: int,
    scaling_w: float,
    ffmpeg: Path,
    output_video_only: Path,
    crf: int,
    frame_cache_root: Path | None,
    frame_cache_mode: str,
) -> Dict[str, Any]:
    if scaling_w > 0 and hasattr(model, "blender") and hasattr(model.blender, "scaling_w"):
        model.blender.scaling_w = float(scaling_w)
    capacity = int(model.get_random_msg().shape[1])
    pointer = hex_to_bits(locator, data_bits)
    message = tensor_from_bits(repeated_bits(pointer, capacity), capacity, device)
    frames, fps, cache_hit, cache_file = load_video_frames(
        input_video, device, resize, 0, frame_cache_root, frame_cache_mode
    )
    chunks: List[torch.Tensor] = []
    for start in range(0, int(frames.shape[0]), max(1, batch_frames)):
        with torch.no_grad():
            watermarked = model.embed(
                frames[start:start + max(1, batch_frames)],
                msgs=message,
                is_video=True,
                lowres_attenuation=True,
            )["imgs_w"].clamp(0, 1)
        chunks.append(watermarked.detach())
    if not chunks:
        raise ValueError(f"decoded no frames from {input_video}")
    write_h264_video(torch.cat(chunks), fps, output_video_only, ffmpeg, crf)
    return {
        "video_pointer_bits": data_bits,
        "videoseal_capacity_bits": capacity,
        "video_pointer_hex": bits_to_hex(pointer),
        "video_embed_frames": int(frames.shape[0]),
        "video_fps": fps,
        "video_resize": resize,
        "video_batch_frames": max(1, batch_frames),
        "videoseal_scaling_w": float(scaling_w),
        "frame_cache_hit": cache_hit,
        "frame_cache_file": cache_file,
    }


def neighbor_masks(n_bits: int, radius: int) -> List[Tuple[int, int]]:
    masks: List[Tuple[int, int]] = [(0, 0)]
    for distance in range(1, radius + 1):
        for positions in combinations(range(n_bits), distance):
            mask = 0
            for position in positions:
                mask |= 1 << (n_bits - 1 - position)
            masks.append((distance, mask))
    return masks


def lookup_candidates(
    value: int,
    pointer_index: Dict[int, List[str]],
    masks: List[Tuple[int, int]],
) -> List[Tuple[int, str]]:
    candidates: List[Tuple[int, str]] = []
    seen = set()
    for distance, mask in masks:
        for record_hash in pointer_index.get(value ^ mask, []):
            if record_hash not in seen:
                seen.add(record_hash)
                candidates.append((distance, record_hash))
    return sorted(candidates, key=lambda item: (item[0], item[1]))


def recover_audio_claim(
    detector: Any,
    media_path: Path,
    work_wav: Path,
    ffmpeg: Path,
    master_key: str,
    decode_window_sec: float,
    decode_hop_sec: float,
    edge_guard_sec: float,
    device: str,
) -> Dict[str, object]:
    extract_audio(media_path, work_wav, ffmpeg)
    wav, sr = load_wav(work_wav, device)
    scan = scan_packets(
        detector, wav, sr, decode_window_sec, decode_hop_sec, edge_guard_sec, master_key
    )
    recovered = recover_tag(scan["valid_packets"])
    bits = recovered["recovered_bits"][:32] if recovered["complete"] else []
    return {
        "audio_tag_recovered": bool(recovered["complete"]),
        "audio_recovered_tag_hex": bits_to_hex(bits),
        "audio_valid_packets": len(scan["valid_packets"]),
        "audio_total_windows": scan["total_windows"],
    }


def recover_video_claim(
    model: Any,
    media_path: Path,
    video_bits: int,
    resize: int,
    max_frames: int,
    device: str,
) -> Dict[str, object]:
    frames, _, _, _ = load_video_frames(
        media_path, device, resize, max_frames, None, "off"
    )
    raw, decoder_meta = decode_frames_bits(model, frames, 256)
    voted, margins, _ = majority_vote_with_margins(raw, video_bits)
    return {
        "video_pointer_recovered": bool(voted),
        "video_recovered_pointer_hex": bits_to_hex(voted),
        "video_recovered_pointer_value": bits_to_int(voted),
        "video_vote_margin_min": min(margins) if margins else 0.0,
        "video_vote_margin_mean": sum(margins) / len(margins) if margins else 0.0,
        "video_frames_decoded": int(frames.shape[0]),
        "video_decoder": str(decoder_meta.get("decoder", "")),
    }
