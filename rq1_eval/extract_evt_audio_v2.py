#!/usr/bin/env python3
"""Extract EVT-audio-v2 global, windowed, and transcript features.

Unlike extract_features.py, this extractor is intentionally independent of
ArcFace and the other provenance channels. It uses Whisper encoder tokens only
over the valid (non-padding) audio span and exports sliding-window content
features for local splice detection.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import multiprocessing as mp
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


META_COLUMNS = [
    "attack_level", "attack_family", "attack_scope",
    "target_start_sec", "target_end_sec", "generator_role",
    "attack_variant", "splice_duration_sec", "splice_position",
    "codec_bandwidth", "enhancement_model", "attack_language",
]


def l2norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    return (v / (float(np.linalg.norm(v)) + eps)).astype(np.float32)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def extract_wav(video_path: Path, sr: int = 16000) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-ac", "1", "-ar", str(sr), "-vn", "-f", "wav", tmp.name,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return tmp.name


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", str(text).lower(), flags=re.UNICODE))


class WhisperEvtAudioV2:
    def __init__(
        self,
        device: str,
        model_name: str,
        window_span_sec: float,
        window_hop_sec: float,
        window_batch_size: int,
        transcribe: bool,
    ):
        import torch
        import whisper

        self.torch = torch
        self.whisper = whisper
        self.device = device
        self.model = whisper.load_model(model_name, device=device)
        self.window_span_sec = window_span_sec
        self.window_hop_sec = window_hop_sec
        self.window_batch_size = window_batch_size
        self.transcribe = transcribe
        self.sr = int(whisper.audio.SAMPLE_RATE)
        self.max_samples = int(whisper.audio.N_SAMPLES)

    def _valid_token_count(self, n_samples: int, n_tokens: int) -> int:
        # Whisper mel frames are 100 Hz and the encoder downsamples by two.
        seconds = min(n_samples, self.max_samples) / self.sr
        return max(1, min(n_tokens, int(np.ceil(seconds * 50.0))))

    def _encode_batch(self, waves: List[np.ndarray]) -> np.ndarray:
        mels = []
        valid_samples = []
        for wav in waves:
            valid_samples.append(min(len(wav), self.max_samples))
            padded = self.whisper.pad_or_trim(np.asarray(wav, dtype=np.float32))
            mels.append(self.whisper.log_mel_spectrogram(padded))
        mel = self.torch.stack(mels).to(self.model.device)
        with self.torch.inference_mode():
            enc = self.model.encoder(mel).float().detach().cpu().numpy()
        out = []
        for tokens, n_samples in zip(enc, valid_samples):
            valid = self._valid_token_count(n_samples, len(tokens))
            out.append(l2norm(tokens[:valid].mean(axis=0)))
        return np.stack(out)

    def _window_starts(self, n_samples: int) -> List[float]:
        duration = n_samples / self.sr
        if duration <= self.window_span_sec:
            return [0.0]
        starts = list(np.arange(
            0.0,
            max(0.0, duration - self.window_span_sec) + 1e-6,
            self.window_hop_sec,
        ))
        last = max(0.0, duration - self.window_span_sec)
        if not starts or abs(starts[-1] - last) > 1e-4:
            starts.append(last)
        return [float(x) for x in starts]

    def __call__(self, video_path: Path) -> Dict[str, object]:
        wav_path = extract_wav(video_path, sr=self.sr)
        try:
            audio = self.whisper.load_audio(wav_path)
            global_vec = self._encode_batch([audio])[0]
            starts = self._window_starts(len(audio))
            span_samples = max(1, int(round(self.window_span_sec * self.sr)))
            waves = []
            for start in starts:
                begin = int(round(start * self.sr))
                waves.append(audio[begin:begin + span_samples])
            window_vecs = []
            for i in range(0, len(waves), self.window_batch_size):
                window_vecs.append(self._encode_batch(waves[i:i + self.window_batch_size]))
            transcript = ""
            if self.transcribe:
                result = self.model.transcribe(
                    wav_path,
                    fp16=self.device.startswith("cuda"),
                    verbose=False,
                )
                transcript = normalize_text(result.get("text", ""))
            return {
                "global": global_vec,
                "windows": np.concatenate(window_vecs, axis=0),
                "window_starts_sec": np.asarray(starts, dtype=np.float32),
                "transcript": transcript,
                "duration_sec": len(audio) / self.sr,
            }
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)


def normalize_row(row: Dict[str, str]) -> Dict[str, str]:
    row = dict(row)
    if row.get("population") == "orig" and not row.get("enroll_id"):
        row["enroll_id"] = row["sample_id"]
    if not row.get("enroll_id"):
        raise ValueError(f"row {row.get('sample_id', '?')} missing enroll_id")
    return row


def output_paths(root: Path, sample_id: str) -> Dict[str, Path]:
    sample = root / sample_id
    return {
        "evt_audio_global_path": sample / "evt_audio_global.npy",
        "evt_audio_windows_path": sample / "evt_audio_windows.npy",
        "evt_audio_window_starts_path": sample / "evt_audio_window_starts.npy",
        "evt_audio_meta_path": sample / "evt_audio_meta.json",
    }


def extract_rows(args, rows: List[Dict[str, str]], worker: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    root = Path(args.out_root)
    path_root = Path(args.path_root) if args.path_root else Path(args.manifest).parent
    extractor = WhisperEvtAudioV2(
        args.device,
        args.whisper_model,
        args.window_span_sec,
        args.window_hop_sec,
        args.window_batch_size,
        args.transcribe,
    )
    success, failures = [], []
    for i, raw in enumerate(rows, 1):
        try:
            row = normalize_row(raw)
            sid = row["sample_id"]
            paths = output_paths(root, sid)
            if args.resume and all(p.exists() for p in paths.values()):
                pass
            else:
                video = Path(row["video_path"])
                if not video.is_absolute():
                    video = path_root / video
                feats = extractor(video)
                paths["evt_audio_global_path"].parent.mkdir(parents=True, exist_ok=True)
                np.save(paths["evt_audio_global_path"], feats["global"])
                np.save(paths["evt_audio_windows_path"], feats["windows"])
                np.save(paths["evt_audio_window_starts_path"], feats["window_starts_sec"])
                paths["evt_audio_meta_path"].write_text(
                    json.dumps({
                        "transcript": feats["transcript"],
                        "duration_sec": feats["duration_sec"],
                        "window_span_sec": args.window_span_sec,
                        "window_hop_sec": args.window_hop_sec,
                    }, ensure_ascii=False),
                    encoding="utf-8",
                )
            out = {
                "sample_id": sid,
                "enroll_id": row["enroll_id"],
                "source_id": row["source_id"],
                "population": row["population"],
                "cond": row["cond"],
                "generator": row["generator"],
                "split": row["split"],
            }
            for col in META_COLUMNS:
                out[col] = row.get(col, "")
            for col, path in paths.items():
                out[col] = str(path.resolve())
            success.append(out)
            print(f"[{worker} {i}/{len(rows)}] ok     {sid}", flush=True)
        except Exception as exc:
            fail = dict(raw)
            fail["error"] = str(exc)[-500:]
            failures.append(fail)
            print(f"[{worker} {i}/{len(rows)}] FAILED {raw.get('sample_id', '?')}: {fail['error']}", flush=True)
    return success, failures


def worker_main(args, rows, device, worker_id, success_csv, failure_csv) -> None:
    args = copy.deepcopy(args)
    args.device = device
    success, failures = extract_rows(args, rows, f"gpu{worker_id}:{device}")
    write_csv(Path(success_csv), success)
    if failures:
        write_csv(Path(failure_csv), failures)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--path-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--devices", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--window-span-sec", type=float, default=2.0)
    ap.add_argument("--window-hop-sec", type=float, default=0.5)
    ap.add_argument("--window-batch-size", type=int, default=16)
    ap.add_argument("--transcribe", action="store_true")
    args = ap.parse_args()

    rows = read_csv(Path(args.manifest))
    if args.limit:
        rows = rows[:args.limit]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()] or [args.device]
    out = Path(args.out_manifest)
    fail = out.with_suffix(".failures.csv")

    if len(devices) == 1:
        success, failures = extract_rows(args, rows, devices[0])
    else:
        ctx = mp.get_context("spawn")
        shards = [rows[i::len(devices)] for i in range(len(devices))]
        jobs, parts = [], []
        for i, (device, shard) in enumerate(zip(devices, shards)):
            ok = out.with_name(f"{out.stem}.worker{i}.csv")
            bad = out.with_name(f"{out.stem}.worker{i}.failures.csv")
            for stale in (ok, bad):
                if stale.exists():
                    stale.unlink()
            job = ctx.Process(target=worker_main, args=(args, shard, device, i, str(ok), str(bad)))
            job.start()
            jobs.append(job)
            parts.append((ok, bad))
        for job in jobs:
            job.join()
        failed = [job.pid for job in jobs if job.exitcode != 0]
        if failed:
            raise RuntimeError(f"EVT-audio-v2 workers failed: {failed}")
        success, failures = [], []
        for ok, bad in parts:
            if ok.exists():
                success.extend(read_csv(ok))
            if bad.exists():
                failures.extend(read_csv(bad))
    write_csv(out, success)
    if failures:
        write_csv(fail, failures)
    elif fail.exists():
        fail.unlink()
    print(f"[evt-audio-v2] -> {out}  ok={len(success)} failed={len(failures)}")


if __name__ == "__main__":
    main()
