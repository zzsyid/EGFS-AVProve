#!/usr/bin/env python3
"""Offline four-channel feature extraction for RQ1.

Input:  a video manifest containing original, same-source attacked, and tamper
        samples.
Output: four .npy features per sample plus a feature manifest consumable by
        build_feature_table.py.

Required input columns:

    sample_id,source_id,population,cond,generator,split,video_path

For every non-orig row, add:

    enroll_id

where enroll_id points to the registered original event. For orig rows,
enroll_id defaults to sample_id if omitted.

Default channels:

    id_audio -> SpeechBrain ECAPA-TDNN speaker embedding
    id_video -> InsightFace ArcFace buffalo_l face embedding
    evt      -> concat(Whisper encoder audio-content, CLIP visual-event)
    sync     -> signal-level AV sync vector:
                cross-correlation curve between audio RMS and mouth motion

The sync default is intentionally model-free so Stage-0 can start before a
Synchformer feature extractor is wired. Replace it later with Synchformer .npy
features if desired; build_feature_table.py only requires fixed-dimensional
vectors.
"""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


CHANNELS = ["id_audio", "id_video", "evt", "sync"]
META_COLUMNS = [
    "attack_level", "attack_family", "attack_scope",
    "target_start_sec", "target_end_sec", "generator_role",
    "attack_variant", "splice_duration_sec", "splice_position",
    "codec_bandwidth", "enhancement_model", "attack_language",
]


def l2norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return (v / (n + eps)).astype(np.float32)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def require_cols(rows: List[Dict[str, str]], cols: List[str]) -> None:
    if not rows:
        raise ValueError("input manifest is empty")
    missing = [c for c in cols if c not in rows[0]]
    if missing:
        raise ValueError(f"input manifest missing columns: {missing}")


def resolve_path(p: str, root: Path) -> Path:
    q = Path(p)
    return q if q.is_absolute() else root / q


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


def load_wav_16k(video_path: Path) -> np.ndarray:
    wav_path = extract_wav(video_path, sr=16000)
    try:
        try:
            import soundfile as sf
            wav, sr = sf.read(wav_path, dtype="float32")
            if sr != 16000:
                raise RuntimeError(f"unexpected sr={sr}")
            return np.asarray(wav, dtype=np.float32).reshape(-1)
        except ImportError:
            import torchaudio
            wav, sr = torchaudio.load(wav_path)
            wav = wav.mean(dim=0)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            return wav.cpu().numpy().astype(np.float32)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def sample_frames(
    video_path: Path,
    n_frames: int = 16,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> List[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total <= 0:
        cap.release()
        raise RuntimeError(f"no frames in {video_path}")
    if fps <= 0:
        fps = 25.0
    first = max(0, min(total - 1, int(np.floor(start_sec * fps))))
    last = total - 1
    if end_sec is not None:
        last = max(first, min(total - 1, int(np.ceil(end_sec * fps)) - 1))
    count = last - first + 1
    idx = set(np.linspace(first, last, min(n_frames, count), dtype=int).tolist())
    frames: List[np.ndarray] = []
    for i in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if i in idx:
            frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"failed to sample frames from {video_path}")
    return frames


_ORT_CUDA_PRELOADED = False


def preload_ort_cuda_runtime() -> None:
    global _ORT_CUDA_PRELOADED
    if _ORT_CUDA_PRELOADED:
        return
    candidates = [
        'cuda_runtime/lib/libcudart.so.12',
        'cublas/lib/libcublasLt.so.12',
        'cublas/lib/libcublas.so.12',
        'cufft/lib/libcufft.so.11',
        'curand/lib/libcurand.so.10',
        'cudnn/lib/libcudnn.so.9',
    ]
    roots = sorted(Path(sys.prefix).glob('lib/python*/site-packages/nvidia'))
    mode = getattr(ctypes, 'RTLD_GLOBAL', 0)
    loaded_any = False
    for root in roots:
        for rel in candidates:
            lib_path = root / rel
            if not lib_path.exists():
                continue
            ctypes.CDLL(str(lib_path), mode=mode)
            loaded_any = True
    if loaded_any:
        _ORT_CUDA_PRELOADED = True



class ECAPAExtractor:
    def __init__(self, device: str, source: str, savedir: str):
        import torch
        try:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier
        except ImportError as exc:
            raise ImportError(
                "ECAPA needs speechbrain. Install e.g. `pip install speechbrain`."
            ) from exc
        self.device = device
        self.model = EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir,
            run_opts={"device": device},
        )

    def __call__(self, video_path: Path) -> np.ndarray:
        import torch
        wav_path = extract_wav(video_path, sr=16000)
        try:
            import torchaudio
            wav, sr = torchaudio.load(wav_path)
            wav = wav.mean(dim=0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            if wav.shape[-1] < 16000:
                raise RuntimeError("audio shorter than 1s")
            with torch.inference_mode():
                emb = self.model.encode_batch(wav.to(self.device)).squeeze()
            return l2norm(emb.detach().cpu().numpy())
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)


class ArcFaceExtractor:
    def __init__(self, device: str, n_frames: int = 16):
        import torch
        if device.startswith("cuda") and torch.cuda.is_available():
            preload_ort_cuda_runtime()
        from insightface.app import FaceAnalysis
        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        ctx_id = int(device.split(":", 1)[1]) if use_cuda and ":" in device else (0 if use_cuda else -1)
        providers = (
            [("CUDAExecutionProvider", {"device_id": ctx_id}), "CPUExecutionProvider"]
            if use_cuda else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.app_hi = FaceAnalysis(name="buffalo_l", providers=providers)
        # Retry difficult talking-head clips at a larger detector resolution
        # and a lower confidence threshold. The primary path remains strict;
        # the fallback improves coverage for small or redrawn faces.
        self.app_hi.prepare(ctx_id=ctx_id, det_thresh=0.25, det_size=(1280, 1280))
        self.n_frames = n_frames

    def _embed_with(self, app, video_path: Path, n_frames: int) -> Optional[np.ndarray]:
        embs = []
        for frame in sample_frames(video_path, n_frames=n_frames):
            faces = app.get(frame)
            if not faces:
                continue
            face = max(
                faces,
                key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            )
            embs.append(np.asarray(face.normed_embedding, dtype=np.float32))
        if not embs:
            return None
        return l2norm(np.mean(np.stack(embs, axis=0), axis=0))

    def __call__(self, video_path: Path) -> np.ndarray:
        return self.extract_window(video_path)

    def _embed_window_with(
        self, app, video_path: Path, n_frames: int,
        start_sec: float = 0.0, end_sec: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        embs = []
        for frame in sample_frames(
            video_path, n_frames=n_frames, start_sec=start_sec, end_sec=end_sec,
        ):
            faces = app.get(frame)
            if not faces:
                continue
            face = max(
                faces,
                key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
            )
            embs.append(np.asarray(face.normed_embedding, dtype=np.float32))
        if not embs:
            return None
        return l2norm(np.mean(np.stack(embs, axis=0), axis=0))

    def extract_window(
        self, video_path: Path, start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> np.ndarray:
        emb = self._embed_window_with(
            self.app, video_path, self.n_frames, start_sec, end_sec,
        )
        if emb is None:
            emb = self._embed_window_with(
                self.app_hi, video_path, max(64, self.n_frames), start_sec, end_sec,
            )
        if emb is None:
            raise RuntimeError("no face detected")
        return emb


class WhisperContentExtractor:
    def __init__(self, device: str, model_name: str):
        try:
            import whisper
        except ImportError as exc:
            raise ImportError(
                "Whisper evt-audio backend needs openai-whisper. "
                "Install e.g. `pip install -U openai-whisper`."
            ) from exc
        self.whisper = whisper
        self.device = device
        self.model = whisper.load_model(model_name, device=device)

    def __call__(self, video_path: Path) -> np.ndarray:
        import torch
        wav_path = extract_wav(video_path, sr=16000)
        try:
            audio = self.whisper.load_audio(wav_path)
            audio = self.whisper.pad_or_trim(audio)
            mel = self.whisper.log_mel_spectrogram(audio).to(self.model.device)
            with torch.inference_mode():
                enc = self.model.encoder(mel.unsqueeze(0)).squeeze(0)
            return l2norm(enc.float().mean(dim=0).detach().cpu().numpy())
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)


class CLIPVisualExtractor:
    def __init__(self, device: str, model_id: str, n_frames: int = 8):
        import torch
        try:
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "CLIP evt-video backend needs transformers and pillow. "
                "Install e.g. `pip install transformers pillow`."
            ) from exc
        self.Image = Image
        self.device = device
        self.n_frames = n_frames
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()

    def __call__(self, video_path: Path) -> np.ndarray:
        return self.extract_window(video_path)

    def extract_window(
        self, video_path: Path, start_sec: float = 0.0,
        end_sec: Optional[float] = None,
    ) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        import cv2
        frames = sample_frames(
            video_path, n_frames=self.n_frames, start_sec=start_sec, end_sec=end_sec,
        )
        imgs = [self.Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        inputs = self.processor(images=imgs, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            feats = self.model.get_image_features(**inputs)
        feats = F.normalize(feats.float(), dim=-1).mean(dim=0)
        return l2norm(feats.detach().cpu().numpy())


class SyncSignalCorrExtractor:
    """Fixed-length sync vector from cross-correlation over lags."""

    def __init__(self, max_lag: int = 20, n_frames: int = 64):
        self.max_lag = max_lag
        self.n_frames = n_frames

    @staticmethod
    def _mouth_motion(video_path: Path, n_frames: int) -> np.ndarray:
        import cv2
        frames = sample_frames(video_path, n_frames=n_frames)
        vals = []
        for frame in frames:
            h, w = frame.shape[:2]
            mouth = frame[int(h * 0.58):int(h * 0.92), int(w * 0.25):int(w * 0.75)]
            gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
            vals.append(float(gray.var()))
        vals = np.asarray(vals, dtype=np.float32)
        motion = np.abs(np.diff(vals, prepend=vals[0]))
        return (motion - motion.mean()) / (motion.std() + 1e-6)

    @staticmethod
    def _audio_rms(video_path: Path, n_frames: int) -> np.ndarray:
        wav = load_wav_16k(video_path)
        if len(wav) < n_frames:
            wav = np.pad(wav, (0, n_frames - len(wav)))
        chunks = np.array_split(wav, n_frames)
        rms = np.asarray([np.sqrt(np.mean(c * c) + 1e-12) for c in chunks], dtype=np.float32)
        return (rms - rms.mean()) / (rms.std() + 1e-6)

    def __call__(self, video_path: Path) -> np.ndarray:
        m = self._mouth_motion(video_path, self.n_frames)
        a = self._audio_rms(video_path, self.n_frames)
        vals = []
        for lag in range(-self.max_lag, self.max_lag + 1):
            if lag < 0:
                x, y = m[-lag:], a[:lag]
            elif lag > 0:
                x, y = m[:-lag], a[lag:]
            else:
                x, y = m, a
            if len(x) < 4 or float(x.std()) < 1e-6 or float(y.std()) < 1e-6:
                vals.append(0.0)
            else:
                vals.append(float(np.corrcoef(x, y)[0, 1]))
        return l2norm(np.asarray(vals, dtype=np.float32))


class FeatureExtractor:
    def __init__(self, args):
        import torch
        device = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
        print(f"[extract] device={device}")
        self.id_audio = ECAPAExtractor(device, args.ecapa_source, args.ecapa_savedir)
        self.id_video = ArcFaceExtractor(device, n_frames=args.face_frames)
        self.evt_audio = WhisperContentExtractor(device, args.whisper_model)
        self.evt_video = CLIPVisualExtractor(device, args.clip_model, n_frames=args.clip_frames)
        self.sync = SyncSignalCorrExtractor(max_lag=args.sync_max_lag, n_frames=args.sync_frames)

    def extract_all(self, video_path: Path) -> Dict[str, np.ndarray]:
        a_id = self.id_audio(video_path)
        v_id = self.id_video(video_path)
        evt_a = self.evt_audio(video_path)
        evt_v = self.evt_video(video_path)
        evt = l2norm(np.concatenate([evt_a, evt_v], axis=0))
        sync = self.sync(video_path)
        return {
            "id_audio": a_id,
            "id_video": v_id,
            "evt": evt,
            "sync": sync,
        }


def normalize_input_row(row: Dict[str, str]) -> Dict[str, str]:
    row = dict(row)
    pop = row.get("population", "")
    if pop == "orig" and not row.get("enroll_id"):
        row["enroll_id"] = row["sample_id"]
    if not row.get("enroll_id"):
        raise ValueError(f"row {row.get('sample_id')} missing enroll_id")
    return row


def extract_rows(args, in_rows: List[Dict[str, str]], worker_name: str = "main") -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    out_root = Path(args.out_root)
    path_root = Path(args.path_root) if args.path_root else Path(args.manifest).parent
    extractor = FeatureExtractor(args)

    feature_rows: List[Dict[str, str]] = []
    failures: List[Dict[str, str]] = []
    for i, raw in enumerate(in_rows, 1):
        try:
            row = normalize_input_row(raw)
            sid = row["sample_id"]
            video_path = resolve_path(row["video_path"], path_root)
            sample_dir = out_root / sid
            paths = {ch: sample_dir / f"{ch}.npy" for ch in CHANNELS}

            if args.resume and all(p.exists() for p in paths.values()):
                pass
            else:
                feats = extractor.extract_all(video_path)
                sample_dir.mkdir(parents=True, exist_ok=True)
                for ch, vec in feats.items():
                    np.save(paths[ch], l2norm(vec))

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
                if col in row:
                    out[col] = row[col]
            for ch in CHANNELS:
                out[f"{ch}_path"] = str(paths[ch].resolve())
            feature_rows.append(out)
            print(f"[{worker_name} {i}/{len(in_rows)}] ok     {sid}", flush=True)
        except Exception as exc:
            fail = dict(raw)
            fail["error"] = str(exc)[-500:]
            failures.append(fail)
            print(f"[{worker_name} {i}/{len(in_rows)}] FAILED "
                  f"{raw.get('sample_id', '?')}: {fail['error']}", flush=True)
    return feature_rows, failures


def worker_main(args, in_rows: List[Dict[str, str]], device: str,
                worker_id: int, success_csv: str, failure_csv: str) -> None:
    worker_args = copy.deepcopy(args)
    worker_args.device = device
    name = f"gpu{worker_id}:{device}"
    print(f"[{name}] starting shard with {len(in_rows)} rows", flush=True)
    feature_rows, failures = extract_rows(worker_args, in_rows, worker_name=name)
    write_csv(Path(success_csv), feature_rows)
    if failures:
        write_csv(Path(failure_csv), failures)
    print(f"[{name}] done ok={len(feature_rows)} failed={len(failures)}", flush=True)


def parse_devices(device: str, devices: str) -> List[str]:
    out = [d.strip() for d in devices.split(",") if d.strip()] if devices else [device]
    if not out:
        raise ValueError("no devices configured")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="input video manifest CSV")
    ap.add_argument("--out-root", required=True, help="directory for .npy features")
    ap.add_argument("--out-manifest", required=True, help="feature manifest CSV for build_feature_table.py")
    ap.add_argument("--path-root", default=None, help="root for relative video_path values")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--devices", default="",
                    help="comma-separated multi-GPU devices, e.g. cuda:0,cuda:1; "
                         "overrides --device")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ecapa-source", default="speechbrain/spkrec-ecapa-voxceleb",
                    help="HF id or local path for ECAPA")
    ap.add_argument("--ecapa-savedir", default="pretrained_models/ecapa_voxceleb")
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--face-frames", type=int, default=16)
    ap.add_argument("--clip-frames", type=int, default=8)
    ap.add_argument("--sync-frames", type=int, default=64)
    ap.add_argument("--sync-max-lag", type=int, default=20)
    args = ap.parse_args()

    in_rows = read_csv(Path(args.manifest))
    require_cols(
        in_rows,
        ["sample_id", "source_id", "population", "cond", "generator", "split", "video_path"],
    )
    if args.limit and args.limit > 0:
        in_rows = in_rows[:args.limit]

    devices = parse_devices(args.device, args.devices)
    out_manifest = Path(args.out_manifest)
    fail_path = out_manifest.with_suffix(".failures.csv")

    if len(devices) == 1:
        args.device = devices[0]
        feature_rows, failures = extract_rows(args, in_rows)
    else:
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        shards = [in_rows[i::len(devices)] for i in range(len(devices))]
        ctx = mp.get_context("spawn")
        jobs = []
        shard_paths = []
        for worker_id, (device, shard) in enumerate(zip(devices, shards)):
            success_csv = out_manifest.with_name(f"{out_manifest.stem}.worker{worker_id}.csv")
            failure_csv = out_manifest.with_name(f"{out_manifest.stem}.worker{worker_id}.failures.csv")
            for stale in (success_csv, failure_csv):
                if stale.exists():
                    stale.unlink()
            shard_paths.append((success_csv, failure_csv))
            job = ctx.Process(
                target=worker_main,
                args=(args, shard, device, worker_id, str(success_csv), str(failure_csv)),
            )
            job.start()
            jobs.append(job)
        for job in jobs:
            job.join()
        failed_jobs = [job.pid for job in jobs if job.exitcode != 0]
        if failed_jobs:
            raise RuntimeError(f"feature workers failed: pids={failed_jobs}")
        feature_rows, failures = [], []
        for success_csv, failure_csv in shard_paths:
            if success_csv.exists():
                feature_rows.extend(read_csv(success_csv))
            if failure_csv.exists():
                failures.extend(read_csv(failure_csv))

    write_csv(out_manifest, feature_rows)
    if failures:
        write_csv(fail_path, failures)
        print(f"[extract] failures -> {fail_path}")
    elif fail_path.exists():
        fail_path.unlink()
    print(f"[extract] feature manifest -> {out_manifest}  ok={len(feature_rows)} failed={len(failures)}")
    print("[extract] next:")
    print(f"  python build_feature_table.py --manifest {out_manifest} --out {out_manifest.parent / 'fe_passfail.csv'}")


if __name__ == "__main__":
    main()
