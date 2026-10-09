#!/usr/bin/env python3
"""Convert one source WAV toward a donor voice with kNN-VC (bshall/knn-vc).

kNN-VC is NOT a single-reference converter like FreeVC. It works as:

    query_seq    = WavLM features of the SOURCE utterance        (T, 1024)
    matching_set = pooled WavLM features of the TARGET speaker   (N, 1024)
    out_wav      = match(query_seq, matching_set, topk)          -> 16 kHz

Each output frame is reconstructed from its k nearest target frames, so the
conversion quality scales with how much TARGET audio is in the matching set.
The original paper pools minutes of target speech. The extended-RQ1 dispatcher
only hands us ONE donor clip (donor_video_path -> donor.wav, ~7 s), which is a
small matching set and will sound rougher than a full-reference run.

To keep this honest and extensible without changing the dispatcher contract:
  * --target-audio        the single donor wav the dispatcher extracts (primary)
  * --extra-target-audio  repeatable; add more target wavs to the matching set
  * --target-glob         glob of additional target wavs (e.g. all clips of the
                          donor source) -> the recommended way to grow the set
The wrapper warns (or errors with --min-target-sec) when the pooled target
duration is too short to expect good kNN-VC quality.

Model loading uses torch.hub. With --knnvc-root pointing at a local checkout it
loads offline via source='local'; otherwise it pulls 'bshall/knn-vc' from
GitHub. prematched=True selects the prematched HiFi-GAN vocoder (recommended).

This produces an L3 cross-source voice-conversion sample: the source's words in
the donor's timbre, later remuxed onto the ORIGINAL video by the dispatcher.
Speaker identity changes, so id_audio is the channel expected to flag it.
"""

from __future__ import annotations

import argparse
import glob
import sys
import wave
from pathlib import Path
from typing import List


def _gather_targets(primary: Path, extra: List[Path], pattern: str,
                    require_exists: bool = True) -> List[Path]:
    targets: List[Path] = [primary]
    targets.extend(extra)
    if pattern:
        targets.extend(Path(p) for p in sorted(glob.glob(pattern)))
    # de-dup, preserve order, keep only existing files
    seen, out = set(), []
    for t in targets:
        rt = t.resolve()
        if rt in seen:
            continue
        seen.add(rt)
        if require_exists and not t.exists():
            raise FileNotFoundError(f"target audio does not exist: {t}")
        out.append(t)
    return out


def _total_seconds(paths: List[Path]) -> float:
    total = 0.0
    for p in paths:
        # Prefer backend-free readers for plain WAV metadata.
        try:
            with wave.open(str(p), "rb") as wav_f:
                frames = wav_f.getnframes()
                sample_rate = wav_f.getframerate()
            if frames and sample_rate:
                total += frames / sample_rate
                continue
        except wave.Error:
            pass

        try:
            import soundfile as sf  # local import so --help works without audio deps

            info = sf.info(str(p))
            if info.frames and info.samplerate:
                total += info.frames / info.samplerate
                continue
        except Exception:
            pass

        import torchaudio  # local import so --help works without torch

        info = torchaudio.info(str(p))
        if info.num_frames and info.sample_rate:
            total += info.num_frames / info.sample_rate
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-audio", type=Path, required=True,
                        help="source utterance (content to keep)")
    parser.add_argument("--target-audio", type=Path, required=True,
                        help="primary donor reference (timbre to impose)")
    parser.add_argument("--output-audio", type=Path, required=True)
    parser.add_argument("--extra-target-audio", type=Path, action="append",
                        default=[], help="additional target wav (repeatable)")
    parser.add_argument("--target-glob", default="",
                        help="glob of additional target wavs for the matching set")
    parser.add_argument("--knnvc-root", type=Path, default=None,
                        help="local checkout of bshall/knn-vc; omit to use GitHub hub")
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--prematched", action="store_true", default=True,
                        help="use the prematched vocoder (recommended; default on)")
    parser.add_argument("--no-prematched", dest="prematched", action="store_false")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-target-sec", type=float, default=0.0,
                        help="error if pooled target duration is below this (0=warn only)")
    parser.add_argument("--matching-vad-trigger-level", type=float, default=0.0,
                        help="VAD trigger level for donor matching-set trimming; 0 disables trimming")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = _gather_targets(
        args.target_audio, args.extra_target_audio, args.target_glob,
        require_exists=not args.dry_run,
    )
    print(f"[knnvc-wrapper] source={args.source_audio}", flush=True)
    print(f"[knnvc-wrapper] matching set = {len(targets)} target wav(s)", flush=True)

    if args.dry_run:
        print("[knnvc-wrapper] dry-run: skipping model load and conversion")
        return

    if not args.source_audio.exists():
        raise FileNotFoundError(args.source_audio)

    target_sec = _total_seconds(targets)
    msg = f"[knnvc-wrapper] pooled target duration = {target_sec:.1f} s"
    if target_sec < 30.0:
        msg += "  (WARNING: <30 s matching set; kNN-VC quality is degraded -- " \
               "pass --target-glob with more donor-source clips for a fair run)"
    print(msg, flush=True)
    if args.min_target_sec and target_sec < args.min_target_sec:
        raise ValueError(
            f"pooled target duration {target_sec:.1f}s < --min-target-sec "
            f"{args.min_target_sec}s; supply more target audio")

    import torch
    import torchaudio

    if args.knnvc_root is not None:
        root = args.knnvc_root
        if not (root / "hubconf.py").exists():
            raise FileNotFoundError(
                f"--knnvc-root {root} has no hubconf.py; expected a bshall/knn-vc checkout")
        print(f"[knnvc-wrapper] loading kNN-VC from local checkout {root}", flush=True)
        knn_vc = torch.hub.load(str(root), "knn_vc", source="local",
                                prematched=args.prematched, trust_repo=True,
                                pretrained=True, device=args.device)
    else:
        print("[knnvc-wrapper] loading kNN-VC from GitHub hub bshall/knn-vc", flush=True)
        knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc",
                                prematched=args.prematched, trust_repo=True,
                                pretrained=True, device=args.device)

    query_seq = knn_vc.get_features(str(args.source_audio))
    matching_set = knn_vc.get_matching_set(
        [str(t) for t in targets],
        vad_trigger_level=args.matching_vad_trigger_level,
    )
    out_wav = knn_vc.match(query_seq, matching_set, topk=args.topk)

    out_wav = out_wav.detach().to("cpu")
    if out_wav.dim() == 1:
        out_wav = out_wav.unsqueeze(0)
    args.output_audio.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output_audio), out_wav, 16000)

    if not args.output_audio.exists():
        raise FileNotFoundError(f"kNN-VC did not create {args.output_audio}")
    print(f"[knnvc-wrapper] complete -> {args.output_audio}", flush=True)


if __name__ == "__main__":
    main()
