#!/usr/bin/env python3
"""Run extended-RQ1 attack wrappers from one task CSV.

Built-in wrapper:

    ffmpeg_splice
    ffmpeg_video_splice
    ffmpeg_full_video_replacement
    ffmpeg_recompress

External wrappers are declared in a JSON config. Two modes are supported:

    command_audio
        Run an external command that writes {output_audio_path}; remux the
        generated audio with the original video into {output_video_path}.

    command_video
        Run an external command that writes {output_video_path} directly.

Commands are JSON argument arrays and execute without a shell. This keeps path
handling predictable across external repositories and conda environments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REQUIRED_TASK_COLUMNS = {
    "task_id", "generator", "input_video_path", "output_video_path",
    "target_start_sec", "target_end_sec",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run(cmd: List[str], dry_run: bool) -> None:
    print("[wrapper]", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def probe_duration(ffprobe: str, video: Path) -> float:
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(result.stdout.strip())
    if duration <= 0:
        raise ValueError(f"invalid duration for {video}: {duration}")
    return duration


def extract_audio(ffmpeg: str, video: Path, wav: Path, dry_run: bool) -> None:
    run([
        ffmpeg, "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "24000",
        "-c:a", "pcm_s16le", str(wav),
    ], dry_run)


def remux_audio(ffmpeg: str, original_video: Path, generated_audio: Path,
                output_video: Path, dry_run: bool) -> None:
    if not dry_run:
        output_video.parent.mkdir(parents=True, exist_ok=True)
    run([
        ffmpeg, "-y", "-i", str(original_video), "-i", str(generated_audio),
        "-map", "0:v:0?", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-shortest", str(output_video),
    ], dry_run)


def run_ffmpeg_splice(row: Dict[str, str], ffmpeg: str, dry_run: bool) -> None:
    original = Path(row["input_video_path"])
    donor = Path(row["donor_video_path"])
    output = Path(row["output_video_path"])
    if not row.get("target_start_sec") or not row.get("target_end_sec"):
        raise ValueError("ffmpeg_splice requires target_start_sec and target_end_sec")
    start = float(row["target_start_sec"])
    end = float(row["target_end_sec"])
    if start < 0 or end <= start:
        raise ValueError(f"invalid splice interval {start}..{end}")
    duration = end - start
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
    graph = (
        f"[0:a]atrim=start=0:end={start},asetpts=PTS-STARTPTS[a0];"
        f"[1:a]atrim=start=0:end={duration},asetpts=PTS-STARTPTS[a1];"
        f"[0:a]atrim=start={end},asetpts=PTS-STARTPTS[a2];"
        "[a0][a1][a2]concat=n=3:v=0:a=1[aout]"
    )
    run([
        ffmpeg, "-y", "-i", str(original), "-i", str(donor),
        "-filter_complex", graph, "-map", "0:v:0?", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(output),
    ], dry_run)


def run_ffmpeg_video_splice(row: Dict[str, str], ffmpeg: str, dry_run: bool) -> None:
    original = Path(row["input_video_path"])
    donor = Path(row["donor_video_path"])
    output = Path(row["output_video_path"])
    if not row.get("target_start_sec") or not row.get("target_end_sec"):
        raise ValueError("ffmpeg_video_splice requires target_start_sec and target_end_sec")
    start = float(row["target_start_sec"])
    end = float(row["target_end_sec"])
    if start < 0 or end <= start:
        raise ValueError(f"invalid splice interval {start}..{end}")
    duration = end - start
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
    graph = (
        f"[1:v]trim=start=0:end={duration},setpts=PTS-STARTPTS,fps=25[donor0];"
        "[donor0][0:v]scale2ref=w=iw:h=ih[donor1][base0];"
        "[donor1]setsar=1,format=yuv420p[donor];"
        "[base0]fps=25,setsar=1,format=yuv420p,split=2[base1][base2];"
        f"[base1]trim=start=0:end={start},setpts=PTS-STARTPTS[v0];"
        f"[base2]trim=start={end},setpts=PTS-STARTPTS[v2];"
        "[v0][donor][v2]concat=n=3:v=1:a=0[vout]"
    )
    run([
        ffmpeg, "-y", "-i", str(original), "-i", str(donor),
        "-filter_complex", graph, "-map", "[vout]", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy", "-shortest", str(output),
    ], dry_run)


def run_ffmpeg_full_video_replacement(row: Dict[str, str], ffmpeg: str,
                                      ffprobe: str, dry_run: bool) -> None:
    original = Path(row["input_video_path"])
    donor = Path(row["donor_video_path"])
    output = Path(row["output_video_path"])
    duration = 10.0 if dry_run else probe_duration(ffprobe, original)
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
    run([
        ffmpeg, "-y", "-stream_loop", "-1", "-i", str(donor), "-i", str(original),
        "-t", f"{duration:.6f}",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", str(output),
    ], dry_run)


def run_ffmpeg_recompress(row: Dict[str, str], ffmpeg: str, dry_run: bool) -> None:
    original = Path(row["input_video_path"])
    output = Path(row["output_video_path"])
    crf = str(row.get("video_crf", "") or "36")
    bitrate = str(row.get("audio_bitrate", "") or "32k")
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
    run([
        ffmpeg, "-y", "-i", str(original), "-c:v", "libx264", "-crf", crf,
        "-preset", "medium", "-c:a", "aac", "-b:a", bitrate, str(output),
    ], dry_run)


def format_command(parts: List[str], values: Dict[str, str]) -> List[str]:
    try:
        return [str(part).format_map(values) for part in parts]
    except KeyError as exc:
        raise ValueError(f"unknown command placeholder: {exc}") from exc


def run_external(row: Dict[str, str], spec: Dict, ffmpeg: str,
                 work_root: Path, device: str, dry_run: bool) -> None:
    mode = spec.get("mode")
    command = spec.get("command")
    if mode not in {"command_audio", "command_video"}:
        raise ValueError(f"unsupported external wrapper mode {mode!r}")
    if not isinstance(command, list) or not command:
        raise ValueError("external wrapper command must be a non-empty JSON array")

    task_dir = work_root / row["task_id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    input_audio = task_dir / "input.wav"
    donor_audio = task_dir / "donor.wav"
    output_audio = task_dir / "generated.wav"
    output_video = Path(row["output_video_path"])
    if not dry_run:
        output_video.parent.mkdir(parents=True, exist_ok=True)

    values = dict(row)
    values.update(
        {
            "device": device,
            "input_audio_path": str(input_audio),
            "donor_audio_path": str(donor_audio),
            "output_audio_path": str(output_audio),
            "output_video_path": row["output_video_path"],
            "work_dir": str(task_dir),
        }
    )
    if mode == "command_audio":
        extract_audio(ffmpeg, Path(row["input_video_path"]), input_audio, dry_run)
        if row.get("donor_video_path"):
            extract_audio(ffmpeg, Path(row["donor_video_path"]), donor_audio, dry_run)
        run(format_command(command, values), dry_run)
        if not dry_run and not output_audio.exists():
            raise FileNotFoundError(f"audio wrapper did not create {output_audio}")
        remux_audio(ffmpeg, Path(row["input_video_path"]), output_audio,
                    output_video, dry_run)
    else:
        run(format_command(command, values), dry_run)
        if not dry_run and not output_video.exists():
            raise FileNotFoundError(f"video wrapper did not create {output_video}")


def load_config(path: Optional[Path]) -> Dict[str, Dict]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    wrappers = data.get("wrappers", data)
    if not isinstance(wrappers, dict):
        raise ValueError("wrapper config must contain an object named 'wrappers'")
    return wrappers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--config", type=Path,
                    help="JSON external wrapper config; optional for built-ins")
    ap.add_argument("--generators", nargs="+",
                    help="run only these generators, e.g. ffmpeg_splice XTTS")
    ap.add_argument("--work-root", type=Path, default=Path("rq1_wrapper_work"))
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--failures-csv", type=Path)
    ap.add_argument(
        "--allow-failures",
        action="store_true",
        help="write failures CSV but exit 0 so long-running shard jobs can continue",
    )
    args = ap.parse_args()

    rows = read_csv(args.tasks)
    missing = REQUIRED_TASK_COLUMNS - set(rows[0]) if rows else REQUIRED_TASK_COLUMNS
    if missing:
        raise ValueError(f"task CSV missing columns: {sorted(missing)}")
    selected = set(args.generators or [])
    if selected:
        rows = [row for row in rows if row["generator"] in selected]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < --num-shards")
    rows = rows[args.shard_index::args.num_shards]
    if args.limit:
        rows = rows[:args.limit]
    wrappers = load_config(args.config)
    failures = []
    n_ok = n_skipped = 0

    for index, row in enumerate(rows, start=1):
        output = Path(row["output_video_path"])
        if args.resume and output.exists():
            n_skipped += 1
            print(f"[wrapper {index}/{len(rows)}] skip   {row['task_id']}", flush=True)
            continue
        try:
            if row["generator"] == "ffmpeg_splice":
                run_ffmpeg_splice(row, args.ffmpeg, args.dry_run)
            elif row["generator"] == "ffmpeg_video_splice":
                run_ffmpeg_video_splice(row, args.ffmpeg, args.dry_run)
            elif row["generator"] == "ffmpeg_full_video_replacement":
                run_ffmpeg_full_video_replacement(
                    row, args.ffmpeg, args.ffprobe, args.dry_run
                )
            elif row["generator"] == "ffmpeg_recompress":
                run_ffmpeg_recompress(row, args.ffmpeg, args.dry_run)
            elif row["generator"] in wrappers:
                run_external(row, wrappers[row["generator"]], args.ffmpeg,
                             args.work_root, args.device, args.dry_run)
            else:
                raise ValueError(
                    f"no wrapper configured for generator {row['generator']!r}; "
                    "run a configured subset with --generators or add it to --config"
                )
            n_ok += 1
            print(f"[wrapper {index}/{len(rows)}] ok     {row['task_id']}", flush=True)
        except Exception as exc:
            fail = dict(row)
            fail["error"] = str(exc)[-1000:]
            failures.append(fail)
            print(f"[wrapper {index}/{len(rows)}] FAILED {row['task_id']}: {exc}", flush=True)

    failure_path = args.failures_csv or args.tasks.with_suffix(".wrapper_failures.csv")
    if failures:
        write_csv(failure_path, failures)
        print(f"[wrapper] failures -> {failure_path}")
    elif failure_path.exists():
        failure_path.unlink()
    print(f"[wrapper] done ok={n_ok} skipped={n_skipped} failed={len(failures)}")
    if failures and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
