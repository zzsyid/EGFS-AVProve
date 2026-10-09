#!/usr/bin/env python3
"""Measure temporal localization on N1000 interval-splice attacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from quantize import LSHQuantizer


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def interval_overlap(start: float, end: float, target_start: float, target_end: float) -> bool:
    return min(end, target_end) > max(start, target_start)


def merge_intervals(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    merged: List[Tuple[float, float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_length(intervals: Iterable[Tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def temporal_iou(
    predicted: Iterable[Tuple[float, float]], target_start: float, target_end: float
) -> float:
    predicted_merged = merge_intervals(predicted)
    target_length = max(0.0, target_end - target_start)
    intersection = sum(
        max(0.0, min(end, target_end) - max(start, target_start))
        for start, end in predicted_merged
    )
    predicted_length = interval_length(predicted_merged)
    union = predicted_length + target_length - intersection
    return float(intersection / union) if union > 0 else 0.0


def binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels, dtype=bool)
    scores_array = np.asarray(scores, dtype=float)
    positives = int(labels_array.sum())
    negatives = int((~labels_array).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores_array, kind="mergesort")
    sorted_scores = scores_array[order]
    ranks = np.empty(len(scores_array), dtype=float)
    start = 0
    while start < len(scores_array):
        end = start + 1
        while end < len(scores_array) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = float(ranks[labels_array].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def summarize_windows(rows: pd.DataFrame, modality: str) -> Dict[str, float]:
    tp = int((rows.target_positive & rows.predicted_positive).sum())
    fp = int((~rows.target_positive & rows.predicted_positive).sum())
    fn = int((rows.target_positive & ~rows.predicted_positive).sum())
    tn = int((~rows.target_positive & ~rows.predicted_positive).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "modality": modality,
        "samples": int(rows.sample_id.nunique()),
        "windows": int(len(rows)),
        "positive_windows": int(rows.target_positive.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "window_precision_pct": 100.0 * precision,
        "window_recall_pct": 100.0 * recall,
        "window_f1_pct": 100.0 * f1,
        "localization_auc_pct": 100.0 * binary_auc(rows.target_positive, rows.score),
    }


def sample_metrics(rows: pd.DataFrame, modality: str) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for sample_id, group in rows.groupby("sample_id", sort=True):
        target_start = float(group.target_start_sec.iloc[0])
        target_end = float(group.target_end_sec.iloc[0])
        predicted = [
            (float(row.window_start_sec), float(row.window_end_sec))
            for row in group.itertuples(index=False)
            if bool(row.predicted_positive)
        ]
        positives = group.target_positive
        recalled = int((group.target_positive & group.predicted_positive).sum())
        records.append({
            "modality": modality,
            "sample_id": sample_id,
            "target_start_sec": target_start,
            "target_end_sec": target_end,
            "window_count": int(len(group)),
            "positive_window_count": int(positives.sum()),
            "alarmed_window_count": int(group.predicted_positive.sum()),
            "temporal_iou": temporal_iou(predicted, target_start, target_end),
            "window_recall": recalled / int(positives.sum()) if positives.any() else 0.0,
            "any_alarm_overlaps_target": bool(recalled > 0),
        })
    return pd.DataFrame(records)


def video_rows(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path).fillna("")
    data = data[data.generator.eq("ffmpeg_video_splice")].copy()
    data["target_start_sec"] = pd.to_numeric(data.target_start_sec)
    data["target_end_sec"] = pd.to_numeric(data.target_end_sec)
    data["window_start_sec"] = pd.to_numeric(data.window_start_sec)
    data["window_end_sec"] = pd.to_numeric(data.window_end_sec)
    data["score"] = pd.to_numeric(data.window_score)
    data["predicted_positive"] = data.window_alarm.map(bool_value)
    data["target_positive"] = data.apply(
        lambda row: interval_overlap(
            row.window_start_sec, row.window_end_sec, row.target_start_sec, row.target_end_sec
        ),
        axis=1,
    )
    data["modality"] = "Video"
    return data[
        [
            "modality", "sample_id", "target_start_sec", "target_end_sec",
            "window_start_sec", "window_end_sec", "target_positive",
            "predicted_positive", "score",
        ]
    ]


def read_manifests(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path).fillna("") for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["sample_id"], keep="last")


def audio_rows(
    scores_path: Path,
    query_manifests: Sequence[Path],
    source_manifest: Path,
    threshold_bits: int,
) -> pd.DataFrame:
    scores = pd.read_csv(scores_path).fillna("")
    scores = scores[scores.generator.eq("ffmpeg_splice")].copy()
    query = read_manifests(query_manifests).set_index("sample_id", drop=False)
    source = pd.read_csv(source_manifest).fillna("")
    source = source[source.population.eq("orig")].drop_duplicates("enroll_id").set_index("enroll_id")
    first_path = Path(str(source.iloc[0].evt_audio_windows_path))
    in_dim = int(np.load(first_path, mmap_mode="r").shape[1])
    quantizer = LSHQuantizer(in_dim, 511, key=b"rq1-stage0-real", tag="evt_audio_v2")

    rows: List[Dict[str, object]] = []
    for score_row in scores.itertuples(index=False):
        sample_id = str(score_row.sample_id)
        if sample_id not in query.index:
            raise ValueError(f"missing audio query features for {sample_id}")
        query_row = query.loc[sample_id]
        if score_row.enroll_id not in source.index:
            raise ValueError(f"missing audio reference features for {score_row.enroll_id}")
        source_row = source.loc[score_row.enroll_id]

        query_features = np.load(str(query_row.evt_audio_windows_path))
        query_starts = np.load(str(query_row.evt_audio_window_starts_path)).reshape(-1)
        source_features = np.load(str(source_row.evt_audio_windows_path))
        source_starts = np.load(str(source_row.evt_audio_window_starts_path)).reshape(-1)
        query_codes = quantizer.encode_batch(query_features)
        source_codes = quantizer.encode_batch(source_features)
        target_start = float(score_row.target_start_sec)
        target_end = float(score_row.target_end_sec)

        for code, start in zip(query_codes, query_starts):
            reference_index = int(np.argmin(np.abs(source_starts - start)))
            hamming = int(np.count_nonzero(code != source_codes[reference_index]))
            end = float(start) + 0.5
            rows.append({
                "modality": "Audio",
                "sample_id": sample_id,
                "target_start_sec": target_start,
                "target_end_sec": target_end,
                "window_start_sec": float(start),
                "window_end_sec": end,
                "target_positive": interval_overlap(float(start), end, target_start, target_end),
                "predicted_positive": hamming > threshold_bits,
                "score": hamming,
            })
    return pd.DataFrame(rows)


def write_paper_table(out_dir: Path, overall: pd.DataFrame) -> None:
    rows = []
    md_rows = []
    for row in overall.itertuples(index=False):
        rows.append(
            f"{row.modality} splice & {row.samples} & {row.windows} & "
            f"{row.temporal_iou_pct:.2f} & {row.window_precision_pct:.2f} & "
            f"{row.window_recall_pct:.2f} & {row.localization_auc_pct:.2f} \\\\"
        )
        md_rows.append(
            f"| {row.modality} splice | {row.samples} | {row.windows} | "
            f"{row.temporal_iou_pct:.2f} | {row.window_precision_pct:.2f} | "
            f"{row.window_recall_pct:.2f} | {row.localization_auc_pct:.2f} |"
        )
    latex = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Temporal localization on interval-level audio and video splice attacks in the closed N1000 subset. IoU is computed from the union of alarmed windows and the annotated replacement interval.}",
        r"\label{tab:n1000_temporal_localization}",
        r"\begin{tabular}{l r r r r r r}",
        r"\toprule",
        "Attack & N & Windows & Temporal IoU (\\%) & Precision (\\%) & Recall (\\%) & AUC (\\%) \\\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    (out_dir / "TEMPORAL_LOCALIZATION.tex").write_text(latex, encoding="utf-8")
    markdown = "\n".join([
        "# N1000 Temporal Localization",
        "",
        "| Attack | N | Windows | Temporal IoU (%) | Precision (%) | Recall (%) | AUC (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *md_rows,
        "",
        "Targets are the attack intervals recorded by the task generator. Video alarms use the calibrated video-window detector; audio alarms use BCH radius failures (t=54).",
        "",
    ])
    (out_dir / "TEMPORAL_LOCALIZATION.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("rq1_eval/artifacts_m7_m6_e2e_scale1000/incremental")
    parser.add_argument(
        "--video-window-scores", type=Path,
        default=root / "video_window_evaluation/m2_video_window_scores.csv",
    )
    parser.add_argument(
        "--audio-window-scores", type=Path,
        default=root / "audio_window_span05_hop025/segmented_evt_bch_scores.csv",
    )
    parser.add_argument(
        "--audio-query-manifests", type=Path, nargs="+",
        default=[
            root / "features/evt_audio_span05_hop025_features.worker0.csv",
            root / "features/evt_audio_span05_hop025_features.worker1.csv",
        ],
    )
    parser.add_argument(
        "--audio-source-manifest", type=Path,
        default=root / "features/source_clean/evt_audio_span05_hop025_features.csv",
    )
    parser.add_argument("--audio-threshold-bits", type=int, default=54)
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("rq1_eval/artifacts_reviewer_evidence_20260911/temporal_localization"),
    )
    args = parser.parse_args()

    video = video_rows(args.video_window_scores)
    audio = audio_rows(
        args.audio_window_scores,
        args.audio_query_manifests,
        args.audio_source_manifest,
        args.audio_threshold_bits,
    )
    all_windows = pd.concat([audio, video], ignore_index=True)
    sample_results = pd.concat(
        [sample_metrics(audio, "Audio"), sample_metrics(video, "Video")], ignore_index=True
    )
    overall_rows = []
    for modality, windows in (("Audio", audio), ("Video", video)):
        metrics = summarize_windows(windows, modality)
        samples = sample_results[sample_results.modality.eq(modality)]
        metrics["temporal_iou_pct"] = 100.0 * float(samples.temporal_iou.mean())
        metrics["median_temporal_iou_pct"] = 100.0 * float(samples.temporal_iou.median())
        metrics["any_overlap_pct"] = 100.0 * float(samples.any_alarm_overlaps_target.mean())
        overall_rows.append(metrics)
    overall = pd.DataFrame(overall_rows)

    summary = {
        "scope": "Interval-splice localization on the closed N1000 media-realized subset.",
        "ground_truth": "Attack intervals emitted by the task generator.",
        "audio_alarm": f"BCH radius failure with t={args.audio_threshold_bits}.",
        "video_alarm": "Calibrated M2 video-window alarm.",
        "overall": overall.to_dict(orient="records"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_windows.to_csv(args.out_dir / "temporal_localization_windows.csv", index=False)
    sample_results.to_csv(args.out_dir / "temporal_localization_by_sample.csv", index=False)
    overall.to_csv(args.out_dir / "temporal_localization_summary.csv", index=False)
    (args.out_dir / "temporal_localization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_paper_table(args.out_dir, overall)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
