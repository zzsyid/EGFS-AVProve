#!/usr/bin/env python3
"""Evaluate N1000 provenance-state predictions against attack-construction labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCAL = "localized-tamper"
MODAL = "tampered/modal-replacement"
LEGACY_MODAL = "modal-replacement"
VERIFIED = "source-and-content-verified"
CONTENT = "content-unverified"
REJECTED = "rejected"
LEGACY_REJECTED = "unknown"
STATE_ORDER = [VERIFIED, LOCAL, MODAL, CONTENT, REJECTED]

TARGETS: Dict[str, Dict[str, str]] = {
    "kNN-VC": {
        "state": MODAL,
        "basis": "Cross-source speech replacement changes the audio modality.",
    },
    "LivePortrait": {
        "state": MODAL,
        "basis": "Cross-source portrait animation replaces the visual modality.",
    },
    "ffmpeg_splice": {
        "state": LOCAL,
        "basis": "A bounded audio interval is replaced.",
    },
    "ffmpeg_video_splice": {
        "state": LOCAL,
        "basis": "A bounded video interval is replaced.",
    },
    "AniPortrait": {
        "state": LOCAL,
        "basis": "The generated face region is a spatially localized edit.",
    },
    "EchoMimicV3": {
        "state": LOCAL,
        "basis": "The generated talking-face region is a spatially localized edit.",
    },
    "ffmpeg_full_video_replacement": {
        "state": LOCAL,
        "basis": "The face-swap task replaces the face region over the clip duration.",
    },
}


def canonical_id(value: object) -> str:
    result = str(value)
    for suffix in ("__M7locatorv2", "__M7shortptr"):
        if result.endswith(suffix):
            result = result[: -len(suffix)]
    return result


def safe_state(value: object) -> str:
    raw = str(value).strip()
    aliases = {
        "verified": VERIFIED,
        "source-and-content verified": VERIFIED,
        "localized tamper": LOCAL,
        "modal replacement": MODAL,
        "modal_replacement": MODAL,
        LEGACY_MODAL: MODAL,
        LEGACY_REJECTED: REJECTED,
    }
    return aliases.get(raw, raw)


def div(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def fmt(value: float) -> str:
    return f"{100.0 * value:.2f}"


def write_confusion_figure(counts: pd.DataFrame, output: Path) -> None:
    active = [state for state in STATE_ORDER if counts.loc[state].sum() or counts[state].sum()]
    shown = counts.loc[active, active]
    normalized = shown.div(shown.sum(axis=1).replace(0, 1), axis=0) * 100.0

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    image = ax.imshow(normalized.to_numpy(), cmap="Blues", vmin=0, vmax=100)
    labels = [state.replace("tampered/", "").replace("-", " ") for state in active]
    ax.set_xticks(range(len(active)), labels, rotation=22, ha="right")
    ax.set_yticks(range(len(active)), labels)
    ax.set_xlabel("Predicted state")
    ax.set_ylabel("Attack-construction target")
    for row in range(len(active)):
        for col in range(len(active)):
            value = normalized.iat[row, col]
            count = int(shown.iat[row, col])
            color = "white" if value >= 55 else "black"
            ax.text(col, row, f"{count}\n({value:.1f}%)", ha="center", va="center", color=color)
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label("Row-normalized rate (%)")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_tables(
    out_dir: Path,
    per_attack: pd.DataFrame,
    metrics: pd.DataFrame,
    counts: pd.DataFrame,
    summary: Dict[str, object],
) -> None:
    attack_rows = []
    tex_rows = []
    for row in per_attack.itertuples(index=False):
        attack_rows.append(
            f"| {row.generator} | {row.target_state} | {row.n} | {row.correct} | {row.target_match_pct:.2f} |"
        )
        name = str(row.generator).replace("_", r"\_")
        target = str(row.target_state).replace("-", " ")
        tex_rows.append(
            f"{name} & {target} & {row.n} & {row.correct} & {row.target_match_pct:.2f} \\\\"
        )

    accuracy = float(summary["overall_accuracy_pct"])
    macro_f1 = float(summary["macro_f1_pct"])
    markdown = "\n".join([
        "# Independent Provenance-State Correctness",
        "",
        "Targets are fixed from the attack construction and do not use M4 or M8.1 predictions.",
        "",
        "| Attack model | Independent target | N | Correct | Target match (%) |",
        "|---|---|---:|---:|---:|",
        *attack_rows,
        "",
        f"Overall accuracy: **{accuracy:.2f}%**; macro-F1: **{macro_f1:.2f}%**.",
        "",
        "## Per-State Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Confusion Matrix (Counts)",
        "",
        counts.to_markdown(),
        "",
    ])
    (out_dir / "STATE_CORRECTNESS.md").write_text(markdown, encoding="utf-8")

    latex = "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Independent provenance-state correctness on the closed N1000 media-realized attack subset. Target states are assigned from the attack construction without using verifier outputs.}",
        r"\label{tab:n1000_independent_state_correctness}",
        r"\begin{tabular}{l l r r r}",
        r"\toprule",
        "Attack Model & Independent Target & N & Correct & Target Match (\\%) \\\\",
        r"\midrule",
        *tex_rows,
        r"\midrule",
        f"Overall & -- & {int(summary['n'])} & {int(summary['correct'])} & {accuracy:.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ])
    (out_dir / "STATE_CORRECTNESS.tex").write_text(latex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("rq1_eval/artifacts_m81_n1000_eval_counterfactual_v03_e10/m81_n1000_predictions.csv"),
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path("rq1_eval/artifacts_m7_m6_e2e_scale1000/m7_m6_scale1000_incremental_tasks.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("rq1_eval/artifacts_reviewer_evidence_20260911/state_correctness"),
    )
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions).fillna("")
    tasks = pd.read_csv(args.tasks).fillna("")
    predictions["join_id"] = predictions["sample_id"].map(canonical_id)
    tasks["join_id"] = tasks["sample_id"].map(canonical_id)
    task_columns = [
        "join_id", "generator", "cond", "attack_scope", "attack_family",
        "target_start_sec", "target_end_sec", "m6_attack_type",
    ]
    merged = predictions.merge(
        tasks[task_columns], on="join_id", how="left", suffixes=("_prediction", "_task"), validate="one_to_one"
    )
    if merged["generator_task"].eq("").any():
        missing = merged.loc[merged["generator_task"].eq(""), "sample_id"].tolist()
        raise ValueError(f"missing independent task metadata for {len(missing)} predictions")
    unknown_generators = sorted(set(merged["generator_task"]) - set(TARGETS))
    if unknown_generators:
        raise ValueError(f"no independent target mapping for: {unknown_generators}")

    merged["target_state"] = merged["generator_task"].map(lambda value: TARGETS[value]["state"])
    merged["target_basis"] = merged["generator_task"].map(lambda value: TARGETS[value]["basis"])
    merged["predicted_state"] = merged["m81_masked_state"].map(safe_state)
    merged["target_match"] = merged["target_state"] == merged["predicted_state"]

    observed = set(merged["target_state"]) | set(merged["predicted_state"])
    order = [state for state in STATE_ORDER if state in observed]
    order.extend(sorted(observed - set(order)))
    counts = pd.crosstab(merged["target_state"], merged["predicted_state"]).reindex(
        index=STATE_ORDER, columns=STATE_ORDER, fill_value=0
    )
    normalized = counts.div(counts.sum(axis=1).replace(0, 1), axis=0) * 100.0

    metric_rows: List[Dict[str, object]] = []
    f1_values = []
    for state in order:
        tp = int(((merged.target_state == state) & (merged.predicted_state == state)).sum())
        fp = int(((merged.target_state != state) & (merged.predicted_state == state)).sum())
        fn = int(((merged.target_state == state) & (merged.predicted_state != state)).sum())
        precision = div(tp, tp + fp)
        recall = div(tp, tp + fn)
        f1 = div(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        metric_rows.append({
            "State": state,
            "Support": int((merged.target_state == state).sum()),
            "Precision (%)": round(100 * precision, 2),
            "Recall (%)": round(100 * recall, 2),
            "F1 (%)": round(100 * f1, 2),
        })
    metrics = pd.DataFrame(metric_rows)

    per_attack = (
        merged.groupby(["generator_task", "target_state"], as_index=False)
        .agg(n=("sample_id", "size"), correct=("target_match", "sum"))
        .rename(columns={"generator_task": "generator"})
    )
    per_attack["target_match_pct"] = 100.0 * per_attack["correct"] / per_attack["n"]
    summary: Dict[str, object] = {
        "scope": "Closed scale1000 media-realized subset with independent attack-construction targets.",
        "n": int(len(merged)),
        "correct": int(merged.target_match.sum()),
        "overall_accuracy_pct": float(100.0 * merged.target_match.mean()),
        "macro_f1_pct": float(100.0 * np.mean(f1_values)),
        "false_verified": int((merged.predicted_state == VERIFIED).sum()),
        "rejected_outputs": int((merged.predicted_state == REJECTED).sum()),
        "target_definition": TARGETS,
        "prediction_field": "m81_masked_state",
        "target_source": str(args.tasks),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_dir / "state_correctness_rows.csv", index=False)
    per_attack.to_csv(args.out_dir / "state_correctness_by_attack.csv", index=False)
    metrics.to_csv(args.out_dir / "state_correctness_per_state.csv", index=False)
    counts.to_csv(args.out_dir / "state_confusion_counts.csv")
    normalized.to_csv(args.out_dir / "state_confusion_row_percent.csv")
    (args.out_dir / "state_correctness_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_confusion_figure(counts, args.out_dir / "state_confusion_matrix")
    write_tables(args.out_dir, per_attack, metrics, counts, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
