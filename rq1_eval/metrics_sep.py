#!/usr/bin/env python3
"""RQ1 outputs (1) and (2): per-channel FE pass/fail + margin vs attack-level.

Consumes the shared base table `artifacts/fe_passfail.parquet` (or .csv) produced
by run.py / populations.py, with the canonical schema documented below, and emits:

  Output (1)  per-channel FE pass/fail summary
              -> artifacts/output1_passfail_summary.csv
  Output (2)  margin vs attack-level (the headline figure) + separability metrics
              -> artifacts/output2_margin_metrics.csv
              -> artifacts/output2_margin.png   (5 panels: id_audio/id_video/evt/sync/fused)

Separability is measured on the *quantized* Hamming distance d_ham (the true FE
pass/fail basis); d_cos is carried only for diagnosis. Metrics (AUC / EER /
d-prime) compare P_same(level) against P_tamper per channel. The "fused" channel
is the four-channel AND: a sample is accepted iff every channel is within radius,
so its separability score is max_channel(d_ham) (worst channel governs the AND),
matching the Level-B worst-candidate logic in RQ1_eval_spec.md sec.7.

Pure-numpy stats (scipy.stats / sklearn are unavailable in the target env):
  * AUC via tie-aware Mann-Whitney U
  * EER via threshold sweep
  * d-prime via pooled-std standardized mean gap

Canonical fe_passfail schema (one row per (sample, channel)):
  clip_id     str
  source_id   str
  channel     str   in {id_audio, id_video, evt, sync}
  population  str   in {orig, same, tamper, cross}
  cond        str   level (L1/L2/L3) for same; tamper type for tamper;
                    "clean" for orig; "cross" for cross
  generator   str   generator name or "none"
  split       str   in {calib, eval}
  ham_bits    int   Hamming(Q(z), Q(z')) in bits
  n           int   code length (bits)
  d_ham       float ham_bits / n   (normalized, for plotting/scoring)
  d_cos       float continuous cosine distance (diagnostic only)
  t           int   FE correction radius (bits); pass == (ham_bits <= t)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CHANNELS = ["id_audio", "id_video", "evt", "sync"]
SAME_LEVELS = ["L1", "L2", "L3"]


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_passfail(path: Path) -> pd.DataFrame:
    """Load the base table; supports .parquet (if engine present) and .csv."""
    path = Path(path)
    if path.suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # pyarrow/fastparquet may be missing
            csv_alt = path.with_suffix(".csv")
            if csv_alt.exists():
                print(f"[warn] parquet read failed ({exc}); using {csv_alt}")
                return pd.read_csv(csv_alt)
            raise
    return pd.read_csv(path)


def _ensure_pass(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pass" not in df.columns:
        df["pass"] = df["ham_bits"] <= df["t"]
    df["pass"] = df["pass"].astype(bool)
    if "d_ham" not in df.columns:
        df["d_ham"] = df["ham_bits"] / df["n"]
    return df


# --------------------------------------------------------------------------- #
# Pure-numpy separability statistics
# --------------------------------------------------------------------------- #
def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """1-based average ranks (handles ties)."""
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = r
        i = j + 1
    return ranks


def auc(same: np.ndarray, tamper: np.ndarray) -> float:
    """AUC for ranking tamper (high d_ham) above same (low d_ham).

    Equivalent to P(score_tamper > score_same) with ties counted at 0.5.
    """
    same = np.asarray(same, float)
    tamper = np.asarray(tamper, float)
    n_t, n_s = len(tamper), len(same)
    if n_t == 0 or n_s == 0:
        return float("nan")
    ranks = _avg_ranks(np.concatenate([tamper, same]))
    r_tamper = ranks[:n_t].sum()
    u = r_tamper - n_t * (n_t + 1) / 2.0
    return float(u / (n_t * n_s))


def eer(same: np.ndarray, tamper: np.ndarray) -> Tuple[float, float]:
    """Equal error rate and the (normalized) threshold achieving it.

    same  should be accepted (low score); tamper should be rejected (high score).
    At threshold thr: FAR = P(tamper <= thr) (tamper wrongly accepted),
                      FRR = P(same  >  thr) (same  wrongly rejected).
    """
    same = np.asarray(same, float)
    tamper = np.asarray(tamper, float)
    if len(same) == 0 or len(tamper) == 0:
        return float("nan"), float("nan")
    cand = np.unique(np.concatenate([same, tamper]))
    cand = np.concatenate([[cand[0] - 1e-9], cand, [cand[-1] + 1e-9]])
    best_gap, best = np.inf, (float("nan"), float("nan"))
    for thr in cand:
        far = float(np.mean(tamper <= thr))
        frr = float(np.mean(same > thr))
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            best = ((far + frr) / 2.0, float(thr))
    return best


def dprime(same: np.ndarray, tamper: np.ndarray) -> float:
    same = np.asarray(same, float)
    tamper = np.asarray(tamper, float)
    if len(same) < 2 or len(tamper) < 2:
        return float("nan")
    pooled = np.sqrt(0.5 * (same.var(ddof=1) + tamper.var(ddof=1)))
    if pooled == 0:
        return float("inf") if tamper.mean() != same.mean() else 0.0
    return float((tamper.mean() - same.mean()) / pooled)


# --------------------------------------------------------------------------- #
# Fused (four-channel AND) view
# --------------------------------------------------------------------------- #
def build_fused(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 4 channels into a synthetic 'fused' channel per sample.

    fused pass  = AND over channels (all within radius)
    fused score = max_channel(d_ham)   (worst channel governs the AND)
    A sample key is `sample_id` when present (one attacked instance), else
    (clip_id, source_id, population, cond, generator, split).
    Only samples that have all CHANNELS present are kept.
    """
    if "sample_id" in df.columns:
        keys = ["sample_id", "source_id", "population", "cond", "generator", "split"]
    else:
        keys = ["clip_id", "source_id", "population", "cond", "generator", "split"]
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        present = set(g["channel"])
        if not set(CHANNELS).issubset(present):
            continue
        d = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        d["channel"] = "fused"
        d["d_ham"] = float(g["d_ham"].max())
        d["pass"] = bool(g["pass"].all())
        d["ham_bits"] = int(g["ham_bits"].max())
        d["n"] = int(g["n"].iloc[0])
        d["t"] = int(g["t"].iloc[0])
        d["d_cos"] = float(g["d_cos"].max())
        rows.append(d)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Output (1): pass/fail summary
# --------------------------------------------------------------------------- #
def output1_passfail(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    df = _ensure_pass(df)
    grp = (
        df.groupby(["split", "channel", "population", "cond"], dropna=False)
        .agg(
            n_samples=("pass", "size"),
            pass_rate=("pass", "mean"),
            d_ham_mean=("d_ham", "mean"),
            d_ham_med=("d_ham", "median"),
            d_ham_p99=("d_ham", lambda s: float(np.quantile(s, 0.99))),
        )
        .reset_index()
        .sort_values(["split", "channel", "population", "cond"])
    )
    out = outdir / "output1_passfail_summary.csv"
    grp.to_csv(out, index=False)
    print(f"[output1] wrote {out}  ({len(grp)} rows)")
    return grp


# --------------------------------------------------------------------------- #
# Output (2): margin vs attack-level
# --------------------------------------------------------------------------- #
def output2_margin(
    df: pd.DataFrame,
    outdir: Path,
    split: str = "eval",
) -> pd.DataFrame:
    df = _ensure_pass(df)
    fused = build_fused(df)
    full = pd.concat([df, fused], ignore_index=True)
    sub = full[full["split"] == split]
    if len(sub) == 0:
        print(f"[output2][warn] no rows for split={split}; using all rows")
        sub = full

    channels = CHANNELS + ["fused"]
    metric_rows: List[Dict] = []
    panels: Dict[str, Dict] = {}

    for ch in channels:
        cdf = sub[sub["channel"] == ch]
        tamper = cdf[cdf["population"] == "tamper"]["d_ham"].to_numpy()
        # per-level same distributions
        same_by_level = {
            lvl: cdf[(cdf["population"] == "same") & (cdf["cond"] == lvl)]["d_ham"].to_numpy()
            for lvl in SAME_LEVELS
        }
        same_l1l2 = np.concatenate(
            [same_by_level["L1"], same_by_level["L2"]]
        ) if (len(same_by_level["L1"]) + len(same_by_level["L2"])) else np.array([])

        # separability of each level vs all tamper
        for lvl in SAME_LEVELS:
            s = same_by_level[lvl]
            if len(s) == 0 or len(tamper) == 0:
                continue
            a, (e, thr) = auc(s, tamper), eer(s, tamper)
            metric_rows.append({
                "channel": ch, "same_level": lvl,
                "n_same": len(s), "n_tamper": len(tamper),
                "auc": a, "eer": e, "eer_thr": thr,
                "dprime": dprime(s, tamper),
            })

        # suggested t-placement window (normalized), informational only
        t_lo = float(np.quantile(same_l1l2, 0.99)) if len(same_l1l2) else float("nan")
        t_hi = float(np.quantile(tamper, 0.01)) if len(tamper) else float("nan")
        t_norm = float(cdf["t"].iloc[0] / cdf["n"].iloc[0]) if len(cdf) else float("nan")

        panels[ch] = {
            "same_by_level": same_by_level,
            "tamper": tamper,
            "t_norm": t_norm,
            "t_lo": t_lo,
            "t_hi": t_hi,
        }

    metrics = pd.DataFrame(metric_rows)
    out_csv = outdir / "output2_margin_metrics.csv"
    metrics.to_csv(out_csv, index=False)
    print(f"[output2] wrote {out_csv}  ({len(metrics)} rows)")

    _plot_margin(panels, channels, outdir, split)
    return metrics


def output2_localization(df: pd.DataFrame, outdir: Path, split: str = "eval") -> pd.DataFrame:
    """Per-channel x per-tamper-type AUC (channel d_ham: same-L1L2 vs each tamper).

    High cell = that channel detects that tamper. This is the honest 'separability'
    view: a single id channel is NOT expected to flag cross-modal tampers it does
    not touch; the design relies on EACH tamper being caught by at least one
    channel (a high cell per column) and on the fused AND. Foreshadows output (4)
    localization.
    """
    df = _ensure_pass(df)
    sub = df[df["split"] == split]
    if len(sub) == 0:
        sub = df
    tamper_types = sorted(sub[sub.population == "tamper"]["cond"].unique())
    rows = []
    for ch in CHANNELS:
        cdf = sub[sub.channel == ch]
        same = cdf[(cdf.population == "same") & (cdf.cond.isin(["L1", "L2"]))]["d_ham"].to_numpy()
        rec = {"channel": ch}
        for tt in tamper_types:
            t_d = cdf[(cdf.population == "tamper") & (cdf.cond == tt)]["d_ham"].to_numpy()
            rec[tt] = auc(same, t_d) if (len(same) and len(t_d)) else float("nan")
        rows.append(rec)
    mat = pd.DataFrame(rows).set_index("channel")
    out = outdir / "output2_localization_auc.csv"
    mat.to_csv(out)
    print(f"[output2-loc] wrote {out}")

    # heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1.1 * len(tamper_types) + 2, 3.2))
        im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(tamper_types)))
        ax.set_xticklabels(tamper_types, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(CHANNELS)))
        ax.set_yticklabels(CHANNELS)
        for i in range(len(CHANNELS)):
            for j in range(len(tamper_types)):
                v = mat.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="w" if v < 0.6 else "k", fontsize=7)
        ax.set_title(f"Per-channel detection AUC by tamper type [split={split}]")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(outdir / "output2_localization.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[output2-loc] wrote {outdir / 'output2_localization.png'}")
    except Exception as exc:
        print(f"[output2-loc][warn] heatmap skipped: {exc}")
    return mat


def _plot_margin(panels: Dict, channels: List[str], outdir: Path, split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, ch in zip(axes, channels):
        p = panels[ch]
        groups, labels = [], []
        for lvl in SAME_LEVELS:
            arr = p["same_by_level"][lvl]
            if len(arr):
                groups.append(arr)
                labels.append(f"same\n{lvl}")
        if len(p["tamper"]):
            groups.append(p["tamper"])
            labels.append("tamper")
        if groups:
            bp = ax.boxplot(groups, labels=labels, showfliers=False, patch_artist=True)
            for i, box in enumerate(bp["boxes"]):
                box.set_facecolor("#9ecae1" if labels[i].startswith("same") else "#fc9272")
        if np.isfinite(p["t_norm"]):
            ax.axhline(p["t_norm"], color="k", ls="--", lw=1.2, label=f"t={p['t_norm']:.3f}")
        if np.isfinite(p["t_lo"]) and np.isfinite(p["t_hi"]) and p["t_lo"] < p["t_hi"]:
            ax.axhspan(p["t_lo"], p["t_hi"], color="green", alpha=0.12, label="t window")
        ax.set_title(ch)
        ax.set_ylabel("d_ham (Hamming / n)")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Output (2): margin vs attack-level  [split={split}]", y=1.02)
    fig.tight_layout()
    out_png = outdir / "output2_margin.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[output2] wrote {out_png}")


# --------------------------------------------------------------------------- #
# Synthetic base table (pipeline validation before real data exists)
# --------------------------------------------------------------------------- #
def synth_passfail(
    n_sources: int = 40,
    clips_per_source: int = 4,
    n_bits: int = 255,
    t: int = 10,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a plausible fe_passfail table to validate (1)/(2) end to end.

    d_ham means encode the intended geometry:
      id channels: tiny under L1/L2, moderate under L3, large under tamper
      evt/sync:    moderate under benign, large under swap/splice/lipsync tamper
    """
    rng = np.random.default_rng(seed)
    tamper_types = ["audio_swap", "face_swap", "same_text_TTS",
                    "reenactment", "audio_splice", "lip_sync", "AV_recomp"]
    # per-channel mean d_ham by condition (normalized)
    mu = {
        "id_audio": {"L1": 0.02, "L2": 0.05, "L3": 0.14, "tamper": 0.42},
        "id_video": {"L1": 0.02, "L2": 0.06, "L3": 0.16, "tamper": 0.40},
        "evt":      {"L1": 0.04, "L2": 0.09, "L3": 0.20, "tamper": 0.33},
        "sync":     {"L1": 0.05, "L2": 0.10, "L3": 0.22, "tamper": 0.31},
    }
    sd = 0.03
    rows = []

    def draw(ch, key):
        return float(np.clip(rng.normal(mu[ch][key], sd), 0.0, 0.5))

    for si in range(n_sources):
        split = "calib" if si < n_sources // 2 else "eval"
        src = f"src{si:03d}"
        for ci in range(clips_per_source):
            clip = f"{src}_clip{ci}"
            # orig (sanity, Hamming 0)
            for ch in CHANNELS:
                rows.append(_row(clip, src, ch, "orig", "clean", "none", split,
                                 0.0, n_bits, t, 0.0))
            # same-source attacks L1/L2/L3
            for lvl in SAME_LEVELS:
                gen = {"L1": "ffmpeg", "L2": "encodec",
                       "L3": rng.choice(["FreeVC", "kNN-VC"])}[lvl]
                for ch in CHANNELS:
                    d = draw(ch, lvl)
                    rows.append(_row(clip, src, ch, "same", lvl, str(gen), split,
                                     d, n_bits, t, d * 0.9))
            # tamper
            for tt in tamper_types:
                gen = rng.choice(["SadTalker", "LivePortrait", "XTTS", "DiffPure"])
                for ch in CHANNELS:
                    d = draw(ch, "tamper")
                    rows.append(_row(clip, src, ch, "tamper", tt, str(gen), split,
                                     d, n_bits, t, d * 0.9))
    return pd.DataFrame(rows)


def _row(clip, src, ch, pop, cond, gen, split, d_ham, n, t, d_cos):
    ham_bits = int(round(d_ham * n))
    return {
        "clip_id": clip, "source_id": src, "channel": ch,
        "population": pop, "cond": cond, "generator": gen, "split": split,
        "ham_bits": ham_bits, "n": n, "d_ham": ham_bits / n,
        "d_cos": d_cos, "t": t, "pass": ham_bits <= t,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", default=None,
                    help="path to fe_passfail.parquet/.csv; omit with --selftest")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--split", default="eval",
                    help="split used for output (2) margin (default: eval)")
    ap.add_argument("--selftest", action="store_true",
                    help="run on synthetic data to validate the pipeline")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.selftest or args.table is None:
        print("[selftest] generating synthetic fe_passfail table")
        df = synth_passfail()
        synth_path = outdir / "fe_passfail_synth.csv"
        df.to_csv(synth_path, index=False)
        print(f"[selftest] wrote {synth_path}  ({len(df)} rows)")
    else:
        df = load_passfail(Path(args.table))

    output1_passfail(df, outdir)
    metrics = output2_margin(df, outdir, split=args.split)
    loc = output2_localization(df, outdir, split=args.split)

    # acceptance read-out (RQ1_eval_spec sec.11)
    if len(metrics):
        fused_auc = (
            metrics[(metrics.channel == "fused") & metrics.same_level.isin(["L1", "L2"])]
            ["auc"].mean()
        )
        print(f"[acceptance] FUSED (4-channel AND) AUC, same-L1L2 vs tamper = "
              f"{fused_auc:.4f}  {'OK' if fused_auc >= 0.95 else 'LOW'}")
    if len(loc):
        # each tamper type should be caught by at least one channel (column max high)
        print("[acceptance] per-tamper best-channel detection AUC "
              "(each should be >= ~0.95):")
        col_max = loc.max(axis=0)
        worst = float(col_max.min())
        for tt, v in col_max.items():
            print(f"   {tt:16s}: {v:.3f}  (by {loc[tt].idxmax()})")
        print(f"   -> worst-covered tamper AUC = {worst:.3f} "
              f"{'OK' if worst >= 0.95 else 'GAP'}")
        # epsilon-like: fused tamper acceptance on eval split
        fused = build_fused(_ensure_pass(df))
        ev = fused[fused.split == args.split]
        if len(ev):
            tacc = float(ev[ev.population == "tamper"]["pass"].mean())
            print(f"[acceptance] fused tamper accept-rate (provisional t) on "
                  f"split={args.split}: {tacc:.4f}  (-> epsilon_sem proxy, want ~0)")
        (outdir / "output2_localization_auc.json").write_text(
            loc.to_json(orient="index")
        )


if __name__ == "__main__":
    main()
