#!/usr/bin/env python3
"""Output (3): empirical adaptive false-accept  hat{epsilon}_sem^adapt(tau).

This estimates the ONLY experimentally-determined term in the Main Theorem:

    hat{epsilon}_sem = #{ X* in P_tamper(eval) : (for all channels) d_ham_. <= t_.
                          AND CrossModalOK } / |P_tamper(eval)|

computed strictly on the EVAL (held-out generator) split. Per-channel t comes from
calibrate_tau.py (calibrated_tau.json); fall back to the table's provisional t.

Levels (RQ1_eval_spec.md sec.7):
  Level A  non-adaptive : judge each tamper sample directly.
  Level B  helper-data-aware selection : per target, the adversary submits N
           candidates and keeps the best one (argmin over candidates of the
           worst-channel margin); we judge that best candidate. Operationalized
           here using the reps per (clip_id, cond, generator) as the candidate
           pool -> ε_B >= ε_A by construction (adversary picks best of N).
  Level C  surrogate-gradient : needs differentiable f_. (Stage-1) -> not here.

CrossModalOK: not computable feature-only (Stage-0). We report the four-channel
AND only; since CrossModalOK is an ADDITIONAL required condition, AND-only epsilon
is a conservative UPPER BOUND on the true epsilon (the verifier can only reject
more, never fewer). Flagged in the output.

CIs: Wilson (closed form) + Clopper-Pearson (exact, via a pure-python regularized
incomplete-beta inverse, since scipy.stats is broken in the target env). 95% by
default. When k=0 events, the CP upper bound (~ the 'rule of three' 3/n) is the
reportable security number.

NOTE: on mock data these epsilon values are circular artifacts (STATUS.md); this
validates the estimator + CI machinery, not the method.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from metrics_sep import CHANNELS, _ensure_pass


# --------------------------------------------------------------------------- #
# Confidence intervals (pure python; scipy.stats unavailable)
# --------------------------------------------------------------------------- #
def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for mi in range(1, MAXIT + 1):
        m2 = 2 * mi
        aa = mi * (b - mi) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + mi) * (qab + mi) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _betai_inv(a: float, b: float, target: float) -> float:
    """Invert I_x(a,b)=target for x via bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if betai(a, b, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else _betai_inv(k, n - k + 1, alpha / 2)
    hi = 1.0 if k == n else _betai_inv(k + 1, n - k, 1 - alpha / 2)
    return (lo, hi)


# --------------------------------------------------------------------------- #
# Acceptance (four-channel AND) per sample
# --------------------------------------------------------------------------- #
def _t_norm_map(df: pd.DataFrame, tau_json: Optional[Path]) -> Dict[str, float]:
    if tau_json and Path(tau_json).exists():
        d = json.loads(Path(tau_json).read_text())
        return {ch: d["per_channel"][ch]["t_norm"] for ch in CHANNELS}
    # fallback: provisional t / n from table
    n = int(df["n"].iloc[0])
    t = int(df["t"].iloc[0])
    print(f"[epsilon][warn] no tau json; using provisional t={t}/n={n} for all channels")
    return {ch: t / n for ch in CHANNELS}


def _per_sample(df: pd.DataFrame, tnorm: Dict[str, float]) -> pd.DataFrame:
    """One row per sample_id with per-channel d_ham, fused accept, worst margin."""
    df = _ensure_pass(df)
    keys = ["sample_id", "clip_id", "source_id", "population", "cond",
            "generator", "split"]
    rows = []
    for key, g in df.groupby("sample_id", dropna=False):
        if not set(CHANNELS).issubset(set(g["channel"])):
            continue
        rec = {k: g[k].iloc[0] for k in keys if k in g.columns}
        dh = {ch: float(g[g.channel == ch]["d_ham"].iloc[0]) for ch in CHANNELS}
        # worst-channel excess over its radius (<=0 means accept)
        excess = max(dh[ch] - tnorm[ch] for ch in CHANNELS)
        rec["fused_accept"] = bool(excess <= 0)   # AND of (d_ham_ch <= t_ch)
        rec["worst_excess"] = excess
        rows.append(rec)
    return pd.DataFrame(rows)


def _ci_block(k: int, n: int) -> Dict:
    w = wilson_ci(k, n)
    cp = clopper_pearson(k, n)
    return {
        "events": k, "n": n,
        "point": (k / n if n else float("nan")),
        "wilson95": [w[0], w[1]],
        "clopper_pearson95": [cp[0], cp[1]],
    }


def estimate(df: pd.DataFrame, tau_json: Optional[Path] = None,
             split: str = "eval") -> Dict:
    df = _ensure_pass(df)
    tnorm = _t_norm_map(df, tau_json)
    samp = _per_sample(df, tnorm)
    sub = samp[samp["split"] == split]
    tamper = sub[sub["population"] == "tamper"]

    # Level A: every eval tamper sample
    nA = len(tamper)
    kA = int(tamper["fused_accept"].sum())

    # Level B: per target (clip, cond, generator), adversary keeps best candidate.
    # Real candidate pools can have unequal sizes (for example, several av_shift
    # severities but only one audio_swap). In that case, compare Level B with a
    # pool-matched non-adaptive reference rather than sample-weighted Level A.
    nB = kB = 0
    pool_sizes: List[int] = []
    pool_accept_rates: List[float] = []
    if len(tamper):
        for _, grp in tamper.groupby(["clip_id", "cond", "generator"], dropna=False):
            nB += 1
            pool_sizes.append(int(len(grp)))
            pool_accept_rates.append(float(grp["fused_accept"].mean()))
            best = grp.loc[grp["worst_excess"].idxmin()]   # most-likely-to-pass
            kB += int(bool(best["fused_accept"]))
    poolmatched_A = float(np.mean(pool_accept_rates)) if pool_accept_rates else float("nan")
    levelB_rate = (kB / nB) if nB else float("nan")
    equal_pool_sizes = (len(set(pool_sizes)) <= 1) if pool_sizes else None

    # per-channel marginal accept on eval tamper (diagnostic)
    per_channel = {}
    sub_rows = _ensure_pass(df)
    et = sub_rows[(sub_rows.split == split) & (sub_rows.population == "tamper")]
    for ch in CHANNELS:
        cc = et[et.channel == ch]
        if len(cc):
            acc = int((cc["d_ham"] <= tnorm[ch]).sum())
            per_channel[ch] = _ci_block(acc, len(cc))

    # same-source acceptance, reported BY LEVEL (spec sec.8 ideal form:
    #   L1/L2-same  <  t  <  L3-same  <~  tamper
    # so L1/L2 fused-accept SHOULD be high (low FRR), while L3-same content has
    # been laundered through a generator and is EXPECTED to fall outside the ball
    # (that is the boundary the verifier defends -- L3-same is NOT a free pass).
    # Pooling L1+L2+L3 hides this and looks like a spuriously high FRR.
    same = sub[sub["population"] == "same"]
    same_overall = _ci_block(int(same["fused_accept"].sum()), len(same)) if len(same) else None
    same_by_level = {}
    if len(same):
        for cond, g in same.groupby("cond", dropna=False):
            same_by_level[str(cond)] = _ci_block(int(g["fused_accept"].sum()), len(g))
    # L1 U L2 = benign-recompression band; this is the true FRR proxy
    benign_same = same[same["cond"].isin(["L1", "L2"])]
    same_benign_block = (_ci_block(int(benign_same["fused_accept"].sum()), len(benign_same))
                         if len(benign_same) else None)

    return {
        "split": split,
        "t_norm_used": tnorm,
        "tau_source": str(tau_json) if tau_json else "provisional(table)",
        "crossmodal_note": "AND-only; true epsilon <= this (CrossModalOK only rejects more)",
        "level_A_nonadaptive": _ci_block(kA, nA),
        "level_A_poolmatched_reference": {
            "point": poolmatched_A,
            "n_pools": nB,
            "note": "Mean within-pool non-adaptive acceptance. Compare this with "
                    "Level B when candidate-pool sizes are unequal.",
        },
        "level_B_helperaware": _ci_block(kB, nB),
        "level_B_candidate_pools": {
            "n_pools": nB,
            "min_size": min(pool_sizes) if pool_sizes else None,
            "max_size": max(pool_sizes) if pool_sizes else None,
            "mean_size": float(np.mean(pool_sizes)) if pool_sizes else None,
            "equal_sizes": equal_pool_sizes,
        },
        "monotonic_sample_A_le_B": ((kA / nA <= levelB_rate) if (nA and nB and equal_pool_sizes)
                                    else None),
        "monotonic_poolmatched_A_le_B": ((poolmatched_A <= levelB_rate + 1e-12)
                                        if nB else None),
        "per_channel_marginal_far": per_channel,
        "same_source_fused_accept_overall": same_overall,
        "same_source_fused_accept_by_level": same_by_level,
        "frr_proxy_L1L2_same": same_benign_block,
        "frr_note": "L1/L2-same accept should be HIGH (low FRR). L3-same is the "
                    "most-stressed same-source band (laundered own content): expect it "
                    "to be the LOWEST same-source accept (L1/L2 >= L3), sitting nearest "
                    "the tamper boundary -- ordering matters, not a fixed target.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", required=True)
    ap.add_argument("--tau-json", default="artifacts/calibrated_tau.json")
    ap.add_argument("--outdir", default="artifacts")
    ap.add_argument("--split", default="eval")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.table) if args.table.endswith(".parquet") else pd.read_csv(args.table)
    tau = Path(args.tau_json) if Path(args.tau_json).exists() else None

    res = estimate(df, tau_json=tau, split=args.split)
    out = outdir / "epsilon_sem.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[epsilon] wrote {out}")
    backends = sorted(set(df["backend"])) if "backend" in df.columns else []
    if backends == ["mock"]:
        print("[epsilon] WARNING: backend=mock; epsilon is a circular artifact (STATUS.md)")
    else:
        print(f"[epsilon] backend={backends or ['unknown']}; interpret epsilon with the "
              "reported CI and held-out attack coverage")

    A, B = res["level_A_nonadaptive"], res["level_B_helperaware"]
    print(f"  CrossModalOK note: {res['crossmodal_note']}")
    print(f"  Level A (non-adaptive)  eps_hat = {A['point']:.4g}  "
          f"({A['events']}/{A['n']})  CP95=[{A['clopper_pearson95'][0]:.4g}, "
          f"{A['clopper_pearson95'][1]:.4g}]")
    print(f"  Level B (helper-aware)  eps_hat = {B['point']:.4g}  "
          f"({B['events']}/{B['n']})  CP95=[{B['clopper_pearson95'][0]:.4g}, "
          f"{B['clopper_pearson95'][1]:.4g}]")
    poolref = res["level_A_poolmatched_reference"]
    pools = res["level_B_candidate_pools"]
    print(f"  Level A pool-matched reference = {poolref['point']:.4g}  "
          f"(pools={poolref['n_pools']}, size={pools['min_size']}..{pools['max_size']})")
    print(f"  monotonic pool-matched-A<=B: {res['monotonic_poolmatched_A_le_B']}")
    if res["frr_proxy_L1L2_same"]:
        s = res["frr_proxy_L1L2_same"]
        print(f"  [sanity] FRR proxy: L1/L2-same fused accept = {s['point']:.3f} "
              f"({s['events']}/{s['n']})  (should be HIGH)")
    if res["same_source_fused_accept_by_level"]:
        print(f"  [sanity] same-source fused accept by level "
              f"(expect ordering L1/L2 >= L3; L3 = most-stressed band):")
        for cond, s in sorted(res["same_source_fused_accept_by_level"].items()):
            print(f"             {cond:6s} = {s['point']:.3f}  ({s['events']}/{s['n']})")


def _selftest():
    print("== CI sanity ==")
    assert clopper_pearson(0, 100)[0] == 0.0
    lo, hi = clopper_pearson(0, 100)
    print(f"  CP(0/100) = [{lo:.4f}, {hi:.4f}]  (rule-of-three ~0.03)")
    assert 0.02 < hi < 0.04
    lo, hi = clopper_pearson(5, 100)
    w = wilson_ci(5, 100)
    print(f"  CP(5/100) = [{lo:.4f}, {hi:.4f}]  Wilson=[{w[0]:.4f}, {w[1]:.4f}]")
    assert lo < 0.05 < hi and w[0] < 0.05 < w[1]
    # betai monotone
    assert betai(2, 5, 0.1) < betai(2, 5, 0.5) < betai(2, 5, 0.9)
    print("  betai monotone OK\n  epsilon_sem CI self-test PASSED")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _selftest()
    else:
        main()
