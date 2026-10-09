# EGFS-AVProve

EGFS-AVProve is a fail-closed audio-visual provenance verification research
prototype. It combines short audio/video provenance locators, a signed registry,
multimodal semantic commitments, hierarchical evidence-dependency reasoning,
and constrained state decoding.

This repository contains the minimum runnable implementation and four supported
workflow entry points.
Datasets, generated media, model weights, private signing keys, paper drafts,
review correspondence, and machine-specific outputs are intentionally excluded.

## Repository layout

- `rq1_eval/`: authenticated locator, evidence extraction, hierarchical graph,
  constrained decoding, and held-out evaluation code.
- `rq1_eval/external_wrappers/`: adapters for AniPortrait, EchoMimicV3,
  LivePortrait, and kNN-VC.
- `server_scripts/`: curated formal launchers with anonymized path defaults; see
  `server_scripts/README.md` for the supported entry points.
- `configs/`: public carrier policy configuration.
- `docs/`: setup, data, privacy, and reproducibility notes.
- `tools/privacy_scan.py`: release-time privacy and secret scanner.

## Setup

Create an isolated Python environment and install the common dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The carrier path requires AudioSeal and VideoSeal. Generative attacks require
their corresponding upstream repositories. See `docs/EXTERNAL_DEPENDENCIES.md`.

## Paths and data

The release does not contain datasets. Replace `/path/to/...` defaults through
the corresponding command-line options or launcher environment variables. The
most common variables are `ROOT`, `TDSC_ROOT`, `DATA_ROOT`, `OUTPUT_ROOT`,
`HF_HOME`, and `TORCH_HOME`.

Expected public datasets include FakeAVCeleb, LAV-DF, VoxCeleb2, and the
project's in-domain evaluation split. Users are responsible for obtaining each
dataset under its original license and constructing manifests with paths valid
on their own systems.

## Minimal code checks

```bash
python -m compileall -q rq1_eval tools
python tools/privacy_scan.py .
```

The final verifier uses the hierarchical evidence graph implemented in
`rq1_eval/train_m81_windowed_fs_avprove_net.py`. The public package does not
include superseded M8 verifier implementations or dedicated ablation variants.

The retained scripts expose their arguments through `--help`. For example:

```bash
python rq1_eval/train_m81_windowed_fs_avprove_net.py --help
python rq1_eval/verify_m7_authenticated_claim_locator_v2.py --help
python rq1_eval/evaluate_n1000_state_correctness.py --help
```

