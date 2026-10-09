# Reproducibility

## Required retained inputs

Reproduction requires dataset manifests, split metadata, external model
checkpoint identifiers, method configurations, and random seeds. Large media
products and decoded caches can be regenerated from those inputs.

## Recommended execution order

1. Run `server_scripts/run_m4_windowed_registry_fusion.sh` to build and sign the
   semantic/window registry.
2. Run `server_scripts/run_media_attack_pipeline.sh` to build the authenticated
   locator registry, embed Audio32/Video24 locators, and realize attacks.
3. Run `server_scripts/run_media_verification_pipeline.sh` to resolve claims and
   recompute semantic, synchronization, and window evidence from attacked media.
4. Run `server_scripts/run_m81_windowed_fs_avprove_net.sh` to train the released
   hierarchical evidence graph, or load a frozen checkpoint for evaluation.
5. Use `eval_m81_n1000_evidence.py`, `evaluate_n1000_state_correctness.py`, and
   `evaluate_n1000_temporal_localization.py` for held-out evaluation.

## Frozen evaluation policy

Evaluation must keep dataset splits, carrier decoding policy, semantic
extractors, registry verification, thresholds, and state-routing policy frozen.
Calibration samples must remain disjoint from held-out evaluation samples.

## Released verifier architecture

The released M8.1 verifier is the final hierarchical evidence graph. It first
contextualizes evidence tokens, applies typed sparse dependency routing, pools
one slot per evidence type, and performs a second dependency-constrained graph
step over those slots. Predicate-factorized positive-support decoding and hard
cryptographic/semantic constraints produce the final provenance state. Earlier
flat or non-hierarchical verifier implementations are intentionally excluded.

The carrier workflow uses only the authenticated locator-v2 schema. The 32-bit
audio locator, 24-bit video locator, version ordinal, family digest, and
collision-resolution salt are covered by the signed registry record. Claim
resolution indexes decoded locators and does not use the expected sample hash.

## Excluded assets

This release excludes datasets, generated attacks, model checkpoints, signed
registries containing experiment-specific identifiers, and private signing
keys. Public release users must regenerate these assets in accordance with the
original dataset and model licenses.
