# Execution Entry Points

Run these four supported workflows from the repository root. Override every
`/path/to/...` placeholder with environment variables before execution.

| Script | Purpose |
| --- | --- |
| `run_m4_windowed_registry_fusion.sh` | Build and sign windowed semantic commitments. |
| `run_media_attack_pipeline.sh` | Sign and embed authenticated locators, execute attacks, and collect outputs. |
| `run_media_verification_pipeline.sh` | Resolve authenticated claims, recompute evidence, and verify attacked media. |
| `run_m81_windowed_fs_avprove_net.sh` | Train the final hierarchical evidence-graph verifier. |

`run_media_attack_pipeline.sh` is controlled by `M6_TASKS`, `GENERATORS`,
`LIMIT`, `PER_GENERATOR_LIMIT`, `RUN_ATTACKS`, and `DEVICE`. A zero `LIMIT`
keeps all eligible rows. Pass the same `OUT` directory to
`run_media_verification_pipeline.sh`. External datasets use the same generic
manifest path; no dataset-specific launcher is required.
