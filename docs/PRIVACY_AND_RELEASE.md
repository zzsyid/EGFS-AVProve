# Privacy and release policy

The public package intentionally excludes raw audio/video, face and voice
features, dataset subject identifiers, per-sample path-bearing records, private
keys, credentials, machine logs, paper drafts, and reviewer correspondence.

Machine-specific paths are replaced with `/path/to/...` placeholders. The
privacy scanner rejects known local usernames, local absolute paths, private-key
markers, common credential assignments, and likely API tokens.

Before every public release:

1. Run `python tools/privacy_scan.py .`.
2. Inspect the archive manifest.
3. Verify that no `.env`, key, checkpoint, media, CSV, or JSONL file is present.
4. Extract the archive into an empty directory and rerun syntax and privacy
   checks there.
