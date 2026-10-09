#!/usr/bin/env python3
"""Fail a release check when private paths, credentials, or unsafe files remain."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

FORBIDDEN_SUFFIXES = {
    ".avi",
    ".ckpt",
    ".csv",
    ".flac",
    ".jsonl",
    ".key",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".p12",
    ".pem",
    ".pfx",
    ".pt",
    ".pth",
    ".safetensors",
    ".wav",
}

KNOWN_USERS = ("zz" + "syid", "de" + "ll")

PATTERNS = {
    "private data path": re.compile(r"/(?:data\d*|quick_data)/", re.I),
    "private home path": re.compile(r"/home/(?!user(?:/|\b))[^/\s'\"]+", re.I),
    "known local username": re.compile(
        r"\b(?:" + "|".join(map(re.escape, KNOWN_USERS)) + r")\b", re.I
    ),
    "e-mail address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "embedded dataset subject path": re.compile(
        r"AV-Deepfake1M/(?:train|test|val)/id\d+/[^/\s'\"]+", re.I
    ),
    "private key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "credential assignment": re.compile(
        r"(?i)(?:api[_-]?key|secret[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"][^'\"]+"
    ),
    "likely bearer token": re.compile(r"\b(?:ghp|github_pat|sk)-[A-Za-z0-9_-]{20,}\b"),
}


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() or ".git" in path.parts:
            continue
        rel = path.relative_to(root)
        if rel == Path("tools/privacy_scan.py"):
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"forbidden file type: {rel}")
            continue
        if path.name == ".env":
            findings.append(f"environment file: {rel}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"unexpected binary file: {rel}")
            continue
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{label}: {rel}:{line}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    findings = scan(Path(args.root).resolve())
    if findings:
        print("Privacy scan failed:")
        print("\n".join(f"- {item}" for item in findings))
        return 1
    print("Privacy scan passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
