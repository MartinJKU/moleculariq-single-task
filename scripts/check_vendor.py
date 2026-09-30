#!/usr/bin/env python3
"""Verify the vendored official code still matches upstream.

    python scripts/check_vendor.py            # against the local third_party clone
    python scripts/check_vendor.py --fetch    # against GitHub

The reward path copies moleculariq-evals extractor verbatim so that training
and the benchmark read a models answer the same way. If upstream changes that
file and we do not, the two quietly diverge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from miqgrpo.paths import REPO_ROOT

VENDOR_DIR = REPO_ROOT / "src" / "miqgrpo" / "vendor"
RAW = "https://raw.githubusercontent.com/ml-jku/moleculariq-eval/{commit}/{path}"


def _upstream_bytes(entry: dict, fetch: bool) -> bytes | None:
    if fetch:
        import urllib.request

        url = RAW.format(commit=entry["commit"], path=entry["path"])
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    local = REPO_ROOT / "third_party" / "moleculariq-eval" / entry["path"]
    return local.read_bytes() if local.exists() else None


def main() -> None:
    """Compare the vendored extractor and system prompt with upstream; exit 1 on any difference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="download from GitHub")
    args = parser.parse_args()

    manifest = json.loads((VENDOR_DIR / "VENDOR.json").read_text())
    failures = []

    entry = manifest["moleculariq_extractors.py"]
    upstream = _upstream_bytes(entry, args.fetch)
    if upstream is None:
        print(
            "  ! upstream copy not available; run with --fetch or after 00_setup_env.sh"
        )
        sys.exit(2)

    digest = hashlib.sha256(upstream).hexdigest()
    if digest != entry["upstream_sha256"]:
        failures.append(
            f"extractors.py: upstream at {entry['commit'][:8]} hashes {digest[:12]}, "
            f"manifest says {entry['upstream_sha256'][:12]}"
        )

    vendored = (VENDOR_DIR / "moleculariq_extractors.py").read_text()
    body = vendored.split('"""', 2)[-1].lstrip("\n")
    if body.strip() != upstream.decode().strip():
        failures.append("extractors.py: vendored body differs from upstream")

    from miqgrpo.vendor import SYSTEM_PROMPT

    prompt_digest = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    if prompt_digest != manifest["moleculariq_system_prompt.py"]["sha256"]:
        failures.append("system prompt: hash differs from VENDOR.json")

    if failures:
        print("vendor check FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("vendored files match upstream")


if __name__ == "__main__":
    main()
