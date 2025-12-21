#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Download and extract a YOLO-format playing-cards dataset.

This script is intentionally generic because public card datasets vary a lot.

Usage:
  python scripts/download_cards_dataset.py --url <ZIP_URL> --to datasets/cards

Notes:
- The ZIP should contain a `data.yaml` compatible with Ultralytics.
- For Roboflow, you usually need an API key; paste the signed download URL.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path


def _download(url: str, dst: Path) -> None:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Dataset ZIP URL")
    ap.add_argument("--to", default="datasets/cards", help="Destination directory")
    args = ap.parse_args()

    out_dir = Path(args.to)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / "dataset.zip"
    print(f"Downloading to: {zip_path}")
    _download(args.url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

    # Try to locate a data.yaml
    data_yaml = next(out_dir.rglob("data.yaml"), None)
    if not data_yaml:
        print("Warning: data.yaml not found under destination. You may need to point --data manually.")
    else:
        print(f"Found data.yaml: {data_yaml}")

    try:
        zip_path.unlink()
    except Exception:
        pass

    print("Done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
