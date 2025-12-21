#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train a YOLO playing-cards detector and export to `model/cards.pt`.

Expected dataset format: Ultralytics/YOLOv5-style with a `data.yaml`.

Example:
  python scripts/train_cards.py --data datasets/cards/data.yaml --epochs 50 --imgsz 640

Notes:
- Class names should encode rank/suit (recommended): AS, 10H, KD, 2C
  (also supported: A_spades, 10_hearts, king_of_diamonds, etc.)
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to data.yaml")
    ap.add_argument("--base", default="yolov8n.pt", help="Base model (e.g. yolov8n.pt)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--device", default=os.getenv("AIGLASS_DEVICE", ""), help="cuda:0 / cpu / ''")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="runs/cards")
    ap.add_argument("--name", default="train")
    ap.add_argument("--out", default=os.getenv("CARDS_MODEL_PATH", str(Path("model") / "cards.pt")))
    args = ap.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"data.yaml not found: {data}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.base)

    train_kwargs = dict(
        data=str(data),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        workers=int(args.workers),
        project=str(args.project),
        name=str(args.name),
        verbose=True,
    )
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    # Resolve best.pt
    best_pt = None
    try:
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
    except Exception:
        best_pt = None

    if not best_pt or not best_pt.exists():
        # Fallback to common ultralytics path
        cand = Path(args.project) / args.name / "weights" / "best.pt"
        if cand.exists():
            best_pt = cand

    if not best_pt or not best_pt.exists():
        raise SystemExit("Training finished but best.pt not found; check the run folder.")

    shutil.copy2(best_pt, out_path)
    print(f"Exported: {out_path} (from {best_pt})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
