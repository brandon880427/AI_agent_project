import os

import numpy as np


def main() -> int:
    from ultralytics import YOLO

    model_path = os.environ.get("CARDS_MODEL_PATH", "model/cards.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model at {model_path}")

    model = YOLO(model_path)

    img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(img, imgsz=640, conf=0.25, iou=0.5, verbose=False)

    names = getattr(model, "names", None)
    boxes = 0
    if results and getattr(results[0], "boxes", None) is not None:
        boxes = len(results[0].boxes)

    print("loaded:", model_path)
    print("num_classes:", len(names) if isinstance(names, dict) else (len(names) if names else "?"))
    if names:
        try:
            # show a few class names for sanity
            if isinstance(names, dict):
                sample = [names[i] for i in sorted(names.keys())[:8]]
            else:
                sample = list(names)[:8]
            print("sample_names:", sample)
        except Exception:
            pass
    print("boxes_on_blank:", boxes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
