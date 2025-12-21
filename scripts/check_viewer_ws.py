import argparse
import asyncio
from pathlib import Path

import websockets


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8081/ws/viewer")
    ap.add_argument("--out", default="recordings/viewer_frame.jpg")
    ap.add_argument("--timeout", type=float, default=8.0)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async with websockets.connect(args.url, ping_interval=20, ping_timeout=20, max_size=20 * 1024 * 1024) as ws:
        msg = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
        if not isinstance(msg, (bytes, bytearray)):
            raise RuntimeError(f"Expected binary JPEG bytes, got: {type(msg)}")

        out_path.write_bytes(bytes(msg))
        print(f"saved: {out_path} ({len(msg)} bytes)")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
