import argparse
import asyncio

import websockets


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8081/ws_audio")
    ap.add_argument("--text", default="撲克牌辨識")
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()

    async with websockets.connect(args.url, ping_interval=20, ping_timeout=20, max_size=2 * 1024 * 1024) as ws:
        payload = f"PROMPT:{args.text}"
        await ws.send(payload)
        print("sent:", payload)

        # Best-effort: server usually replies OK:PROMPT_ACCEPTED
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
                print("recv:", msg)
                if isinstance(msg, str) and (msg.startswith("OK:") or msg.startswith("ERR:") or msg.startswith("RESTART")):
                    break
        except asyncio.TimeoutError:
            print("recv: (timeout)")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
