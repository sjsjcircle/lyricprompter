import asyncio, json, time, sys
import websockets
async def main():
    async with websockets.connect("ws://127.0.0.1:18931/ws/prompter") as ws:
        t0 = time.perf_counter(); n = 0; last = None
        while time.perf_counter() - t0 < 3.0:
            last = json.loads(await ws.recv()); n += 1
        print(f"received {n} msgs in 3 s -> {n/3:.1f} Hz; last={last}")
    async with websockets.connect("ws://127.0.0.1:18931/ws/control") as ws:
        await ws.send(json.dumps({"op": "seek", "line": 3})); print("control:", await ws.recv())
asyncio.run(main())
