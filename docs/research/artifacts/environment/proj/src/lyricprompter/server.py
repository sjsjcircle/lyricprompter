"""Minimal FastAPI app: tracker thread -> asyncio hub -> WebSocket push at ~25 Hz (latest-value semantics)."""
import asyncio, json, threading, time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

PUSH_HZ = 25


class Hub:
    def __init__(self) -> None:
        self.state: dict = {"seq": 0}
        self.version = 0
        self.changed = asyncio.Event()
        self.loop: asyncio.AbstractEventLoop | None = None

    # called from ANY thread (audio/tracker thread)
    def publish_threadsafe(self, **state) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._set, state)

    def _set(self, state: dict) -> None:
        self.version += 1
        self.state = {**state, "seq": self.version, "t_server": time.time()}
        self.changed.set()
        self.changed = asyncio.Event()  # fresh event for the next update


hub = Hub()


def fake_tracker(stop: threading.Event) -> None:
    """Stands in for the OLTW tracker: publishes reference position every 40 ms (hop 640 @ 16 kHz)."""
    t0 = time.perf_counter()
    while not stop.is_set():
        ref_t = time.perf_counter() - t0
        hub.publish_threadsafe(ref_time=round(ref_t, 3), rate=1.0, line=int(ref_t // 4), syl=int(ref_t * 3) % 12, conf=0.9)
        stop.wait(0.04)


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.loop = asyncio.get_running_loop()
    stop = threading.Event()
    th = threading.Thread(target=fake_tracker, args=(stop,), daemon=True)
    th.start()
    yield
    stop.set()
    th.join(timeout=1)


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/prompter")
async def ws_prompter(ws: WebSocket):
    """Event-driven push: send as soon as the tracker publishes, capped at PUSH_HZ, latest-wins."""
    await ws.accept()
    min_dt = 1 / PUSH_HZ
    last_sent, t_last = -1, 0.0
    try:
        while True:
            if hub.version == last_sent:
                try:
                    await asyncio.wait_for(hub.changed.wait(), timeout=1.0)
                except TimeoutError:
                    await ws.send_text('{"type":"ping"}')  # keepalive; also detects dead sockets
                    continue
            wait = min_dt - (time.perf_counter() - t_last)
            if wait > 0:
                await asyncio.sleep(wait)
            last_sent, t_last = hub.version, time.perf_counter()
            await ws.send_text(json.dumps(hub.state, separators=(",", ":")))
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.websocket("/ws/control")
async def ws_control(ws: WebSocket):
    """Operator console -> server commands (mode switch, seek, start/stop)."""
    await ws.accept()
    try:
        while True:
            cmd = await ws.receive_json()
            await ws.send_json({"ack": cmd.get("op")})
    except WebSocketDisconnect:
        pass


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
