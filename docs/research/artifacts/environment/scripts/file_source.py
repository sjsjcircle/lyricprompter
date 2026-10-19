import sys, time, threading, queue
import numpy as np, soundfile as sf, soxr

TARGET_SR, HOP = 16000, 640  # 40 ms hops like the reference repo

class FileSource:
    """'Simulate live' input: reads a file block-by-block, resamples to 16 kHz, paces to wall clock.
    speed>1 plays faster (pure pacing change, pitch unaffected because we don't alter samples)."""
    def __init__(self, path, channels=None, speed=1.0, out_q=None, block_sec=0.01):
        self.path, self.channels, self.speed = path, channels, speed
        self.q = out_q or queue.Queue(maxsize=500)
        self.block_sec = block_sec
        self._stop = threading.Event()

    def start(self):
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start(); return self

    def stop(self): self._stop.set()

    def _run(self):
        with sf.SoundFile(self.path) as f:
            sr = f.samplerate
            blk = int(sr * self.block_sec)
            rs = soxr.ResampleStream(sr, TARGET_SR, 1, dtype="float32", quality="HQ")
            buf = np.zeros(0, np.float32)
            t0 = time.perf_counter(); n_in = 0
            for x in f.blocks(blocksize=blk, dtype="float32", always_2d=True):
                if self._stop.is_set(): break
                x = x[:, self.channels] if self.channels is not None else x
                mono = x.mean(axis=1)
                last = len(x) < blk
                buf = np.concatenate([buf, rs.resample_chunk(mono, last=last)])
                while len(buf) >= HOP:
                    self.q.put((n_in / sr, buf[:HOP].copy())); buf = buf[HOP:]
                n_in += len(x)
                # pace: sleep until the wall-clock time this block "would have arrived"
                deadline = t0 + (n_in / sr) / self.speed
                d = deadline - time.perf_counter()
                if d > 0: time.sleep(d)
        self.q.put(None)

if __name__ == "__main__":
    src = FileSource(sys.argv[1], speed=float(sys.argv[2]) if len(sys.argv) > 2 else 1.0).start()
    t0 = time.perf_counter(); n = 0
    while (item := src.q.get()) is not None:
        n += 1
    wall = time.perf_counter() - t0
    print(f"hops={n} audio={n*HOP/TARGET_SR:.2f}s wall={wall:.2f}s speed={src.speed}")
