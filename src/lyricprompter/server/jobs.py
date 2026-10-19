"""Background jobs: `lyricprompter prep ...` runs as a subprocess, one at a time (it loads large
models), and its `PROGRESS <stage> <percent> <message>` stdout lines become job status."""
from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..config import PROJECT_ROOT

log = logging.getLogger(__name__)

PROGRESS_RE = re.compile(r"^PROGRESS\s+(\S+)\s+(-?\d+(?:\.\d+)?)(?:\s+(.*))?$")
FINAL_STATES = ("done", "failed", "cancelled")


@dataclass
class Job:
    id: str
    argv: list[str]
    kind: str = "prep"
    song_id: str | None = None
    title: str = ""
    status: str = "queued"          # queued | running | done | failed | cancelled
    stage: str = ""
    percent: float = 0.0
    message: str = ""
    returncode: int | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    log: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    cleanup: Path | None = None     # directory removed once the job is finished
    expect: Path | None = None      # file that must exist for a zero exit to count as success
    output: Path | None = None      # package directory the job builds (see JobManager.submit)
    output_existed: bool = False    # `output` was already there at submit time

    def to_dict(self, log_lines: int = 30) -> dict:
        return {"id": self.id, "kind": self.kind, "song_id": self.song_id, "title": self.title,
                "status": self.status, "stage": self.stage, "percent": round(self.percent, 1),
                "message": self.message, "returncode": self.returncode, "error": self.error,
                "created": self.created, "started": self.started, "finished": self.finished,
                "log": list(self.log)[-log_lines:] if log_lines else []}


def parse_progress(line: str) -> tuple[str, float, str] | None:
    m = PROGRESS_RE.match(line.strip())
    if not m:
        return None
    return m.group(1), max(0.0, min(100.0, float(m.group(2)))), (m.group(3) or "").strip()


def build_prep_argv(audio: Path, lyrics_file: Path, *, title: str, song_id: str, songs_dir: Path,
                    artist: str = "", language: str = "auto", lead: bool = False, refine: str | None = None,
                    sep_model: str | None = None, style: str | None = None) -> list[str]:
    """Arguments for `lyricprompter prep` (see prep/pipeline.py `cli`).

    Values use the `--opt=value` form so free text starting with '-' (a title like "-Intro-", the
    artist "-M-") is never mistaken for an option by argparse.
    """
    opts: list[tuple[str, str]] = [("--audio", str(audio)), ("--lyrics", str(lyrics_file)), ("--title", title),
                                   ("--id", song_id), ("--songs-dir", str(songs_dir)), ("--lang", language or "auto")]
    if artist:
        opts.append(("--artist", artist))
    if refine:
        opts.append(("--refine", refine))
    if sep_model:
        opts.append(("--sep-model", sep_model))
    if style:
        opts.append(("--style", style))
    argv = ["prep", *(f"{k}={v}" for k, v in opts)]
    if lead:
        argv.append("--lead")
    return argv


class JobManager:
    """FIFO of subprocess jobs executed by one worker thread."""

    def __init__(self, command_prefix: list[str] | None = None, *, cwd: Path | None = None,
                 env: dict[str, str] | None = None, max_jobs: int = 50,
                 on_change: Callable[[Job], None] | None = None):
        self.command_prefix = command_prefix or [sys.executable, "-m", "lyricprompter.cli"]
        self.cwd = Path(cwd or PROJECT_ROOT)
        self.env = {**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})}
        self.max_jobs, self.on_change = max_jobs, on_change
        self._jobs: dict[str, Job] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._proc: subprocess.Popen[str] | None = None
        self._current: str | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._work, name="JobManager", daemon=True)
        self._thread.start()

    # -- public API -----------------------------------------------------------------------------
    def submit(self, argv: list[str], *, kind: str = "prep", song_id: str | None = None, title: str = "",
               cleanup: Path | None = None, expect: Path | None = None, output: Path | None = None) -> Job:
        """Queue a job. `output`: the package directory it builds; if the job fails or is cancelled
        and leaves that directory without a song.json, it is removed, unless it existed before (a
        partial package would otherwise hide from the library while blocking its song id)."""
        job = Job(id=uuid.uuid4().hex[:12], argv=list(argv), kind=kind, song_id=song_id, title=title,
                  cleanup=cleanup, expect=expect, output=output,
                  output_existed=output is not None and output.exists(), message="waiting for the previous job")
        with self._wake:
            if self._closed:
                raise RuntimeError("job manager is shut down")
            self._jobs[job.id] = job
            self._queue.append(job.id)
            self._prune()
            self._wake.notify()
        self._changed(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def active_song_ids(self) -> set[str]:
        with self._lock:
            return {j.song_id for j in self._jobs.values() if j.song_id and j.status in ("queued", "running")}

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in FINAL_STATES:
                return False
            if job.status == "queued":
                self._queue.remove(job_id)
                self._finish(job, "cancelled", error="cancelled before start")
                return True
            proc = self._proc if self._current == job_id else None
            job.status, job.message = "cancelled", "cancelling"
        if proc is not None:
            _terminate(proc)
        return True

    def wait(self, job_id: str, timeout: float | None = None) -> Job | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            job = self.get(job_id)
            if job is None or job.status in FINAL_STATES and job.finished is not None:
                return job
            if deadline is not None and time.monotonic() > deadline:
                return job
            time.sleep(0.05)

    def shutdown(self, timeout: float = 5.0) -> None:
        with self._wake:
            self._closed = True
            for jid in list(self._queue):
                self._finish(self._jobs[jid], "cancelled", error="server shutting down")
            self._queue.clear()
            proc = self._proc
            self._wake.notify_all()
        if proc is not None:
            _terminate(proc)
        self._thread.join(timeout)

    # -- worker -----------------------------------------------------------------------------------
    def _prune(self) -> None:
        done = sorted((j for j in self._jobs.values() if j.status in FINAL_STATES), key=lambda j: j.created)
        for j in done[:max(0, len(self._jobs) - self.max_jobs)]:
            del self._jobs[j.id]

    def _changed(self, job: Job) -> None:
        if self.on_change is not None:
            try:
                self.on_change(job)
            except Exception:
                log.exception("job on_change callback failed")

    def _finish(self, job: Job, status: str, *, error: str | None = None) -> None:
        job.status, job.finished = status, time.time()
        if error:
            job.error = error
        if status == "done":
            job.percent, job.message = 100.0, job.message or "done"
        if job.cleanup is not None:
            shutil.rmtree(job.cleanup, ignore_errors=True)
        out = job.output
        if (status != "done" and out is not None and not job.output_existed and out.is_dir()
                and not (out / "song.json").exists()):
            log.info("job %s %s: removing its partial package %s", job.id, status, out)
            shutil.rmtree(out, ignore_errors=True)

    def _work(self) -> None:
        while True:
            with self._wake:
                while not self._queue and not self._closed:
                    self._wake.wait()
                if self._closed:
                    return
                job = self._jobs[self._queue.popleft()]
                job.status, job.started, job.message = "running", time.time(), "starting"
                self._current = job.id
            self._changed(job)
            try:
                self._run(job)
            except Exception as e:
                log.exception("job %s crashed", job.id)
                with self._lock:
                    self._finish(job, "failed", error=f"{type(e).__name__}: {e}")
            finally:
                with self._lock:
                    self._current, self._proc = None, None
            self._changed(job)

    def _run(self, job: Job) -> None:
        cmd = [*self.command_prefix, *job.argv]
        log.info("job %s: %s", job.id, " ".join(cmd))
        proc = subprocess.Popen(cmd, cwd=self.cwd, env=self.env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                                encoding="utf-8", errors="replace", bufsize=1, start_new_session=True)
        with self._lock:
            self._proc = proc
            cancelled = job.status == "cancelled"
        if cancelled:
            _terminate(proc)
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                prog = parse_progress(line)
                with self._lock:
                    if prog is not None:
                        job.stage, job.percent, job.message = prog
                    elif line.strip():
                        job.log.append(line)
                self._changed(job)
            rc = proc.wait()
        except BaseException:
            # Never leave the process behind (stdout undrained, models loaded) while the worker
            # moves on: the next job must not start until this one is gone.
            _terminate(proc)
            proc.wait()
            raise
        finally:
            proc.stdout.close()
        with self._lock:
            job.returncode = rc
            if job.status == "cancelled":
                self._finish(job, "cancelled", error="cancelled")
            elif rc != 0:
                tail = next((ln for ln in reversed(job.log) if ln.strip()), "")
                self._finish(job, "failed", error=f"prep exited with code {rc}" + (f": {tail}" if tail else ""))
            elif job.expect is not None and not job.expect.exists():
                self._finish(job, "failed", error=f"finished without producing {job.expect.name}")
            else:
                self._finish(job, "done")


def _terminate(proc: subprocess.Popen[str], grace: float = 5.0) -> None:
    """SIGTERM the job's process group, then SIGKILL after `grace` seconds."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()

    def reap() -> None:
        try:
            proc.wait(grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
    threading.Thread(target=reap, daemon=True).start()
