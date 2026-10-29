# lyricprompter_oscsender — implementation spec (internal contract)

A performance version of lyricprompter: instead of showing lyrics, it **sends OSC messages at timestamps
written on a reference recording's time axis**, while following a live performance with lyricprompter's
tracker (same algorithm, both styles classic / pop, both modes mix / vocal).

Everything lives in `lyricprompter_oscsender/` (its own uv project, Python 3.12, no torch). Run commands
from that folder: `uv run pytest -q`, `uv run ruff check`, `uv run oscsender ...`.

Code style: match the parent project (`../src/lyricprompter`): `from __future__ import annotations`,
dataclasses, module docstrings that explain behaviour, comments only where they add information,
line length 110, English in code and UI. Tests in `tests/test_<module>.py`; anything needing real
time > ~20 s or audio devices is `@pytest.mark.slow` or uses fakes.

## 0. Package layout

```
lyricprompter_oscsender/
  pyproject.toml  uv.lock  .python-version  .gitignore  README.md (Korean, user guide)
  docs/SPEC.md (this file)  docs/DESIGN.md (Korean, written last)
  projects/                      user data (gitignored): one folder per project + .settings.json
  src/lyricprompter_oscsender/
    __init__.py config.py cli.py songfile.py           [done]
    audio_io.py                                         [copied from lyricprompter, unchanged]
    features/extractor.py  tracking/{oltw,relocate,tracker}.py  live/audio_input.py
                                                        [copied unchanged; tracker imports STYLES from config]
    osc.py                                              [done: codec + OscSender + Target]
    cues.py        cue file format: parse, format, edit (W1-cues)
    settings.py    persisted console settings (W1-cues)
    project.py     project store, tuning, waveform peaks, CLI add/list (W1-project)
    synth.py       synthetic live takes (copied from lyricprompter eval) + synthetic music for tests (W1-project)
    scheduler.py   CueScheduler (pure) + CueRunner (thread) (W1-scheduler)
    latency.py     echo RTT, echo responder, audio loopback, CLI (W1-latency)
    live/session.py  OscSession (W2-session) ; simulate / eval CLIs in live/simulate.py (W2-session)
    server/app.py  server/static/{console.html,console.css,console.js} (W2-server, W2-ui)
  tests/  (tracker/feature/audio_input tests copied from lyricprompter already pass: keep them green)
```

`cli.py` COMMANDS (each module exposes `cli(argv) -> int`): `serve` (server.app), `add` / `list`
(project), `simulate` / `eval` (live.simulate), `echo` / `latency` (latency), `devices`
(live.audio_input, already exists there), `check` (cues: validate a cue file and print the cues).
Whoever owns a module adds its command(s) to `cli.COMMANDS` (one line each; keep alphabetical-ish order).

## 1. Time axis and tracker facts you must rely on

* All times are seconds on the **reference axis** (the reference recording, `audio/mix.wav`).
* `LiveTracker.state` → `TrackerState(status, ref_time, tempo, confidence, live_time, frame, active,
  transpose, key_auto, key_suggest)`. `status` ∈ waiting | tracking | coasting | hold | stopped.
  `ref_time` is the published position **including** `tempo * (latency_s + lag_comp_s)`; `tempo` is the
  *rate* (ref s per live s) at which the published position currently advances: tempo estimate while
  tracking, coasting rate while coasting, **0 when the position does not move** (hold, waiting at a
  vocal-mode coast limit). The published position is monotonic except explicit jump / nudge, vocal relock
  settling (backward over the positions it coasted through: up to `relock_back_s`, not before where the voice
  stopped minus `relock_stop_back_s`), verified re-localisations (any size, either direction), key corrections.
* Frames are 40 ms (25 fps). Tracker updates arrive on the audio worker thread after each audio block that
  completes ≥ 1 frame.
* `tracker.onsets` (numpy int frame indices) are the vocal-mode phrase onsets (without lyrics: rising
  edges of the reference vocal activity). `tracker._free_run` (property) is True in pop style once locked
  (no coast limit). `tracker.set_latency(s)`, `jump(t)`, `nudge(dt)`, `hold(on)`, `set_transpose(k)`,
  `set_auto_key(on)`, `set_style(s)`, `start(t)`, `stop()`, `push(frames)`.
* `features.extractor.FeatureConfig.for_mode(mode, tuning)`, `FeatureExtractor(cfg).push(y16k)`,
  `load_ref_features(project_dir, stem, cfg)` (cached in `<project>/cache/`), `tracking.tracker.
  TrackerParams.for_mode(mode, style)`, `ref_activity(rms_db, fps, rel_db=..., min_gap_s=..., min_on_s=...)`.
* Live sources: `live.audio_input.DeviceInput(device, channels(1-based), samplerate)` and
  `FileInput(path, realtime, speed, monitor, start_s)`; both `.start(on_audio)` / `.stop()` / `.stats()` /
  `.level_db` / `.finished` / `.wait(timeout)`; deliver 16 kHz mono float32 blocks on their worker thread.
* Reference implementation to adapt: `../src/lyricprompter/live/session.py` (LiveSession, json_safe,
  reference_stem, file_signature), `../src/lyricprompter/server/app.py` (Hub, WSClient, OriginGuard,
  broadcaster, session endpoints, line edit + undo), `../src/lyricprompter/server/static/console.*` and
  `common.js` (RefClock, connectWS, fmtTime), `../src/lyricprompter/eval/track_eval.py` (synthetic lives).

## 2. Cue file format (`cues.py`) — the user-facing contract

UTF-8 text (a BOM is ignored), one cue per line; `\n`, `\r\n` or a lone `\r` (old Mac / Excel exports,
stray CRs) end a line, and are preserved when the file is edited. Other characters that some editors show
as line breaks (`\v`, `\f`, `\x1c`-`\x1e`, `\x85`, U+2028, U+2029) are read as spaces, with a warning on
that line (not inside a quoted string).

```
# lines starting with # are comments; blank lines are ignored
@target lights 192.168.0.50:8000
@target video  127.0.0.1:7000  latency=80ms

0:00.000                 /show/start
0:12.480     lights      /cue/go 12                # verse 1 wash
1:05.2       video       /clip/play "intro.mp4" 0.5
75.300                   /mixer/fader f:0.75 i:3 true
[01:20.00]               /scene "two words" nil
```

Cue line = `<time> [<target>] <address> [<arg> ...] [# note]`, tokens separated by spaces / tabs.

* **time**: `S[.fff]` (seconds, e.g. `75.3`), `M:SS[.fff]` (`1:05.2`, `01:05.200`), `H:MM:SS[.fff]`.
  Optional surrounding `[ ]` (LRC style). `,` is accepted as the decimal separator (`1:05,2`). Seconds /
  minutes fields after the first must be < 60. Negative times are an error. 4-part timecodes
  (`HH:MM:SS:FF`) are an error ("frames are not supported: write HH:MM:SS.fff"). A time later than
  `MAX_TIME_S` = 100:00:00.000 is an error (parse and edits), so a cue time is always finite.
  `parse_time(token) -> float`; `time_style(token) -> "s" | "ms" | "hms"` (and whether bracketed);
  `format_time(t, style="ms", brackets=False) -> str` always with 3 decimals: s → `75.300`,
  ms → `1:05.200` (minutes unpadded unless the original was zero-padded: keep a `pad` flag),
  hms → `1:01:05.200`. Times are rounded to the millisecond when written.
* **target** (optional): a token that does not start with `/`, matching `[A-Za-z_][A-Za-z0-9_.-]*`.
  Absent → the default target (console setting).
* **address**: first token starting with `/`; validated with `osc.validate_address`.
* **args** (each token → one OSC argument):
  * quoted `"..."` or `'...'` → string `s` (backslash escapes the next character inside quotes);
    `""` is an empty string.
  * `true` / `false` → `T` / `F`; `nil` / `null` → `N` (case-insensitive bare words).
  * integer `[+-]?\d+` → `i` (int32), or `h` if it does not fit int32 (must fit int64, else error).
  * float (`1.5`, `-.5`, `1e3`, `2.`) → `f` (float32); error if beyond float32 range ("write d:...").
  * explicit type prefix `i:`, `h:`, `f:`, `d:`, `s:` (e.g. `f:1`, `d:0.1`, `s:123`, `s:"a b"`); a bad
    value for the type is an error. `s:` takes the rest of the token (or a quoted string).
  * any other bare word → string `s`.
* **note**: an unquoted `#` that starts a token (preceded by whitespace) begins a comment; its text (after
  `#`, stripped) is the cue's `note`. A `#` inside a quoted string is literal. A `#` inside a bare
  argument (`12# verse`) is part of that string argument, with a warning ("write a space before # to start
  a note, or quote the string"); quoted or `s:` arguments get no warning.
* **directives**: `@target <name> <host>:<port> [latency=<n>[ms]]` declares a target (IPv6 as
  `[::1]:9000`; latency in ms, `ms` suffix optional, 0..5000). Unknown `@...` directive → warning, ignored.
* Errors never abort parsing: the bad line is skipped and reported as a `CueIssue(line, message,
  "error")`. Warnings (`"warning"`): exact duplicate cue (same time, target, address, args) on another
  line; a target declared twice (the first wins); a bare argument containing `#`; a line-break-like
  character (see above); a cue line whose time is earlier than a previous cue
  line's time is fine (cues are sorted) — no warning.
* `CueSheet.cues` sorted by `(time, id)`; `id` = 1-based line number; `message` = the source text from
  the address through the last argument (verbatim, trimmed), `note` = comment text.
* Server-side extra warnings (not in `parse_cues`): unknown target names (not in settings and not
  declared), `@target` lines that differ from the OSC target of that name (`settings.declared_conflicts`:
  the settings win), cues after the reference duration.

API (pure functions; editing functions take and return the whole text, keep every other line byte for
byte, keep the line ending style, and raise `CueEditError(ValueError)` with a user-readable message):

```python
parse_cues(text: str) -> CueSheet
sheet_rev(text: str) -> str                         # sha1(text.encode()).hexdigest()[:12]
parse_time(token) -> float ; format_time(t, style="ms", *, pad=False, brackets=False) -> str
parse_message(text: str) -> tuple[str | None, str, tuple[OscArg, ...], str]   # (target, address, args, note)
format_arg(tag, value) -> str ; format_args(args) -> str     # canonical cue syntax, round-trips through
                                                             # parse_message (strings quoted when needed,
                                                             # floats shortest repr with a '.', h/d prefixed)
set_cue_time(text, cue_id, t) -> str      # rewrite only the time token, same style (s / ms / hms, pad, [ ])
shift_cue(text, cue_id, dt) -> str        # error if the result < 0
set_cue_message(text, cue_id, message) -> str   # message = "[target] /address args [# note]"; validates;
                                                # replaces target..note (keeps the time token + spacing)
add_cue(text, t, message) -> tuple[str, int]    # inserts "<time> <message>" after the last cue line whose
                                                # time <= t (before the first cue line if none; the end of an
                                                # empty file); time style of the nearest cue line (default ms);
                                                # returns (text, new line number)
delete_cue(text, cue_id) -> str
```
`cue_id` that is not a valid cue line → `CueEditError`. Inserting / deleting a line shifts later ids:
callers re-parse.

`settings.py` (same owner): persisted console settings, `projects/.settings.json`, atomic write.

```python
SKIP_POLICIES = ("skip", "latest", "all")
@dataclass
class Settings:
    targets: list[osc.Target]            # default [Target("main", "127.0.0.1", 9000)]
    default_target: str = "main"
    armed: bool = True                   # OSC output on (off = dry run: cues are logged, not sent)
    skip_policy: str = "latest"          # cues jumped over: skip | latest (per target+address) | all
    catchup_s: float = 1.0               # forward moves up to this send every crossed cue (late)
    input_latency_ms: float = 0.0        # tracker latency compensation (audio input + processing)
load_settings(path, warnings=None) -> Settings
                                         # never raises: missing file -> defaults; unreadable file (also bad
                                         # JSON) -> defaults; each invalid field -> its default; a BOM is
                                         # accepted. An invalid target entry keeps its name and valid fields
                                         # but is disabled; a saved default_target that is not a target is
                                         # added as a disabled target (cues are never sent to another device
                                         # than the file named). Each problem is logged and appended to
                                         # `warnings` (user-readable); after a lossy load the file is copied to
                                         # .settings.json.bad-<hash> so the next save cannot lose it
save_settings(path, s) -> bool           # temp file + fsync + rename; False (logged) when it fails
validate_settings(s) -> Settings         # raises ValueError: unique non-empty target names matching the
                                         # target-name rule, host non-empty, 1 <= port <= 65535,
                                         # 0 <= latency_ms <= 5000, policy in SKIP_POLICIES,
                                         # 0.1 <= catchup_s <= 30, -500 <= input_latency_ms <= 2000,
                                         # default_target names a target
merge_declared(s, decls: list[TargetDecl]) -> tuple[Settings, list[str]]
                                         # adds declared targets missing from the settings (name, host, port,
                                         # latency); existing targets are NOT changed; returns added names
declared_conflicts(s, decls) -> list[CueIssue]
                                         # a warning on each @target line whose host, port or declared latency
                                         # differs from the settings target of that name (the settings win)
latency_s(s, cue) -> float               # the cue's target latency in seconds (default target for None,
                                         # 0 for unknown)
```

## 3. Scheduler (`scheduler.py`) — when a cue fires

### 3.1 Pure core

```python
MOVING = ("tracking", "coasting")

@dataclass
class SchedulerParams:
    catchup_s: float = 1.0      # forward move <= this: every crossed cue fires (late); larger: a jump
    rearm_back_s: float = 1.0   # backward move >= this re-arms the cues after the new position
    max_extrap_s: float = 0.1   # extrapolate at most this far past the last tracker update
    skip_policy: str = "latest" # what a jump forward does with the cues it skipped

@dataclass(frozen=True)
class TrackPoint:               # one tracker update as the scheduler sees it
    ref_time: float             # published reference time (s)
    rate: float                 # TrackerState.tempo (0 = not moving)
    status: str
    t: float                    # clock time of the update (same clock as `now` everywhere)
    hold_from: float | None = None   # cues with time >= hold_from wait (vocal mode waiting for the voice)
    settling: bool = False           # a vocal relock may still move the position back over what it coasted

@dataclass(frozen=True)
class Fire:
    cue: Cue
    at: float                   # clock time it fires
    ref_time: float             # extrapolated position (without latency look-ahead) at `at`
    reason: str                 # "track" (crossed in time) | "catchup" (position moved past it at an update /
                                # small forward correction) | "chase" (sent by the skip policy) |
                                # "skipped" (jumped over, NOT sent) | "start" (chase at start)
    send: bool = True           # False for "skipped"
    ref_late: float = 0.0       # ref seconds past the cue: position + rate*latency - cue.time at `at` (>= 0)

class CueScheduler:
    def __init__(self, cues: list[Cue], latency_of: Callable[[Cue], float],
                 params: SchedulerParams | None = None): ...
    def start(self, pt: TrackPoint, boundary: float | None = None) -> list[Fire]   # boundary = requested time
    def update(self, pt: TrackPoint) -> list[Fire]
    def jump(self, pt: TrackPoint, boundary: float | None = None) -> list[Fire]    # operator jump (pt: after it)
    def due(self, now: float) -> list[Fire]               # cues due by `now`; marks them fired
    def next_due_time(self) -> float | None               # clock time of the next due cue, None if unknown
    def position(self, now: float) -> float               # extrapolated reference position
    def upcoming(self, now: float, n: int = 3) -> list[tuple[Cue, float | None]]   # next unfired cues, ETA s
    def set_cues(self, cues: list[Cue], now: float) -> None
    def set_params(self, params: SchedulerParams) -> None
    def set_latency_of(self, fn) -> None
    def fired_ids(self) -> set[int]
```

Semantics (all times on one injected clock, `now`):

* Position model after the last point `p`: `P(now) = p.ref_time + p.rate * clip(now - p.t, 0, max_extrap_s)`
  if `p.status in MOVING`, else `p.ref_time`. Look-ahead rate `v = p.rate if p.status in MOVING else 0`.
* Cue `c` (unfired, armed) is **due** at `now` when `P(now) + v * L(c) >= c.time - 1e-9`, where
  `L(c) = latency_of(c)` (seconds, ≥ 0) — and not guarded: `p.hold_from is None or c.time < p.hold_from`.
* `next_due_time()`: smallest `t` at which some armed unfired unguarded cue becomes due under the current
  point: `p.t + (c.time - p.ref_time - v*L(c)) / v` if `v > 0` and that is `<= p.t + max_extrap_s`
  (already due → `p.t`... i.e. return a time <= now so the runner fires immediately); `None` when nothing
  can become due before the next update. Must be O(cues near the position), not O(all cues).
* `start(pt, boundary)`: all cues armed and unfired. `boundary` is the requested start time (None:
  `pt.ref_time`); the tracker publishes it plus its input-latency look-ahead, and a point behind it (a
  negative input latency) is moved up to it. Cues with `time < boundary - 1e-6` are jumped over → skip
  policy with reasons "start"/"skipped"; returns those Fire records (sent ones first in time order).
  Cues from the boundary to the start position are due at once (returned by the following `due`, reason
  "track"); the next update continues from the published position (no false jump).
* `update(pt)`: `E = P_prev(pt.t)` (previous point extrapolated to the new update time). `d = pt.ref_time - E`.
  * `d > catchup_s` → **jump forward**: unfired cues with `E < time <= pt.ref_time` are jumped over →
    skip policy (`skip`: all "skipped"; `latest`: per `(target, address)` only the latest one "chase",
    the rest "skipped"; `all`: every one "chase", in time order). Returned with `at = pt.t`.
  * `d <= -rearm_back_s` → **jump back**: cues with `time > pt.ref_time + 1e-6` become unfired again.
    Except after positions that did not follow the music: during a run of points that are coasting, carry
    a vocal guard (`hold_from`) or say `settling`, and at the first followed point after such a run, the
    move back is measured from where the run began (the position last followed): only a landing
    `rearm_back_s` or more behind that is a jump back; a vocal relock settling back over coasted positions
    re-fires nothing.
  * otherwise nothing special: cues the position passed fire through `due` (reason "catchup" when their
    ideal due time is earlier than the update by more than one frame, 0.04 s, else "track").
  Then the point becomes current. `update` itself returns only the jump-forward records; callers then
  call `due(now)`.
  * As implemented (scheduler.py docstring): the cues a tracker jump back re-arms are *restorable* (they
    were handled in this pass, at or before its reach: the position it had when it jumped back, plus the
    look-ahead); a later forward move of at least `rearm_back_s` (a jump, or a catch-up move that large when
    `catchup_s > rearm_back_s`) landing less than `rearm_back_s` behind that reach, or beyond it, marks every
    restorable cue handled again silently (look-ahead lanes included), so a wrong re-localisation that is
    undone never re-sends them; landing further behind (a new pass from there), or a smaller forward jump,
    restores only the restorable cues at or before the landing position. A cue sent after the jump back
    makes the restorable cues on its `(target, address)` ordinary armed cues (the device no longer holds
    what they set: on the return they fire or go through the skip policy, "latest" chases them). An
    operator jump or `start` clears the restorable set. Unfired cues the vocal guard held at or behind `E`
    belong to a forward jump's range too (skip policy, time order), never fire late after it.
* `jump(pt, boundary)`: explicit operator jump (Go / jump to time / jump to cue); `boundary` = the
  requested time as in `start` (None: `pt.ref_time`; a point behind it is moved up to it). Re-arm (unfire)
  every cue with `time >= boundary - 1e-6`; cues between the old position and the boundary when moving
  forward → skip policy; cues from the boundary to the published position are due at once ("track"). A
  jump exactly to a cue's time makes it due at once (operator "go to cue N" fires N), whatever the input
  latency.
  The point becomes current (no jump detection against it).
* `set_cues(cues, now)` (file reloaded / edited): fired state carries over by `Cue.key()`; a new cue whose
  `time < P(now) - 1e-6` is marked fired without sending (it is behind the position: never fire it late).
* Reasons: a cue fired by `due` because the runner's timer reached its ideal time is "track"; fired at
  an update because the position (or a small correction) moved past it, "catchup". `ref_late` is always
  computed at `at`.
* Determinism: equal-time cues fire in id (file line) order. A cue fires at most once until re-armed.

### 3.2 Realtime runner

```python
@dataclass
class FireRecord:          # what the console / show log shows
    seq: int               # increasing from 1 per runner
    cue_id: int; time: float; target: str; address: str; message: str
    reason: str; sent: bool; error: str | None
    wall: float            # time.time() when fired
    ref_time: float; ref_late_ms: float; live_time: float | None; dry_run: bool

class CueRunner:
    def __init__(self, scheduler: CueScheduler, send: Callable[[Cue], SendResult], *,
                 clock=time.perf_counter, armed: bool = True, default_target: str = "main",
                 on_fire: Callable[[FireRecord], None] | None = None, live_time: Callable[[], float] | None = None,
                 log_size: int = 500)
    def start(self, pt: TrackPoint, boundary: float | None = None) -> None   # timer thread; scheduler.start
    def stop(self) -> None
    def feed(self, pt: TrackPoint) -> None      # audio thread: update + due(now) + fire, then wake the timer
    def jump(self, pt: TrackPoint, boundary: float | None = None) -> None    # operator jump
    def fire_manual(self, cue: Cue) -> FireRecord    # sends now regardless of `armed` (reason "manual")
    armed: bool (property with setter) ; set_cues(cues) ; set_params(p) ; set_latency_of(fn) ; set_default_target(name)
    def records(self, after_seq: int = 0, limit: int = 100) -> list[FireRecord]
    def upcoming(self, n=3) ; position() ; counts() -> {"sent", "skipped", "errors", "dry"}
```

* One lock guards the scheduler; the timer thread waits on a `threading.Condition` until
  `min(next_due_time(), now + 0.05)`, then fires `scheduler.due(clock())`. `feed` notifies the condition.
* Sending: `send(cue)` (session-provided; returns `osc.SendResult`) is called for records with `send=True`
  only when `armed`; when disarmed the record is `dry_run=True, sent=False`. Skipped records are logged
  (`sent=False`) but never sent. Records are appended to a bounded deque and passed to `on_fire`.
* Fires are sent in order and never while holding the lock longer than the send itself (UDP send is
  ~µs; DNS is resolved beforehand by OscSender).
* Thread safety: `feed` is called from the audio worker thread, everything else from the server's
  threads; `stop()` joins the timer thread (timeout 1 s).

## 4. Session (`live/session.py`) — OscSession

Adapt `../src/lyricprompter/live/session.py` LiveSession (keep its structure: SessionDeps for testable
construction, one RLock, `_on_audio`, `start` / `stop`, `align_source_start`, operator controls,
`snapshot`, json_safe). Differences: no Song / Timeline / lines; the package is a project
(`project.load_meta(pdir)` gives `tuning`, `style`, `duration`; stems `mix` / `vocals` via
`songfile.stem_path`); the tracker gets `line_starts=None`; a `CueRunner` is fed after every tracker
step.

```python
class OscSession:
    def __init__(self, project_dir: Path, mode: str, source, params=None, *, cues: list[Cue],
                 sender: OscSender, settings: Settings, style: str | None = None,
                 start_time: float = 0.0, deps: SessionDeps | None = None, clock=time.perf_counter,
                 log_dir: Path | None = None, on_update=None)
    start(); stop(); running; source_finished; align_source_start(preroll)
    jump_time(t); jump_cue(cue_id); hold(on); nudge(dt); set_transpose(k); set_auto_key(on); set_style(s)
    set_input_latency(ms)           # tracker.set_latency(ms / 1000)
    set_cues(cues); apply_settings(settings)   # targets/latency map/default target/skip policy/catch-up/armed
    set_armed(on); fire_cue(cue_id) -> FireRecord
    snapshot() -> dict
```

* `TrackPoint` per tracker step: `ref_time = state.ref_time`, `rate = state.tempo`, `status`,
  `t` = the clock time at which its newest frame ended: `clock()` right after `tracker.push` minus the samples
  of the block that came after that frame (`excess / sr`; the offline simulator uses `min(state.live_time, t_k)`),
  `hold_from` = vocal waiting guard: in mode "vocal",
  when not `tracker._free_run` and (`status == "coasting"` or not `state.active`): the next phrase onset
  at/after the position, `(onsets[k] + 1) / fps - 0.1` (k = first onset >= position frame - 0.5), else None;
  `settling` = vocal mode and the tracker's relock is provisional (`tracker._prov`) or its OLTW may still
  step back (`tracker.oltw.free > 0`).
* `send(cue)`: target = `cue.target or settings.default_target`; `sender.send(target, cue.address, cue.args)`.
* Show log: when `log_dir` is given, every FireRecord is appended to
  `log_dir/show-YYYYmmdd-HHMMSS.csv` (header: wall_iso, live_time, ref_time, cue_id, cue_time, target,
  address, message, reason, sent, dry_run, error, ref_late_ms) by a writer thread (never in the firing path).
* `snapshot()` (20 Hz, JSON-safe):
  `{"type": "state", "project_id", "mode", "style", "status", "ref_time", "position" (extrapolated now),
  "tempo", "confidence", "live_time", "level_db", "t", "age", "frame", "active", "transpose", "key_auto",
  "key_suggest", "input_latency_ms", "input_latency_reported_ms" (PortAudio input latency if known, else
  null), "armed", "ended", "error", "warning", "source": {...}, "upcoming": [{"id","time","target","message",
  "eta"}], "fired_seq": last seq, "fired": [last 30 records as dicts], "counts": {...},
  "targets": sender.stats()}`. `status` is "stopped" once the source ended or stop() was called.

As implemented (live/session.py), beyond the signature above:
* `settings=None` means `Settings()`; keyword-only extras `on_fire(record)` (every FireRecord, in order,
  from the firing thread; must not block) and `timer=True` (False: no runner timer thread, for tests and
  offline tools). The constructor already applies `settings.input_latency_ms`; `apply_settings` applies
  it again only when it changed (calling `set_input_latency` with the same value is harmless).
* Members: `cue(id)` (`UnknownCueError(KeyError, ValueError)`), `cues`, `settings` (a copy), `armed`,
  `stopped`, `input_latency_reported_ms()`, `project_id`, `stem`, `style`, `runner`, `scheduler`,
  `tracker`, `show_log`. Vocal mode without a vocal stem raises `StemMissingError(FileNotFoundError,
  ValueError)`.
* At `start()` and at an operator jump the runner gets the tracker's published position (the requested time
  plus the input-latency look-ahead; frame 0 as 0.04 s) with the requested time as `boundary`: a cue at 0:00
  or exactly at the jump target fires at once (also with a negative input latency), the cues from there to
  the published position too, and the next update is no false jump. Jump points carry no vocal guard; the
  start point keeps it.
* Show log names: `show-YYYYmmdd-HHMMSS.csv`, `-2`, `-3` ... when taken; created at `start()`; booleans
  `true`/`false`, times with 3 decimals, `ref_late_ms` with 1. A writer failure shows in `warning`.
* A session whose source ended stays the server's session (status "stopped", `ended` true, its fired log
  still shown) until it is stopped or replaced: it still counts as live for DELETE and device rescans.

`live/simulate.py`:
* `oscsender simulate --project ID|--project-dir DIR --file live.wav [--mode mix|vocal] [--style]
  [--start-time S] [--transpose K] [--auto-key/--no-auto-key] [--latency-ms N] [--realtime] [--send]`.
  Default (offline): drives the tracker as fast as possible with a **virtual clock = live time**
  (frames / fps): after each tracker update at live time t_k, fire every cue whose `next_due_time() <=
  t_{k+1}` at its exact virtual time, then the next update. Prints one line per fire: live time, cue
  time, ref position, reason, target, message. `--realtime` uses FileInput(realtime) + CueRunner +
  wall clock; `--send` actually sends (else dry run).
* `simulate_cues(ref_project_dir, live_audio_16k, mode, style, settings, ...) -> list[Fire-like dicts]`
  reusable function (used by eval and tests).
* `oscsender eval --project ID [--seeds 3] [--mode mix] [--style classic|pop] [--tempo LO,HI]
  [--transpose K]`: synthetic live takes of the project's reference (synth.make_synthetic: piecewise
  tempo, noise, reverb, EQ), simulated cue firing, error per cue = fire live time − ground-truth live time
  of the cue (synth.ref_to_live). Report per take and overall: n cues, fired, missed, median |err| ms,
  p90, max, signed median (bias), % within 50 / 100 / 200 ms. `--json` output option.

## 5. Project store (`project.py`)

```
projects/<id>/project.json   {"schema": 1, "id", "title", "created" (ISO 8601 UTC), "duration" (s, of mix),
                              "tuning" (semitones, librosa estimate as in lyricprompter prep), "style",
                              "sources": {"audio": original file name, "vocals": name | null, "cues": name | null},
                              "stems": ["mix"] or ["mix", "vocals"]}
projects/<id>/audio/mix.wav     canonical 44.1 kHz stereo PCM24 (audio_io.to_canonical_wav; needs ffmpeg)
projects/<id>/audio/vocals.wav  optional vocal stem, converted the same way and padded / trimmed to the
                                mix length (warn in the result when lengths differ by > 0.5 s)
projects/<id>/cues.txt          the cue file (source of truth; may be edited by hand while the server runs)
projects/<id>/cache/            feature cache (extractor) + waveform peaks
projects/<id>/logs/             show logs
projects/.settings.json  projects/.trash/
```

```python
@dataclass
class ProjectMeta: id, title, created, duration, tuning, style="classic", sources: dict, stems: list[str]
check_project_id(pid) -> str             # NFC, one path component, regex \w[\w\-.]* (like lyricprompter)
new_project_id(title, root) -> str       # slugify like lyricprompter.songfile; -2, -3 suffixes
create_project(root, *, title, audio: Path, cues_text: str = "", vocals: Path | None = None,
               style="classic", project_id=None, cues_name=None) -> tuple[Path, list[str]]   # (dir, warnings)
    # builds in projects/.tmp-<id>-<uuid> then renames: never leaves a half project
load_meta(pdir) -> ProjectMeta ; save_meta(pdir, meta) ; list_projects(root) -> list[Path]
cues_path(pdir) ; read_cues(pdir) -> str ; write_cues(pdir, text) -> None (atomic, UTF-8, keeps text as is)
set_vocals(pdir, path) -> list[str]      # add / replace the vocal stem later
waveform(pdir, stem="mix", points=2000) -> {"duration", "points", "min": [...], "max": [...]}  # float lists
    # rounded to 4 decimals; cached in cache/peaks_<stem>_<points>.json keyed by the stem's mtime/size
trash_project(root, pid) -> Path         # move to projects/.trash/<pid>-<timestamp>
estimate_tuning(mix_wav) -> float        # copy of lyricprompter.prep.pipeline.estimate_tuning (+ TUNING_PARAMS),
                                         # streamed: the same value in bounded memory (block-wise read and
                                         # resampling, piptrack on chunks of frames), whatever the mix length
cli: oscsender add --audio A --cues C.txt --title T [--vocals V] [--style] [--id]; oscsender list
```

`synth.py`: copy from `../src/lyricprompter/eval/track_eval.py` unchanged: `SynthSpec`, `synth_time_map`,
`pv_warp`, `_pink`, `degrade`, `make_synthetic`, `ref_to_live`, `map_events`. Plus a deterministic
**synthetic music generator** for tests and demos:
`render_song(score, sr=16000, tempo_map=None, transpose=0) -> np.ndarray` and
`make_score(seconds, seed) -> score` (chord progression of triads with changing harmony every 0.5–1.5 s,
a melody line, kick/snare-like noise bursts, a few rests, ~-12 dBFS), where `tempo_map` maps score time →
performance time (piecewise linear, e.g. `([0, 20, 40], [0, 21.5, 44])`) so a re-rendered "live" take
has an exact ground truth (no phase vocoder). Must be fast (< 1 s for 60 s at 16 kHz).

## 6. Latency (`latency.py`)

OSC output delay = what a receiver needs from our send to its effect. Per target `latency_ms` makes the
scheduler send that target's cues earlier. Three ways to get it (UI: Targets ▸ Measure):

1. **Network echo** (automatic): round trip to a responder. `measure_echo(host, port, *, count=20,
   interval_s=0.05, timeout_s=0.5, message=None, listen_port=None, clock=time.perf_counter) ->
   LatencyReport`. Socket bound to `("0.0.0.0", listen_port or 0)`, sequential pings:
   * `message=None`: send `/oscsender/ping i:seq d:send_time`, accept `/oscsender/pong` whose first arg
     is `seq` (ignore others / stale ones);
   * `message=(address, args)`: send it, accept the first OSC packet that arrives (any address) — for
     receivers that answer queries (e.g. QLab replies) or are patched to echo.
   `estimate_ms = median RTT / 2`.
   `serve_echo(port, host="0.0.0.0", *, echo_all=False, stop=None, on_message=None)`: answers
   `/oscsender/ping` with `/oscsender/pong` + the same args to the sender's address; `echo_all` mirrors
   every message unchanged. CLI `oscsender echo [--port 9001] [--echo-all]` (run it on the receiving
   machine to measure the network path).
2. **Audio loopback** (automatic, for receivers that make sound: samplers, QLab / Ableton audio, a
   click): send the target's test message and detect the sound at the audio input.
   `measure_loopback(send: Callable[[], float], capture: AudioCapture, *, count=8, interval_s=1.5,
   window_s=1.0, threshold_db=20.0, pre_s=0.3, clock=time.perf_counter) -> LatencyReport` where `send()`
   sends and returns the clock time. `AudioCapture(device, channels, samplerate=None, clock=...,
   stream_factory=None)`: sounddevice InputStream (channels 1-based, downmixed); the callback stores each
   block with the clock time of its first sample's ADC capture:
   `clock_at_callback - (time_info.currentTime - time_info.inputBufferAdcTime)` when PortAudio provides
   them (non-zero), else `clock_at_callback - stream.latency - frames / sr`; ring buffer of 15 s;
   `segment(t0, t1) -> (times, samples)`. Detection: 1 ms RMS envelope (dB); floor = median of
   `[t_send - pre_s, t_send)`; onset = first envelope point in `(t_send, t_send + window_s]` above
   `max(floor + threshold_db, -70 dBFS)`; latency = onset - t_send; a trial without an onset counts as a
   miss. Wait until the level is back near the floor (or interval_s passed) before the next trial.
   `estimate_ms = median`. `stream_factory` lets tests inject a fake stream that "plays back" a click
   N ms after each send.
3. **By eye / ear**: the cue editor's preview sends OSC while the reference plays (console), the operator
   adjusts the target latency until the effect lands on the music; and plain manual entry.

```python
@dataclass
class LatencyReport:
    method: str                 # "echo" | "loopback"
    target: str | None; host: str | None; port: int | None
    sent: int; received: int
    samples_ms: list[float]
    min_ms: float | None; median_ms: float | None; p95_ms: float | None; max_ms: float | None
    jitter_ms: float | None     # interquartile range
    estimate_ms: float | None   # suggested target latency
    note: str = ""; error: str | None = None
    def to_dict(self) -> dict
```
CLI `oscsender latency --host H --port P [--count N] [--message "/addr args"] [--listen-port L] [--json]`.

## 7. Server (`server/app.py`)

FastAPI + one 20 Hz WebSocket broadcaster, structured like lyricprompter's (Hub, WSClient latest-wins
outbox, OriginGuard copied unchanged, `create_app(projects_dir=None, *, session_factory=None,
devices=None, broadcast_hz=20.0, allowed_hosts=(), sender=None)`, `cli(argv)` for
`oscsender serve [--host 127.0.0.1] [--port 8800] [--projects-dir] [--allow-host H]`). Default port
**8800**. One OscSender lives in the Hub (targets from settings) and is shared by sessions, Test, Fire and
preview. All errors are `{"detail": message}` with 4xx; JSON bodies; numbers JSON-safe (no NaN).

Projects
* `GET /api/projects` → `[{"id","title","duration","style","stems","n_cues","n_errors","created"}]`
* `POST /api/projects` (multipart: `audio` file required, `title` required, `cues` file optional,
  `cues_text` optional (used when no file), `vocals` file optional, `style` classic|pop) → 201
  `{"project": summary, "warnings": [...]}`; conversion runs in a threadpool; the declared `@target`s
  are merged into the settings (`merge_declared`).
* `GET /api/projects/{id}` → `{"project": meta dict + "stems", "cues": sheet_json, "logs": [file names]}`
* `DELETE /api/projects/{id}` → trash it (409 while it is the live session's project)
* `GET /api/projects/{id}/audio/{stem}` (mix|vocals; FileResponse, supports Range) ;
  `GET /api/projects/{id}/waveform?points=2000` (100..20000) ; `GET /api/projects/{id}/cues.txt` (download)
* `PUT /api/projects/{id}/style {"style"}` → saves; applies to the running session of that project
* `POST /api/projects/{id}/vocals` (multipart `vocals`) → adds / replaces the vocal stem

Cues — `sheet_json` = `{"rev", "text", "cues": [{"id","time","target","address","args": [[tag, value],
...], "message","note"}], "targets": [{"name","host","port","latency_ms","line"}], "issues": [{"line",
"severity","message"}]}` (issues include the server-side warnings: unknown target, after the end).
Blob values never occur in cue files. Every edit: optional `rev` (sheet_rev of the text the client
edited; mismatch → 409 `{"detail", "cues": current sheet}`), saves atomically, pushes the new cues to a
running session of the project, records undo (per project, last 100 texts, in memory), bumps the
project's broadcast rev, returns the new `sheet_json` (+ fields below).
* `GET /api/projects/{id}/cues`
* `PUT /api/projects/{id}/cues {"text", "rev"?}` → replace the whole file (saved even with issues)
* `POST /api/projects/{id}/cues/upload` (multipart `cues`) → replace
* `PATCH /api/projects/{id}/cues/{cue_id} {"rev"?, exactly one of "time" | "shift" | "message"}`
* `POST /api/projects/{id}/cues/add {"rev"?, "time", "message"}` → + `"added": new id`
* `DELETE /api/projects/{id}/cues/{cue_id}?rev=...`
* `POST /api/projects/{id}/cues/undo` → + `"undo_left"`; 409 when nothing to undo or the file changed on
  disk since the edit being undone (the stack is cleared then)
* `POST /api/projects/{id}/cues/{cue_id}/fire {"source": "manual"|"preview"}` → sends now via the
  Hub's sender (ignores `armed`; with a live session of the project, goes through `session.fire_cue`
  so it is logged) → `{"ok","error","target","address"}`
* The broadcaster stats the live / cued project's `cues.txt` twice a second; an external edit is
  re-parsed, pushed to the session and announced (`{"type":"project","id","rev"}`).

Settings and targets
* `GET /api/settings` → settings dict + `"targets_status": sender.stats()` + `"warnings"` (what
  `load_settings` repaired in a damaged settings file at startup; cleared by the next successful save;
  the console shows them above the OSC targets)
* `PUT /api/settings` (partial: any of targets (whole list), default_target, armed, skip_policy,
  catchup_s, input_latency_ms) → validated (422/400 on errors), given to the sender (no host lookup under
  the locks: see "Target hosts" below) and to a running session, saved; broadcast `{"type":"settings","rev"}`.
* `POST /api/session/arm {"on"}` (same as PUT armed, kept for the big ARM button)
* `POST /api/targets/{name}/test {"message"?}` → sends `message` or the target's `test_message` or
  `/oscsender/test` → `{"ok","error"}` (an enabled target without an address is looked up again first,
  as for a manual Fire)
* `POST /api/latency/echo {"target" | "host"+"port", "count"=20, "interval_ms"=50, "timeout_ms"=500,
  "message"?, "listen_port"?}` → LatencyReport dict (threadpool; count 1..200)
* `POST /api/latency/loopback {"target", "device", "channels", "count"=8, "interval_ms"=1500,
  "threshold_db"=20, "message"?}` → report; 409 while a session with a device input runs.
* `GET /api/devices?refresh=false` → input devices (live.audio_input.list_input_devices). `refresh=true`
  re-initialises PortAudio (closing every open stream): 409 while a session exists or is being started /
  stopped, or a loopback measurement runs; it holds the session lock, so a start waits for it (a loopback
  measurement asked for meanwhile: 409).

Session (one at a time; like lyricprompter's)
* `POST /api/session/start {"project_id", "mode": "mix"|"vocal", "style"?, "input": {"kind":
  "device"|"file", "device"?, "channels"?, "samplerate"?, "stem"? (mix|vocals of this project),
  "path"?, "monitor"=false, "speed"=1, "start_s"=0}, "start_time"? | "start_cue"?, "transpose"=0,
  "auto_key"=true}` → snapshot. Vocal mode needs the vocals stem (400 otherwise). A rehearsal from this
  project's own stem with a start position plays 2 s of pre-roll (`align_source_start(2.0)`). Starting a
  session stops a running one.
* `POST /api/session/stop|hold {"on"}|jump {"time"|"cue_id", "rev"?}|nudge {"dt"}|transpose {"semitones"}|
  autokey {"on"}|latency {"ms"}|style {"style"}` ; `GET /api/session`
* WebSocket `/ws`: on connect `{"type":"hello","project_id","settings_rev"}` then 20 Hz
  `{"type":"state", ...}` (session snapshot, or `{"type":"state","status":"idle","project_id":null}`);
  `{"type":"project","id","rev"}` when a project's cue sheet changes (any client edit or disk edit);
  `{"type":"settings","rev"}` when settings change.

As implemented (server/app.py); the fields above are unchanged, these are additions:
* Session factory: `(project_dir, SessionStart, SessionContext) -> session` where `SessionContext` =
  `(projects_dir, cues, sender, settings (a copy), start_time, clock)`. The endpoint reads and parses the
  cue file, resolves `start_cue` to a time (404 when unknown; `start_time` + `start_cue` together: 422;
  with the optional `"rev"` (sheet_rev the cue was picked in) a changed file is 409 + the current sheet, as
  for edits) and copies the settings before calling it. The default factory builds `DeviceInput` /
  `FileInput(realtime)` (`path` relative to the package root), `OscSession(..., log_dir=<project>/logs)`,
  applies the pre-roll, the input latency and transpose / auto-key (lyricprompter's rules). Vocal mode
  without a vocal stem: 400 (checked in the endpoint and in the factory).
* Make before break: when the new device input cannot open while the running device session holds the
  interface, the old session is stopped and a newly built session is started once more (the failed one
  is stopped; it cannot be started again). Settings, cue edits and a project style saved
  (`PUT .../style`) while the session was being built are applied to it once it runs.
* `create_app(..., clock=time.perf_counter)`; the CLI also takes `--log-level`.
* Extra response fields: `sheet_json` + `project_rev` (the project's broadcast rev) + `undo_left`; edits +
  `targets_added`; undo + `undone` (label); create / vocals + `targets_added`; project rows and
  `GET /api/projects/{id}` + `rev`; settings dict + `rev`; Fire + `logged` (+ `seq` through the
  session); Test → `{"ok","error","target","address"}`; WS project message + `sheet_rev`; WS settings
  message + `armed`; the idle state + `armed`, `t`, `targets`; `GET /api/session` →
  `{"session": start request or null, "project_id", "state"}`.
* `GET /api/projects/{id}/logs/{name}` downloads a show log. `JumpReq.project_id` (optional): 409 when it
  is not the live project. `JumpReq.rev` (optional, with `cue_id`): sheet_rev of the sheet the cue was
  picked in; 409 + the current sheet when the live project's cue file changed since (cue ids are line
  numbers: a Go from a stale view would jump to another line's cue).
* `POST /api/session/latency {"ms"}` saves `input_latency_ms` in the settings (settings broadcast) and works
  without a session (returns the idle state).
* Error codes: cue edit errors and settings validation 400; unknown cue id 404; schema errors and "exactly
  one of" violations 422; upload validation (empty title, bad style, empty / unreadable audio, non-UTF-8 cue
  file) 400; a second loopback measurement, or a device session start while one runs, 409. Cue text or
  settings that cannot be saved as UTF-8 (a lone surrogate escaped in JSON) 400, nothing saved.
* Cue files are watched by their own 0.5 s task (not the broadcaster), which holds the edit lock; a disk
  edit also merges its `@target` declarations. Manual Fire and Go by cue id first pass a disk edit the
  watcher has not seen yet to the live session. A change is told by the file signature (mtime, size,
  inode, ctime) and, for a file written in the last 5 s (FAT32 / HFS+ keep mtime in 1–2 s steps, so a
  same-size save there keeps the signature), by its text; a new signature with the text last announced or
  saved is not announced.
* Target hosts: the sender resolves through the Hub's host book, so setting its targets never waits on DNS
  (an unknown `.local` name takes 5 s). Lookups run in worker threads outside every lock, one per
  host:port, only for hosts not looked up yet (a target whose host did not change is not looked up again);
  PUT settings, cue edits that add `@target`s and project creation wait for the lookups of the hosts they
  added before replying (startup too). Enabled targets without an address are looked up again every 5 s
  in the background, on a settings PUT (in the background), and before Test / a manual Fire; resolved host
  names every 60 s (a new DHCP address). A failed lookup never replaces an address that worked (it is
  logged).

## 8. Console UI (`server/static/console.{html,css,js}`, vanilla ES modules, no build, no CDN)

Single page at `/`. Dark operator theme like lyricprompter's console (reuse its CSS approach and
`common.js` helpers by copying what is needed into `console.js` or a local `common.js`). Must work offline.

1. **Header**: app name, connection pill, big **OSC ARM** toggle (armed = red "OSC LIVE"; disarmed =
   grey "SAFE — dry run"), notice line.
2. **Projects** (left): list (title, duration, cue count, error count); **New project** form: title,
   reference audio (file), cue file (file) or pasted cue text, vocals stem (optional), style; progress
   while uploading/converting; delete (with confirm, moves to trash).
3. **Live** (center top): status pill, reference time (large, mm:ss.ss), live time, tempo, confidence and
   input meters, key (−, value, +, Auto, suggestion), **Next cue** with countdown, **Fired log** (newest
   first: time of day, cue #, cue time, target, message, reason, sent/dry/skipped/error, late ms; a new
   session's records start a new run: its seq going back, another project, or another record under the
   newest seq seen). When the WebSocket drops, the status shows Disconnected, the reference greys, the
   countdown blanks, Hold and nudge are disabled and a warning shows, until the next state message.
4. **Session controls**: Listen to (Full band mix / Vocal mic only — disabled without a vocals stem),
   Input (Audio interface: device + channel checkboxes + Rescan; Rehearsal: this project's mix / vocals /
   another file path, Play through speakers), Style (classic / pop), Start at (beginning / cue N / time),
   Start / Stop / Hold (H), nudge −0.5 / +0.5 ([ ]), Input latency (ms). Only the Start button starts
   (Enter in a field of the form does not); while a session runs, Start asks first (restart / replace).
5. **OSC targets**: table name / host / port / latency ms / enabled / Test / Measure; add / remove;
   default target select; skip-policy select (Skip / Send latest per address / Send all); catch-up
   window (s). Edits PUT /api/settings. **Measure** dialog: method (Network echo / Audio loopback), its
   parameters (echo: count, timeout, optional reply message + listen port, hint to run `oscsender echo`
   on the receiver; loopback: device, channels, count, threshold, test message), Run, result (median,
   min / p95 / max, jitter, n), buttons "Use as latency" (estimate) and "Cancel".
6. **Cue editor** (main area / right): issues list (line: message) ; toolbar: Undo (⌘Z), Tap (T), Add cue
   at playhead (prompts target/message), Edit file (modal textarea; Save → PUT text; shows issues),
   Download cues.txt, Replace file (upload); preview transport: Play/Pause (Space), position, Click at
   cues (toggle + volume), **Send OSC while previewing** (toggle, applies target latency; also sent when
   SAFE; while a session runs only after a confirmation, and a session that starts turns it off), Follow live;
   **waveform** canvas (whole song, cue markers colored per target, playhead, live position; click to
   seek; hover shows cue) and a **zoom strip** (±4 s around the playhead, markers with labels, drag a
   marker horizontally to move its cue — commit on release with PATCH time); **cue table**: Go | # | Time
   | Target | Message | Note | Play (from 1 s before) · −0.1 · +0.1 · −0.01 · +0.01 · Set (=playhead) ·
   Fire · Delete. Time cell click → inline input (accepts the cue time syntax); message cell double-click
   → inline input (Enter saves, Esc cancels). Rows: passed cues dimmed relative to the preview playhead;
   the live session's last fired cue highlighted; errors shown.
7. **Preview audio engine**: WebAudio. Fetch `/api/projects/{id}/audio/mix`, `decodeAudioData` once per
   project; play with an `AudioBufferSourceNode` (restart on seek); position = offset + (ctx.currentTime −
   startCtx); the heard position (shown playhead, Set, Tap, Add cue at playhead, pause) is that minus
   outputLatency. Clicks are scheduled on the same AudioContext timeline with a 25 ms look-ahead timer
   (short 1.5 kHz blip ~30 ms; per-target pitch offsets), so a click is sample-aligned with the music.
   Preview OSC: for each cue, POST `/fire` with source "preview" at the context time
   `cueCtx + outputLatency − targetLatency` (outputLatency = `ctx.outputLatency || ctx.baseLatency || 0`)
   using setTimeout from the look-ahead loop; at most once per cue and pass (a cue counts as sent when its
   timer fires; play from a stop or a seek starts a new pass, resume after a pause continues it); a cue
   edit or toggle re-arms: unsent sends are scheduled again (one already due is sent at once).
8. **Tap mode**: T sets the armed cue's time to the preview playhead (rounded to 1 ms) and arms the
   next cue; click a row's # to arm it; Esc leaves. The armed cue and a Start at cue follow their cue
   (same time, target and message; or the same line after an in-place edit) when lines shift; when it is
   gone Tap disarms and Start at becomes its time, with a notice.
9. Edits go through one promise queue (in order); each response carries the new sheet (ids may shift
   after add/delete: re-render from it); on 409 reload the sheet and notify.
10. Keyboard: Space play/pause, T tap, ⌘Z/Ctrl+Z undo, H hold, [ / ] nudge — ignored while typing in a
    field. A focused select is not typing: the keys act and never reach its type-ahead.
11. As implemented: input latency is a setting (`PUT /api/settings {"input_latency_ms"}`, applied to a
    running session by the server). Stop is enabled whenever a session exists, including one whose
    rehearsal file has ended (it still holds its project); Hold and nudge only while it runs. Go sends
    `{"cue_id", "project_id"}`. The WS project message is matched against the sheet's `project_rev`, and
    the console then re-fetches `GET /api/projects/{id}` (meta, sheet, logs).

## 9. Verification expectations

* Unit tests per module (parser incl. every format rule and edit function round trips; OSC codec vs
  python-osc; scheduler semantics with a fake clock: steady tracking with latency look-ahead, sub-frame
  timing, catch-up, jump forward per policy, jump back re-arm, small backward settle without re-fire,
  operator jump to a cue, guard, reload carry-over, disarmed dry run; latency echo against a local
  responder (also packet loss / timeouts) and loopback with a fake stream; project create / waveform /
  trash; server endpoints with fake sessions and a real OscSender to a local UDP socket; websocket).
* End-to-end: a synthetic song (synth.render_song) as reference, a re-rendered live take with a known
  tempo map, simulated cue firing: median |error| ≤ 60 ms and every cue fires exactly once for mix
  mode in both styles.
* `uv run ruff check` clean; the copied tracker tests stay green.
