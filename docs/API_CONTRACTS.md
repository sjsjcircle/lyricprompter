# Module API contracts (v1)

Parallel implementers code against these signatures. Adding optional parameters / extra fields with
defaults is fine; changing or removing anything listed here is NOT — report it as an interface change request.
Shared, already-implemented modules (do not edit): `config.py`, `text/model.py`, `songfile.py`, `cli.py`.
Every module that provides a CLI command exposes `cli(argv: list[str]) -> int` (see `cli.py`).

## text (owner A) — `lyricprompter.text`
```python
def parse_lyrics(raw: str, default_lang: str = "auto") -> list[Line]
    # Lines with words/syllables/units filled, all times None. Repeats ("xN") expanded in sung order
    # (copies get flags ["repeat_of:<i>"]). Section headers / blank lines -> Line.section / Line.stanza.
    # Every AlignUnit.roman matches ^[a-z']*$ (MMS_FA vocabulary). Spans index Line.text.
def detect_lang(text: str) -> str          # "ko" | "ja" | "en" | "x"
```

## prep (owners B1 / B2) — `lyricprompter.prep`, `lyricprompter.audio_io`
```python
# audio_io.py (B1)
def to_canonical_wav(src: Path, dst: Path) -> Path            # ffmpeg -> 44.1 kHz stereo PCM24
def load_mono(path: Path, sr: int = 16000) -> np.ndarray       # float32 mono, resampled (soxr)
def save_wav(path: Path, y: np.ndarray, sr: int) -> Path
def duration(path: Path) -> float

# prep/separate.py (B1)
DEFAULT_SEP_MODEL = "bs_roformer_vocals_resurrection_unwa.ckpt"
LEAD_SEP_MODEL = "bs_roformer_karaoke_frazer_becruily.ckpt"
def separate_vocals(mix_wav: Path, out_dir: Path, *, model: str = DEFAULT_SEP_MODEL, lead: bool = False,
                    progress: Progress | None = None) -> dict[str, Path]
    # -> {"vocals": .., "instrumental": .., ["lead": ..]} written as out_dir/<name>.wav,
    #    each exactly as long as mix_wav (pad/trim).

# prep/align.py (B2)
@dataclass
class AlignReport:
    method: str                      # e.g. "mms_fa+gate" or "mms_fa+gate+qwen"
    line_confidence: list[float]     # per Line, 0..1
    flagged: list[int]               # line indices needing review
    timings: dict[str, float]        # seconds spent per stage
    details: dict                    # free-form (model ids, disagreement stats, ...)
def align_lines(lines: list[Line], vocals: np.ndarray, sr: int = 16000, *, refine: str = "auto",
                progress: Progress | None = None) -> AlignReport
    # Fills start/end (+score) on every AlignUnit/Syllable and start/end/confidence/needs_review on
    # Lines in place. Guarantees: no None times on non-empty lines, times non-decreasing in sung order,
    # syllable.start <= syllable.end. refine: "none" | "qwen" | "auto" (qwen for low-confidence lines
    # if the model is available locally).

# prep/vad.py (B2)
def vocal_activity(y: np.ndarray, sr: int = 16000, hop: int = 320, rel_db: float = -35.0,
                   dilate_s: float = 0.3) -> np.ndarray   # bool per hop frame

# prep/pipeline.py (B1)
def prepare_song(audio: Path, lyrics_text: str, *, title: str, artist: str = "", default_lang: str = "auto",
                 song_id: str | None = None, songs_root: Path | None = None,
                 sep_model: str = DEFAULT_SEP_MODEL, lead: bool = False, refine: str = "auto",
                 vocals: Path | None = None, progress: Progress | None = None) -> Path   # song dir

Progress = Callable[[str, float, str], None]    # (stage, percent 0..100, message)
# CLI form prints:  PROGRESS <stage> <percent> <message>
```

## features / tracking / timeline (owner C)
```python
# features/extractor.py
@dataclass(frozen=True)
class FeatureConfig:
    kind: str = "mix"            # "mix" | "vocal"
    sr: int = 16000
    hop: int = 640
    n_fft: int = 2048
    tuning: float = 0.0
    transpose: int = 0
    # ... further tunables with defaults
    def key(self) -> str          # stable short hash for caching
@dataclass
class FeatureFrames:
    feats: np.ndarray            # [n, D] float32
    rms_db: np.ndarray           # [n] float32
class FeatureExtractor:
    def __init__(self, cfg: FeatureConfig): ...
    def reset(self) -> None: ...
    def push(self, samples: np.ndarray) -> FeatureFrames      # 16 kHz mono float32, any length -> 0..k frames
def compute_offline(y: np.ndarray, cfg: FeatureConfig) -> FeatureFrames   # same code path as push()
def load_ref_features(song_dir: Path, stem: str, cfg: FeatureConfig) -> FeatureFrames  # cached in cache/

# tracking/tracker.py
@dataclass
class TrackerParams:
    mode: str = "mix"            # "mix" | "vocal"
    latency_s: float = 0.0       # added (x tempo) to the reported ref_time
    # ... further tunables with defaults
@dataclass
class TrackerState:
    status: str                  # waiting | tracking | coasting | hold | stopped
    ref_time: float
    tempo: float
    confidence: float            # 0..1
    live_time: float
    frame: int
class LiveTracker:
    def __init__(self, ref: FeatureFrames, fps: float, params: TrackerParams, *,
                 ref_active: np.ndarray | None = None, line_starts: list[float] | None = None): ...
    def start(self, ref_time: float = 0.0) -> None
    def push(self, frames: FeatureFrames) -> TrackerState
    def jump(self, ref_time: float) -> None
    def hold(self, on: bool) -> None
    def nudge(self, dt: float) -> None
    def set_transpose(self, semitones: int) -> None
    def stop(self) -> None
    @property
    def state(self) -> TrackerState

# timeline.py
@dataclass
class Cursor:
    line_idx: int                # current or most recently finished line, -1 before the first
    syl_idx: int                 # index into line.syllables(), -1 if none active
    syl_progress: float          # 0..1 within the active syllable
    next_line_idx: int           # -1 if none
    in_gap: bool                 # between lines (instrumental / breath)
class Timeline:
    def __init__(self, lines: list[Line]): ...
    def locate(self, ref_time: float) -> Cursor
    def line_start(self, idx: int) -> float
```

## live / server (owner D)
```python
# live/audio_input.py
def list_input_devices() -> list[dict]     # {index, name, channels, default_samplerate, hostapi}
class DeviceInput:
    def __init__(self, device: int | str | None = None, channels: list[int] | None = None,   # 1-based
                 samplerate: float | None = None, blocksize: int = 0): ...
    def start(self, on_audio: Callable[[np.ndarray], None]) -> None   # delivers 16 kHz mono float32 from a worker thread
    def stop(self) -> None
class FileInput:
    def __init__(self, path: Path, realtime: bool = True, speed: float = 1.0, monitor: bool = False,
                 start_s: float = 0.0): ...
    def start(self, on_audio) -> None
    def stop(self) -> None

# live/session.py
class LiveSession:
    def __init__(self, song_dir: Path, mode: str, source, params: TrackerParams | None = None): ...
    def start(self) -> None
    def stop(self) -> None
    def jump_line(self, idx: int) -> None
    def hold(self, on: bool) -> None
    def nudge(self, dt: float) -> None
    def set_transpose(self, k: int) -> None
    def set_latency(self, s: float) -> None
    def snapshot(self) -> dict
        # {"type":"state","song_id","mode","status","ref_time","tempo","confidence","live_time",
        #  "level_db","line_idx","syl_idx","t": unix_time}
```
WebSocket `/ws` pushes `snapshot()` at ~20 Hz (one broadcaster task, not one loop per client) plus
`{"type":"song","id":...}` when the session's song changes. The prompter page extrapolates
`ref_time + tempo * (now - t)` with requestAnimationFrame and locates the cursor client-side.

## v1.1 — additions made during implementation (all backward-compatible, now part of the contract)

**features**
- `FeatureFrames.groups: tuple[(name, start, stop)]` (e.g. `("chroma", 0, 12)`), helpers `group()`, `slice()`, `empty()`, `concat()`. `set_transpose` / auto key need the chroma group.
- `FeatureConfig.for_mode(mode, tuning=0.0)`: tuned per-mode defaults. The live extractor must use the same config as the reference.

**tracking**
- `TrackerParams.for_mode(mode)`; extra tunables: `lag_comp_s`, `reloc_*`, `relock_*`, `onset_rest_max_s`, `inline_wait_s`, and `auto_key` / `key_*` (automatic key detection, default on).
- `TrackerState`
  - extra fields: `active`, `transpose` (current key, operator- or auto-set), `key_auto`, `key_suggest` (a better-fitting key the search is not sure about).
  - `tempo` is the rate at which the published position advances: 0 while parked (coast limit, hold, waiting, stopped).
- `LiveTracker`
  - `set_latency(s)`, `set_line_starts(starts)` (timing edits during a session).
  - `set_auto_key(on)`; `set_transpose(k)` turns auto key off.
  - counters: `relocations`, `key_switches`.
- `ref_activity(rms_db, fps, rel_db, min_gap_s, min_on_s)`: reference vocal activity for vocal mode.
- `FrameSyncOLTW.anchor(i0, prior, free=n)` / `StepResult.free`: settling steps that may move backwards.

**prep**
- `align_lines(..., options: AlignOptions | None = None, refine_options: RefineOptions | None = None)`, `align.release_models()`.
- `AlignReport.line_confidence: list[float | None]`: None = the line has no lead acoustic unit (backing / symbols only), like `Line.confidence`.
- `vocal_activity(..., floor_db, noise_margin_db, min_rel_db, noise_pct)`.
- `separate_vocals(..., reuse=True)`.
- The `lyricprompter prep` CLI flags are relied on by `server/jobs.py` (guarded by `test_prep_argv_matches_pipeline_cli`).

**live / server**
- `LiveSession(..., prefer_lead, start_line, start_time, deps, on_update)`, `jump_time(t)`, `reload_lines()`, `set_auto_key(on)`.
- `snapshot()` extra fields: `age` (s since the state; prompters extrapolate from arrival time − age, so clocks need not be synced), `syl_progress`, `next_line_idx`, `in_gap`, `frame`, `active`, `stem`, `transpose`, `key_auto`, `key_suggest`, `latency`, `ended`, `error`, `source`. Statuses also include `idle`, `manual`, `error`.
- WS messages
  - `{"type":"song","id","rev"}`: rev increments on every timing edit, so prompters re-fetch.
  - `{"type":"display","lead_time","font_scale","mirror","preview_lines","layout","page_lines","wipe","rev"}`
    (`layout` / `page_lines` / `wipe`: feedback round 2026-10-13, see "Prompter display" below).
- REST
  - `POST /api/session/start {..., transpose, auto_key=true}`: a non-zero `transpose` with auto_key seeds the search; `auto_key=false` makes `transpose` the operator key.
  - `POST /api/session/autokey {on}`.

**v1.1 final-review additions**
- The session's `song_id` and the WS `{"type":"song","id"}` are the **package folder name** (NFC), not `meta.id`. Song ids are NFC-normalised at every API boundary.
- `{"type":"song","rev"}` also increments when song.json changes on disk (e.g. a CLI re-prep), so prompters re-fetch.
- Display settings (`PUT /api/display`) persist in `<songs_dir>/.display.json`.
- `lyricprompter simulate`: `--transpose N` (operator key, auto off), `--transpose N --auto-key` (seed only), `--no-auto-key` (original key).
- `snapshot()["warning"]`: str | None. For example, the song's mix changed on disk during the session, so the operator should restart it.
- `snapshot()["source"]["stalled_s"]`: seconds since the last device callback. `last_error` also reports a device that stopped.
- Song ids: NFC, one path component, `\w[\w.-]*` without `..`. The same rule applies to `prep --id` and the server.

## Timing edits: line edit + undo (feedback round 2026-10-13)
`PUT /api/songs/{id}/lines/{idx}`, body: exactly one of `{"start": s}`, `{"shift": ds}`, `{"end": s}` (finite seconds).
An edit always succeeds except at either end of the song: lines keep their order and the neighbours make room.
- `start` / `shift`: the line moves rigidly (syllables and align units keep their rhythm), anywhere in `[0, duration)`.
  A line without timing gets it from `start`: end = `min(start + max(1 s, 0.25 s × syllables), next timed start)`,
  syllables spread evenly, each align unit takes its syllable's span. `shift` / `end` on an untimed line: 422.
- `end`: the line's times are scaled linearly from `[start, old end]` onto `[start, end]`; needs
  `end >= start + 0.1 s` and `end <= duration`.
- Pushes: a following timed line that would start less than `MIN_LINE_DUR` (0.25 s) after the previous line's new
  start (for an `end` edit: before the new end) moves rigidly later, cascading only as far as needed; preceding lines
  symmetrically earlier. A pushed line gets flag `pushed` and `needs_review: true` (its start is a placeholder now).
- Fit: every line whose start or whose next timed line's start changed is fitted before that next start (the last
  line before `duration`). If all its onsets (line / syllable / unit starts) are at least 0.05 s before it, only the
  tails past it are clipped; otherwise all its times are scaled into `[start, next start]`. A line never ends after
  the next timed line starts.
- Flags: the edited line gets `manual_timing`, `needs_review: false` and loses `pushed`. A line that is only
  clipped or squeezed keeps its flags.
- Errors (nothing saved): 422 for a malformed body, a start outside `[0, duration)`, or an edit that would push a
  line before 0 s or to / past `duration` (the message names the lines); 404 for an unknown line.
- Response: `{"song_id", "shift", "clamped": false, "pushed": [idx, ...], "rev", "undo_left", "lines": {"idx": Line}}`.
  `shift` is how far the edited line's start moved (0 for an `end` edit or a first timing); `clamped` is always false
  (kept for older clients); `lines` holds every modified line. `lines` is `{}` when the edit changed nothing: then
  nothing is saved, the rev stays and no undo step is added. A saved edit bumps the song's rev (every screen reloads
  the song) and hands a running session of that song the new lines (`LiveSession.reload_lines`).

`POST /api/songs/{id}/undo` reverts the song's last saved timing edit.
- History: in the server's memory, per song, up to `UNDO_DEPTH` (100) edits, lost on a restart. A step keeps the
  lines the edit changed (as they were) and a fingerprint of the song's lines right after the edit.
- 409 `"Nothing to undo"`. 409 as well when the lines in song.json differ from the state right after that edit
  (e.g. a re-prep); the song's history is then dropped. A change of the song's meta only (title, style) does not
  block undo. An edit saved after such an on-disk change starts a fresh history.
- Response: `{"song_id", "undone": "line 31 start" | "line 31 end", "undo_left", "rev", "lines": {"idx": Line}}`
  with the restored lines. Bumps the rev and reloads a running session like an edit.

Python (`server/app.py`): `move_line(song, idx, *, start=None, shift=None, end=None) -> LineMove(shift, touched,
pushed, before)` edits a `Song` in place (IndexError / ValueError, song unchanged; `before` = the touched lines as
dicts, for undo). Constants: `MIN_LINE_DUR`, `FIT_MARGIN`, `MIN_END_DUR`, `UNTIMED_SYL_DUR`, `UNTIMED_MIN_DUR`,
`UNDO_DEPTH`.

## Tracking style: classic / pop (feedback round 2026-10-13)
`style` = `"classic"` | `"pop"` (UI labels "Classical — flexible tempo" / "Pop & band — steady tempo"); behaviour
and evaluation in [eval/TRACKING.md §8](eval/TRACKING.md).
- Song: `SongMeta.style` (song.json `meta.style`, default `"classic"`; a song.json without it loads as classic).
  `GET /api/songs` rows and `GET /api/songs/{id}` (top level, besides `meta.style`) include `style`.
- Prep: `lyricprompter prep --style classic|pop`; without it a re-prep keeps the package's style (read when the new
  song.json is written, so a style saved meanwhile survives), a new song is classic. `prepare_song(..., style=None)`.
  Upload: `POST /api/songs` form field `options` JSON may hold `"style"` (422 for another value) and is passed on.
- `PUT /api/songs/{id}/style {"style"}`: saves song.json (only when it changes: a new song rev, so every screen
  re-fetches the song) and switches a running session of that song.
  Response `{"song_id", "style", "rev", "session": bool}` (`session`: a live session was switched).
- `POST /api/session/start {..., "style": null}`: null / absent = the song's style.
- `POST /api/session/style {"style"}`: switches the running session only (song.json unchanged; 409 without a
  session); returns the state message like the other session controls. `snapshot()["style"]`: the session's style.
- Python: `TrackerParams.style` and `pop_*` tunables (read only by pop); `TrackerParams.for_mode(mode, style="classic")`;
  `tracking.tracker.style_fields(mode, style)` = the preset fields a style switch changes (today only `style`);
  `LiveTracker.set_style(style)` (live; other params / overrides and the latency stay), counters `snaps`,
  `reanchors`; `LiveSession(..., style=None)` (None: the song's style), `LiveSession.set_style(style)`,
  `live.session.with_style(params, mode, style)`; `text.model.STYLES`. ValueError for an unknown style.
- `lyricprompter simulate --style classic|pop` (default: the song's style); `lyricprompter eval-track --styles
  classic,pop` (rows get `style`; with several styles the summary keys are `"<style>/<mode>"`), new pairs
  `steady_<song>` / `pop_<song>` (kind `"pop"`, `Pair.pop: PopSpec`, `make_pop_live`).

## Prompter display: layout / page_lines / wipe (feedback round 2026-10-13)
New `DisplaySettings` / `PUT /api/display` keys, pushed to every prompter (a screen can override each one locally,
like the other display keys; `common.js DISPLAY_KEYS`):
- `layout`: `"line"` (default; "Line by line", one line at a time) | `"page"` ("Multiple lines", a page of a stanza).
- `page_lines`: 1–8 (default 4), "Max lines" per page in the page layout.
- `wipe`: bool (default true), "Syllable highlight": the per-syllable karaoke wipe on the focus line. Off: the
  focus line is plain white (the page layout keeps its accent bar).
- 422 for another value. A `.display.json` saved before these keys existed loads with their defaults.
- Pages (`common.js paginate(lines, maxLines)` -> `[first, end)` line ranges): consecutive lines of one stanza
  (`Line.stanza`, and the same `Line.section`) form a block; a block longer than `page_lines` is split into
  `ceil(n / page_lines)` pages of balanced size, larger first (11 lines at max 5 -> 4/4/3). Stanzas are never merged.
  The page of the focus line (lead time as in the line layout) is shown; `preview_lines` then means that many
  first lines of the next page.
- Prompter keys: L toggles the layout, W the syllable highlight (this screen's override, like M).
