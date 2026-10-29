"""Console settings kept across server restarts (`projects/.settings.json`): the OSC targets, the default
target, OSC arming, what a jump does with the cues it skips, the catch-up window and the tracker's input
latency compensation.

`load_settings` never fails: a missing or unreadable file gives the defaults, and each invalid field falls
back on its own, with a logged warning, so one bad value does not lose the others. A target entry that is
not valid keeps its name and its valid fields but is disabled, and the default target keeps its name (a
disabled stand-in when it is gone): cues are never sent to another device than the one the file named.
After such a lossy load the file is copied aside (`.settings.json.bad-<hash>`), so the next save does not
lose what was in it. `validate_settings` is strict (ValueError with a user-readable message) and is what
the server applies to edits. `save_settings` replaces the file atomically; a failed save is logged, never
raised (the settings in memory still apply, which is what matters during a show).
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cues import MAX_DECL_LATENCY_MS, TARGET_NAME_RE, CueIssue, TargetDecl, parse_message
from .osc import Target

log = logging.getLogger(__name__)

SETTINGS_FILE = ".settings.json"       # in the projects folder
SKIP_POLICIES = ("skip", "latest", "all")
TARGET_LATENCY_MS = (0.0, MAX_DECL_LATENCY_MS)
CATCHUP_S = (0.1, 30.0)
INPUT_LATENCY_MS = (-500.0, 2000.0)


class SettingsError(ValueError):
    """Invalid settings (the message is user-readable)."""


def default_targets() -> list[Target]:
    return [Target("main", "127.0.0.1", 9000)]


@dataclass
class Settings:
    targets: list[Target] = field(default_factory=default_targets)
    default_target: str = "main"
    armed: bool = True                   # OSC output on (off = dry run: cues are logged, not sent)
    skip_policy: str = "latest"          # cues jumped over: skip | latest (per target+address) | all
    catchup_s: float = 1.0               # forward moves up to this send every crossed cue (late)
    input_latency_ms: float = 0.0        # tracker latency compensation (audio input + processing)

    def target(self, name: str | None) -> Target | None:
        return next((t for t in self.targets if t.name == name), None)

    def copy(self) -> Settings:
        return dataclasses.replace(self, targets=[dataclasses.replace(t) for t in self.targets])

    def to_dict(self) -> dict[str, Any]:
        return {"targets": [dataclasses.asdict(t) for t in self.targets], "default_target": self.default_target,
                "armed": self.armed, "skip_policy": self.skip_policy, "catchup_s": self.catchup_s,
                "input_latency_ms": self.input_latency_ms}


def settings_path(projects_dir: Path) -> Path:
    return Path(projects_dir) / SETTINGS_FILE


# ----------------------------------------------------------------------------------------- validation


def _number(v: Any, what: str, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SettingsError(f"{what} must be a number, got {v!r}")
    try:
        f = float(v)
    except OverflowError:                                  # an int too large for a float
        raise SettingsError(f"{what} must be between {lo:g} and {hi:g}") from None
    if not math.isfinite(f):
        raise SettingsError(f"{what} must be a number, got {v!r}")
    if not lo <= f <= hi:
        raise SettingsError(f"{what} must be between {lo:g} and {hi:g}, got {f:g}")
    return f


def _bool(v: Any, what: str) -> bool:
    if not isinstance(v, bool):
        raise SettingsError(f"{what} must be true or false, got {v!r}")
    return v


def _target_name(v: Any, what: str = "target name") -> str:
    if not isinstance(v, str) or not TARGET_NAME_RE.fullmatch(v):
        raise SettingsError(f"{what} {v!r} is not valid: use letters, digits, _ . - (starting with a letter or _)")
    return v


def _skip_policy(v: Any) -> str:
    if v not in SKIP_POLICIES:
        raise SettingsError(f"skip policy must be one of {', '.join(SKIP_POLICIES)}, got {v!r}")
    return v


def check_target(t: Target) -> Target:
    """A validated, normalised copy of a target (host stripped, numbers as int / float); ValueError."""
    name = _target_name(t.name)
    if not isinstance(t.host, str) or not t.host.strip():
        raise SettingsError(f"target {name!r}: the host is empty")
    host = t.host.strip()
    if any(c.isspace() for c in host):
        raise SettingsError(f"target {name!r}: the host {host!r} contains spaces")
    port = t.port
    if isinstance(port, float) and port.is_integer():
        port = int(port)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise SettingsError(f"target {name!r}: the port must be a whole number 1..65535, got {t.port!r}")
    latency = _number(t.latency_ms, f"target {name!r}: latency (ms)", *TARGET_LATENCY_MS)
    enabled = _bool(t.enabled, f"target {name!r}: enabled")
    if not isinstance(t.test_message, str):
        raise SettingsError(f"target {name!r}: the test message must be text")
    if t.test_message.strip():
        try:
            other, *_ = parse_message(t.test_message)
        except ValueError as e:
            raise SettingsError(f"target {name!r}: test message: {e}") from None
        if other not in (None, name):
            raise SettingsError(f"target {name!r}: the test message names another target ({other!r})")
    return Target(name, host, port, latency, enabled, t.test_message)


_FIELDS: dict[str, Callable[[Any], Any]] = {
    "default_target": lambda v: _target_name(v, "default target"),
    "armed": lambda v: _bool(v, "armed"),
    "skip_policy": _skip_policy,
    "catchup_s": lambda v: _number(v, "catch-up window (s)", *CATCHUP_S),
    "input_latency_ms": lambda v: _number(v, "input latency (ms)", *INPUT_LATENCY_MS),
}


def validate_settings(s: Settings) -> Settings:
    """A validated, normalised copy of `s`; ValueError (user-readable) for the first problem found."""
    if not isinstance(s.targets, list) or not s.targets:
        raise SettingsError("at least one OSC target is needed")
    targets: list[Target] = []
    for t in s.targets:
        if not isinstance(t, Target):
            raise SettingsError(f"not a target: {t!r}")
        t = check_target(t)
        if any(o.name == t.name for o in targets):
            raise SettingsError(f"target name {t.name!r} is used twice")
        targets.append(t)
    values = {k: check(getattr(s, k)) for k, check in _FIELDS.items()}
    if not any(t.name == values["default_target"] for t in targets):
        raise SettingsError(f"the default target {values['default_target']!r} is not one of the targets")
    return Settings(targets=targets, **values)


def target_from_dict(d: Any) -> Target:
    """A Target from JSON data (missing fields take the Target defaults); ValueError when malformed."""
    if not isinstance(d, dict):
        raise SettingsError(f"a target is an object with name, host, port, ..., got {d!r}")
    known = {f.name for f in dataclasses.fields(Target)}
    if "name" not in d:
        raise SettingsError("a target needs a name")
    return check_target(Target(**{k: v for k, v in d.items() if k in known}))


def settings_from_dict(data: Any, base: Settings | None = None) -> Settings:
    """Settings from JSON data (a full or partial dict: missing keys keep `base`'s values, unknown keys
    are ignored), validated; ValueError when anything is invalid."""
    if not isinstance(data, dict):
        raise SettingsError("settings must be a JSON object")
    s = (base or Settings()).copy()
    if "targets" in data:
        if not isinstance(data["targets"], list):
            raise SettingsError("targets must be a list")
        s.targets = [target_from_dict(d) for d in data["targets"]]
    for k in _FIELDS:
        if k in data:
            setattr(s, k, data[k])
    return validate_settings(s)


# ----------------------------------------------------------------------------------------- persistence

_BAD_ERRORS = (TypeError, ValueError, ArithmeticError, RecursionError)     # what malformed JSON data can raise


def _stand_in(d: Any) -> Target | None:
    """A disabled stand-in for a saved target entry that is not valid: its name and each of its fields that
    is valid on its own (the others take the defaults: 127.0.0.1, 9000, 0 ms); None without a valid name."""
    name = d.get("name") if isinstance(d, dict) else None
    if not isinstance(name, str) or not TARGET_NAME_RE.fullmatch(name):
        return None
    t = Target(name, enabled=False)
    for k in ("host", "port", "latency_ms", "test_message"):
        if k in d:
            with contextlib.suppress(*_BAD_ERRORS):
                t = check_target(dataclasses.replace(t, **{k: d[k]}))
    return t


def load_settings(path: Path, warnings: list[str] | None = None) -> Settings:
    """Settings saved by `save_settings`. Missing file -> defaults; unreadable file or invalid fields ->
    defaults for what is wrong, an invalid target entry -> disabled (see the module doc). Each problem is
    logged and, when `warnings` is given, appended to it (user-readable). The result always passes
    `validate_settings`."""
    path = Path(path)
    s = Settings()
    found: list[str] = []

    def warn(msg: str) -> None:
        log.warning("%s: %s", path, msg)
        found.append(msg)

    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return s
    except OSError as e:
        warn(f"cannot read the saved settings ({e}): using the defaults")
        data = None
    if data is not None:
        _load_into(s, data, warn)
        if found:
            backup = path.with_name(f"{path.name}.bad-{hashlib.sha1(data).hexdigest()[:10]}")
            try:
                if not backup.exists():
                    backup.write_bytes(data)
                warn(f"the settings file is kept as it was in {backup.name}")
            except OSError as e:
                warn(f"cannot keep a copy of the settings file as {backup.name}: {e}")
    if warnings is not None:
        warnings.extend(found)
    return validate_settings(s)


def _load_into(s: Settings, data: bytes, warn: Callable[[str], None]) -> None:
    """The file's settings into the defaults `s`, field by field (`warn` for each problem)."""
    try:
        raw = json.loads(data.decode("utf-8-sig"))
    except _BAD_ERRORS as e:
        warn(f"ignoring the saved settings, they are not valid JSON ({e}): using the defaults")
        return
    if not isinstance(raw, dict):
        warn("ignoring the saved settings, they are not a JSON object: using the defaults")
        return
    if "targets" in raw:
        targets: list[Target] = []
        items = raw["targets"] if isinstance(raw["targets"], list) else []
        if not isinstance(raw["targets"], list):
            warn("the saved targets are not a list: using the default target")
        for item in items:
            try:
                t = target_from_dict(item)
            except _BAD_ERRORS as e:
                t = _stand_in(item)
                if t is None:
                    warn(f"ignoring a saved target: {e}")
                    continue
                warn(f"{e}: the target is kept but disabled until you fix it in OSC targets")
            if any(o.name == t.name for o in targets):
                warn(f"ignoring a second target named {t.name!r}")
                continue
            targets.append(t)
        if targets:
            s.targets = targets
        elif items:
            warn("no valid target: using the default target")
    for k, check in _FIELDS.items():
        if k in raw:
            try:
                setattr(s, k, check(raw[k]))
            except _BAD_ERRORS as e:
                warn(f"{e} (using the default {getattr(s, k)!r})")
    if s.target(s.default_target) is None:
        if "default_target" in raw and raw["default_target"] == s.default_target:
            # the file names this default: never send its cues to another device instead
            warn(f"the default target {s.default_target!r} is not one of the saved targets: it is added, "
                 "disabled, so cues without a target are not sent until you fix it in OSC targets")
            s.targets.append(Target(s.default_target, enabled=False))
        else:                                              # none (or an invalid one) saved
            warn(f"the default target {s.default_target!r} is not one of the targets: using {s.targets[0].name!r}")
            s.default_target = s.targets[0].name


def save_settings(path: Path, s: Settings) -> bool:
    """Write the settings atomically (temporary file + rename). Returns False (and logs) when it fails."""
    path = Path(path)
    data = json.dumps(s.to_dict(), ensure_ascii=False, indent=2) + "\n"
    tmp: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = None
        return True
    except (OSError, ValueError) as e:
        log.warning("cannot save the settings to %s: %s", path, e)
        return False
    finally:
        if tmp is not None:
            Path(tmp).unlink(missing_ok=True)


# ----------------------------------------------------------------------------------------- helpers


def merge_declared(s: Settings, decls: Iterable[TargetDecl]) -> tuple[Settings, list[str]]:
    """Add the cue file's declared targets that the settings do not have yet (name, host, port, latency);
    existing targets are not changed. Returns (new settings, names added). An invalid declaration is
    skipped with a logged warning."""
    out = s.copy()
    added: list[str] = []
    for d in decls:
        if out.target(d.name) is not None:
            continue
        try:
            t = check_target(Target(d.name, d.host, d.port, float(d.latency_ms or 0.0)))
        except (TypeError, ValueError) as e:
            log.warning("ignoring the declared target %r: %s", d.name, e)
            continue
        out.targets.append(t)
        added.append(t.name)
    return out, added


def _hostport(host: str, port: int) -> str:
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def declared_conflicts(s: Settings, decls: Iterable[TargetDecl]) -> list[CueIssue]:
    """A warning for each cue file @target whose host, port or latency (when declared) differs from the
    settings target of that name: the settings win (`merge_declared` does not change existing targets), so
    the console sends somewhere else than the cue file says."""
    out: list[CueIssue] = []
    for d in decls:
        t = s.target(d.name)
        if t is None:
            continue
        same = (d.host.lower() == t.host.lower() and d.port == t.port
                and (d.latency_ms is None or abs(d.latency_ms - t.latency_ms) < 1e-6))
        if not same:
            declared, used = _hostport(d.host, d.port), _hostport(t.host, t.port)
            if d.latency_ms is not None:
                declared, used = f"{declared} latency {d.latency_ms:g} ms", f"{used} latency {t.latency_ms:g} ms"
            out.append(CueIssue(d.line, f"target {d.name!r} is declared as {declared}, but the console sends to "
                                f"{used} (the OSC targets setting wins): edit the target in OSC targets to use "
                                "this declaration", "warning"))
    return out


def latency_s(s: Settings, cue: Any) -> float:
    """Output latency in seconds of the cue's target (`cue.target`, or the default target when None);
    0 for an unknown target."""
    t = s.target(getattr(cue, "target", None) or s.default_target)
    return max(0.0, float(t.latency_ms)) / 1000.0 if t is not None else 0.0
