from __future__ import annotations

import dataclasses
import json
import logging
import os

import pytest

from lyricprompter_oscsender.cues import Cue, TargetDecl, parse_cues
from lyricprompter_oscsender.osc import Target
from lyricprompter_oscsender.settings import (
    SETTINGS_FILE,
    SKIP_POLICIES,
    Settings,
    SettingsError,
    declared_conflicts,
    latency_s,
    load_settings,
    merge_declared,
    save_settings,
    settings_from_dict,
    settings_path,
    validate_settings,
)


def full() -> Settings:
    return Settings(targets=[Target("main", "127.0.0.1", 9000),
                             Target("lights", "192.168.0.50", 8000, 12.5, False, "/cue/go 1"),
                             Target("v6", "::1", 7000, 80.0)],
                    default_target="lights", armed=False, skip_policy="all", catchup_s=2.5,
                    input_latency_ms=-40.0)


# ----------------------------------------------------------------------------------------- defaults


def test_defaults():
    s = Settings()
    assert s.targets == [Target("main", "127.0.0.1", 9000)]
    assert (s.default_target, s.armed, s.skip_policy, s.catchup_s, s.input_latency_ms) == \
        ("main", True, "latest", 1.0, 0.0)
    assert validate_settings(s) == s
    assert SKIP_POLICIES == ("skip", "latest", "all")
    assert Settings().targets is not Settings().targets             # no shared mutable default


def test_settings_path():
    assert settings_path(os.path.join("x", "projects")).name == SETTINGS_FILE == ".settings.json"


def test_copy_is_deep_for_targets():
    s = full()
    c = s.copy()
    c.targets[0].port = 1
    c.targets.append(Target("x"))
    assert s.targets[0].port == 9000 and len(s.targets) == 3


def test_target_lookup_and_to_dict():
    s = full()
    assert s.target("lights").port == 8000 and s.target("nope") is None and s.target(None) is None
    d = json.loads(json.dumps(s.to_dict()))
    assert d["targets"][1] == {"name": "lights", "host": "192.168.0.50", "port": 8000, "latency_ms": 12.5,
                               "enabled": False, "test_message": "/cue/go 1"}
    assert {k: d[k] for k in d if k != "targets"} == {"default_target": "lights", "armed": False,
                                                      "skip_policy": "all", "catchup_s": 2.5,
                                                      "input_latency_ms": -40.0}


# ----------------------------------------------------------------------------------------- validation


def test_validate_normalises():
    s = Settings(targets=[Target("main", "  host.local ", 9000.0, 5)], catchup_s=2, input_latency_ms=10)
    v = validate_settings(s)
    assert v.targets[0] == Target("main", "host.local", 9000, 5.0)
    assert type(v.targets[0].port) is int and type(v.targets[0].latency_ms) is float
    assert type(v.catchup_s) is float and type(v.input_latency_ms) is float
    assert v is not s and v.targets[0] is not s.targets[0]
    assert validate_settings(full()) == full()


def _with(**kw) -> Settings:
    return dataclasses.replace(full(), **kw)


def _target(**kw) -> Settings:
    return Settings(targets=[dataclasses.replace(Target("main"), **kw)])


@pytest.mark.parametrize("settings, fragment", [
    (_with(targets=[]), "at least one"),
    (_with(targets="main"), "at least one"),
    (_with(targets=[Target("main"), Target("main", port=1)], default_target="main"), "used twice"),
    (_with(targets=[Target("main"), "x"], default_target="main"), "not a target"),
    (_target(name=""), "not valid"),
    (_target(name="1x"), "not valid"),
    (_target(name="a b"), "not valid"),
    (_target(name=None), "not valid"),
    (_target(host=""), "host is empty"),
    (_target(host="   "), "host is empty"),
    (_target(host="a b"), "contains spaces"),
    (_target(host=None), "host is empty"),
    (_target(port=0), "port"),
    (_target(port=65536), "port"),
    (_target(port=True), "port"),
    (_target(port="9000"), "port"),
    (_target(port=9000.5), "port"),
    (_target(latency_ms=-1), "between 0 and 5000"),
    (_target(latency_ms=5000.1), "between 0 and 5000"),
    (_target(latency_ms=float("nan")), "must be a number"),
    (_target(latency_ms=None), "must be a number"),
    (_target(enabled="yes"), "true or false"),
    (_target(test_message=5), "test message must be text"),
    (_target(test_message="no address"), "test message"),
    (_target(test_message="other /a"), "names another target"),
    (_with(default_target="nope"), "not one of the targets"),
    (_with(default_target=""), "not valid"),
    (_with(armed=1), "true or false"),
    (_with(skip_policy="none"), "skip policy"),
    (_with(catchup_s=0.05), "between 0.1 and 30"),
    (_with(catchup_s=31), "between 0.1 and 30"),
    (_with(catchup_s=float("inf")), "must be a number"),
    (_with(input_latency_ms=-501), "between -500 and 2000"),
    (_with(input_latency_ms=2001), "between -500 and 2000"),
    (_with(input_latency_ms="0"), "must be a number"),
])
def test_validate_errors(settings, fragment):
    with pytest.raises(SettingsError, match=fragment):
        validate_settings(settings)
    assert issubclass(SettingsError, ValueError)            # SPEC: validate_settings raises ValueError


def test_validate_limits_inclusive():
    validate_settings(_with(catchup_s=0.1, input_latency_ms=-500))
    validate_settings(_with(catchup_s=30, input_latency_ms=2000))
    validate_settings(Settings(targets=[Target("main", "h", 1, 0), Target("b", "h", 65535, 5000)]))
    validate_settings(Settings(targets=[Target("main", test_message="main /x 1")]))   # its own name is fine


# ----------------------------------------------------------------------------------------- from dict


def test_settings_from_dict_partial():
    base = full()
    s = settings_from_dict({"armed": True, "catchup_s": 3}, base)
    assert s.armed is True and s.catchup_s == 3.0 and s.targets == base.targets and base.armed is False
    s = settings_from_dict({"targets": [{"name": "solo", "host": "h", "port": 1}], "default_target": "solo",
                            "targets_status": {}, "unknown": 1})
    assert s.targets == [Target("solo", "h", 1)] and s.default_target == "solo"
    assert settings_from_dict(full().to_dict()) == full()


@pytest.mark.parametrize("data, fragment", [
    ([], "JSON object"),
    ({"targets": {}}, "must be a list"),
    ({"targets": ["x"]}, "is an object"),
    ({"targets": [{"host": "h"}]}, "needs a name"),
    ({"targets": [{"name": "main", "port": "x"}]}, "port"),
    ({"targets": [{"name": "other"}]}, "default target 'main' is not one"),
    ({"skip_policy": "nope"}, "skip policy"),
])
def test_settings_from_dict_errors(data, fragment):
    with pytest.raises(ValueError, match=fragment):
        settings_from_dict(data)


# ----------------------------------------------------------------------------------------- load / save


def test_save_load_round_trip(tmp_path):
    p = tmp_path / "projects" / SETTINGS_FILE                  # parent folder is created
    assert save_settings(p, full()) is True
    assert load_settings(p) == full()
    assert json.loads(p.read_text(encoding="utf-8"))["default_target"] == "lights"
    assert [f.name for f in p.parent.iterdir()] == [SETTINGS_FILE]          # no temporary files left


def test_save_overwrites_atomically(tmp_path):
    p = tmp_path / SETTINGS_FILE
    save_settings(p, Settings())
    save_settings(p, full())
    assert load_settings(p) == full()
    assert sorted(f.name for f in tmp_path.iterdir()) == [SETTINGS_FILE]


def test_save_failure_is_logged_not_raised(tmp_path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with caplog.at_level(logging.WARNING):
        assert save_settings(blocker / SETTINGS_FILE, Settings()) is False
    assert "cannot save the settings" in caplog.text


def test_load_missing_file_is_silent(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert load_settings(tmp_path / "missing.json") == Settings()
    assert caplog.text == ""


@pytest.mark.parametrize("content", ["", "{", "[1, 2]", "null", "\xff\xfe"])
def test_load_unreadable_gives_defaults(tmp_path, caplog, content):
    p = tmp_path / SETTINGS_FILE
    p.write_bytes(content.encode("latin-1"))
    with caplog.at_level(logging.WARNING):
        assert load_settings(p) == Settings()
    assert "settings" in caplog.text


def test_load_directory_gives_defaults(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert load_settings(tmp_path) == Settings()
    assert caplog.text


def test_load_invalid_fields_fall_back_one_by_one(tmp_path, caplog):
    p = tmp_path / SETTINGS_FILE
    p.write_text(json.dumps({
        "targets": [{"name": "a", "host": "h", "port": 1, "latency_ms": 9999},       # invalid latency
                    {"name": "b", "host": "h", "port": 2, "extra": "ignored"},
                    {"name": "b", "host": "other", "port": 3},                        # duplicate name
                    "junk", {"host": "nameless"}],
        "default_target": "a",                                                       # dropped target
        "armed": "yes", "skip_policy": "all", "catchup_s": 100, "input_latency_ms": 25,
    }), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        s = load_settings(p)
    assert s.targets == [Target("a", "h", 1, 0.0, False), Target("b", "h", 2)]      # a: kept, disabled
    assert s.default_target == "a" and s.armed is True and s.catchup_s == 1.0
    assert s.skip_policy == "all" and s.input_latency_ms == 25.0
    assert validate_settings(s) == s
    for fragment in ("latency", "disabled", "second target", "armed", "catch-up"):
        assert fragment in caplog.text


def test_load_no_valid_target_keeps_default(tmp_path, caplog):
    p = tmp_path / SETTINGS_FILE
    p.write_text(json.dumps({"targets": [{"name": "1bad"}], "default_target": "main", "armed": False}))
    with caplog.at_level(logging.WARNING):
        s = load_settings(p)
    assert s == Settings(armed=False)
    p.write_text(json.dumps({"targets": "x"}))
    assert load_settings(p) == Settings()


def _bad_main(tmp_path, port="9000"):
    p = tmp_path / SETTINGS_FILE
    p.write_text(json.dumps({"targets": [{"name": "lights", "host": "192.168.0.50", "port": 8000},
                                         {"name": "main", "host": "192.168.0.60", "port": port, "latency_ms": 20}],
                             "default_target": "main", "armed": True}), encoding="utf-8")
    return p


def test_load_never_moves_the_default_target_to_another_device(tmp_path, caplog):
    p = _bad_main(tmp_path, port="9001")                       # hand edit: the port written as a string
    warnings: list[str] = []
    with caplog.at_level(logging.WARNING):
        s = load_settings(p, warnings)
    assert s.default_target == "main"                                       # not "lights"
    assert s.targets == [Target("lights", "192.168.0.50", 8000),
                         Target("main", "192.168.0.60", 9000, 20.0, enabled=False)]     # its readable fields
    assert validate_settings(s) == s
    assert any("'main'" in w and "disabled" in w for w in warnings)
    assert all(w in caplog.text for w in warnings)


def test_load_default_target_missing_from_the_targets_is_added_disabled(tmp_path):
    p = tmp_path / SETTINGS_FILE
    p.write_text(json.dumps({"targets": [{"name": "lights", "host": "10.0.0.5", "port": 8000}],
                             "default_target": "mian"}))
    warnings: list[str] = []
    s = load_settings(p, warnings)
    assert s.default_target == "mian"
    assert s.targets == [Target("lights", "10.0.0.5", 8000), Target("mian", enabled=False)]
    assert any("cues without a target are not sent" in w for w in warnings)
    p.write_text(json.dumps({"targets": [{"name": "lights", "host": "10.0.0.5", "port": 8000}]}))
    assert load_settings(p).default_target == "lights"        # no default saved: the first target, as before


def test_lossy_load_keeps_a_copy_of_the_file(tmp_path):
    p = _bad_main(tmp_path)
    original = p.read_bytes()
    warnings: list[str] = []
    s = load_settings(p, warnings)
    backups = sorted(tmp_path.glob(f"{SETTINGS_FILE}.bad-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert any(backups[0].name in w for w in warnings)
    load_settings(p)                                           # the same content: no second copy
    assert len(list(tmp_path.glob(f"{SETTINGS_FILE}.bad-*"))) == 1
    save_settings(p, s)                                        # the next save cannot lose the original
    assert load_settings(p, w2 := []) == s and w2 == []
    assert backups[0].read_bytes() == original
    p.write_text("{", encoding="utf-8")                        # unreadable: a copy too
    load_settings(p)
    assert len(list(tmp_path.glob(f"{SETTINGS_FILE}.bad-*"))) == 2


def test_clean_or_missing_load_writes_no_copy(tmp_path):
    p = tmp_path / SETTINGS_FILE
    warnings: list[str] = []
    load_settings(p, warnings)
    save_settings(p, full())
    load_settings(p, warnings)
    assert warnings == [] and [f.name for f in tmp_path.iterdir()] == [SETTINGS_FILE]


def test_load_accepts_a_bom(tmp_path):
    p = tmp_path / SETTINGS_FILE
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps(full().to_dict()).encode("utf-8"))
    warnings: list[str] = []
    assert load_settings(p, warnings) == full() and warnings == []


@pytest.mark.parametrize("content", [
    '{"catchup_s": ' + "1" * 400 + "}", '{"input_latency_ms": -' + "1" * 400 + "}",
    '{"targets": [{"name": "main", "host": "h", "port": 1, "latency_ms": ' + "1" * 400 + "}]}",
    "[" * 100000 + "]" * 100000, '{"a": ' + "[" * 100000 + "]" * 100000 + "}",
])
def test_load_never_raises_for_huge_or_deep_values(tmp_path, content):
    p = tmp_path / SETTINGS_FILE
    p.write_text(content, encoding="utf-8")
    s = load_settings(p)
    assert validate_settings(s) == s
    if '"targets"' in content:
        assert s.targets == [Target("main", "h", 1, 0.0, enabled=False)]


def test_huge_numbers_are_settings_errors():
    for data in ({"catchup_s": 10 ** 400}, {"input_latency_ms": -10 ** 400},
                 {"targets": [{"name": "a", "host": "h", "port": 1, "latency_ms": 10 ** 400}]},
                 {"targets": [{"name": "a", "host": "h", "port": 10 ** 400}]}):
        with pytest.raises(SettingsError):
            settings_from_dict(data)


def test_save_fsyncs_before_the_rename(tmp_path, monkeypatch):
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (calls.append("replace"), real_replace(a, b))[1])
    assert save_settings(tmp_path / SETTINGS_FILE, full()) is True
    assert calls == ["fsync", "replace"]


# ----------------------------------------------------------------------------------------- merge / latency


def test_merge_declared():
    s = full()
    decls = parse_cues("@target lights 10.0.0.9:1 latency=50\n@target video 127.0.0.1:7000 latency=80ms\n"
                       "@target v6b [::1]:9001\n@target video 1.1.1.1:1\n").targets
    out, added = merge_declared(s, decls)
    assert added == ["video", "v6b"]
    assert out.target("lights") == s.target("lights")                    # existing targets unchanged
    assert out.target("video") == Target("video", "127.0.0.1", 7000, 80.0)
    assert out.target("v6b") == Target("v6b", "::1", 9001, 0.0)
    assert len(s.targets) == 3 and out is not s                          # input not modified
    assert validate_settings(out) == out
    out2, added2 = merge_declared(out, decls)
    assert added2 == [] and out2 == out


def test_merge_declared_skips_invalid(caplog):
    with caplog.at_level(logging.WARNING):
        out, added = merge_declared(Settings(), [TargetDecl("x", "h", 70000), TargetDecl("y", "h", 1, 6000.0),
                                                 TargetDecl("z", "h", 2)])
    assert added == ["z"] and [t.name for t in out.targets] == ["main", "z"]
    assert "ignoring the declared target 'x'" in caplog.text


def test_declared_conflicts():
    s = full()                                    # lights 192.168.0.50:8000 12.5 ms, v6 ::1:7000 80 ms
    decls = parse_cues("@target lights 10.1.1.20:8000 latency=40ms\n@target v6 [::1]:7000\n"
                       "@target main 127.0.0.1:9000 latency=0\n@target video 1.2.3.4:5\n"
                       "@target V6 [::1]:7001\n@target v6x [::1]:7000 latency=79\n").targets
    issues = declared_conflicts(s, decls)
    assert [(i.line, i.severity) for i in issues] == [(1, "warning")]
    assert "192.168.0.50:8000" in issues[0].message and "10.1.1.20:8000" in issues[0].message
    assert "OSC targets" in issues[0].message
    assert [i.line for i in declared_conflicts(s, parse_cues("@target v6 [::1]:7001 latency=80\n"
                                                             "@target lights 192.168.0.50:8000 latency=12\n"
                                                             "@target LIGHTS 1.1.1.1:1\n").targets)] == [1, 2]
    assert declared_conflicts(s, parse_cues("@target Main localhost:9000\n@target main 127.0.0.1:9000\n"
                                            "@target lights 192.168.0.50:8000 latency=12.5\n").targets) == []


def test_latency_s():
    s = full()                                                           # default target: lights (12.5 ms)
    assert latency_s(s, Cue(1, 0.0, "/a")) == pytest.approx(0.0125)
    assert latency_s(s, Cue(1, 0.0, "/a", target="v6")) == pytest.approx(0.08)
    assert latency_s(s, Cue(1, 0.0, "/a", target="main")) == 0.0
    assert latency_s(s, Cue(1, 0.0, "/a", target="unknown")) == 0.0
    s.default_target = "gone"
    assert latency_s(s, Cue(1, 0.0, "/a")) == 0.0
