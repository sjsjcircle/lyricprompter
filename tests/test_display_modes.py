"""Prompter display modes: page layout ("Multiple lines", page_lines) and the syllable-highlight toggle
(wipe). Server validation / persistence of the settings, the stanza paging helper and the screen's
display-state merge (common.js), and the real static/prompter.js driven headless with a stub DOM."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lyricprompter.server import app as app_mod
from lyricprompter.server.app import DisplaySettings, create_app, load_display
from lyricprompter.server.jobs import JobManager
from lyricprompter.text.model import AlignUnit, Line, Song, SongMeta, Syllable, Word

STATIC = Path(app_mod.__file__).parent / "static"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def make_line(idx: int, text: str, start: float, stanza: int, section: str | None = None,
              dur: float = 1.5) -> Line:
    syls, step = [], dur / len(text)
    for k, ch in enumerate(text):
        t0, t1 = round(start + k * step, 4), round(start + (k + 1) * step, 4)
        syls.append(Syllable(text=ch, span=(k, k + 1), units=[AlignUnit(ch, "a", t0, t1, 0.5)], start=t0, end=t1))
    word = Word(text=text, span=(0, len(text)), lang="ko", syllables=syls)
    return Line(idx=idx, text=text, words=[word], start=start, end=round(start + dur, 4), stanza=stanza,
                section=section, confidence=0.8)


TEXTS = ["가나다", "라마바", "사아자", "차카타", "파하가", "나다라", "마바사"]


def stanza_song(sizes: list[int], sections: list[str | None] | None = None) -> Song:
    """Lines 2 s apart (line i starts at 10 + 2 i, 1.5 s long, 3 syllables), stanza k has sizes[k] lines."""
    lines: list[Line] = []
    for k, n in enumerate(sizes):
        for _ in range(n):
            i = len(lines)
            lines.append(make_line(i, TEXTS[i % len(TEXTS)], 10.0 + 2 * i, k, (sections or [None] * len(sizes))[k]))
    return Song(meta=SongMeta(id="s", title="Stanzas", default_lang="ko"), lines=lines)


def run_node(tmp_path: Path, script: str, *args: str, module: str = "common.js") -> dict:
    path = tmp_path / "check.mjs"
    path.write_text(script.replace("%MODULE%", json.dumps((STATIC / module).as_uri())), encoding="utf-8")
    out = subprocess.run(["node", str(path), *args], capture_output=True, text=True, check=True, timeout=60).stdout
    return json.loads(out.strip().splitlines()[-1])


@pytest.fixture
def song_root(tmp_path: Path) -> Path:
    root = tmp_path / "songs"
    root.mkdir()
    return root


def client(song_root: Path) -> TestClient:
    return TestClient(create_app(song_root, jobs=JobManager(), devices=lambda refresh: [],
                                 allowed_hosts=("testserver",)))


# ----------------------------------------------------------------------------- server settings


def test_display_defaults_and_validation(song_root: Path) -> None:
    with client(song_root) as c:
        d = c.get("/api/display").json()
        assert (d["layout"], d["page_lines"], d["wipe"]) == ("line", 4, True)
        for bad in ({"layout": "grid"}, {"layout": ""}, {"page_lines": 0}, {"page_lines": 9}, {"page_lines": 2.5},
                    {"wipe": "maybe"}, {"wipe": None, "layout": "pages"}):
            assert c.put("/api/display", json=bad).status_code == 422, bad
        d = c.put("/api/display", json={"layout": "page", "page_lines": 5, "wipe": False}).json()
        assert (d["layout"], d["page_lines"], d["wipe"], d["rev"]) == ("page", 5, False, 1)
        # a patch carries only the fields the operator changed: the others stay
        d = c.put("/api/display", json={"page_lines": 8}).json()
        assert (d["layout"], d["page_lines"], d["wipe"], d["lead_time"]) == ("page", 8, False, 1.0)
        assert c.get("/api/session").json()["display"]["layout"] == "page"
        with c.websocket_connect("/ws") as ws:
            pushed = [ws.receive_json() for _ in range(2)]
        msg = next(m for m in pushed if m["type"] == "display")
        assert (msg["layout"], msg["page_lines"], msg["wipe"], msg["rev"]) == ("page", 8, False, 2)


def test_display_modes_survive_a_restart_and_old_files_still_load(song_root: Path) -> None:
    with client(song_root) as c:
        c.put("/api/display", json={"layout": "page", "page_lines": 6, "wipe": False, "mirror": True})
    saved = json.loads((song_root / app_mod.DISPLAY_FILE).read_text())
    assert (saved["layout"], saved["page_lines"], saved["wipe"], saved["mirror"]) == ("page", 6, False, True)
    with client(song_root) as c:
        d = c.get("/api/display").json()
    assert (d["layout"], d["page_lines"], d["wipe"], d["mirror"]) == ("page", 6, False, True)
    # a .display.json written before these settings existed: its values load, the new ones default
    old = song_root / app_mod.DISPLAY_FILE
    old.write_text('{"lead_time": 2.5, "font_scale": 1.3, "mirror": true, "preview_lines": 1}')
    assert load_display(old) == DisplaySettings(lead_time=2.5, font_scale=1.3, mirror=True, preview_lines=1)
    assert (load_display(old).layout, load_display(old).page_lines, load_display(old).wipe) == ("line", 4, True)
    old.write_text('{"layout": "grid", "mirror": true}')                    # invalid: all defaults (as before)
    assert load_display(old) == DisplaySettings()


# ----------------------------------------------------------------------------- paging helper (node)

PAGING_SCRIPT = """
import { paginate, prepareSong } from %MODULE%;
const mk = (stanzas, sections) => stanzas.map((s, i) => ({ stanza: s, section: sections ? sections[i] : null }));
const runs = (sizes) => sizes.flatMap((n, k) => Array(n).fill(k));       // [4, 3] -> stanza of each line
const sizes = (pages) => pages.map(([a, b]) => b - a);
const out = {};
out.balanced = sizes(paginate(mk(runs([11])), 5));                       // 11 at max 5: 4/4/3, not 5/5/1
out.stanzas_of_4_at_5 = sizes(paginate(mk(runs([4, 4, 4])), 5));         // the user's example: 4 per screen
out.no_breaks_20_at_5 = sizes(paginate(mk(runs([20])), 5));              // ... and 5 per screen without breaks
out.no_breaks_12_at_5 = sizes(paginate(mk(runs([12])), 5));
out.never_merged = sizes(paginate(mk(runs([1, 2, 1])), 8));
out.replicant = sizes(paginate(mk(runs([4, 3, 4, 3, 4, 2])), 2));       // songs/06 shape at max 2
out.june = sizes(paginate(mk(runs([45])), 4));                           // songs/07: one 45-line stanza
out.ranges = paginate(mk(runs([5, 2])), 3);
// a section label change starts a block even within one stanza number; no stanza at all: one block
out.section_split = sizes(paginate(mk([0, 0, 0, 0], [null, null, "Chorus", "Chorus"]), 8));
out.no_stanza = sizes(paginate([{}, {}, {}, {}, {}], 4));
out.clamped = [paginate(mk(runs([3])), 0), paginate(mk(runs([10])), 99), paginate(mk(runs([5])), "x")].map(sizes);
out.empty = paginate([], 4);
const lines = prepareSong(JSON.parse(process.argv[2]));
out.prepared = lines.map((l) => l.stanza);
out.prepared_pages = paginate(lines, 2);
const old = JSON.parse(process.argv[2]);
old.lines.forEach((l) => delete l.stanza);                               // song.json without stanza
out.old_stanza = prepareSong(old).map((l) => l.stanza);
console.log(JSON.stringify(out));
"""


@needs_node
def test_paging_splits_stanzas_into_balanced_pages(tmp_path: Path) -> None:
    song = stanza_song([3, 2], ["Verse", "Chorus"])
    out = run_node(tmp_path, PAGING_SCRIPT, json.dumps(song.to_dict()))
    assert out["balanced"] == [4, 4, 3]
    assert out["stanzas_of_4_at_5"] == [4, 4, 4]
    assert out["no_breaks_20_at_5"] == [5, 5, 5, 5]
    assert out["no_breaks_12_at_5"] == [4, 4, 4]
    assert out["never_merged"] == [1, 2, 1]
    assert out["replicant"] == [2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2]
    assert out["june"] == [4] * 9 + [3] * 3 and sum(out["june"]) == 45
    assert out["ranges"] == [[0, 3], [3, 5], [5, 7]]
    assert out["section_split"] == [2, 2]
    assert out["no_stanza"] == [3, 2]
    assert out["clamped"] == [[1, 1, 1], [5, 5], [3, 2]]                 # max 1..8; not a number: 4
    assert out["empty"] == []
    assert out["prepared"] == [0, 0, 0, 1, 1]
    assert out["prepared_pages"] == [[0, 2], [2, 3], [3, 5]]
    assert out["old_stanza"] == [0, 0, 0, 0, 0]


# ----------------------------------------------------------------------------- screen display state (node)

MERGE_SCRIPT = """
import { DISPLAY_KEYS, mergeDisplay, resetDisplay, restoreDisplayState, setLocalDisplay } from %MODULE%;
const D = { lead_time: 1.0, font_scale: 1.0, mirror: false, preview_lines: 2, layout: "line", page_lines: 4, wipe: true };
const srv = (patch, rev) => ({ type: "display", ...D, ...patch, rev });
const reload = (st) => restoreDisplayState(D, JSON.parse(JSON.stringify(st)));
const out = { keys: DISPLAY_KEYS };
let st = restoreDisplayState(D, null);
mergeDisplay(st, srv({}, 0));
setLocalDisplay(st, { layout: "page" });                       // L pressed on this screen
out.wipe_changed = mergeDisplay(st, srv({ wipe: false }, 1));  // the console turns the highlight off
out.after_wipe = { ...st.settings };
out.lines_changed = mergeDisplay(st, srv({ wipe: false, page_lines: 6 }, 2));
st = reload(st);                                               // page reload: init message again
mergeDisplay(st, srv({ wipe: false, page_lines: 6 }, 2));
out.after_reload = { ...st.settings };
mergeDisplay(st, srv({ wipe: false, page_lines: 6, layout: "page" }, 3));   // the console picks a layout ...
mergeDisplay(st, srv({ wipe: false, page_lines: 6, layout: "line" }, 4));   // ... and changes it: it wins
out.console_layout = st.settings.layout;
out.overrides = st.overrides;
// old server (no new keys in its messages): this screen keeps its own / default values
const legacy = restoreDisplayState(D, null);
out.legacy_changed = mergeDisplay(legacy, { type: "display", lead_time: 2, font_scale: 1, mirror: false, preview_lines: 2 });
out.legacy = { ...legacy.settings };
// a stale / hand-edited localStorage entry never yields a layout the screen cannot draw
out.bad = restoreDisplayState(D, { settings: { layout: "grid", page_lines: 12, wipe: "no", mirror: true },
                                   server: { layout: "page", page_lines: 2.5 }, overrides: ["layout", "zoom"] });
out.good = restoreDisplayState(D, { settings: { layout: "page", page_lines: 8, wipe: false }, overrides: ["wipe"] });
let fresh = restoreDisplayState(D, { settings: { ...D, layout: "page" }, overrides: ["layout"], server: {} });
mergeDisplay(fresh, srv({ layout: "line" }, 5));               // first contact keeps this screen's layout
out.first_contact = fresh.settings.layout;
out.reset_changed = resetDisplay(fresh);
out.after_reset = fresh.settings.layout;
console.log(JSON.stringify(out));
"""


@needs_node
def test_screen_display_state_with_layout_and_wipe(tmp_path: Path) -> None:
    out = run_node(tmp_path, MERGE_SCRIPT)
    assert {"layout", "page_lines", "wipe"} <= set(out["keys"])
    assert out["wipe_changed"] is True
    assert out["after_wipe"]["layout"] == "page" and out["after_wipe"]["wipe"] is False
    assert out["lines_changed"] is True
    assert out["after_reload"] == {"lead_time": 1.0, "font_scale": 1.0, "mirror": False, "preview_lines": 2,
                                   "layout": "page", "page_lines": 6, "wipe": False}
    assert out["console_layout"] == "line" and "layout" not in out["overrides"]
    assert out["legacy_changed"] is True and out["legacy"]["layout"] == "line" and out["legacy"]["wipe"] is True
    assert out["bad"]["settings"]["layout"] == "line" and out["bad"]["settings"]["page_lines"] == 4
    assert out["bad"]["settings"]["wipe"] is True and out["bad"]["settings"]["mirror"] is True
    assert out["bad"]["server"] == {"layout": "page"} and out["bad"]["overrides"] == ["layout"]
    assert (out["good"]["settings"]["layout"], out["good"]["settings"]["page_lines"],
            out["good"]["settings"]["wipe"]) == ("page", 8, False)
    assert out["first_contact"] == "page"
    assert out["reset_changed"] is True and out["after_reset"] == "line"


# ----------------------------------------------------------------------------- prompter.js headless

PROMPTER_SCRIPT = r"""
// The real static/prompter.js on a small stub DOM (tree, classList, dataset): page layout drawing,
// focus colours, page turns, the font fit, the wipe toggle and the L / W keys.
const SONG = JSON.parse(process.argv[2]);
const PAGE_H = 600;                                  // #page height; #page-in is 1.4 em per child
class CL {
  constructor() { this.s = new Set(); }
  add(...c) { c.forEach((x) => this.s.add(x)); } remove(...c) { c.forEach((x) => this.s.delete(x)); }
  toggle(c, f) { const on = f === undefined ? !this.s.has(c) : !!f; if (on) this.s.add(c); else this.s.delete(c); return on; }
  contains(c) { return this.s.has(c); }
}
class N {
  constructor(tag, id = "") {
    this.tag = tag; this.id = id; this.children = []; this.parentNode = null; this.classList = new CL();
    this.dataset = {}; this.style = { setProperty(k, v) { this[k] = v; } }; this.hidden = false; this._text = "";
    this.value = ""; this.checked = false; this.disabled = false; this.handlers = {}; this.lang = "";
  }
  set className(v) { this.classList = new CL(); this.classList.add(...v.split(/\s+/).filter(Boolean)); }
  get className() { return [...this.classList.s].join(" "); }
  append(...ns) {
    for (const n of ns) {
      if (n.parentNode) n.parentNode.children = n.parentNode.children.filter((c) => c !== n);
      n.parentNode = this; this.children.push(n);
    }
  }
  set textContent(v) { for (const c of this.children) c.parentNode = null; this.children = []; this._text = String(v); }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  addEventListener(t, f) { (this.handlers[t] ||= []).push(f); }
  removeAttribute(k) { if (k === "data-section") delete this.dataset.section; if (k === "lang") this.lang = ""; }
  querySelector() { return (this._q ||= new N("b")); }
  closest() { return null; }
  get offsetWidth() { return 0; }
  get clientHeight() { return this.id === "page" ? PAGE_H : 0; }
  get offsetHeight() { return this.id === "page-in" ? parseFloat(this.style.fontSize) * 1.4 * this.children.length : 0; }
}
const byId = new Map();
const $ = (id) => { if (!byId.has(id)) byId.set(id, new N("x", id)); return byId.get(id); };
const docKeys = [];
globalThis.document = {
  getElementById: $, querySelector: (s) => $(`q:${s}`), createElement: (t) => new N(t),
  documentElement: new N("html"), body: new N("body"), visibilityState: "hidden",
  addEventListener: (t, f) => { if (t === "keydown") docKeys.push(f); },
};
const store = {};
globalThis.localStorage = { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; } };
globalThis.location = { protocol: "http:", host: "stage:8000" };
globalThis.matchMedia = () => ({ matches: false });
globalThis.getComputedStyle = (el) => ({ fontSize: el.id === "page" ? "100px" : "16px" });
let raf = null;
globalThis.requestAnimationFrame = (f) => { raf = f; return 1; };
let sock = null;
globalThis.WebSocket = class { constructor() { sock = this; } close() {} };
globalThis.fetch = async () => ({ ok: true, headers: { get: () => "application/json" }, json: async () => SONG });

await import(%MODULE%);
const tick = (ms) => new Promise((r) => setTimeout(r, ms));
sock.onopen();
const send = (m) => sock.onmessage({ data: JSON.stringify(m) });
const D = { lead_time: 1, font_scale: 1, mirror: false, preview_lines: 2, layout: "page", page_lines: 3, wipe: true };
send({ type: "display", ...D, rev: 1 });
send({ type: "song", id: "s", rev: 0 });
await tick(20);
let now = 1000;
const at = (t) => { send({ type: "state", status: "hold", ref_time: t, tempo: 0, age: 0 }); now += 16; raf(now); };
const key = (k) => docKeys.forEach((f) => f({ key: k, target: $("x") }));
const box = $("page-in");
const row = (c) => {
  if (c.tag === "hr") return "--";
  if (c.tag === "div") return `${c.classList.contains("focus") ? "F" : c.classList.contains("past") ? "P" : "U"}:${c.children[0].textContent}`;
  return `${c.className}:${c.textContent}${c.dataset.section ? "@" + c.dataset.section : ""}`;
};
const focusRow = () => box.children.find((c) => c.classList.contains("focus"));
const snap = () => ({
  reel: $("reel").hidden, page: $("page").hidden, rows: box.children.map(row), font: box.style.fontSize,
  turn: box.classList.contains("turn"), wipe: document.body.classList.contains("wipe"),
  cue: $("cue").parentNode ? ($("cue").parentNode.id || $("cue").parentNode.children[0].textContent) : null,
  syls: focusRow() ? focusRow().children[0].children.map((s) => (s.classList.contains("done") ? "d" : s.classList.contains("active") ? "a" : "-")).join("") : null,
});
const out = {};
at(10.5); out.first = snap();
const firstRows = [...box.children];
at(12.5); out.second = snap();
out.sameRows = firstRows.every((c, i) => box.children[i] === c);   // no redraw while the focus walks down
at(15.2); out.lastOfPage = snap();
at(15.6); out.turned = snap();                                     // lead 1 s before line 3 (16 s): page 2
at(20.5); out.chorus = snap();
key("w"); at(20.6); out.noWipe = snap();
key("l"); at(20.7); out.line = { ...snap(), cur: $("ln-cur").textContent, n1: $("ln-next1").textContent };
send({ type: "display", ...D, page_lines: 2, rev: 2 });           // the console edits Max lines only
out.afterConsole = JSON.parse(store["lyricprompter.prompter"]);
key("l"); at(20.8); out.pagedAgain = snap();
console.log(JSON.stringify(out));
process.exit(0);
"""


@needs_node
def test_prompter_page_layout_headless(tmp_path: Path) -> None:
    song = stanza_song([5, 2], [None, "Chorus"])
    t = [ln.text for ln in song.lines]
    out = run_node(tmp_path, PROMPTER_SCRIPT, json.dumps(song.to_dict()), module="prompter.js")
    first = out["first"]
    assert (first["reel"], first["page"]) == (True, False)
    assert first["rows"] == [f"F:{t[0]}", f"U:{t[1]}", f"U:{t[2]}", "--", f"ln pv:{t[3]}", f"ln pv:{t[4]}"]
    assert first["font"] == "71px"            # 6 children x 1.4 em <= 600 px: the largest whole px, cap 100
    assert first["turn"] is False and first["cue"] == t[0] and first["wipe"] is True
    assert first["syls"] == "da-"             # 10.5 s: first syllable sung, second just starting
    assert out["second"]["rows"][:3] == [f"P:{t[0]}", f"F:{t[1]}", f"U:{t[2]}"] and out["sameRows"] is True
    assert out["second"]["font"] == "71px" and out["second"]["cue"] == t[1]
    assert out["lastOfPage"]["rows"][:3] == [f"P:{t[0]}", f"P:{t[1]}", f"F:{t[2]}"]
    turned = out["turned"]
    assert turned["rows"] == [f"F:{t[3]}", f"U:{t[4]}", "--", f"ln pv:{t[5]}@Chorus", f"ln pv:{t[6]}"]
    assert turned["turn"] is True and turned["font"] == "85px" and turned["cue"] == t[3]
    chorus = out["chorus"]
    assert chorus["rows"] == ["sec:Chorus", f"F:{t[5]}", f"U:{t[6]}"] and chorus["font"] == "100px"   # last page
    assert out["noWipe"]["wipe"] is False and out["noWipe"]["syls"] == "---"
    line = out["line"]
    assert (line["reel"], line["page"]) == (False, True) and line["cue"] == "focus-wrap"
    assert (line["cur"], line["n1"]) == (t[5], t[6])
    saved = out["afterConsole"]
    assert saved["settings"]["layout"] == "line" and saved["settings"]["page_lines"] == 2   # L stays on this screen
    assert sorted(saved["overrides"]) == ["layout", "wipe"]
    again = out["pagedAgain"]                 # max 2: stanza 1 is still its own page
    assert again["rows"] == ["sec:Chorus", f"F:{t[5]}", f"U:{t[6]}"] and again["page"] is False


def test_display_controls_are_wired() -> None:
    html = (STATIC / "prompter.html").read_text(encoding="utf-8")
    for needle in ('id="page"', 'id="page-in"', 'id="s-layout"', 'id="s-lines"', 'id="s-wipe"', "L line by line",
                   "W syllable highlight"):
        assert needle in html, needle
    console_html = (STATIC / "console.html").read_text(encoding="utf-8")
    for needle in ('name="layout"', 'value="page">Multiple lines', 'name="page_lines"', 'name="wipe"',
                   "Syllable highlight", "Max lines"):
        assert needle in console_html, needle
    console = (STATIC / "console.js").read_text(encoding="utf-8")
    assert "el.page_lines.disabled" in console and '"wipe", "mirror"' in console
