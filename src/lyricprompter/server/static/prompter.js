import {
  api, connectWS, focusLine, locate, mergeDisplay, paginate, paintLine, prepareSong, RefClock, renderLine,
  resetDisplay, restoreDisplayState, setLocalDisplay, shownStatus, STATUS_TEXT,
} from "./common.js";

const $ = (id) => document.getElementById(id);
const root = document.documentElement;
const els = {
  reel: $("reel"), prev: $("ln-prev"), cur: $("ln-cur"), n1: $("ln-next1"), n2: $("ln-next2"),
  focusWrap: $("focus-wrap"), page: $("page"), pageIn: $("page-in"),
  cue: $("cue"), cueFill: $("cue-fill"), cueSec: $("cue-sec"), status: $("status"), title: $("title"),
  conf: document.querySelector("#conf span"), empty: $("empty"), settings: $("settings"),
};
const STATUS_COLOR = { tracking: "--track", coasting: "--coast", hold: "--hold", manual: "--hold", stale: "--alert" };
const LOCAL_KEY = "lyricprompter.prompter";

const clock = new RefClock();
let lines = [];
let songKey = null;    // `${id}@${rev}` of the song shown (or being fetched)
let shownId = null;    // id of the song whose lines are shown
let wanted = null;     // key of the latest song announcement
let retryTimer = 0;
let retryDelay = 1000;
let focus = null;      // focus line drawn (null: draw again)
let curViews = [];     // syllable views of the focus line (karaoke wipe)
let lastStatus = "offline";
let empty = true;      // no lines: #empty instead of the lyrics
// page layout ("Multiple lines")
let paged = false;     // settings.layout === "page"
let pages = [];        // [first, end) line ranges (common.js paginate)
let pageOf = [];       // line index -> page index
let pageShown = null;  // page drawn in #page-in (null: draw again)
let pageRows = [];     // {idx, wrap, views} of the lines drawn on the page
let needFit = false;   // the page's font size must be fitted again (resize, fonts loaded)

// ------------------------------------------------------------------ settings
// The console's settings apply to every screen; a setting changed here (mirror for a teleprompter
// glass, a bigger font on a far monitor) stays on this screen, across reloads and server restarts,
// until the console changes that same setting (common.js mergeDisplay).
const DEFAULTS = {
  lead_time: 1.0, font_scale: 1.0, mirror: false, preview_lines: 2, layout: "line", page_lines: 4, wipe: true,
};
let saved = null;
try { saved = JSON.parse(localStorage.getItem(LOCAL_KEY) || "null"); } catch { /* private mode */ }
const display = restoreDisplayState(DEFAULTS, saved);
const settings = display.settings;

function saveLocal() {
  try { localStorage.setItem(LOCAL_KEY, JSON.stringify(display)); } catch { /* ignore */ }
}

function applySettings() {
  document.body.classList.toggle("mirror", !!settings.mirror);
  document.body.classList.toggle("wipe", !!settings.wipe);
  root.style.setProperty("--scale", settings.font_scale);
  $("s-layout").value = settings.layout;
  $("s-lines").value = settings.page_lines; $("o-lines").value = String(settings.page_lines);
  $("s-lines").disabled = settings.layout !== "page";
  $("s-lead").value = settings.lead_time; $("o-lead").value = `${Number(settings.lead_time).toFixed(1)} s`;
  $("s-size").value = settings.font_scale; $("o-size").value = `${Math.round(settings.font_scale * 100)}%`;
  $("s-prev").value = settings.preview_lines; $("o-prev").value = String(settings.preview_lines);
  $("s-wipe").checked = !!settings.wipe;
  $("s-mirror").checked = !!settings.mirror;
  paged = settings.layout === "page";
  relayout();
}

function setLocal(patch) {
  setLocalDisplay(display, patch);
  saveLocal();
  applySettings();
}

$("s-layout").addEventListener("change", (e) => setLocal({ layout: e.target.value }));
$("s-lines").addEventListener("input", (e) => setLocal({ page_lines: Number(e.target.value) }));
$("s-lead").addEventListener("input", (e) => setLocal({ lead_time: Number(e.target.value) }));
$("s-size").addEventListener("input", (e) => setLocal({ font_scale: Number(e.target.value) }));
$("s-prev").addEventListener("input", (e) => setLocal({ preview_lines: Number(e.target.value) }));
$("s-wipe").addEventListener("change", (e) => setLocal({ wipe: e.target.checked }));
$("s-mirror").addEventListener("change", (e) => setLocal({ mirror: e.target.checked }));
$("b-close").addEventListener("click", () => toggleSettings(false));
$("b-reset").addEventListener("click", () => { resetDisplay(display); saveLocal(); applySettings(); });
$("b-full").addEventListener("click", toggleFullscreen);

function toggleSettings(show = els.settings.hidden) {
  els.settings.hidden = !show;
  document.body.classList.toggle("pointer", show);
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen?.();
  else document.documentElement.requestFullscreen?.().catch(() => {});
}

document.addEventListener("keydown", (e) => {
  if (e.target.closest?.("input, select") || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "f") toggleFullscreen();
  else if (k === "l") setLocal({ layout: settings.layout === "page" ? "line" : "page" });
  else if (k === "w") setLocal({ wipe: !settings.wipe });
  else if (k === "m") setLocal({ mirror: !settings.mirror });
  else if (k === "+" || k === "=") setLocal({ font_scale: Math.min(2, +(settings.font_scale + 0.05).toFixed(2)) });
  else if (k === "-" || k === "_") setLocal({ font_scale: Math.max(0.5, +(settings.font_scale - 0.05).toFixed(2)) });
  else if (k === "s" || k === "escape") toggleSettings(k === "s" ? undefined : false);
});
$("stage").addEventListener("click", () => toggleSettings());
$("stage").addEventListener("dblclick", toggleFullscreen);

let pointerTimer = 0;
document.addEventListener("mousemove", () => {
  document.body.classList.add("pointer");
  clearTimeout(pointerTimer);
  pointerTimer = setTimeout(() => { if (els.settings.hidden) document.body.classList.remove("pointer"); }, 2000);
});

// keep the screen awake on stage
let wakeLock = null;
async function keepAwake() {
  try { if ("wakeLock" in navigator && document.visibilityState === "visible") wakeLock = await navigator.wakeLock.request("screen"); }
  catch { wakeLock = null; }
}
document.addEventListener("visibilitychange", keepAwake);
keepAwake();

// ------------------------------------------------------------------ song
// The server announces the song only when it (or its rev) changes, so a failed fetch is retried
// with backoff until it loads or another song is announced.
async function loadSong(id, rev) {
  const key = id ? `${id}@${rev}` : null;
  wanted = key;
  if (key === songKey) return;
  songKey = key;
  clearTimeout(retryTimer);
  if (!id) {
    lines = []; relayout(); shownId = null; els.title.textContent = ""; els.empty.textContent = "";
    showEmpty(true);
    return;
  }
  try {
    const song = await api(`/api/songs/${encodeURIComponent(id)}`);
    if (songKey !== key) return;
    lines = prepareSong(song);
    shownId = id;
    retryDelay = 1000;
    const meta = song.meta || {};
    els.title.textContent = [meta.title, meta.artist].filter(Boolean).join(", ");
    relayout();
    els.empty.textContent = "";
    showEmpty(!lines.length);
  } catch (err) {
    if (songKey !== key) return;          // a newer announcement is loading
    songKey = null;
    if (shownId !== id) {                 // a new song: say so; a newer rev of the shown one: keep it up
      lines = []; relayout(); shownId = null; els.title.textContent = "";
      els.empty.textContent = `Could not load the song: ${err.message}. Retrying…`;
      showEmpty(true);
    }
    retryTimer = setTimeout(() => { if (wanted === key && songKey === null) loadSong(id, rev); }, retryDelay);
    retryDelay = Math.min(retryDelay * 2, 10000);
  }
}

// The lines, the layout or a display setting changed: draw everything again on the next frame.
function relayout() {
  pages = paged ? paginate(lines, settings.page_lines) : [];
  pageOf = [];
  pages.forEach(([a, b], p) => { for (let i = a; i < b; i++) pageOf[i] = p; });
  focus = null;
  pageShown = null;
  syncViews();
}

function showEmpty(isEmpty) {
  empty = isEmpty;
  els.empty.hidden = !isEmpty;
  syncViews();
  if (isEmpty && !els.empty.textContent) els.empty.textContent = "Waiting for a song from the console.";
}

function syncViews() {
  els.reel.hidden = empty || paged;
  els.page.hidden = empty || !paged;
}

function fillLine(el, line, showSection) {
  if (!line) { el.textContent = ""; el.removeAttribute("data-section"); return; }
  el.textContent = line.text;
  if (line.lang) el.lang = line.lang; else el.removeAttribute("lang");
  el.classList.toggle("backing", !!line.backing);
  if (showSection && line.section) el.dataset.section = line.section; else el.removeAttribute("data-section");
}

function sectionStarts(i) {
  return i >= 0 && lines[i]?.section && (i === 0 || lines[i - 1].section !== lines[i].section);
}

const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// Restart a CSS entrance animation (class `cls` on `el`).
function replay(el, cls) {
  el.classList.remove(cls);
  void el.offsetWidth;
  el.classList.add(cls);
}

function setFocus(i) {
  if (paged) { setPageFocus(i); return; }
  if (i === focus) return;
  const advance = focus != null && i === focus + 1;
  focus = i;
  const at = (k) => (k >= 0 && k < lines.length ? lines[k] : null);
  fillLine(els.prev, at(i - 1), false);
  const line = at(i);
  curViews = line ? renderLine(line, els.cur) : [];
  els.cur.classList.toggle("backing", !!line?.backing);
  if (sectionStarts(i)) els.cur.dataset.section = line.section; else els.cur.removeAttribute("data-section");
  if (els.cue.parentNode !== els.focusWrap) els.focusWrap.append(els.cue);    // back from the page layout
  fillLine(els.n1, settings.preview_lines >= 1 ? at(i + 1) : null, sectionStarts(i + 1));
  fillLine(els.n2, settings.preview_lines >= 2 ? at(i + 2) : null, sectionStarts(i + 2));
  if (advance && !reducedMotion()) replay(els.reel, "advance");
}

// ------------------------------------------------------------------ page layout
// The page holding the focus line (common.js paginate: a stanza, or a balanced part of a long one)
// at ONE font size: the largest at which the page and its next-page preview fit the screen, at most
// the line layout's size (--fs, which carries the text size setting). The size changes only when the
// page turns or the screen resizes, never while the focus moves down the page: every line keeps its
// place and only its colour changes (sung: past grey, focus: white with the wipe and an amber bar
// in the margin, still to come: preview grey).
const MIN_PAGE_PX = 14;

const node = (tag, cls, parent) => {
  const el = document.createElement(tag);
  el.className = cls;
  parent.append(el);
  return el;
};

function drawPage(p) {
  const box = els.pageIn;
  box.textContent = "";
  pageRows = [];
  pageShown = p;
  const [a, b] = pages[p];
  if (sectionStarts(a)) node("p", "sec", box).textContent = lines[a].section;
  for (let i = a; i < b; i++) {
    const wrap = node("div", "pl", box);
    const ln = node("p", "ln", wrap);
    ln.classList.toggle("backing", !!lines[i].backing);
    pageRows.push({ idx: i, wrap, views: renderLine(lines[i], ln) });
  }
  // the first lines of the next page, small and dim under a thin rule (the "Upcoming lines" setting)
  const nx = pages[p + 1];
  const n = nx ? Math.min(settings.preview_lines, nx[1] - nx[0]) : 0;
  if (n > 0) {
    node("hr", "pdiv", box);
    for (let i = nx[0]; i < nx[0] + n; i++) fillLine(node("p", "ln pv", box), lines[i], i === nx[0] && sectionStarts(i));
  }
}

// Binary search of the font size (whole px, ~7 layouts): text height grows with the size.
function fitPage() {
  const avail = els.page.clientHeight;
  if (pageShown == null || !avail) return;          // hidden or not drawn yet: fitted once it shows
  needFit = false;
  const box = els.pageIn;
  const cap = Math.max(MIN_PAGE_PX, Math.floor(parseFloat(getComputedStyle(els.page).fontSize) || 0));
  const fits = (px) => { box.style.fontSize = `${px}px`; return box.offsetHeight <= avail; };
  if (fits(cap)) return;
  let lo = MIN_PAGE_PX, hi = cap;                   // a page that does not fit even at lo is clipped
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (fits(mid)) lo = mid; else hi = mid;
  }
  fits(lo);
}

function setPageFocus(i) {
  const p = pageOf[i] ?? 0;
  const turn = p !== pageShown;
  if (turn) {
    const animate = pageShown != null && !reducedMotion();    // not on the first draw / a settings change
    drawPage(p);
    markPage(i);              // before the first layout: the new lines get their colours without a fade
    fitPage();
    if (animate) replay(els.pageIn, "turn");
  } else if (i !== focus) {
    markPage(i);
  }
}

function markPage(i) {
  focus = i;
  let row = null;
  for (const r of pageRows) {
    r.wrap.classList.toggle("past", r.idx < i);
    r.wrap.classList.toggle("focus", r.idx === i);
    if (r.idx === i) row = r;
  }
  curViews = row ? row.views : [];
  if (row) row.wrap.append(els.cue);                  // the entrance cue sits under the focus line
}

if (typeof ResizeObserver === "function") new ResizeObserver(() => { needFit = true; }).observe(els.page);
document.fonts?.ready?.then(() => { needFit = true; });

// ------------------------------------------------------------------ live state
function setStatus(status, confidence) {
  if (status !== lastStatus) {
    lastStatus = status;
    els.status.dataset.status = status;
    els.status.querySelector("b").textContent = STATUS_TEXT[status] || status;
    els.conf.style.background = `var(${STATUS_COLOR[status] || "--idle"})`;
  }
  els.conf.style.width = `${Math.round(Math.max(0, Math.min(1, confidence || 0)) * 100)}%`;
}

connectWS((msg) => {
  if (msg.type === "state") {
    clock.update(msg);
    setStatus(shownStatus(msg), msg.confidence);
  } else if (msg.type === "song") {
    loadSong(msg.id, msg.rev);
  } else if (msg.type === "display") {
    if (mergeDisplay(display, msg)) applySettings();
    saveLocal();
  }
}, (connected) => {
  if (!connected) { clock.moving = false; setStatus("offline", 0); }
  // (re)connected, possibly to a restarted server whose revs start again at 0: fetch the announced
  // song again (the lines shown stay up meanwhile)
  else songKey = null;
});

// ------------------------------------------------------------------ render loop (60 fps)
function frame(now) {
  requestAnimationFrame(frame);
  if (!lines.length) return;
  const t = clock.tick(now);
  const cur = locate(lines, t);
  let f = focusLine(lines, cur, t, settings.lead_time);
  if (f < 0) f = Math.max(0, cur.line_idx);
  setFocus(f);
  if (paged && needFit) fitPage();
  if (settings.wipe) paintLine(curViews, t);

  const line = lines[f];
  const rem = line && line.start != null ? line.start - t : -1;
  if (rem > 0) {
    const win = Math.max(settings.lead_time, 4);
    els.cue.hidden = false;
    els.cueFill.style.width = `${(1 - Math.min(rem, win) / win) * 100}%`;
    els.cueSec.textContent = rem > 0.6 ? `${Math.ceil(rem)}` : "";
  } else if (!els.cue.hidden) {
    els.cue.hidden = true;
  }
}

applySettings();
showEmpty(true);
requestAnimationFrame(frame);
