// Shared by the console and the prompter: song preparation, cursor location (same rules as
// lyricprompter.timeline), page layout paging, syllable rendering + karaoke wipe, ref-time
// extrapolation, WebSocket, display settings.

export const MOVING = new Set(["tracking", "coasting"]);

export const STATUS_TEXT = {
  tracking: "Tracking", coasting: "Coasting", hold: "On hold", waiting: "Waiting for audio",
  stopped: "Stopped", idle: "No session", manual: "Manual", error: "Error", offline: "Disconnected",
  stale: "No audio",
};

// A moving state whose tracker has not advanced for this long means the input stalled.
export const STALE_AGE_S = 1.0;

// Status to show for a state message: "stale" when tracking but the audio stopped arriving.
export function shownStatus(msg) {
  return MOVING.has(msg.status) && (msg.age || 0) > STALE_AGE_S ? "stale" : msg.status;
}

export function fmtTime(t, digits = 1) {
  if (t == null || !isFinite(t)) return "–";
  const sign = t < 0 ? "-" : "";
  t = Math.abs(t);
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(digits).padStart(digits ? digits + 3 : 2, "0");
  return `${sign}${m}:${s}`;
}

function majorityLang(words) {
  const n = {};
  for (const w of words || []) if (w.lang && w.lang !== "x") n[w.lang] = (n[w.lang] || 0) + w.text.length;
  let best = null;
  for (const k in n) if (best == null || n[k] > n[best]) best = k;
  return best;
}

const ok = (v) => v != null && Number.isFinite(v);

// Forward-fill nulls; a leading run takes the first known value (`fallback` if none) — timeline._fill.
function fill(values, fallback) {
  let last = values.find((v) => v != null) ?? fallback;
  return values.map((v) => (last = v != null ? v : last));
}

// Linear interpolation at axis position a from [axis, value] pairs (nearest known value outside);
// ties resolve to the first pair, like Python's max()/min() in timeline._interp.
function interp(a, known) {
  let prev = null, next = null;
  for (const q of known) {
    if (q[0] <= a && (prev == null || q[0] > prev[0])) prev = q;
    if (q[0] >= a && (next == null || q[0] < next[0])) next = q;
  }
  if (prev == null || next == null) return (prev || next)[1];
  if (next[0] === prev[0]) return prev[1];
  return prev[1] + (next[1] - prev[1]) * (a - prev[0]) / (next[0] - prev[0]);
}

function upperBound(arr, t) {          // bisect_right(arr, t)
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] <= t) lo = mid + 1; else hi = mid; }
  return lo;
}

// song.json -> display lines. Missing times are filled exactly like lyricprompter.timeline.Timeline
// so the client-side cursor agrees with the server's line_idx / syl_idx.
export function prepareSong(song) {
  const src = song.lines || [];
  const n = src.length;
  const raw = src.map((ln) => (ln.words || []).flatMap((w) => w.syllables || []));
  const starts = [], ends = [];
  src.forEach((ln, i) => {
    const sy = raw[i];
    const s0 = sy.find((x) => ok(x.start))?.start ?? null;
    const e0 = [...sy].reverse().find((x) => ok(x.end))?.end ?? null;
    starts.push(ok(ln.start) ? ln.start : s0);
    ends.push(ok(ln.end) ? ln.end : e0);
  });
  const prevEnd = [];
  let last = null;
  for (let i = 0; i < n; i++) {
    prevEnd.push(last);
    if (starts[i] != null || ends[i] != null) last = ends[i] != null ? ends[i] : starts[i];
  }
  const nxt = fill([...starts].reverse(), 0.0).reverse();
  const fs = [], fe = [];
  for (let i = 0; i < n; i++) {
    let s = starts[i], e = ends[i];
    if (s == null && e == null) s = e = prevEnd[i] != null ? prevEnd[i] : nxt[i];
    else if (s == null) s = e;
    else if (e == null) e = starts.slice(i + 1).find((x) => x != null) ?? s;
    fs.push(s); fe.push(e);
  }
  for (let i = 1; i < n; i++) fs[i] = Math.max(fs[i], fs[i - 1]);
  for (let i = 0; i < n; i++) fe[i] = Math.max(fe[i], fs[i]);

  return src.map((ln, i) => {
    const sy = raw[i];
    // boundaries on a "syllable axis" (start_k at k, end_k at k+1); missing ones are linearly
    // interpolated between the nearest known boundaries, line start / end being the outer anchors
    const pts = [];
    sy.forEach((x, k) => { pts.push([k, ok(x.start) ? x.start : null], [k + 1, ok(x.end) ? x.end : null]); });
    const known = [[0, fs[i]], ...pts.filter((q) => q[1] != null), [sy.length, fe[i]]];
    const vals = pts.map(([a, v]) => (v != null ? v : interp(a, known)));
    const ss = vals.filter((_, j) => j % 2 === 0), se = vals.filter((_, j) => j % 2 === 1);
    for (let k = 1; k < ss.length; k++) ss[k] = Math.max(ss[k], ss[k - 1]);
    return {
      idx: i, text: ln.text, start: fs[i], end: fe[i], timed: ok(ln.start), section: ln.section,
      stanza: ln.stanza ?? 0, backing: ln.backing, confidence: ln.confidence, needs_review: ln.needs_review,
      flags: ln.flags || [], lang: majorityLang(ln.words),
      syls: sy.map((x, k) => ({ text: x.text, span: x.span, start: ss[k], end: Math.max(se[k], ss[k]) })),
    };
  });
}

// Cursor at ref time t: {line_idx, syl_idx, syl_progress, next_line_idx, in_gap} (timeline.locate).
export function locate(lines, t) {
  const n = lines.length;
  if (t == null || Number.isNaN(t)) t = -Infinity;
  if (!lines._starts) Object.defineProperty(lines, "_starts", { value: lines.map((l) => l.start) });
  const li = upperBound(lines._starts, t) - 1;
  if (li < 0) return { line_idx: -1, syl_idx: -1, syl_progress: 0, next_line_idx: n ? 0 : -1, in_gap: true };
  const next = li + 1 < n ? li + 1 : -1;
  const line = lines[li];
  if (t >= line.end) return { line_idx: li, syl_idx: -1, syl_progress: 0, next_line_idx: next, in_gap: true };
  if (!line._ss) Object.defineProperty(line, "_ss", { value: line.syls.map((x) => x.start) });
  const si = upperBound(line._ss, t) - 1;
  if (si < 0) return { line_idx: li, syl_idx: -1, syl_progress: 0, next_line_idx: next, in_gap: false };
  const s = line.syls[si], dur = s.end - s.start;
  const prog = dur <= 0 ? 1 : Math.min(1, Math.max(0, (t - s.start) / dur));
  return { line_idx: li, syl_idx: si, syl_progress: prog, next_line_idx: next, in_gap: false };
}

// Which line the singer should be looking at: the sung line until it ends, then the next line
// from (next.start - lead) on (never before the current line has finished).
export function focusLine(lines, cur, t, lead) {
  if (cur.line_idx < 0) return cur.next_line_idx;
  if (!cur.in_gap || cur.next_line_idx < 0) return cur.line_idx;
  const nx = lines[cur.next_line_idx];
  return nx.start - t <= lead ? cur.next_line_idx : cur.line_idx;
}

export const PAGE_LINES_MAX = 8;

// Pages of the "page" layout as [first, end) line ranges covering every line in order. Consecutive
// lines of one stanza (a blank line in the lyrics starts one, so does a section header) form a
// block; a block longer than `maxLines` is split into ceil(n / maxLines) pages of balanced size,
// the larger ones first (11 lines at max 5 -> 4/4/3, not 5/5/1). Stanzas are never merged.
export function paginate(lines, maxLines) {
  const m = Math.round(Number(maxLines));
  const max = Number.isFinite(m) ? Math.min(PAGE_LINES_MAX, Math.max(1, m)) : 4;
  const pages = [];
  let a = 0;
  for (let i = 1; i <= lines.length; i++) {
    const same = i < lines.length && lines[i].stanza === lines[i - 1].stanza
      && (lines[i].section ?? null) === (lines[i - 1].section ?? null);
    if (same) continue;
    const n = i - a, k = Math.ceil(n / max), base = Math.floor(n / k), extra = n % k;
    for (let p = 0, s = a; p < k; p++) {
      const e = s + base + (p < extra ? 1 : 0);
      pages.push([s, e]);
      s = e;
    }
    a = i;
  }
  return pages;
}

// Render a line as syllable spans; text between syllables sticks to the preceding syllable.
export function renderLine(line, el) {
  el.textContent = "";
  if (line.lang) el.lang = line.lang; else el.removeAttribute("lang");
  const text = line.text || "";
  const views = [];
  if (!line.syls.length) {
    const sp = document.createElement("span");
    sp.className = "syl";
    sp.textContent = text || " ";
    el.append(sp);
    views.push({ el: sp, start: line.start, end: line.end, state: "" });
    return views;
  }
  let pos = 0;
  const lead = line.syls[0].span[0];
  if (lead > 0) {
    const sp = document.createElement("span");
    sp.textContent = text.slice(0, lead);
    el.append(sp);
    pos = lead;
  }
  line.syls.forEach((s, k) => {
    const a = Math.max(pos, s.span[0]);
    const b = k + 1 < line.syls.length ? Math.max(a, line.syls[k + 1].span[0]) : text.length;
    const sp = document.createElement("span");
    sp.className = "syl";
    sp.textContent = text.slice(a, b) || s.text;
    el.append(sp);
    views.push({ el: sp, start: s.start, end: s.end, state: "" });
    pos = b;
  });
  return views;
}

// Karaoke wipe: done / active (with --p fill) / upcoming. Only touches spans whose state changed.
export function paintLine(views, t) {
  for (const v of views) {
    let state, p = 0;
    if (v.start == null || t < v.start) state = "";
    else if (t >= v.end) state = "done";
    else { state = "active"; p = v.end > v.start ? (t - v.start) / (v.end - v.start) : 1; }
    if (state !== v.state) {
      v.el.classList.toggle("done", state === "done");
      v.el.classList.toggle("active", state === "active");
      v.state = state;
    }
    if (state === "active") v.el.style.setProperty("--p", (p * 100).toFixed(1) + "%");
  }
}

// Longest extrapolation past the last tracker update (s).
export const MAX_EXTRAPOLATION_S = 0.35;

// Smoothly extrapolated reference time: ref_time + tempo * (now - t_state), capped, with gentle
// correction toward each new server estimate and hard snaps on jumps. A stale state (audio stopped
// arriving) holds still where the capped extrapolation left it instead of jumping back.
export class RefClock {
  constructor() {
    this.base = 0; this.tempo = 0; this.anchor = 0; this.moving = false; this.disp = null; this.last = null;
  }
  update(msg, now = performance.now()) {
    const age = Math.max(0, msg.age || 0);
    this.base = msg.ref_time ?? 0;
    this.tempo = msg.tempo || 0;
    this.anchor = now - age * 1000;
    const status = shownStatus(msg);
    this.moving = MOVING.has(status);
    if (status === "stale") this.base += this.tempo * Math.min(age, MAX_EXTRAPOLATION_S);
  }
  target(now) {
    if (!this.moving) return this.base;
    const dt = Math.min(Math.max(0, (now - this.anchor) / 1000), MAX_EXTRAPOLATION_S);
    return this.base + this.tempo * dt;
  }
  tick(now = performance.now()) {
    const tgt = this.target(now);
    const dt = this.last == null ? 0 : Math.min(0.1, (now - this.last) / 1000);
    this.last = now;
    if (this.disp == null || Math.abs(tgt - this.disp) > 0.6) {
      this.disp = tgt;
      return this.disp;
    }
    if (!this.moving) {
      // hold / parked / stale: freeze where the display is instead of stepping back to the server's
      // un-extrapolated position (an operator jump or nudge of >= 0.45 s back still shows)
      if (tgt > this.disp || this.disp - tgt >= 0.45) this.disp = tgt;
      return this.disp;
    }
    let pred = this.disp + this.tempo * dt;
    pred += (tgt - pred) * (1 - Math.exp(-dt / 0.12));
    if (pred < this.disp && this.disp - pred < 0.25) pred = this.disp; // no small backward jitter
    this.disp = pred;
    return pred;
  }
}

// Auto-reconnecting WebSocket to /ws. onMessage(obj), onStatus(connected:boolean).
// The server sends ~20 messages/s, so a socket silent for `staleMs` is dead even if it looks
// open (stalled Wi-Fi, or dropped by the server as too slow): abandon it and reconnect.
export function connectWS(onMessage, onStatus, { staleMs = 2500 } = {}) {
  let ws = null, delay = 500, closed = false, last = 0, retryTimer = 0;
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  const retry = () => {
    if (closed) return;
    clearTimeout(retryTimer);
    retryTimer = setTimeout(open, delay);
    delay = Math.min(delay * 2, 5000);
  };
  const abandon = (sock) => {            // do not wait for a close handshake on a stalled link
    sock.onopen = sock.onmessage = sock.onclose = sock.onerror = null;
    try { sock.close(); } catch { /* already closed */ }
  };
  const open = () => {
    const sock = new WebSocket(url);
    ws = sock;
    last = Date.now();
    sock.onopen = () => { delay = 500; last = Date.now(); onStatus?.(true); };
    sock.onmessage = (ev) => {
      last = Date.now();
      try { onMessage(JSON.parse(ev.data)); } catch (e) { console.error(e); }
    };
    sock.onclose = () => {
      if (ws !== sock) return;
      ws = null;
      onStatus?.(false);
      retry();
    };
    sock.onerror = () => sock.close();
  };
  const watchdog = setInterval(() => {
    if (closed || !ws || Date.now() - last <= staleMs) return;
    const sock = ws;
    ws = null;
    abandon(sock);
    onStatus?.(false);
    retry();
  }, Math.max(50, staleMs / 5));
  open();
  return {
    close() {
      closed = true;
      clearInterval(watchdog);
      clearTimeout(retryTimer);
      if (ws) abandon(ws);
      ws = null;
    },
  };
}

// ------------------------------------------------------------------ prompter display settings
// The console's settings apply to every prompter, but a value set on one screen (mirror on a
// teleprompter glass, a bigger font on a far monitor) stays until the console changes that same
// setting. `state` = {settings, overrides: [keys set on this screen], server: {key: last console
// value seen}}; it is persisted per screen, so overrides survive reloads and server restarts.
export const DISPLAY_KEYS = ["lead_time", "font_scale", "mirror", "preview_lines", "layout", "page_lines", "wipe"];

// Values a saved screen state may hold beyond having the default's type (a stale or hand-edited
// localStorage entry must not leave a screen in a layout it cannot draw).
const DISPLAY_VALID = {
  layout: (v) => v === "line" || v === "page",
  page_lines: (v) => Number.isInteger(v) && v >= 1 && v <= PAGE_LINES_MAX,
};
const validDisplay = (defaults, k, v) =>
  k in defaults && typeof v === typeof defaults[k] && (DISPLAY_VALID[k]?.(v) ?? true);

export function newDisplayState(defaults) {
  return { settings: { ...defaults }, overrides: [], server: {} };
}

export function restoreDisplayState(defaults, saved) {
  const st = newDisplayState(defaults);
  if (!saved || typeof saved !== "object") return st;
  for (const k of DISPLAY_KEYS) {
    if (saved.settings && validDisplay(defaults, k, saved.settings[k])) st.settings[k] = saved.settings[k];
    if (saved.server && validDisplay(defaults, k, saved.server[k])) st.server[k] = saved.server[k];
  }
  st.overrides = DISPLAY_KEYS.filter((k) => (saved.overrides || []).includes(k));
  return st;
}

// A setting changed on this screen.
export function setLocalDisplay(state, patch) {
  for (const k of DISPLAY_KEYS) {
    if (!(k in patch)) continue;
    state.settings[k] = patch[k];
    if (!state.overrides.includes(k)) state.overrides.push(k);
  }
}

// Drop this screen's overrides and follow the console again. Returns true if settings changed.
export function resetDisplay(state) {
  let changed = false;
  state.overrides = [];
  for (const k of DISPLAY_KEYS) {
    if (k in state.server && state.settings[k] !== state.server[k]) { state.settings[k] = state.server[k]; changed = true; }
  }
  return changed;
}

// A "display" message from the server. Returns true if the screen's settings changed.
export function mergeDisplay(state, msg) {
  let changed = false;
  for (const k of DISPLAY_KEYS) {
    if (!(k in msg)) continue;
    const seen = k in state.server;
    if (seen && state.server[k] === msg[k]) continue;        // the console did not change it
    state.server[k] = msg[k];
    if (!seen && state.overrides.includes(k)) continue;      // first contact: keep this screen's value
    state.overrides = state.overrides.filter((x) => x !== k);
    if (state.settings[k] !== msg[k]) { state.settings[k] = msg[k]; changed = true; }
  }
  return changed;
}

export async function api(path, { method = "GET", body, form } = {}) {
  const opts = { method, headers: {} };
  if (form) opts.body = form;
  else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers["Content-Type"] = "application/json"; }
  const res = await fetch(path, opts);
  const isJson = (res.headers.get("content-type") || "").includes("json");
  const data = isJson ? await res.json() : await res.text();
  if (!res.ok) {
    let msg = typeof data === "string" ? data : data.detail;
    if (Array.isArray(msg)) msg = msg.map((d) => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join("; ");
    throw new Error(msg || `${res.status} ${res.statusText}`);
  }
  return data;
}
