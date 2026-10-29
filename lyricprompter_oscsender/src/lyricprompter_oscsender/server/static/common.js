// Helpers for the operator console (trimmed from lyricprompter's static/common.js): status texts, time
// formatting, cue time parsing, reference-time extrapolation, the auto-reconnecting WebSocket and the
// JSON API helper.

export const MOVING = new Set(["tracking", "coasting"]);

export const STATUS_TEXT = {
  tracking: "Tracking", coasting: "Coasting", hold: "On hold", waiting: "Waiting for audio",
  stopped: "Stopped", idle: "No session", error: "Error", offline: "Disconnected", stale: "No audio",
};

// A moving state whose tracker has not advanced for this long means the input stalled.
export const STALE_AGE_S = 1.0;

// Status to show for a state message: "stale" when tracking but the audio stopped arriving.
export function shownStatus(msg) {
  return MOVING.has(msg.status) && (msg.age || 0) > STALE_AGE_S ? "stale" : msg.status || "idle";
}

// m:ss.d (digits decimals); hours fold into minutes (75:02.0), like the cue file's M:SS style.
export function fmtTime(t, digits = 1) {
  if (t == null || !Number.isFinite(t)) return "–";
  const sign = t < 0 ? "-" : "";
  t = Math.abs(t);
  // round first so 59.9996 shows as 1:00.000, not 0:60.000
  const k = 10 ** digits;
  t = Math.round(t * k) / k;
  const m = Math.floor(t / 60);
  const s = (t - m * 60).toFixed(digits).padStart(digits ? digits + 3 : 2, "0");
  return `${sign}${m}:${s}`;
}

// Time of day of a wall-clock time (s since the epoch): HH:MM:SS.s
export function fmtClock(wall) {
  if (wall == null || !Number.isFinite(wall)) return "–";
  const d = new Date(wall * 1000);
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}.${Math.floor(d.getMilliseconds() / 100)}`;
}

// The cue file's time syntax (cues.parse_time): S[.fff], M:SS[.fff], H:MM:SS[.fff], optional [ ],
// ',' as the decimal separator. Returns seconds, or null when the text is not a valid time.
export function parseTime(text) {
  let s = String(text ?? "").trim();
  if (s.startsWith("[") && s.endsWith("]")) s = s.slice(1, -1).trim();
  s = s.replace(",", ".");
  if (!/^\d*(?:\.\d*)?$|^\d+(?::\d+){1,2}(?:\.\d*)?$/.test(s) || s === "" || s === ".") return null;
  const parts = s.split(":");
  let total = 0;
  for (let i = 0; i < parts.length; i++) {
    const v = Number(parts[i]);
    if (!Number.isFinite(v)) return null;
    if (i > 0 && v >= 60) return null;
    total = total * 60 + v;
  }
  return total;
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

// Error of a failed API call: the server's `detail` as the message, plus the HTTP status and the
// whole JSON body (a 409 on a cue edit carries the current sheet in `data.cues`).
export class ApiError extends Error {
  constructor(message, status, data) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

function errorText(res, data) {
  let msg = typeof data === "string" ? data : data?.detail;
  if (Array.isArray(msg)) msg = msg.map((d) => `${(d.loc || []).slice(-1)[0]}: ${d.msg}`).join("; ");
  if (msg && typeof msg === "object") msg = JSON.stringify(msg);
  return msg || `${res.status} ${res.statusText || "error"}`;
}

export async function api(path, { method = "GET", body, form, signal } = {}) {
  const opts = { method, headers: {}, signal };
  if (form) opts.body = form;
  else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers["Content-Type"] = "application/json"; }
  const res = await fetch(path, opts);
  const isJson = (res.headers.get("content-type") || "").includes("json");
  const data = isJson ? await res.json() : await res.text();
  if (!res.ok) throw new ApiError(errorText(res, data), res.status, data);
  return data;
}

// Multipart POST with upload progress (fetch cannot report it). onProgress(fraction 0..1, done).
export function upload(path, form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", path);
    xhr.responseType = "text";
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress?.(e.loaded / e.total, false); };
    xhr.upload.onload = () => onProgress?.(1, true);
    xhr.onerror = () => reject(new ApiError("Upload failed: the server could not be reached", 0, null));
    xhr.onload = () => {
      let data = xhr.responseText;
      try { data = JSON.parse(xhr.responseText); } catch { /* plain text */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(new ApiError(errorText({ status: xhr.status, statusText: xhr.statusText }, data), xhr.status, data));
    };
    xhr.send(form);
  });
}

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[c]));
