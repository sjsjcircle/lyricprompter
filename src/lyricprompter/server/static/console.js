import {
  api, connectWS, fmtTime, focusLine, locate, paintLine, prepareSong, RefClock, renderLine, shownStatus, STATUS_TEXT,
} from "./common.js";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const enc = encodeURIComponent;
const fs = $("#f-session");
const player = $("#player");
const tbody = $("#t-lines tbody");

const S = {
  songs: [], jobs: [], selected: null, detail: null, lines: [],
  live: { id: null, key: null, lines: [] }, last: null, devices: [], display: {},
  // the key panel: S.transpose belongs to S.keySong (see keyFor); S.autoKey is a standing choice
  transpose: 0, autoKey: true, keySong: null,
  liveRow: -1, playRow: -1, knownDone: new Set(), openLogs: new Map(),
};
const clock = new RefClock();
let liveFocus = null;
let liveViews = [];

// ------------------------------------------------------------------ notices
let noticeTimer = 0;
function notify(text, error = false, ms = error ? 9000 : 4000) {
  const el = $("#notice");
  el.textContent = text;
  el.classList.toggle("error", error);
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { el.textContent = ""; }, ms);
}
async function run(fn, okText) {
  try {
    const r = await fn();
    if (okText) notify(okText);
    return r;
  } catch (e) {
    notify(e.message, true);
    return null;
  }
}
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ------------------------------------------------------------------ library
async function refreshSongs() {
  const songs = await run(() => api("/api/songs"));
  if (!songs) return;
  S.songs = songs;
  renderSongs();
  if (S.selected && !songs.some((s) => s.id === S.selected)) S.selected = null;
}

function renderSongs() {
  const ul = $("#songs");
  ul.textContent = "";
  for (const s of S.songs) {
    const li = document.createElement("li");
    li.dataset.id = s.id;
    li.setAttribute("aria-selected", String(s.id === S.selected));
    li.tabIndex = 0;
    const review = s.n_review ? `<span class="review">${s.n_review} to review</span>` : "";
    const live = s.id === S.live.id ? `<span class="live-tag">on prompter</span>` : "";
    li.innerHTML = `<span class="t">${esc(s.title)}</span><span class="m">${s.duration ? fmtTime(s.duration, 0) : ""}</span>
      <span class="a">${esc(s.artist || "")} ${live}</span><span class="m">${s.n_lines ?? 0} lines ${review}</span>
      ${s.error ? `<span class="err">${esc(s.error)}</span>` : ""}`;
    ul.append(li);
  }
  $("#songs-empty").hidden = S.songs.length > 0 || S.jobs.some((j) => !isFinal(j));
}

$("#songs").addEventListener("click", (e) => {
  const li = e.target.closest("li");
  if (li) selectSong(li.dataset.id);
});
$("#songs").addEventListener("keydown", (e) => {
  const li = e.target.closest("li");
  if (li && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); selectSong(li.dataset.id); }
});

async function selectSong(id, { keepAudio = false } = {}) {
  const detail = await run(() => api(`/api/songs/${enc(id)}`));
  if (!detail) return;
  const changed = S.selected !== id;
  S.selected = id;
  if (changed && !sessionActive()) keyFor(id);   // while a song is live the panel shows its key
  S.detail = detail;
  S.lines = prepareSong(detail);
  renderSongs();
  renderLines();
  const meta = detail.meta || {};
  $("#lines-title").textContent = meta.title ? `Lines, ${meta.title}` : "Lines";
  $("#b-cue").disabled = false;
  if (changed && !keepAudio) {
    const stem = detail.stems?.includes("mix") ? "mix" : detail.stems?.[0];
    player.src = stem ? `/api/songs/${enc(id)}/audio/${stem}` : "";
  }
  showStyle(detail.style || meta.style);
  const sel = fs.elements.start_line;
  const keep = sel.value;
  sel.innerHTML = `<option value="">Beginning</option>` + S.lines
    .filter((l) => l.timed)
    .map((l) => `<option value="${l.idx}">${l.idx + 1}. ${esc(l.text.slice(0, 40))}</option>`).join("");
  if (!changed) sel.value = keep;
}

// ------------------------------------------------------------------ jobs
const isFinal = (j) => ["done", "failed", "cancelled"].includes(j.status);
let jobTimer = 0;
async function refreshJobs() {
  clearTimeout(jobTimer);
  const jobs = await api("/api/jobs").catch(() => null);
  if (jobs) {
    let newlyDone = false;
    for (const j of jobs) {
      if (j.status === "done" && !S.knownDone.has(j.id)) {
        if (S.jobs.some((o) => o.id === j.id && !isFinal(o))) { newlyDone = true; notify(`${j.title} is ready`); }
        S.knownDone.add(j.id);
      }
    }
    S.jobs = jobs;
    renderJobs();
    if (newlyDone) refreshSongs();
  }
  jobTimer = setTimeout(refreshJobs, S.jobs.some((j) => !isFinal(j)) ? 1000 : 10000);
}

const STAGE_TEXT = { queued: "Waiting for the previous song", running: "Preparing" };
function renderJobs() {
  const ul = $("#jobs");
  const recent = Date.now() / 1000 - 15 * 60;
  const show = S.jobs.filter((j) => !isFinal(j) || (j.status !== "done" && (j.finished || 0) > recent));
  ul.textContent = "";
  for (const j of show) {
    const li = document.createElement("li");
    li.className = j.status;
    const head = j.status === "failed" ? "Preparation failed" : j.status === "cancelled" ? "Cancelled"
      : `${STAGE_TEXT[j.status] || j.status}${j.stage ? `: ${j.stage}` : ""}`;
    const msg = j.error || j.message || "";
    li.innerHTML = `<div class="t"><span>${esc(j.title || j.song_id)}</span>
      ${isFinal(j) ? "" : `<button class="btn ghost small" data-cancel="${j.id}" type="button">Cancel</button>`}</div>
      <div class="msg">${esc(head)}${msg ? `. ${esc(msg)}` : ""}</div>
      ${isFinal(j) ? "" : `<progress max="100" value="${j.status === "running" ? j.percent : 0}"></progress>`}
      ${j.status === "failed" ? `<details data-job="${j.id}" ${S.openLogs.has(j.id) ? "open" : ""}><summary>Show log</summary><pre>${esc(S.openLogs.get(j.id) || "")}</pre></details>` : ""}`;
    ul.append(li);
  }
  $("#songs-empty").hidden = S.songs.length > 0 || show.length > 0;
}

$("#jobs").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-cancel]");
  if (b) await run(() => api(`/api/jobs/${b.dataset.cancel}`, { method: "DELETE" }), "Cancelling");
  refreshJobs();
});
$("#jobs").addEventListener("toggle", async (e) => {
  const d = e.target.closest("details[data-job]");
  if (!d) return;
  const id = d.dataset.job;
  if (!d.open) { S.openLogs.delete(id); return; }
  if (S.openLogs.get(id)) return;
  S.openLogs.set(id, "Loading…");
  const job = await run(() => api(`/api/jobs/${id}`));
  const text = job ? (job.log || []).join("\n") || "(no output)" : "";
  S.openLogs.set(id, text);
  const pre = $(`details[data-job="${id}"] pre`, $("#jobs"));
  if (pre) pre.textContent = text;
}, true);

// ------------------------------------------------------------------ add song
const fAdd = $("#f-add");
function showAdd(show) {
  fAdd.hidden = !show;
  $("#b-add").setAttribute("aria-expanded", String(show));
  if (show) fAdd.elements.audio.focus();
}
$("#b-add").addEventListener("click", () => showAdd(fAdd.hidden));
$("#b-add-cancel").addEventListener("click", () => showAdd(false));
fAdd.elements.audio.addEventListener("change", () => {
  const f = fAdd.elements.audio.files[0];
  if (f && !fAdd.elements.title.value) fAdd.elements.title.value = f.name.replace(/\.[^.]+$/, "");
});
fAdd.addEventListener("submit", async (e) => {
  e.preventDefault();
  const el = fAdd.elements;
  const form = new FormData();
  form.append("audio", el.audio.files[0]);
  for (const k of ["title", "artist", "language", "lyrics"]) form.append(k, el[k].value);
  form.append("options", JSON.stringify({ lead: el.lead.checked, refine: el.refine.value, style: el.style.value }));
  const btn = $("button[type=submit]", fAdd);
  btn.disabled = true;
  btn.textContent = "Uploading…";
  const res = await run(() => api("/api/songs", { method: "POST", form }));
  btn.disabled = false;
  btn.textContent = "Prepare song";
  if (res) {
    notify(`Preparing ${el.title.value}. This takes a few minutes.`);
    fAdd.reset();
    showAdd(false);
    refreshJobs();
  }
});

// ------------------------------------------------------------------ devices
async function loadDevices(refresh = false) {
  const devs = await run(() => api(`/api/devices${refresh ? "?refresh=true" : ""}`));
  if (!devs) return;
  S.devices = devs;
  const sel = fs.elements.device;
  const prev = sel.value;
  sel.innerHTML = devs.map((d) => `<option value="${d.index}">${esc(d.name)} (${d.channels} ch)</option>`).join("")
    || `<option value="">No input devices found</option>`;
  const def = devs.find((d) => String(d.index) === prev) || devs.find((d) => d.is_default) || devs[0];
  if (def) sel.value = String(def.index);
  renderChannels();
}
function renderChannels() {
  const d = S.devices.find((x) => String(x.index) === fs.elements.device.value);
  const box = $("#chans");
  box.textContent = "";
  if (!d) return;
  const n = Math.min(d.channels, 32);
  for (let c = 1; c <= n; c++) {
    const lab = document.createElement("label");
    lab.className = "check";
    lab.innerHTML = `<input type="checkbox" name="ch" value="${c}" ${c === 1 || (d.channels === 2 && c === 2) ? "checked" : ""}> ${c}`;
    box.append(lab);
  }
}
fs.elements.device.addEventListener("change", renderChannels);
$("#b-rescan").addEventListener("click", () => loadDevices(true));

// ------------------------------------------------------------------ session controls
function syncFormVisibility() {
  const kind = fs.elements.kind.value;
  for (const d of $$("[data-kind]", fs)) d.hidden = d.dataset.kind !== kind;
  $("#lead-opt").hidden = fs.elements.mode.value !== "vocal";
  $("[data-path]", fs).hidden = fs.elements.file_src.value !== "path";
}
fs.addEventListener("change", syncFormVisibility);

fs.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!S.selected) { notify("Pick a song first.", true); return; }
  const el = fs.elements;
  const input = { kind: el.kind.value };
  if (input.kind === "device") {
    input.device = el.device.value === "" ? null : Number(el.device.value);
    input.channels = $$("input[name=ch]:checked", fs).map((c) => Number(c.value));
    if (!input.channels.length) { notify("Select at least one input channel.", true); return; }
  } else {
    const src = el.file_src.value;
    if (src === "path") input.path = el.path.value.trim();
    else input.stem = src.split(":")[1];
    input.monitor = el.monitor.checked;
  }
  const body = {
    song_id: S.selected, mode: el.mode.value, lead: el.lead.checked, input,
    start_line: el.start_line.value === "" ? null : Number(el.start_line.value),
    latency: latencyS(), auto_key: S.autoKey, style: el.style.value,
    // a key (detected, or set by hand) belongs to its song and never carries over to another one;
    // with Auto on it only seeds the search
    transpose: S.keySong === S.selected ? S.transpose : 0,
  };
  $("#b-start").disabled = true;
  $("#b-start").textContent = "Starting…";
  const title = S.detail?.meta?.title || S.selected;
  await run(() => api("/api/session/start", { method: "POST", body }), `Following ${title}`);
  $("#b-start").disabled = false;
  $("#b-start").textContent = "Start";
});

const post = (path, body) => run(() => api(`/api/session/${path}`, { method: "POST", body: body ?? {} }));
// tracking style of the selected song: saved to the song, a live session of it switches at once
const STYLE_TEXT = { classic: "Classical — flexible tempo", pop: "Pop & band — steady tempo" };
function showStyle(style) {
  const sel = fs.elements.style;
  sel.disabled = !S.selected;
  if (document.activeElement !== sel) sel.value = STYLE_TEXT[style] ? style : "classic";
}
fs.elements.style.addEventListener("change", async () => {
  const id = S.selected, style = fs.elements.style.value;
  if (!id) return;
  const res = await run(() => api(`/api/songs/${enc(id)}/style`, { method: "PUT", body: { style } }));
  if (!res) { showStyle(S.detail?.style); return; }      // not saved: show the song's style again
  if (S.detail && S.selected === id) S.detail.style = style;
  const song = S.songs.find((x) => x.id === id);
  if (song) song.style = style;
  notify(`${STYLE_TEXT[style]}${res.session ? ", applied to the live session" : ""}.`);
});
$("#b-stop").addEventListener("click", () => post("stop"));
function toggleHold() {
  return S.last?.status === "hold" ? post("resume") : post("hold", { on: true });
}
$("#b-hold").addEventListener("click", toggleHold);
for (const b of $$("[data-nudge]")) b.addEventListener("click", () => post("nudge", { dt: Number(b.dataset.nudge) }));
const fmtKey = (k) => (k > 0 ? "+" : "") + k;
function showKey() {
  $("#o-transpose").value = fmtKey(S.transpose);
  const b = $("#b-autokey");
  b.classList.toggle("on", S.autoKey);
  b.setAttribute("aria-pressed", String(S.autoKey));
}
// The key panel shows the live song's key while a session runs (from its state messages), else
// the key prepared for the selected song. A key belongs to one song: switching to another one
// starts from its original key (0), so neither a detected key nor a hand-set one carries over.
function keyFor(id) {
  if (S.keySong === id) return;
  if (S.keySong != null) S.transpose = 0;    // (a key set before any song was picked goes to the first one)
  S.keySong = id;
  $("#b-keysuggest").hidden = true;
  showKey();
}
const keyOwner = () => (sessionActive() ? S.last.song_id : S.selected);
function setKey(k) {                       // operator key: turns auto detection off
  S.transpose = Math.max(-12, Math.min(12, k));
  S.autoKey = false;
  S.keySong = keyOwner();
  showKey();
  $("#b-keysuggest").hidden = true;
  if (sessionActive()) post("transpose", { semitones: S.transpose });
}
for (const b of $$("[data-transpose]")) {
  b.addEventListener("click", () => setKey(S.transpose + Number(b.dataset.transpose)));
}
$("#b-autokey").addEventListener("click", () => {
  S.autoKey = !S.autoKey;
  S.keySong = keyOwner();
  if (!S.autoKey) $("#b-keysuggest").hidden = true;
  showKey();
  if (sessionActive()) post("autokey", { on: S.autoKey });
});
$("#b-keysuggest").addEventListener("click", (e) => {
  const k = Number(e.currentTarget.dataset.key);
  if (Number.isFinite(k)) setKey(k);
});
// the field's own min / max (ms): typing past them must not reach the tracker
const latencyS = () => Math.max(-500, Math.min(1000, Number(fs.elements.latency.value) || 0)) / 1000;
fs.elements.latency.addEventListener("change", () => {
  if (sessionActive()) post("latency", { seconds: latencyS() });
});
const sessionActive = () => S.last && !["idle", "manual"].includes(S.last.status) && S.last.mode;

// prompter display settings (pushed to every prompter). Only the fields the operator edited are
// sent, so a setting changed on one prompter screen survives edits of the others.
let displayTimer = 0;
const displayPatch = {};
const DISPLAY_FIELDS = ["layout", "page_lines", "lead_time", "font_scale", "preview_lines"];  // + 2 checkboxes
function pushDisplay(e) {
  const el = e.target;
  if (el.type !== "checkbox" && el.value === "") return;
  displayPatch[el.name] = el.type === "checkbox" ? el.checked : el.name === "layout" ? el.value : Number(el.value);
  if (el.name === "layout") fs.elements.page_lines.disabled = el.value !== "page";   // Max lines: page layout only
  clearTimeout(displayTimer);
  displayTimer = setTimeout(() => {
    const body = { ...displayPatch };
    for (const k of Object.keys(displayPatch)) delete displayPatch[k];
    run(() => api("/api/display", { method: "PUT", body }));
  }, 150);
}
for (const n of [...DISPLAY_FIELDS, "wipe", "mirror"]) {
  fs.elements[n].addEventListener("input", pushDisplay);
}
function applyDisplay(d) {
  S.display = d;
  const el = fs.elements;
  for (const n of DISPLAY_FIELDS) if (d[n] != null && document.activeElement !== el[n]) el[n].value = d[n];
  el.page_lines.disabled = d.layout !== "page";
  el.wipe.checked = d.wipe !== false;
  el.mirror.checked = !!d.mirror;
}

document.addEventListener("keydown", (e) => {
  if (e.target.closest?.("input, textarea, select") || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "h" || e.key === "H") { e.preventDefault(); toggleHold(); }
  else if (e.key === "[") post("nudge", { dt: -0.5 });
  else if (e.key === "]") post("nudge", { dt: 0.5 });
});

// ------------------------------------------------------------------ lines table
const FLAG_TEXT = { manual_timing: "edited" };
const flagText = (f) => esc(FLAG_TEXT[f] || f.replace(/[_:]+/g, " "));
// Tap mode: T sets the armed line's start to the player position and arms the next line
const tap = { on: false, armed: -1, song: null };

function renderLines() {
  if (tap.on && tap.song !== S.selected) setTap(false);     // Tap mode belongs to one song
  tbody.textContent = "";
  const onlyReview = $("#only-review").checked;
  const rows = [];
  for (const l of S.lines) {
    if (onlyReview && !l.needs_review) continue;
    const timed = l.timed;
    const conf = l.confidence == null ? null : Math.max(0, Math.min(1, l.confidence));
    const flags = l.flags.filter((f) => !f.startsWith("repeat_of")).map(flagText).join(", ");
    rows.push(`<tr data-idx="${l.idx}" class="${l.needs_review ? "review" : ""} ${timed ? "" : "untimed"} ${tap.on && l.idx === tap.armed ? "is-armed" : ""}">
      <td class="c-jump"><button class="btn small" data-go="${l.idx}" type="button" ${timed ? "" : "disabled"} aria-label="Go to line ${l.idx + 1}">Go</button></td>
      <td class="c-idx">${l.idx + 1}</td>
      <td class="c-start">${timed ? fmtTime(l.start, 2) : "–"}</td>
      <td class="c-conf" title="${conf == null ? "No confidence" : `Alignment fit ${Math.round(conf * 100)}%`}">
        <span class="fit"><span style="width:${conf == null ? 0 : Math.round(conf * 100)}%"></span></span></td>
      <td class="text"${l.lang ? ` lang="${l.lang}"` : ""}>${l.section && (l.idx === 0 || S.lines[l.idx - 1].section !== l.section) ? `<span class="sec">${esc(l.section)}</span>` : ""}${esc(l.text)}${flags ? `<span class="flag">${flags}</span>` : ""}</td>
      <td class="c-edit">
        <button class="btn small" data-play="${l.idx}" type="button" ${timed ? "" : "disabled"} title="Play the recording from this line">Play</button>
        <button class="btn small" data-shift="-0.1" type="button" ${timed ? "" : "disabled"} title="Start 0.1 s earlier">−0.1</button>
        <button class="btn small" data-shift="0.1" type="button" ${timed ? "" : "disabled"} title="Start 0.1 s later">+0.1</button>
        <button class="btn small" data-set="start" type="button" title="Set the start to the player position (in Tap mode: T)">Set start</button>
        <button class="btn small" data-set="end" type="button" ${timed ? "" : "disabled"} title="Set the end to the player position">Set end</button>
      </td></tr>`);
  }
  tbody.innerHTML = rows.join("");
  const empty = $("#lines-empty");
  empty.hidden = S.lines.length > 0;
  empty.textContent = S.selected ? "This song has no lines." : "Pick a song to review its timing and jump between lines.";
  S.liveRow = -1; S.playRow = -1;
  liveFocus = null;          // re-mark the live row on the next frame
  goBlocked = null;
  syncGo();
  if (player.src && player.readyState > 0) trackPlayer();
}

// Go moves the prompter within the song it is following. While another song is live, the lines
// shown here belong to the selected song, so their Go buttons must not move the live one.
let goBlocked = null;
const otherSongLive = () => !!(sessionActive() && S.selected && S.last.song_id !== S.selected);
const liveTitle = () => S.songs.find((s) => s.id === S.last?.song_id)?.title || S.last?.song_id || "Another song";
function syncGo() {
  const blocked = otherSongLive();
  if (blocked === goBlocked) return;
  goBlocked = blocked;
  const why = blocked ? `${liveTitle()} is live: select it to jump between its lines, or stop the session first` : "";
  for (const b of $$("button[data-go]", tbody)) {
    b.disabled = blocked || !S.lines[Number(b.dataset.go)]?.timed;
    b.title = why;
  }
}
$("#only-review").addEventListener("change", renderLines);

// Timing edits (and undos) go to the server one at a time, in order: fast taps never overtake.
let editQueue = Promise.resolve();
function queueEdit(fn) {
  editQueue = editQueue.then(fn).catch((e) => notify(e.message, true));
  return editQueue;
}
// [31, 32, 34] -> "lines 32–33, 35" (the table numbers lines from 1)
function lineList(idxs) {
  const parts = [];
  for (let k = 0; k < idxs.length; k++) {
    let j = k;
    while (j + 1 < idxs.length && idxs[j + 1] === idxs[j] + 1) j++;
    parts.push(j > k ? `${idxs[k] + 1}–${idxs[j] + 1}` : `${idxs[k] + 1}`);
    k = j;
  }
  return `${idxs.length > 1 ? "lines" : "line"} ${parts.join(", ")}`;
}
// Apply the lines an edit / undo response carries (if that song is still the one shown here).
function applyLineEdit(res) {
  if (!S.detail || res.song_id !== S.selected) return false;
  for (const [i, line] of Object.entries(res.lines)) S.detail.lines[Number(i)] = line;
  S.detail.rev = res.rev;
  S.lines = prepareSong(S.detail);
  if (S.live.id === S.selected) { S.live.lines = S.lines; S.live.key = `${S.selected}@${res.rev}`; }
  const wrap = $(".table-wrap"), scroll = wrap.scrollTop;
  renderLines();
  wrap.scrollTop = scroll;
  if (tap.on) armLine(tap.armed, "nearest");
  return true;
}
// body: exactly one of {start}, {shift}, {end} (seconds). Neighbours make room: the server pushes
// the lines in the way and flags them, so an edit only fails at either end of the song.
function editLine(idx, body) {
  const id = S.selected;
  return queueEdit(async () => {
    const res = await run(() => api(`/api/songs/${enc(id)}/lines/${idx}`, { method: "PUT", body }));
    if (!res) return;
    const what = "end" in body ? "ends" : "starts";
    if (!Object.keys(res.lines).length) { notify(`Line ${idx + 1} already ${what} there.`); return; }
    if (!applyLineEdit(res)) return;
    const l = S.lines[idx];
    let text = `Line ${idx + 1} now ${what} at ${fmtTime("end" in body ? l.end : l.start, 2)}`;
    const pushed = res.pushed || [];
    const moved = [[pushed.filter((i) => i > idx), "later"], [pushed.filter((i) => i < idx), "earlier"]]
      .filter(([ls]) => ls.length).map(([ls, dir]) => `${lineList(ls)} pushed ${dir}`);
    if (moved.length) text += ` (${moved.join(", ")}: check ${pushed.length > 1 ? "them" : "it"})`;
    notify(text, false, moved.length ? 9000 : 4000);
  });
}
function undoEdit() {
  const id = S.selected;
  if (!id) { notify("Pick a song first.", true); return; }
  queueEdit(async () => {
    const res = await run(() => api(`/api/songs/${enc(id)}/undo`, { method: "POST" }));
    if (!res || !applyLineEdit(res)) return;
    notify(`Undid the ${res.undone} edit${res.undo_left ? ` (${res.undo_left} more to undo)` : ""}`);
  });
}
$("#b-undo").addEventListener("click", undoEdit);

// The player position for Set start / Set end / T (null, with a notice, before anything is loaded).
function playhead() {
  if (!player.src || player.readyState === 0) { notify("Play the recording first, then set the time.", true); return null; }
  return Math.round(player.currentTime * 100) / 100;
}
function armLine(idx, scroll) {
  tap.armed = idx;
  for (const r of $$("tr.is-armed", tbody)) r.classList.remove("is-armed");
  const row = tap.on && idx >= 0 ? tbody.querySelector(`tr[data-idx="${idx}"]`) : null;
  if (!row) return;
  row.classList.add("is-armed");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (scroll) row.scrollIntoView({ block: scroll, behavior: reduce || scroll === "nearest" ? "auto" : "smooth" });
}
function setTap(on) {
  if (on && !S.selected) { notify("Pick a song first.", true); return; }
  tap.on = on;
  tap.song = on ? S.selected : null;
  const b = $("#b-tap");
  b.classList.toggle("on", on);
  b.setAttribute("aria-pressed", String(on));
  $("#t-lines").classList.toggle("tapping", on);
  if (!on) { armLine(-1); return; }
  // the first line at / after the playhead (prepared starts: untimed lines sit between their neighbours)
  const t = player.readyState > 0 ? player.currentTime : 0;
  const i = S.lines.findIndex((l) => l.start >= t - 0.01);
  armLine(i < 0 ? S.lines.length - 1 : i, "center");
  if (tap.armed >= 0) notify(`Tap mode: press T when line ${tap.armed + 1} starts. Click a line number to tap another line.`);
}
$("#b-tap").addEventListener("click", () => setTap(!tap.on));
function tapNow() {
  const idx = tap.armed;
  if (!(idx >= 0 && idx < S.lines.length)) { notify("All lines tapped. Click a line number to tap it again.", true); return; }
  const t = playhead();
  if (t == null) return;
  armLine(idx + 1 < S.lines.length ? idx + 1 : -1, "center");
  editLine(idx, { start: t });
}
// Its own listener (the H / [ / ] one above ignores modifier keys): Cmd/Ctrl+Z undoes the last
// timing edit; in Tap mode T taps and Escape leaves Tap mode. Typing in a field is left alone.
document.addEventListener("keydown", (e) => {
  if (e.repeat || e.target.closest?.("input, textarea, select, [contenteditable]")) return;
  const k = (e.key || "").toLowerCase();
  if (k === "z" && (e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey) { e.preventDefault(); undoEdit(); }
  else if (!tap.on || e.metaKey || e.ctrlKey || e.altKey) return;
  else if (k === "t") { e.preventDefault(); tapNow(); }
  else if (e.key === "Escape") setTap(false);
});

tbody.addEventListener("click", async (e) => {
  const cell = tap.on && e.target.closest("td.c-idx");
  if (cell) { armLine(Number(cell.closest("tr").dataset.idx)); return; }
  const b = e.target.closest("button");
  if (!b) return;
  const idx = Number(b.closest("tr").dataset.idx);
  const line = S.lines[idx];
  if (b.dataset.go !== undefined) {
    if (otherSongLive()) {
      notify(`${liveTitle()} is live. Select it to jump between its lines, or stop the session first.`, true);
      return;
    }
    if (!sessionActive() && S.live.id !== S.selected && !(await post("cue", { song_id: S.selected }))) return;
    post("jump", { line: idx, song_id: S.selected });   // the server refuses it if another song is live
  } else if (b.dataset.play !== undefined) {
    if (!player.src) return;
    player.currentTime = Math.max(0, line.start - 0.3);
    player.play().catch((err) => notify(`Cannot play: ${err.message}`, true));
  } else if (b.dataset.shift !== undefined) {
    editLine(idx, { shift: Number(b.dataset.shift) });
  } else if (b.dataset.set !== undefined) {
    const t = playhead();
    if (t != null) editLine(idx, { [b.dataset.set]: t });     // set: "start" | "end"
  }
});

function markRow(prev, next, cls, scroll) {
  if (prev === next) return next;
  tbody.querySelector(`tr[data-idx="${prev}"]`)?.classList.remove(cls);
  const row = tbody.querySelector(`tr[data-idx="${next}"]`);
  if (row) {
    row.classList.add(cls);
    if (scroll) row.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }
  return next;
}
function trackPlayer() {
  if (!S.lines.length) return;
  const cur = locate(S.lines, player.currentTime);
  S.playRow = markRow(S.playRow, cur.line_idx, "is-play", false);
}
player.addEventListener("timeupdate", trackPlayer);
player.addEventListener("seeked", trackPlayer);

$("#b-cue").addEventListener("click", () => {
  if (S.selected) post("cue", { song_id: S.selected });
});

// ------------------------------------------------------------------ live state
const lvl = { conf: $("#m-conf"), level: $("#m-level"), peak: $("#m-peak") };
const pct = (db) => `${Math.max(0, Math.min(100, ((db ?? -120) + 60) / 60 * 100))}%`;

function onState(msg) {
  S.last = msg;
  clock.update(msg);
  const st = $("#st");
  const shown = shownStatus(msg);
  if (st.dataset.status !== shown) {
    st.dataset.status = shown;
    $("b", st).textContent = STATUS_TEXT[shown] || shown;
  }
  $("#r-ref").textContent = fmtTime(msg.ref_time, 1);
  $("#r-live").textContent = msg.mode ? fmtTime(msg.live_time, 1) : "–";
  $("#r-tempo").textContent = msg.mode && msg.tempo ? `${Math.round(msg.tempo * 100)}%` : "–";
  lvl.conf.style.width = `${Math.round((msg.confidence || 0) * 100)}%`;
  lvl.conf.style.background = (msg.confidence || 0) < 0.35 ? "var(--warn)" : (msg.confidence || 0) < 0.6 ? "var(--gel)" : "var(--ok)";
  const level = msg.level_db, peak = msg.source?.peak_db;
  lvl.level.style.width = pct(level);
  lvl.level.style.background = (peak ?? -120) > -3 ? "var(--warn)" : (level ?? -120) > -12 ? "var(--gel)" : "var(--ok)";
  lvl.peak.style.left = pct(peak);
  $("#r-level").textContent = level != null && msg.mode ? `${Math.round(level)} dBFS` : "";
  if (msg.mode && msg.style && msg.song_id === S.selected && S.detail) {
    S.detail.style = msg.style;          // (switched from another console)
    showStyle(msg.style);
  }
  $("#b-hold").textContent = msg.status === "hold" ? "Resume" : "Hold";
  $("#b-hold").classList.toggle("on", msg.status === "hold");
  if (typeof msg.transpose === "number" && msg.mode) {
    S.transpose = msg.transpose;
    S.keySong = msg.song_id;
    if (typeof msg.key_auto === "boolean") S.autoKey = msg.key_auto;
    showKey();
  } else if (!msg.mode && S.selected) {
    keyFor(S.selected);                 // no session (any more): the selected song's own key
  }
  // a suggestion only exists for a running session (a stale one would apply an old song's key)
  const sug = $("#b-keysuggest"), k = msg.key_suggest;
  sug.hidden = !(msg.mode && S.autoKey && typeof k === "number" && k !== S.transpose);
  if (!sug.hidden) { sug.dataset.key = String(k); sug.textContent = `${fmtKey(k)}?`; }
  syncGo();
  const warn = [];
  const src = msg.source || {};
  if (msg.error) warn.push(`Tracking error: ${msg.error}`);
  if (msg.warning) warn.push(msg.warning);
  if (src.last_error) warn.push(`Input error: ${src.last_error}`);
  // exact digital silence arriving from the input (permission or channel problem); an input that
  // delivers nothing at all has stopped, which the input error above reports
  if (msg.mode && src.running && src.silent_s > 1.5 && !(src.stalled_s > 1)) {
    warn.push("No signal on the input. Check the channel selection and that this terminal app may use the microphone.");
  }
  if (shown === "stale") warn.push(`No audio has reached the tracker for ${(msg.age || 0).toFixed(1)} s; the prompters are frozen. Check the input.`);
  if (src.overflows) warn.push(`Audio processing fell behind; ${src.overflows} blocks dropped.`);
  if (msg.ended) warn.push("The recording has finished.");
  const w = $("#live-warn");
  w.hidden = !warn.length;
  w.textContent = warn.join(" ");
}

// The song on the prompter. A failed fetch is retried (with backoff) until it loads or the song
// changes: the server announces a song only when it changes.
let songMsg = null, songRetry = 0, songFails = 0;
async function onSong(msg) {
  songMsg = msg;
  const key = msg.id ? `${msg.id}@${msg.rev}` : null;
  if (key === S.live.key) return;
  clearTimeout(songRetry);
  S.live.key = key;
  S.live.id = msg.id;
  liveFocus = null;
  if (!msg.id) { S.live.lines = []; songFails = 0; renderSongs(); return; }
  if (!S.selected) await selectSong(msg.id);
  else if (msg.id === S.selected && S.detail && S.detail.rev !== msg.rev) await selectSong(msg.id, { keepAudio: true });
  if (S.live.key !== key) return;                  // a newer song message took over
  let lines = null;
  if (msg.id === S.selected && S.detail?.rev === msg.rev) lines = S.lines;
  else {
    try {
      lines = prepareSong(await api(`/api/songs/${enc(msg.id)}`));
    } catch (e) {
      if (!songFails) notify(`Cannot load the song on the prompter: ${e.message}. Retrying.`, true);
    }
    if (S.live.key !== key) return;
  }
  if (lines) {
    S.live.lines = lines;
    songFails = 0;
  } else {
    S.live.lines = [];
    S.live.key = null;
    const delay = Math.min(1000 * 2 ** songFails++, 10000);
    songRetry = setTimeout(() => { if (songMsg === msg && S.live.key == null) onSong(msg); }, delay);
  }
  renderSongs();
}

connectWS((msg) => {
  if (msg.type === "state") onState(msg);
  else if (msg.type === "song") onSong(msg);
  else if (msg.type === "display") applyDisplay(msg);
}, (ok) => {
  const c = $("#conn");
  c.dataset.status = ok ? "online" : "offline";
  $("b", c).textContent = ok ? "Connected" : "Disconnected";
  if (!ok) clock.moving = false;
  else {
    // (re)connected, possibly to a restarted server whose revs start again at 0: reload the song
    // on the prompter (and the selected one) when its announcement arrives
    S.live.key = null;
    if (S.detail) S.detail.rev = null;
  }
});

// the console's own small prompter + live row highlight
function frame(now) {
  requestAnimationFrame(frame);
  const lines = S.live.lines;
  if (!lines.length || !S.last || S.last.status === "idle") {
    if (liveFocus !== -1) {
      liveFocus = -1; liveViews = [];
      $("#live-line").textContent = ""; $("#live-next").textContent = "";
      S.liveRow = markRow(S.liveRow, -1, "is-live", false);
    }
    return;
  }
  const t = clock.tick(now);
  const cur = locate(lines, t);
  let f = focusLine(lines, cur, t, Number(S.display.lead_time ?? 1));
  if (f < 0) f = Math.max(0, cur.line_idx);
  if (f !== liveFocus) {
    liveFocus = f;
    liveViews = lines[f] ? renderLine(lines[f], $("#live-line")) : [];
    $("#live-next").textContent = lines[f + 1]?.text || "";
    if (S.live.id === S.selected) S.liveRow = markRow(S.liveRow, f, "is-live", $("#follow").checked);
  }
  paintLine(liveViews, t);
}

// ------------------------------------------------------------------ boot
syncFormVisibility();
refreshSongs();
refreshJobs();
loadDevices();
api("/api/display").then(applyDisplay).catch(() => {});
requestAnimationFrame(frame);
