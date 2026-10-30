// Operator console: projects, the live session (readout, fired log, controls), OSC targets and settings,
// and the cue editor with a WebAudio preview of the reference recording.
//
// The server is the source of truth: every cue edit goes through one promise queue (in order, with the
// rev of the sheet it edits) and every response carries the whole new sheet, which is re-rendered (ids
// are line numbers: they shift after an add / delete). Settings edits PUT /api/settings and re-render
// from the response; the WebSocket announces changes made by other consoles or by hand in cues.txt.
//
// Preview timing: one AudioContext per page; the reference is decoded once per project and played by an
// AudioBufferSourceNode started at `startCtx` from `offset`, so the context position is
// offset + (ctx.currentTime - startCtx) and the heard one (the playhead shown, Set, Tap, Add, pause) is
// that minus the output latency. Cue clicks are scheduled on the same context timeline by a 25 ms
// look-ahead loop (sample-aligned with the music); preview OSC is sent by setTimeout at
// cueCtx + outputLatency - targetLatency, at most once per cue and pass (play from a stop or a seek starts
// a new pass, resume after a pause continues it).
import {
  api, connectWS, esc, fmtClock, fmtTime, parseTime, RefClock, shownStatus, STATUS_TEXT, upload,
} from "./common.js";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const enc = encodeURIComponent;
const fs = $("#f-session");
const tbody = $("#t-cues tbody");

const S = {
  projects: [], selected: null, project: null, sheet: null, cues: [],
  idsSig: "", epoch: 0, projectRev: null, pendingTime: null,
  settings: null, settingsRev: undefined, armed: null, devices: [], targetsStatus: null,
  last: null, lastAt: 0, liveProject: null, wasRunning: false, offline: false,
  // the key panel: S.transpose belongs to S.keyProject (see keyFor); S.autoKey is a standing choice
  transpose: 0, autoKey: true, keyProject: null,
  fired: [], firedKeys: new Set(), firedRun: 0, lastSeq: 0, lastSig: null, firedProject: null, liveFired: null,
};
const clock = new RefClock();
const FIRED_MAX = 200;

// ------------------------------------------------------------------ notices
let noticeTimer = 0;
function notify(text, error = false, ms = error ? 9000 : 4000) {
  const el = $("#notice");
  el.textContent = text;
  el.title = text;
  el.classList.toggle("error", error);
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { el.textContent = ""; el.title = ""; }, ms);
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
const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;

// ------------------------------------------------------------------ targets: colours, click pitch, latency
const PALETTE = ["#5cc8ff", "#4fd1a1", "#c39bff", "#ff7ab6", "#c6e86a", "#ff9f5a", "#3fd0c9", "#ffe066"];
const PITCH = [0, 4, 7, 12, -5, 2, 9, 14];         // semitones from the 1.5 kHz click, per colour slot
let colourSlots = new Map();
let colourSig = "";

function targetNames() {           // settings targets first (stable colours), then declared / used ones
  const names = [];
  const add = (n) => { if (n && !names.includes(n)) names.push(n); };
  for (const t of S.settings?.targets || []) add(t.name);
  for (const t of S.sheet?.targets || []) add(t.name);
  for (const c of S.cues) add(c.target);
  return names;
}
function rebuildColours() {
  const names = targetNames();
  colourSlots = new Map(names.map((n, i) => [n, i % PALETTE.length]));
  const sig = `${names.join(",")}|${defaultTarget()}`;
  const changed = sig !== colourSig;
  colourSig = sig;
  return changed;
}
const defaultTarget = () => S.settings?.default_target || "main";
const effTarget = (cue) => cue.target || defaultTarget();
const colourOf = (name) => PALETTE[colourSlots.get(name) ?? 0];
const pitchOf = (name) => PITCH[colourSlots.get(name) ?? 0];
const settingsTarget = (name) => (S.settings?.targets || []).find((t) => t.name === name);
// the latency the server applies (settings.latency_s): 0 for a target the settings do not know
const latencyOf = (cue) => (settingsTarget(effTarget(cue))?.latency_ms || 0) / 1000;
function tagHtml(cue) {
  const name = effTarget(cue);
  const known = !!settingsTarget(name);
  if (!cue.target) return `<span class="tag default" title="Default target">${esc(name)}</span>`;
  return `<span class="tag${known ? "" : " unknown"}" style="background:${colourOf(name)}"
    ${known ? "" : `title="Not one of the OSC targets: its cues cannot be sent (add it under OSC targets)"`}>${esc(name)}</span>`;
}

// ------------------------------------------------------------------ projects
async function refreshProjects() {
  const list = await run(() => api("/api/projects"));
  if (!list) return;
  S.projects = list;
  if (S.selected && !list.some((p) => p.id === S.selected)) clearSelection();
  renderProjects();
}
let projectsTimer = 0;
const refreshProjectsSoon = () => { clearTimeout(projectsTimer); projectsTimer = setTimeout(refreshProjects, 400); };

function renderProjects() {
  const ul = $("#projects");
  ul.textContent = "";
  for (const p of S.projects) {
    const li = document.createElement("li");
    li.dataset.id = p.id;
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", String(p.id === S.selected));
    li.tabIndex = 0;
    const live = p.id === S.liveProject ? `<span class="live-tag">LIVE</span>` : "";
    const errs = p.n_errors ? ` · <span class="bad">${plural(p.n_errors, "error")}</span>` : "";
    li.innerHTML = `<span class="t" title="${esc(p.title || p.id)}">${esc(p.title || p.id)}${live}</span>
      <button class="btn ghost small del" type="button" data-del="${esc(p.id)}" title="Move to the trash"
        aria-label="Delete ${esc(p.title || p.id)}">✕</button>
      <span class="m">${p.duration ? fmtTime(p.duration, 0) : "–"} · ${p.n_cues == null ? "?" : plural(p.n_cues, "cue")}${errs}</span>
      ${p.error ? `<span class="err">${esc(p.error)}</span>` : ""}`;
    ul.append(li);
  }
  $("#projects-empty").hidden = S.projects.length > 0;
}

$("#projects").addEventListener("click", (e) => {
  const del = e.target.closest("[data-del]");
  if (del) { e.stopPropagation(); deleteProject(del.dataset.del); return; }
  const li = e.target.closest("li");
  if (li) selectProject(li.dataset.id);
});
$("#projects").addEventListener("keydown", (e) => {
  const li = e.target.closest("li");
  if (li && e.target === li && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); selectProject(li.dataset.id); }
});

async function deleteProject(id) {
  const p = S.projects.find((x) => x.id === id);
  if (!confirm(`Move "${p?.title || id}" to the trash? (projects/.trash keeps it)`)) return;
  const res = await run(() => api(`/api/projects/${enc(id)}`, { method: "DELETE" }), `Moved ${p?.title || id} to the trash`);
  if (res !== null && S.selected === id) clearSelection();
  refreshProjects();
}

function clearSelection() {
  S.selected = null; S.project = null; S.sheet = null; S.cues = []; S.idsSig = ""; S.epoch++; S.projectRev = null;
  setTap(false);
  closeAdd();
  preview.unload();
  wave.unload();
  renderCues(); renderIssues(); renderStartAt(); renderLogs([]);
  syncEditorTools();
  syncStemOptions();
  showStyle(null);
  $("#cues-title").textContent = "Cues";
}

// show logs (CSV, one per session) of the selected project, newest first
function renderLogs(names) {
  const box = $("#logs");
  box.hidden = !names.length || !S.selected;
  $("#logs-sum").textContent = `Show logs (${names.length})`;
  $("#logs-list").innerHTML = names.map((n) => `<li><a href="/api/projects/${enc(S.selected)}/logs/${enc(n)}"
    download="${esc(n)}">${esc(n)}</a></li>`).join("");
}

async function selectProject(id) {
  const detail = await run(() => api(`/api/projects/${enc(id)}`));
  if (!detail) return;
  const changed = S.selected !== id;
  if (changed) {
    setTap(false);
    closeAdd();
    preview.unload();
    wave.unload();
    S.projectRev = null;
    if (!sessionRunning()) keyFor(id);     // while a session runs the key panel shows its key
  }
  S.selected = id;
  S.project = detail.project || {};
  if (detail.rev != null) S.projectRev = detail.rev;
  applySheet(sheetOf(detail), { reset: changed });
  renderLogs(detail.logs || []);
  renderProjects();
  $("#cues-title").textContent = `Cues, ${S.project.title || id}`;
  showStyle(S.project.style);
  syncStemOptions();
  syncEditorTools();
  if (changed) {
    wave.load(id);
    preview.load(id);
  }
}

// ------------------------------------------------------------------ new project
const fNew = $("#f-new");
function showNew(show) {
  fNew.hidden = !show;
  $("#b-new").setAttribute("aria-expanded", String(show));
  if (show) fNew.elements.title.focus();
}
$("#b-new").addEventListener("click", () => showNew(fNew.hidden));
$("#b-new-cancel").addEventListener("click", () => showNew(false));
fNew.elements.audio.addEventListener("change", () => {
  const f = fNew.elements.audio.files[0];
  if (f && !fNew.elements.title.value) fNew.elements.title.value = f.name.replace(/\.[^.]+$/, "");
});
fNew.addEventListener("submit", async (e) => {
  e.preventDefault();
  const el = fNew.elements;
  const form = new FormData();
  form.append("title", el.title.value.trim());
  form.append("audio", el.audio.files[0]);
  if (el.cues.files[0]) form.append("cues", el.cues.files[0]);
  else if (el.cues_text.value.trim()) form.append("cues_text", el.cues_text.value);
  if (el.vocals.files[0]) form.append("vocals", el.vocals.files[0]);
  form.append("style", el.style.value);
  const btn = $("button[type=submit]", fNew);
  const box = $("#new-progress"), bar = $("progress", box), msg = $(".msg", box);
  btn.disabled = true;
  box.hidden = false;
  bar.value = 0;
  msg.textContent = "Uploading…";
  try {
    const res = await upload("/api/projects", form, (f, done) => {
      if (done) { bar.removeAttribute("value"); msg.textContent = "Converting the audio and analysing it…"; }
      else { bar.value = f; msg.textContent = `Uploading ${Math.round(f * 100)}%`; }
    });
    const proj = res.project || res;
    const warns = res.warnings || [];
    notify(`Created ${proj.title || proj.id}${warns.length ? `. ${warns.join(" ")}` : ""}`, false, warns.length ? 12000 : 4000);
    fNew.reset();
    showNew(false);
    box.hidden = true;
    await refreshProjects();
    if (proj.id) selectProject(proj.id);
  } catch (err) {
    bar.value = 0;
    msg.textContent = err.message;
    notify(err.message, true);
  } finally {
    btn.disabled = false;
  }
});

// ------------------------------------------------------------------ cue sheet
// The edit responses are the sheet itself (+ "added" / "undo_left"); GET /api/projects/{id} nests it
// under "cues". Returns the sheet in either case (null when there is none).
function sheetOf(res) {
  if (res && Array.isArray(res.cues)) return res;
  if (res?.cues && Array.isArray(res.cues.cues)) return res.cues;
  return null;
}

function applySheet(sheet, { reset = false } = {}) {
  if (!sheet) return;
  const sig = (sheet.cues || []).map((c) => c.id).join(",");
  // queued edits refer to cues by line number: a new line layout makes them unsafe (see cueEdit)
  if (reset || sig !== S.idsSig) S.epoch++;
  // the tap-armed cue and the Start at cue are chosen by line number: follow them to their new lines
  const armedOld = tap.on && !reset ? cueById(tap.armed) : null;
  const at = fs.elements.start_at.value;
  const startOld = at.startsWith("cue:") ? cueById(Number(at.slice(4))) : null;
  const sameLines = lineSet(S.cues) === lineSet(sheet.cues || []);
  S.idsSig = sig;
  S.sheet = sheet;
  S.cues = sheet.cues || [];
  S.pendingTime = null;
  if (sheet.project_rev != null) S.projectRev = sheet.project_rev;   // its own announcement is then a no-op
  const lost = [];
  if (armedOld) {
    const c = sameCue(armedOld, sameLines);
    tap.armed = c ? c.id : null;
    if (!c) lost.push(`Tap: cue ${armedOld.id} (${cueBody(armedOld)}) is gone, so no cue is armed. Click a cue number to arm one.`);
  }
  let startAt = at;
  if (startOld) {                          // another project (reset) starts from its beginning
    const c = reset ? null : sameCue(startOld, sameLines);
    startAt = c ? `cue:${c.id}` : reset ? "" : "time";
    if (!c && !reset) {                    // keep its position, visibly, rather than another cue's
      fs.elements.start_time.value = fmtTime(startOld.time, 3);
      lost.push(`Start at: cue ${startOld.id} (${cueBody(startOld)}) is gone, so Start at is now its time ${fmtTime(startOld.time, 3)}.`);
    }
  }
  rebuildColours();
  renderIssues();
  renderCues();
  renderStartAt(startAt);
  renderAddTargets();
  // (shown after the notice of the edit or reload that caused it, which would otherwise replace it)
  if (lost.length) setTimeout(() => notify(lost.join(" "), true), 0);
  preview.cuesChanged();
  wave.dirty = true;
  const p = S.projects.find((x) => x.id === S.selected);
  if (p) {
    const errs = (sheet.issues || []).filter((i) => i.severity === "error").length;
    if (p.n_cues !== S.cues.length || p.n_errors !== errs) { p.n_cues = S.cues.length; p.n_errors = errs; renderProjects(); }
  }
}

// The cue of S.cues that is `old` (a cue of the previous sheet): the one with the same time, target and
// message (on the same line if there are several); else, when no line was added or removed (an edit in
// place), the same line if its target and message are unchanged (its time was edited). null: it is gone.
const lineSet = (cues) => cues.map((c) => c.id).sort((a, b) => a - b).join(",");
function sameCue(old, sameLines) {
  const key = cueKey(old);
  const same = S.cues.filter((c) => cueKey(c) === key);
  if (same.length) return same.find((c) => c.id === old.id) || same[0];
  const c = sameLines ? cueById(old.id) : null;
  return c && (c.target ?? null) === (old.target ?? null) && c.message === old.message ? c : null;
}

async function reloadSheet(pid = S.selected) {
  if (!pid) return;
  const res = await run(() => api(`/api/projects/${enc(pid)}/cues`));
  if (res && S.selected === pid) applySheet(sheetOf(res));
}

function renderIssues() {
  const box = $("#issues"), list = $("#issues-list");
  const issues = S.sheet?.issues || [];
  box.hidden = !issues.length;
  if (!issues.length) { list.textContent = ""; return; }
  const errs = issues.filter((i) => i.severity === "error").length;
  const warns = issues.length - errs;
  box.classList.toggle("has-errors", errs > 0);
  $("#issues-sum").textContent = [errs ? plural(errs, "error") + " (lines ignored)" : "", warns ? plural(warns, "warning") : ""]
    .filter(Boolean).join(", ");
  if (errs) box.open = true;
  list.innerHTML = issueItems(issues);
}
const issueItems = (issues) => issues.map((i) => `<li class="${i.severity === "error" ? "error" : "warning"}" data-line="${i.line}">
  <span class="ln">${i.line ? `line ${i.line}` : "file"}</span> ${esc(i.message)}</li>`).join("");

$("#issues-list").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-line]");
  if (!li) return;
  const line = Number(li.dataset.line);
  const row = tbody.querySelector(`tr[data-id="${line}"]`);
  if (row) { row.scrollIntoView({ block: "center" }); flashRow(line); }
  else openFileEditor(line);                // an ignored line: show it in the file
});

const cueBody = (c) => {
  const text = c.target ? `${c.target} ${c.message}` : c.message;
  return c.note ? `${text} # ${c.note}` : text;
};

function renderCues() {
  inline = null;
  if (tap.on && tap.project !== S.selected) setTap(false);
  const byLine = new Map();
  for (const i of S.sheet?.issues || []) {
    if (!byLine.has(i.line)) byLine.set(i.line, []);
    byLine.get(i.line).push(i.message);
  }
  tbody.innerHTML = S.cues.map((c) => {
    const warn = byLine.get(c.id);
    const cls = [warn ? "warn" : "", tap.on && c.id === tap.armed ? "is-armed" : ""].join(" ");
    return `<tr data-id="${c.id}" class="${cls}"${warn ? ` title="${esc(warn.join("; "))}"` : ""}>
      <td class="c-go"><button class="btn small" data-act="go" type="button" aria-label="Go to cue ${c.id}">Go</button></td>
      <td class="c-idx" title="Line ${c.id} of cues.txt">${c.id}</td>
      <td class="c-time" title="Click to edit (cue time syntax: 75.3, 1:05.200 …)">${fmtTime(c.time, 3)}</td>
      <td class="c-tgt">${tagHtml(c)}</td>
      <td class="c-msg" title="Double-click to edit">${esc(c.message)}${c.note ? `<span class="note-inline"># ${esc(c.note)}</span>` : ""}</td>
      <td class="c-note">${esc(c.note)}</td>
      <td class="c-act"><div class="acts">
        <button class="btn small" data-act="play" type="button" title="Preview from 1 s before this cue">Play</button>
        <span class="seg" role="group" aria-label="Move cue ${c.id}">
          <button class="btn small" data-act="shift" data-dt="-0.1" type="button" title="0.1 s earlier" aria-label="0.1 s earlier">−.1</button>
          <button class="btn small" data-act="shift" data-dt="0.1" type="button" title="0.1 s later" aria-label="0.1 s later">+.1</button>
          <button class="btn small" data-act="shift" data-dt="-0.01" type="button" title="10 ms earlier" aria-label="10 ms earlier">−.01</button>
          <button class="btn small" data-act="shift" data-dt="0.01" type="button" title="10 ms later" aria-label="10 ms later">+.01</button>
        </span>
        <button class="btn small" data-act="set" type="button" title="Set the time to the preview playhead (in Tap mode: T)">Set</button>
        <span class="sep"></span>
        <button class="btn small" data-act="fire" type="button" title="Send this cue's OSC now (also when SAFE)">Fire</button>
        <button class="btn small danger" data-act="delete" type="button" title="Delete this cue line" aria-label="Delete cue ${c.id}">✕</button>
      </div></td></tr>`;
  }).join("");
  const empty = $("#cues-empty");
  empty.hidden = S.cues.length > 0;
  empty.textContent = !S.selected ? "Pick a project to edit its cues and preview them against the reference."
    : "No cues yet. Play the preview and use Add cue at playhead, or Edit file.";
  marks.passed = 0;
  marks.fired = null;
  marks.next = null;
  goBlocked = null;
  syncGo();
  markLive();
  if (tap.on) armCue(tap.armed);
}

function flashRow(id) {
  const row = tbody.querySelector(`tr[data-id="${id}"]`);
  if (!row) return;
  row.classList.remove("flash");
  void row.offsetWidth;                     // restart the animation
  row.classList.add("flash");
}

// Row marks that follow time: passed (relative to the preview playhead), live fired / next cue.
const marks = { passed: 0, fired: null, next: null };
function updatePassed(pos) {
  let lo = 0, hi = S.cues.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (S.cues[mid].time < pos - 5e-4) lo = mid + 1; else hi = mid; }
  if (lo === marks.passed) return;
  const rows = tbody.rows;
  for (let i = Math.min(lo, marks.passed); i < Math.max(lo, marks.passed); i++) rows[i]?.classList.toggle("passed", i < lo);
  marks.passed = lo;
}
function markRow(prev, next, cls, scroll) {
  if (prev === next) return next;
  if (prev != null) tbody.querySelector(`tr[data-id="${prev}"]`)?.classList.remove(cls);
  const row = next == null ? null : tbody.querySelector(`tr[data-id="${next}"]`);
  if (row) {
    row.classList.add(cls);
    if (scroll) row.scrollIntoView({ block: "nearest", behavior: reduceMotion() ? "auto" : "smooth" });
  }
  return next;
}
const reduceMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
function markLive() {
  const here = liveHere();
  const fired = here && S.liveFired?.project === S.selected ? S.liveFired.id : null;
  const next = here && !S.offline ? S.last.upcoming?.[0]?.id ?? null : null;
  const follow = $("#follow").checked && !tap.on && !inline;
  marks.fired = markRow(marks.fired, fired, "is-fired", follow);
  marks.next = markRow(marks.next, next, "is-next", false);
}

// ------------------------------------------------------------------ edit queue
// Cue edits (and undos) go to the server one at a time, in order: fast taps never overtake.
let editQueue = Promise.resolve();
function queueEdit(fn) {
  editQueue = editQueue.then(fn).catch((e) => notify(e.message, true));
  return editQueue;
}

// One cue edit. It runs after the edits queued before it, with the rev of the sheet it then edits. An
// edit that names a cue by id (line number) is dropped when the line layout changed after it was asked
// for (an add / delete / undo / reload in between), instead of hitting another line.
function cueEdit(method, path, { body, form, rev = true, anyEpoch = false } = {}) {
  const pid = S.selected, epoch = S.epoch;
  if (!pid || !S.sheet) { notify("Pick a project first.", true); return Promise.resolve(null); }
  let out = null;
  return queueEdit(async () => {
    if (S.selected !== pid || !S.sheet) return;
    if (!anyEpoch && S.epoch !== epoch) {
      notify("The cue list changed before this edit ran, so it was not applied. Please do it again.", true);
      return;
    }
    let url = `/api/projects/${enc(pid)}/cues${path}`;
    let b = body;
    if (rev && method === "DELETE") url += `?rev=${enc(S.sheet.rev)}`;
    else if (rev && !form) b = { ...(body || {}), rev: S.sheet.rev };
    let res;
    try {
      res = await api(url, { method, body: b, form });
    } catch (e) {
      S.pendingTime = null;
      if (e.status === 409) {
        // the file changed under the edit (the body carries the current sheet), or nothing to undo
        const cur = sheetOf(e.data);
        if (cur && S.selected === pid) applySheet(cur);
        else await reloadSheet(pid);
        notify(e.message, true);
      } else {
        notify(e.message, true);
      }
      return;
    }
    if (S.selected === pid) applySheet(sheetOf(res));
    out = res;
  }).then(() => out);
}

const cueById = (id) => S.cues.find((c) => c.id === id);
async function patchCue(id, body) {
  const res = await cueEdit("PATCH", `/${id}`, { body });
  if (!res) return;
  const c = cueById(id);
  if (!c) return;
  notify("message" in body ? `Cue ${id}: ${cueBody(c)}` : `Cue ${id} now at ${fmtTime(c.time, 3)}`);
  flashRow(id);
}
async function deleteCue(id) {
  const c = cueById(id);
  if (!c || !confirm(`Delete cue ${id} (${fmtTime(c.time, 3)} ${c.message})? Undo can restore it.`)) return;
  const res = await cueEdit("DELETE", `/${id}`);
  if (res) notify(`Deleted cue ${id}. Undo (⌘Z / Ctrl+Z) restores it.`);
}
async function undoEdit() {
  if (!S.selected) { notify("Pick a project first.", true); return; }
  const res = await cueEdit("POST", "/undo", { rev: false, anyEpoch: true });
  if (res) notify(`Undid ${res.undone || "the last edit"}${res.undo_left ? ` (${res.undo_left} more to undo)` : ""}`);
}
$("#b-undo").addEventListener("click", undoEdit);

async function fireCue(id) {
  const pid = S.selected;
  const res = await run(() => api(`/api/projects/${enc(pid)}/cues/${id}/fire`,
    { method: "POST", body: { source: "manual", rev: S.sheet?.rev } }));
  if (!res) return;
  if (res.ok) notify(`Sent cue ${id}: ${res.address || ""} to ${res.target || "the default target"}`);
  else notify(`Cue ${id} not sent: ${res.error || "unknown error"}`, true);
}

// ------------------------------------------------------------------ cue table actions + inline editing
// Go: with a live session of this project, jump it to the cue (which fires it); without one, start there.
let goBlocked = null;
const otherProjectLive = () => !!(sessionRunning() && S.selected && S.last.project_id !== S.selected);
function syncGo() {
  const blocked = otherProjectLive();
  if (blocked === goBlocked) return;
  goBlocked = blocked;
  const why = blocked ? `${liveTitle()} is live: select it to jump between its cues, or stop the session first`
    : "Live session: jump here (fires this cue). No session: start here.";
  for (const b of $$("button[data-act=go]", tbody)) { b.disabled = blocked; b.title = why; }
}
const liveTitle = () => S.projects.find((p) => p.id === S.last?.project_id)?.title || S.last?.project_id || "Another project";

tbody.addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-id]");
  if (!tr) return;
  const id = Number(tr.dataset.id);
  const cue = cueById(id);
  if (!cue) return;
  if (tap.on && e.target.closest("td.c-idx")) { armCue(id); return; }
  const timeCell = e.target.closest("td.c-time");
  if (timeCell && !timeCell.querySelector("input")) { startInline(timeCell, id, "time"); return; }
  const b = e.target.closest("button[data-act]");
  if (!b) return;
  const act = b.dataset.act;
  if (act === "go") goToCue(cue);
  else if (act === "play") preview.playFrom(Math.max(0, cue.time - 1));
  else if (act === "shift") patchCue(id, { shift: Number(b.dataset.dt) });
  else if (act === "set") patchCue(id, { time: playhead() });
  else if (act === "fire") fireCue(id);
  else if (act === "delete") deleteCue(id);
});
tbody.addEventListener("dblclick", (e) => {
  const cell = e.target.closest("td.c-msg, td.c-note, td.c-tgt");
  const tr = e.target.closest("tr[data-id]");
  if (cell && tr && !cell.querySelector("input")) {
    startInline(tr.querySelector("td.c-msg"), Number(tr.dataset.id), "message");
  }
});

function goToCue(cue) {
  if (otherProjectLive()) { notify(`${liveTitle()} is live. Stop it first to jump in this project.`, true); return; }
  if (liveHere()) { sessionPost("jump", { cue_id: cue.id, project_id: S.selected, rev: S.sheet?.rev }); return; }
  fs.elements.start_at.value = `cue:${cue.id}`;
  syncFormVisibility();
  notify(`Start at cue ${cue.id} (${fmtTime(cue.time, 3)}): press Start.`);
}

// Inline editors: the time cell (cue time syntax, sent as text: the server parses it) and the message
// (the cue line after its time: "[target] /address args [# note]"). Enter saves, Esc or leaving cancels.
let inline = null;
function startInline(td, id, kind) {
  cancelInline();
  const cue = cueById(id);
  if (!cue || !td) return;
  const orig = kind === "time" ? fmtTime(cue.time, 3) : cueBody(cue);
  const html = td.innerHTML;
  const input = document.createElement("input");
  input.value = orig;
  input.spellcheck = false;
  input.setAttribute("aria-label", kind === "time" ? `Time of cue ${id}` : `Message of cue ${id}`);
  input.title = "Enter saves, Esc cancels";
  td.textContent = "";
  td.append(input);
  inline = { td, id, kind, html, input };
  input.focus();
  input.select();
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); commitInline(); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); cancelInline(); }
  });
  input.addEventListener("blur", () => { if (inline?.input === input) cancelInline(); });
}
function cancelInline() {
  if (!inline) return;
  const { td, html } = inline;
  inline = null;
  td.innerHTML = html;
}
function commitInline() {
  if (!inline) return;
  const { id, kind, input } = inline;
  const text = input.value.trim();
  const orig = kind === "time" ? fmtTime(cueById(id)?.time, 3) : cueBody(cueById(id) || {});
  cancelInline();
  if (!text || text === orig) return;
  patchCue(id, kind === "time" ? { time: text } : { message: text });
}

// ------------------------------------------------------------------ tap mode
// T sets the armed cue's time to the preview playhead (1 ms) and arms the next cue; click a # to arm.
const tap = { on: false, armed: null, project: null };
function armCue(id, scroll) {
  tap.armed = id;
  for (const r of $$("tr.is-armed", tbody)) r.classList.remove("is-armed");
  const row = tap.on && id != null ? tbody.querySelector(`tr[data-id="${id}"]`) : null;
  if (!row) return;
  row.classList.add("is-armed");
  if (scroll) row.scrollIntoView({ block: scroll, behavior: reduceMotion() || scroll === "nearest" ? "auto" : "smooth" });
}
function setTap(on) {
  if (on && !S.selected) { notify("Pick a project first.", true); return; }
  if (on && !S.cues.length) { notify("This project has no cues to tap yet.", true); return; }
  tap.on = on;
  tap.project = on ? S.selected : null;
  const b = $("#b-tap");
  b.classList.toggle("on", on);
  b.setAttribute("aria-pressed", String(on));
  $("#t-cues").classList.toggle("tapping", on);
  if (!on) { armCue(null); return; }
  const t = preview.heard();
  const next = S.cues.find((c) => c.time >= t - 0.01) || S.cues[S.cues.length - 1];
  armCue(next.id, "center");
  notify(`Tap mode: press T at cue ${next.id}. Click a cue number to tap another cue. Esc leaves.`);
}
$("#b-tap").addEventListener("click", () => setTap(!tap.on));
function tapNow() {
  const idx = S.cues.findIndex((c) => c.id === tap.armed);
  if (idx < 0) { notify("No cue is armed (all tapped, or the armed cue is gone). Click a cue number to tap it.", true); return; }
  const id = tap.armed;
  armCue(S.cues[idx + 1]?.id ?? null, "center");
  patchCue(id, { time: playhead() });
}
const playhead = () => Math.round(preview.heard() * 1000) / 1000;      // what is heard now (1 ms)

// ------------------------------------------------------------------ add cue at playhead
const fAdd = $("#f-add");
function renderAddTargets() {
  const sel = fAdd.elements.target;
  const keep = sel.value;
  sel.innerHTML = `<option value="">Default (${esc(defaultTarget())})</option>`
    + targetNames().map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  sel.value = [...sel.options].some((o) => o.value === keep) ? keep : "";
}
function openAdd() {
  if (!S.selected) { notify("Pick a project first.", true); return; }
  renderAddTargets();
  fAdd.elements.time.value = fmtTime(playhead(), 3);
  fAdd.hidden = false;
  $("#b-add").setAttribute("aria-expanded", "true");
  fAdd.elements.message.focus();
}
function closeAdd() {
  fAdd.hidden = true;
  $("#b-add").setAttribute("aria-expanded", "false");
}
$("#b-add").addEventListener("click", () => (fAdd.hidden ? openAdd() : closeAdd()));
$("#b-add-cancel").addEventListener("click", closeAdd);
fAdd.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); closeAdd(); } });
fAdd.addEventListener("submit", async (e) => {
  e.preventDefault();
  const el = fAdd.elements;
  const msg = el.message.value.trim();
  if (!msg) { notify("Write the OSC message, e.g. /cue/go 12", true); el.message.focus(); return; }
  const text = el.time.value.trim();
  const t = parseTime(text);
  const message = el.target.value ? `${el.target.value} ${msg}` : msg;
  const res = await cueEdit("POST", "/add", { body: { time: t ?? text, message }, anyEpoch: true });
  if (!res) return;
  el.message.value = "";
  closeAdd();
  const id = res.added;
  notify(`Added cue ${id ?? ""} at ${fmtTime(cueById(id)?.time ?? t, 3)}`);
  if (id != null) {
    tbody.querySelector(`tr[data-id="${id}"]`)?.scrollIntoView({ block: "center" });
    flashRow(id);
  }
});

// ------------------------------------------------------------------ edit file / download / replace
const dFile = $("#d-file");
const fFile = $("#f-file");
const fileState = { rev: null, crlf: false };
function openFileEditor(line) {
  if (!S.sheet) { notify("Pick a project first.", true); return; }
  const text = S.sheet.text || "";
  fileState.rev = S.sheet.rev;
  fileState.crlf = text.includes("\r\n");
  fFile.elements.text.value = text;
  showFileIssues(S.sheet.issues || [], "");
  dFile.showModal();
  if (line) selectLine(line);
  else fFile.elements.text.focus();
}
function showFileIssues(issues, msg, error = false) {
  $("#file-issues").innerHTML = issueItems(issues);
  const m = $("#file-msg");
  m.textContent = msg;
  m.classList.toggle("error", error);
}
function selectLine(line) {
  const ta = fFile.elements.text;
  const lines = ta.value.split("\n");
  let a = 0;
  for (let i = 0; i < Math.min(line - 1, lines.length); i++) a += lines[i].length + 1;
  const b = a + (lines[line - 1] || "").length;
  ta.focus();
  ta.setSelectionRange(a, b);
  const lh = parseFloat(getComputedStyle(ta).lineHeight) || 20;
  ta.scrollTop = Math.max(0, (line - 4) * lh);
}
$("#file-issues").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-line]");
  if (li && Number(li.dataset.line)) selectLine(Number(li.dataset.line));
});
$("#b-editfile").addEventListener("click", () => openFileEditor());
$("#b-file-cancel").addEventListener("click", () => dFile.close());
$("#b-file-save").addEventListener("click", () => {
  const pid = S.selected;
  // a textarea gives \n line breaks: keep the file's \r\n style
  let text = fFile.elements.text.value;
  if (fileState.crlf) text = text.replace(/\r?\n/g, "\r\n");
  const btn = $("#b-file-save");
  btn.disabled = true;
  queueEdit(async () => {
    try {
      if (S.selected !== pid) return;
      let res;
      try {
        res = await api(`/api/projects/${enc(pid)}/cues`, { method: "PUT", body: { text, rev: fileState.rev } });
      } catch (e) {
        if (e.status !== 409) { showFileIssues([], e.message, true); return; }
        const cur = sheetOf(e.data);
        if (cur) applySheet(cur); else await reloadSheet(pid);
        fileState.rev = S.sheet?.rev ?? null;
        showFileIssues(S.sheet?.issues || [], "The file changed meanwhile (another console or an editor). "
          + "Your text was not saved: Save again to replace the file with it, or Close.", true);
        return;
      }
      const sheet = sheetOf(res);
      applySheet(sheet);
      fileState.rev = sheet?.rev ?? null;
      const issues = sheet?.issues || [];
      if (!issues.length) { dFile.close(); notify("Cue file saved."); return; }
      const errs = issues.filter((i) => i.severity === "error").length;
      showFileIssues(issues, `Saved, with ${[errs ? plural(errs, "error") : "", issues.length - errs ? plural(issues.length - errs, "warning") : ""]
        .filter(Boolean).join(" and ")}${errs ? " (lines with errors are ignored)" : ""}.`, errs > 0);
    } finally {
      btn.disabled = false;
    }
  });
});

$("#b-replace").addEventListener("click", () => {
  if (!S.selected) { notify("Pick a project first.", true); return; }
  $("#i-replace").click();
});
$("#i-replace").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  e.target.value = "";
  if (!file || !S.selected) return;
  if (!confirm(`Replace the cue file of ${S.project?.title || S.selected} with ${file.name}? Undo can bring the old one back.`)) return;
  const form = new FormData();
  form.append("cues", file);
  const res = await cueEdit("POST", "/upload", { form, rev: false, anyEpoch: true });
  if (res) notify(`Replaced the cue file with ${file.name}: ${plural(S.cues.length, "cue")}.`);
});

function syncEditorTools() {
  const has = !!S.selected;
  for (const id of ["#b-undo", "#b-tap", "#b-add", "#b-editfile", "#b-replace"]) $(id).disabled = !has;
  const a = $("#a-download");
  a.href = has ? `/api/projects/${enc(S.selected)}/cues.txt` : "#";
  a.setAttribute("download", has ? `${S.selected}-cues.txt` : "cues.txt");
  a.setAttribute("aria-disabled", String(!has));
  $("#b-play").disabled = !has;
}

// ------------------------------------------------------------------ preview audio engine
const LOOKAHEAD_MS = 25;          // scheduler period
const AHEAD_S = 0.12;             // how far ahead clicks / OSC sends are scheduled
const CLICK_HZ = 1500;

const preview = {
  ctx: null, master: null, clickBus: null, buffer: null, pid: null, abort: null, loading: false, wantPlay: false,
  src: null, playing: false, paused: false, offset: 0, startCtx: 0, timer: 0,
  // oscTimers: cue key -> pending send; sent: the cue keys whose OSC this pass has sent
  clickQ: [], ci: 0, oscQ: [], oi: 0, clickNodes: new Set(), oscTimers: new Map(), sent: new Set(),

  ensureCtx() {
    if (this.ctx) return this.ctx;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    this.ctx = new Ctx({ latencyHint: "interactive" });
    this.master = this.ctx.createGain();
    this.master.connect(this.ctx.destination);
    this.clickBus = this.ctx.createGain();
    this.clickBus.gain.value = Number($("#c-vol").value);
    this.clickBus.connect(this.ctx.destination);
    return this.ctx;
  },
  duration() { return this.buffer?.duration || S.project?.duration || wave.data?.duration || 0; },
  position() {
    let t = this.offset;
    if (this.playing && this.ctx) t += Math.max(0, this.ctx.currentTime - this.startCtx);
    const d = this.duration();
    return d ? Math.min(Math.max(0, t), d) : Math.max(0, t);
  },
  // seconds from the context's timeline to the speakers (the preview OSC sends wait for it too)
  outLat() { return this.ctx ? this.ctx.outputLatency || this.ctx.baseLatency || 0 : 0; },
  // The position being heard: the context runs ahead of the speakers by the output latency. The playhead
  // shown, Set, Tap and Add cue at playhead use it, so a cue tapped on a heard event lands on it.
  heard() {
    if (!this.playing || !this.ctx) return this.position();
    const t = this.offset + Math.max(0, this.ctx.currentTime - this.startCtx - this.outLat());
    const d = this.duration();
    return d ? Math.min(t, d) : t;
  },

  async load(pid) {
    this.abort?.abort();
    const ac = new AbortController();
    this.abort = ac;
    this.loading = true;
    loadText("Loading audio…");
    try {
      const res = await fetch(`/api/projects/${enc(pid)}/audio/mix`, { signal: ac.signal });
      if (!res.ok) throw new Error(`audio: ${res.status} ${res.statusText}`);
      const data = await res.arrayBuffer();
      if (ac.signal.aborted) return;
      loadText("Decoding…");
      const buf = await this.ensureCtx().decodeAudioData(data);
      if (ac.signal.aborted || S.selected !== pid) return;
      this.buffer = buf;
      this.pid = pid;
      loadText("");
      wave.dirty = true;
      if (this.wantPlay) this.play();
    } catch (e) {
      if (ac.signal.aborted) return;
      loadText(`Preview unavailable: ${e.message}`, true);
    } finally {
      if (this.abort === ac) { this.loading = false; this.abort = null; }
      this.wantPlay = false;
    }
  },
  unload() {
    this.pause();
    this.paused = false;
    this.abort?.abort();
    this.abort = null;
    this.loading = false;
    this.wantPlay = false;
    this.buffer = null;
    this.pid = null;
    this.offset = 0;
    loadText("");
  },

  play() {
    if (!S.selected) return;
    const ctx = this.ensureCtx();
    ctx.resume?.();                         // inside the click / key press that asked for it
    if (!this.buffer) {
      if (this.loading) { this.wantPlay = true; loadText("Loading audio… (plays when ready)"); }
      else notify("The reference audio is not loaded.", true);
      return;
    }
    if (this.playing) this.stopSource();
    // resuming after a pause continues the pass; play from a stop, a seek or the end starts a new one
    let resume = this.paused;
    this.paused = false;
    if (this.offset >= this.buffer.duration - 0.01) { this.offset = 0; resume = false; }
    const src = ctx.createBufferSource();
    src.buffer = this.buffer;
    src.connect(this.master);
    this.startCtx = ctx.currentTime + 0.05;
    src.start(this.startCtx, this.offset);
    src.onended = () => {
      if (this.src !== src) return;         // stopped on purpose (pause / seek)
      this.offset = this.buffer?.duration || this.offset;
      this.src = null;
      this.playing = false;
      this.paused = false;
      this.cancelScheduled();
      clearInterval(this.timer);
    };
    this.src = src;
    this.playing = true;
    if (!resume) this.sent.clear();         // a new pass: every cue may send once again
    this.rebuild(this.offset, resume);
    clearInterval(this.timer);
    this.timer = setInterval(() => this.tick(), LOOKAHEAD_MS);
    this.tick();
  },
  pause() {
    if (!this.playing) return;
    this.offset = this.heard();             // where the operator heard it stop (Set uses it)
    this.stopSource();
    this.paused = true;
  },
  stopSource() {
    const src = this.src;
    this.src = null;
    this.playing = false;
    clearInterval(this.timer);
    this.cancelScheduled();
    if (src) { src.onended = null; try { src.stop(); } catch { /* not started */ } src.disconnect(); }
  },
  toggle() { if (this.playing) this.pause(); else this.play(); },
  seek(t) {
    const d = this.duration();
    t = Math.max(0, d ? Math.min(t, d) : t);
    this.paused = false;                    // a seek starts a new pass
    if (this.playing) { this.stopSource(); this.offset = t; this.play(); }
    else this.offset = t;
    if (tap.on) {
      const next = S.cues.find((c) => c.time >= t - 0.01);
      if (next) armCue(next.id, "nearest");
    }
  },
  playFrom(t) {
    this.seek(t);
    if (!this.playing) this.play();
  },

  // Queues of the cues still ahead of `from` (song time): clicks in cue time order, OSC sends in send time
  // order (each target's latency moves its sends earlier). Continuing a pass (`resume`), the OSC queue
  // also keeps the cues before `from` whose send (after the output latency) is still due, and the ones
  // in `pending` (cue keys whose send a re-arm cancelled: one already due whose timer had not run yet
  // is sent at once, not dropped).
  rebuild(from, resume = false, pending = null) {
    const ahead = S.cues.filter((c) => c.time >= from - 1e-6);
    this.clickQ = ahead;
    this.ci = 0;
    const outLat = this.outLat();
    const osc = resume ? S.cues.filter((c) => c.time >= from - 1e-6 || c.time + outLat - latencyOf(c) >= from - 1e-6
      || pending?.has(cueKey(c))) : ahead;
    this.oscQ = [...osc].sort((a, b) => (a.time - latencyOf(a)) - (b.time - latencyOf(b)) || a.time - b.time || a.id - b.id);
    this.oi = 0;
  },
  // The cues or the toggles changed while playing: drop what is scheduled and not yet due, and schedule
  // again from the playhead. Cues already sent in this pass stay sent (this.sent).
  rearm() {
    if (!this.playing) return;
    const pending = new Set(this.oscTimers.keys());
    this.cancelScheduled();
    this.rebuild(this.position(), true, pending);
    this.tick();
  },
  cuesChanged() { this.rearm(); },
  cancelScheduled() {                       // (a cancelled send is not sent: a rebuild may schedule it again)
    for (const n of this.clickNodes) { try { n.stop(0); } catch { /* ended */ } n.disconnect(); }
    this.clickNodes.clear();
    for (const t of this.oscTimers.values()) clearTimeout(t);
    this.oscTimers.clear();
  },

  tick() {
    if (!this.playing || !this.ctx) return;
    const ctx = this.ctx, now = ctx.currentTime, horizon = now + AHEAD_S;
    const ctxAt = (t) => this.startCtx + (t - this.offset);
    const clicks = $("#c-click").checked;
    while (this.ci < this.clickQ.length) {
      const c = this.clickQ[this.ci];
      const at = ctxAt(c.time);
      if (at >= horizon) break;
      this.ci++;
      if (clicks && at >= now - 0.005) this.click(Math.max(at, now), c);   // a late click (hidden tab) is dropped
    }
    const osc = $("#c-osc").checked;
    const outLat = this.outLat();
    while (this.oi < this.oscQ.length) {
      const c = this.oscQ[this.oi];
      const at = ctxAt(c.time) + outLat - latencyOf(c);
      if (at >= horizon) break;
      this.oi++;
      if (!osc) continue;
      const key = cueKey(c);
      if (this.sent.has(key) || this.oscTimers.has(key)) continue;
      // sent once the timer fires: a re-arm before that cancels it and schedules it again
      const timer = setTimeout(() => { this.oscTimers.delete(key); this.sent.add(key); firePreview(c); },
        Math.max(0, (at - now) * 1000));
      this.oscTimers.set(key, timer);
    }
  },
  click(at, cue) {
    const ctx = this.ctx;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = CLICK_HZ * 2 ** (pitchOf(effTarget(cue)) / 12);
    g.gain.setValueAtTime(0.0001, at);
    g.gain.exponentialRampToValueAtTime(1, at + 0.002);
    g.gain.exponentialRampToValueAtTime(0.0001, at + 0.03);
    osc.connect(g);
    g.connect(this.clickBus);
    osc.start(at);
    osc.stop(at + 0.035);
    this.clickNodes.add(osc);
    osc.onended = () => { this.clickNodes.delete(osc); g.disconnect(); };
  },
};
const cueKey = (c) => `${c.time.toFixed(4)}|${c.target ?? ""}|${c.message}`;

let previewFailNoticeAt = 0;
function firePreview(cue) {
  const pid = S.selected;
  if (!pid || !preview.playing || !$("#c-osc").checked) return;
  api(`/api/projects/${enc(pid)}/cues/${cue.id}/fire`, { method: "POST", body: { source: "preview", rev: S.sheet?.rev } })
    .then((res) => {
      if (res && res.ok === false) previewFailed(`Preview OSC of cue ${cue.id} not sent: ${res.error || "error"}`);
    })
    .catch((e) => { if (e.status !== 409) previewFailed(`Preview OSC: ${e.message}`); });
}
function previewFailed(text) {          // at most one notice every 3 s while previewing
  const now = performance.now();
  if (now - previewFailNoticeAt < 3000) return;
  previewFailNoticeAt = now;
  notify(text, true);
}

function loadText(text, error = false) {
  const el = $("#p-load");
  el.textContent = text;
  el.classList.toggle("error", error);
}
$("#b-play").addEventListener("click", () => preview.toggle());
$("#c-click").addEventListener("change", () => preview.rearm());
$("#c-vol").addEventListener("input", (e) => { if (preview.clickBus) preview.clickBus.gain.value = Number(e.target.value); });
// Preview OSC goes to the live targets (also when SAFE): while a session runs only after a confirmation,
// and a session that starts turns it off (see onState).
$("#c-osc").addEventListener("change", (e) => {
  if (e.target.checked && sessionRunning()
      && !confirm("A live session is running. Send OSC from the preview as well? The targets receive both, also when SAFE.")) {
    e.target.checked = false;
    return;
  }
  preview.rearm();
  if (e.target.checked) notify("Preview sends OSC to the targets (also when SAFE), early by each target's latency.");
});

// ------------------------------------------------------------------ waveform + zoom strip
const waveCv = $("#wave"), zoomCv = $("#zoom");
const ZOOM_HALF_S = 4;
const wave = {
  data: null, zoomData: null, pid: null, dirty: true, staticCv: document.createElement("canvas"),
  hoverX: null, zoomHoverX: null, drag: null, press: null, lastKey: "", lastZoomKey: "", fetchTimer: 0,

  unload() { this.data = null; this.zoomData = null; this.pid = null; this.dirty = true; this.drag = null; },
  wantPoints() { return Math.min(20000, Math.max(100, Math.round(waveCv.clientWidth * (window.devicePixelRatio || 1)))); },
  async load(pid) {
    const points = this.wantPoints();
    let d;
    try {
      d = await api(`/api/projects/${enc(pid)}/waveform?points=${points}`);
    } catch (e) {
      if (S.selected === pid) notify(`Waveform: ${e.message}`, true);
      return;
    }
    if (S.selected !== pid) return;
    this.data = d;
    this.pid = pid;
    this.dirty = true;
    // the zoom strip needs more detail: ~200 peaks per second (the server allows up to 20000)
    const zp = Math.min(20000, Math.max(100, Math.round((d.duration || 0) * 200)));
    if (zp <= d.points) { this.zoomData = d; return; }
    try {
      const z = await api(`/api/projects/${enc(pid)}/waveform?points=${zp}`);
      if (S.selected === pid) { this.zoomData = z; this.lastZoomKey = ""; }
    } catch { this.zoomData = d; }
  },
  resized() {
    this.dirty = true;
    this.lastKey = "";
    this.lastZoomKey = "";
    if (!this.data || !S.selected) return;
    const want = this.wantPoints();
    if (Math.abs(want - this.data.points) / this.data.points < 0.15) return;
    clearTimeout(this.fetchTimer);
    const pid = S.selected;
    this.fetchTimer = setTimeout(async () => {
      try {
        const d = await api(`/api/projects/${enc(pid)}/waveform?points=${this.wantPoints()}`);
        if (S.selected === pid) { this.data = d; this.dirty = true; }
      } catch { /* keep the old peaks */ }
    }, 300);
  },
};
const dpr = () => window.devicePixelRatio || 1;
function fitCanvas(cv) {
  const w = Math.max(1, Math.round(cv.clientWidth * dpr())), h = Math.max(1, Math.round(cv.clientHeight * dpr()));
  if (cv.width === w && cv.height === h) return false;
  cv.width = w; cv.height = h;
  return true;
}
new ResizeObserver(() => { fitCanvas(waveCv); fitCanvas(zoomCv); wave.resized(); }).observe($("#editor"));

const songDur = () => wave.data?.duration || preview.duration() || 0;
const cueTime = (c) => (S.pendingTime && S.pendingTime.id === c.id ? S.pendingTime.t : c.time);

// min / max of peaks[i0, i1) (at least one point)
function peakRange(d, i0, i1) {
  i0 = Math.max(0, Math.min(d.points - 1, i0));
  i1 = Math.max(i0 + 1, Math.min(d.points, i1));
  let lo = Infinity, hi = -Infinity;
  for (let i = i0; i < i1; i++) { if (d.min[i] < lo) lo = d.min[i]; if (d.max[i] > hi) hi = d.max[i]; }
  return [lo, hi];
}
function tickStep(span, px, minPx) {             // a "nice" tick interval (s) at least minPx apart
  for (const s of [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600]) if (s / span * px >= minPx) return s;
  return 1200;
}

function renderWaveStatic() {
  const cv = wave.staticCv;
  cv.width = waveCv.width; cv.height = waveCv.height;
  const g = cv.getContext("2d");
  const k = dpr(), W = waveCv.width / k, H = waveCv.height / k;
  g.setTransform(k, 0, 0, k, 0, 0);
  g.fillStyle = "#171a1e";
  g.fillRect(0, 0, W, H);
  const d = wave.data, dur = songDur();
  if (!d || !(dur > 0)) {
    g.fillStyle = "#9ba3ae";
    g.font = "12px system-ui, sans-serif";
    g.fillText(S.selected ? "Loading waveform…" : "No project", 10, H / 2 + 4);
    return;
  }
  const mid = H / 2, amp = H / 2 - 6;
  g.fillStyle = "#56606e";
  const cols = waveCv.width;
  for (let i = 0; i < cols; i++) {
    const [lo, hi] = peakRange(d, Math.floor(i / cols * d.points), Math.floor((i + 1) / cols * d.points));
    g.fillRect(i / k, mid - hi * amp, 1 / k, Math.max(1 / k, (hi - lo) * amp));
  }
  // time ticks along the bottom
  const step = tickStep(dur, W, 70);
  g.fillStyle = "#6b7380";
  g.font = "10px system-ui, sans-serif";
  for (let t = step; t < dur; t += step) {
    const x = t / dur * W;
    g.fillRect(x, H - 5, 1, 5);
    g.fillText(fmtTime(t, 0), x + 3, H - 2);
  }
  // cue markers
  for (const c of S.cues) {
    const x = Math.round(cueTime(c) / dur * W * k) / k;
    g.fillStyle = colourOf(effTarget(c));
    g.globalAlpha = 0.85;
    g.fillRect(x - 0.5, 0, 1.5, H);
    g.globalAlpha = 1;
    g.beginPath(); g.moveTo(x - 4, 0); g.lineTo(x + 4.5, 0); g.lineTo(x + 0.25, 6); g.closePath(); g.fill();
  }
  wave.dirty = false;
}

function drawLine(g, x, H, colour, w = 2) {
  g.fillStyle = colour;
  g.fillRect(x - w / 2, 0, w, H);
}
function drawWave(pos, livePos) {
  const key = `${pos.toFixed(3)}|${livePos?.toFixed(3)}|${wave.hoverX}|${wave.dirty}|${waveCv.width}`;
  if (key === wave.lastKey && !wave.dirty) return;
  if (wave.dirty) renderWaveStatic();
  wave.lastKey = key;
  const g = waveCv.getContext("2d");
  const k = dpr(), W = waveCv.width / k, H = waveCv.height / k, dur = songDur();
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.drawImage(wave.staticCv, 0, 0);
  if (!(dur > 0)) return;
  g.setTransform(k, 0, 0, k, 0, 0);
  // the zoom strip's window
  const zc = zoomCenter(pos, livePos);
  g.fillStyle = "rgba(111, 168, 255, .10)";
  g.fillRect((zc - ZOOM_HALF_S) / dur * W, 0, 2 * ZOOM_HALF_S / dur * W, H);
  if (livePos != null) drawLine(g, livePos / dur * W, H, "#ffb23f", 2);
  drawLine(g, pos / dur * W, H, "#6fa8ff", 2);
  if (wave.hoverX != null) { g.fillStyle = "rgba(236, 234, 228, .5)"; g.fillRect(wave.hoverX, 0, 1, H); }
}

// cue nearest to x (CSS px) within `tol` px, on a time axis given by timeAt
function cueNear(x, xOfTime, tol) {
  let best = null, bd = Infinity;
  for (const c of S.cues) {
    const dx = Math.abs(xOfTime(cueTime(c)) - x);
    if (dx <= tol && dx < bd) { bd = dx; best = c; }
  }
  return best;
}
const cueLabel = (c) => `#${c.id}  ${fmtTime(cueTime(c), 3)}  ${effTarget(c)}  ${c.message}${c.note ? `  # ${c.note}` : ""}`;
function showTip(tip, cv, x, text) {
  tip.hidden = !text;
  if (!text) return;
  tip.textContent = text;
  const w = cv.clientWidth;
  tip.style.left = x < w / 2 ? `${Math.max(0, x + 10)}px` : "";
  tip.style.right = x < w / 2 ? "" : `${Math.max(0, w - x + 10)}px`;
}

waveCv.addEventListener("mousemove", (e) => {
  const dur = songDur();
  if (!(dur > 0)) return;
  const W = waveCv.clientWidth, x = e.offsetX;
  wave.hoverX = x;
  const c = cueNear(x, (t) => t / dur * W, 4);
  showTip($("#wave-tip"), waveCv, x, c ? cueLabel(c) : fmtTime(x / W * dur, 2));
});
waveCv.addEventListener("mouseleave", () => { wave.hoverX = null; $("#wave-tip").hidden = true; });
waveCv.addEventListener("click", (e) => {
  const dur = songDur();
  if (!(dur > 0)) return;
  const W = waveCv.clientWidth;
  const c = cueNear(e.offsetX, (t) => t / dur * W, 4);
  preview.seek(c ? c.time : e.offsetX / W * dur);
});

// The zoom strip is centred on the preview playhead; on the live position instead when Follow live is
// on, the live session plays this project and the preview is not playing.
function zoomCenter(pos, livePos) {
  if (wave.drag) return wave.drag.center;
  if (livePos != null && $("#follow").checked && !preview.playing) return livePos;
  return pos;
}
function drawZoom(pos, livePos) {
  const center = zoomCenter(pos, livePos);
  const d = wave.zoomData, dur = songDur();
  const key = `${center.toFixed(4)}|${pos.toFixed(3)}|${livePos?.toFixed(3)}|${wave.zoomHoverX}|${wave.drag?.t}|${S.epoch}|${S.sheet?.rev}|${colourSig}|${zoomCv.width}|${!!d}`;
  if (key === wave.lastZoomKey) return;
  wave.lastZoomKey = key;
  const g = zoomCv.getContext("2d");
  const k = dpr(), W = zoomCv.width / k, H = zoomCv.height / k;
  g.setTransform(k, 0, 0, k, 0, 0);
  g.fillStyle = "#171a1e";
  g.fillRect(0, 0, W, H);
  if (!S.selected) return;
  const t0 = center - ZOOM_HALF_S, span = 2 * ZOOM_HALF_S;
  const xOf = (t) => (t - t0) / span * W;
  // peaks
  if (d && dur > 0) {
    const mid = H / 2 + 6, amp = H / 2 - 16, cols = zoomCv.width, per = d.points / dur;
    g.fillStyle = "#4a5361";
    for (let i = 0; i < cols; i++) {
      const a = t0 + i / cols * span, b = t0 + (i + 1) / cols * span;
      if (b <= 0 || a >= dur) continue;
      const [lo, hi] = peakRange(d, Math.floor(a * per), Math.ceil(b * per));
      g.fillRect(i / k, mid - hi * amp, 1 / k, Math.max(1 / k, (hi - lo) * amp));
    }
  }
  // ruler
  g.font = "10px system-ui, sans-serif";
  for (let t = Math.ceil(t0 / 0.5) * 0.5; t <= t0 + span; t += 0.5) {
    if (t < 0) continue;
    const x = xOf(t), whole = Math.abs(t - Math.round(t)) < 1e-6;
    g.fillStyle = whole ? "#7d848e" : "#4b5360";
    g.fillRect(x, H - (whole ? 8 : 4), 1, whole ? 8 : 4);
    if (whole) { g.fillStyle = "#7d848e"; g.fillText(fmtTime(t, 0), x + 3, H - 2); }
  }
  if (dur > 0 && t0 + span > dur) { g.fillStyle = "rgba(0, 0, 0, .45)"; g.fillRect(xOf(dur), 0, W - xOf(dur), H); }
  if (t0 < 0) { g.fillStyle = "rgba(0, 0, 0, .45)"; g.fillRect(0, 0, xOf(0), H); }
  // cue markers with labels (two label rows so neighbours do not overlap)
  g.font = "11px ui-monospace, Menlo, monospace";
  const rowEnd = [-Infinity, -Infinity];
  for (const c of S.cues) {
    const dragged = wave.drag?.id === c.id;
    const t = dragged ? wave.drag.t : cueTime(c);
    const x = xOf(t);
    if (x < -200 || x > W + 4) continue;
    const colour = colourOf(effTarget(c));
    g.fillStyle = colour;
    g.globalAlpha = dragged ? 1 : 0.9;
    g.fillRect(x - (dragged ? 1 : 0.75), 0, dragged ? 2 : 1.5, H);
    g.globalAlpha = 1;
    let label = `${c.id} ${c.message}`;
    const maxW = 170;
    while (label.length > 4 && g.measureText(label).width > maxW) label = `${label.slice(0, -2)}…`;
    const lw = g.measureText(label).width + 8;
    const r = rowEnd[0] <= x ? 0 : rowEnd[1] <= x ? 1 : 0;
    rowEnd[r] = x + lw + 2;
    const y = 2 + r * 15;
    g.fillStyle = colour;
    g.fillRect(x, y, lw, 14);
    g.fillStyle = "#111";
    g.fillText(label, x + 4, y + 11);
    if (dragged) {
      const txt = fmtTime(t, 3);
      const tw = g.measureText(txt).width + 10;
      const tx = Math.min(Math.max(0, x - tw / 2), W - tw);
      g.fillStyle = "#eceae4";
      g.fillRect(tx, H - 34, tw, 16);
      g.fillStyle = "#111";
      g.fillText(txt, tx + 5, H - 22);
    }
  }
  if (livePos != null && livePos >= t0 && livePos <= t0 + span) drawLine(g, xOf(livePos), H, "#ffb23f", 2);
  if (pos >= t0 && pos <= t0 + span) drawLine(g, xOf(pos), H, "#6fa8ff", 2);
  if (wave.zoomHoverX != null && !wave.drag) { g.fillStyle = "rgba(236, 234, 228, .35)"; g.fillRect(wave.zoomHoverX, 0, 1, H); }
}

// Zoom strip pointer: drag a marker horizontally to move its cue (commit on release, 1 ms snap, Esc
// cancels); a click without a drag seeks the preview (to the cue's own time on a marker).
const zoomX = (e) => e.clientX - zoomCv.getBoundingClientRect().left;
const zoomTimeAt = (x, center) => center - ZOOM_HALF_S + x / zoomCv.clientWidth * 2 * ZOOM_HALF_S;
function zoomCueAt(x, center) {
  const W = zoomCv.clientWidth;
  return cueNear(x, (t) => (t - (center - ZOOM_HALF_S)) / (2 * ZOOM_HALF_S) * W, 6);
}
let lastCenter = 0;
zoomCv.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || !S.selected) return;
  const x = zoomX(e), center = lastCenter;
  const c = zoomCueAt(x, center);
  wave.press = { x0: x, cue: c, center, moved: false };
  if (c) {
    wave.drag = { id: c.id, orig: c.time, t: c.time, grab: zoomTimeAt(x, center) - c.time, center, epoch: S.epoch };
    try { zoomCv.setPointerCapture(e.pointerId); } catch { /* pointer already gone */ }
    zoomCv.classList.add("dragging");
  }
});
zoomCv.addEventListener("pointermove", (e) => {
  const x = zoomX(e);
  if (wave.press && Math.abs(x - wave.press.x0) > 2) wave.press.moved = true;
  if (wave.drag) {
    const t = Math.max(0, Math.round((zoomTimeAt(x, wave.drag.center) - wave.drag.grab) * 1000) / 1000);
    wave.drag.t = t;
    showTip($("#zoom-tip"), zoomCv, x, `#${wave.drag.id} → ${fmtTime(t, 3)} (${t >= wave.drag.orig ? "+" : "−"}${Math.abs(t - wave.drag.orig).toFixed(3)} s)  Esc cancels`);
    return;
  }
  wave.zoomHoverX = x;
  const c = zoomCueAt(x, lastCenter);
  zoomCv.classList.toggle("grab", !!c);
  showTip($("#zoom-tip"), zoomCv, x, c ? `${cueLabel(c)}  (drag to move)` : "");
});
zoomCv.addEventListener("pointerleave", () => {
  if (wave.drag) return;
  wave.zoomHoverX = null;
  $("#zoom-tip").hidden = true;
});
zoomCv.addEventListener("pointerup", (e) => {
  const press = wave.press, drag = wave.drag;
  wave.press = null;
  endZoomDrag();
  if (drag && press?.moved) {
    if (drag.t !== drag.orig && drag.epoch === S.epoch) {
      S.pendingTime = { id: drag.id, t: drag.t };      // shown there until the response re-renders
      wave.dirty = true;
      patchCue(drag.id, { time: drag.t });
    }
    return;
  }
  if (press && !press.moved) preview.seek(press.cue ? press.cue.time : zoomTimeAt(zoomX(e), press.center));
});
zoomCv.addEventListener("pointercancel", () => { wave.press = null; endZoomDrag(); });
function endZoomDrag() {
  if (!wave.drag) return;
  wave.drag = null;
  zoomCv.classList.remove("dragging");
  $("#zoom-tip").hidden = true;
  wave.lastZoomKey = "";
}
function cancelZoomDrag() {
  if (!wave.drag) return false;
  wave.press = null;
  endZoomDrag();
  notify("Move cancelled.");
  return true;
}

// ------------------------------------------------------------------ devices
async function loadDevices(refresh = false) {
  const devs = await run(() => api(`/api/devices${refresh ? "?refresh=true" : ""}`));
  if (!devs) return;
  S.devices = devs;
  fillDevices(fs.elements.device);
  renderChannels(fs.elements.device, $("#chans"), "ch");
  if (dMeasure.open) {
    fillDevices(fMeasure.elements.device);
    renderChannels(fMeasure.elements.device, $("#m-chans"), "mch");
  }
}
function fillDevices(sel) {
  const prev = sel.value;
  sel.innerHTML = S.devices.map((d) => `<option value="${d.index}">${esc(d.name)} (${d.channels} ch)</option>`).join("")
    || `<option value="">No input devices found</option>`;
  const def = S.devices.find((d) => String(d.index) === prev) || S.devices.find((d) => d.is_default) || S.devices[0];
  if (def) sel.value = String(def.index);
}
function renderChannels(sel, box, name) {
  const d = S.devices.find((x) => String(x.index) === sel.value);
  box.textContent = "";
  if (!d) return;
  for (let c = 1; c <= Math.min(d.channels, 32); c++) {
    const lab = document.createElement("label");
    lab.className = "check";
    lab.innerHTML = `<input type="checkbox" name="${name}" value="${c}" ${c === 1 || (d.channels === 2 && c === 2) ? "checked" : ""}> ${c}`;
    box.append(lab);
  }
}
const checkedChannels = (box, name) => $$(`input[name=${name}]:checked`, box).map((c) => Number(c.value));
fs.elements.device.addEventListener("change", () => renderChannels(fs.elements.device, $("#chans"), "ch"));
$("#b-rescan").addEventListener("click", () => {
  if (S.last?.mode) { notify("Stop the session before rescanning the devices.", true); return; }
  loadDevices(true);
});

// ------------------------------------------------------------------ session controls
const sessionRunning = () => !!(S.last?.mode && !["idle", "stopped"].includes(S.last.status));
const liveHere = () => sessionRunning() && !!S.selected && S.last.project_id === S.selected;
const hasVocals = () => !!S.project?.stems?.includes("vocals");

function syncFormVisibility() {
  const el = fs.elements;
  const kind = el.kind.value;
  for (const d of $$("[data-kind]", fs)) d.hidden = d.dataset.kind !== kind;
  $("[data-path]", fs).hidden = el.file_src.value !== "path";
  $("[data-start-time]", fs).hidden = el.start_at.value !== "time";
}
fs.addEventListener("change", syncFormVisibility);

// Add / replace the vocal stem of the selected project (vocal mode follows the singer against it). The
// server's project announcement then re-reads the project and enables Vocal mic only.
$("#b-vocals").addEventListener("click", () => {
  if (!S.selected) { notify("Pick a project first.", true); return; }
  $("#f-vocals").click();
});
$("#f-vocals").addEventListener("change", async (e) => {
  const file = e.target.files[0], id = S.selected;
  e.target.value = "";
  if (!file || !id) return;
  const form = new FormData();
  form.append("vocals", file);
  const b = $("#b-vocals");
  b.disabled = true;
  try {
    const res = await upload(`/api/projects/${enc(id)}/vocals`, form, (f, done) => {
      b.textContent = done ? "Converting…" : `Uploading ${Math.round(f * 100)}%`;
    });
    const warn = res?.warnings?.length ? ` (${res.warnings.join("; ")})` : "";
    notify(`Vocal stem added to ${res?.project?.title || id}${warn}`, !!warn);
  } catch (err) {
    notify(err.message, true);
  } finally {
    b.disabled = false;
    b.textContent = hasVocals() ? "Replace vocal stem…" : "Add vocal stem…";
  }
});

function syncStemOptions() {
  const el = fs.elements, vocals = hasVocals();
  $("#b-vocals").textContent = vocals ? "Replace vocal stem…" : "Add vocal stem…";
  const vocalRadio = $("input[name=mode][value=vocal]", fs);
  vocalRadio.disabled = !vocals;
  $("#mode-vocal").title = vocals ? "Follow the singer's microphone against the vocal stem" : "Needs a vocal stem in the project";
  if (!vocals && el.mode.value === "vocal") $("input[name=mode][value=mix]", fs).checked = true;
  const opt = $("option[value='stem:vocals']", el.file_src);
  opt.disabled = !vocals;
  if (!vocals && el.file_src.value === "stem:vocals") el.file_src.value = "stem:mix";
  syncFormVisibility();
}

function renderStartAt(keep = fs.elements.start_at.value) {
  const sel = fs.elements.start_at;
  sel.innerHTML = `<option value="">Beginning</option>`
    + S.cues.map((c) => `<option value="cue:${c.id}">Cue ${c.id} · ${fmtTime(c.time, 3)} ${esc(c.message.slice(0, 32))}</option>`).join("")
    + `<option value="time">Time…</option>`;
  sel.value = [...sel.options].some((o) => o.value === keep) ? keep : "";
  syncFormVisibility();
}

// Start is a plain button and the form never submits: Enter in one of its fields (Input latency, the
// start time, a file path) must not start a session, nor replace the running one.
fs.addEventListener("submit", (e) => e.preventDefault());
$("#b-start").addEventListener("click", async () => {
  if (!S.selected) { notify("Pick a project first.", true); return; }
  const el = fs.elements;
  const input = { kind: el.kind.value };
  if (input.kind === "device") {
    input.device = el.device.value === "" ? null : Number(el.device.value);
    input.channels = checkedChannels($("#chans"), "ch");
    if (!input.channels.length) { notify("Select at least one input channel.", true); return; }
  } else {
    const src = el.file_src.value;
    if (src === "path") {
      input.path = el.path.value.trim();
      if (!input.path) { notify("Give the path of the recording to play.", true); return; }
    } else input.stem = src.split(":")[1];
    input.monitor = el.monitor.checked;
  }
  const body = {
    project_id: S.selected, mode: el.mode.value, style: el.style.value, input,
    auto_key: S.autoKey,
    // a key (detected, or set by hand) belongs to its project and never carries over to another one;
    // with Auto on it only seeds the search
    transpose: S.keyProject === S.selected ? S.transpose : 0,
  };
  const at = el.start_at.value;
  if (at.startsWith("cue:")) { body.start_cue = Number(at.slice(4)); body.rev = S.sheet?.rev; }
  else if (at === "time") {
    const t = parseTime(el.start_time.value);
    if (t == null) { notify("Write the start time as 75.3, 1:05.200 or 1:01:05.2", true); el.start_time.focus(); return; }
    body.start_time = t;
  }
  if (sessionRunning()) {                  // starting replaces the running session: ask first
    const title = S.project?.title || S.selected;
    const from = body.start_cue != null ? `at cue ${body.start_cue} (${fmtTime(cueById(body.start_cue)?.time, 3)})`
      : body.start_time != null ? `at ${fmtTime(body.start_time, 3)}` : "from the beginning";
    const ask = S.last.project_id === S.selected
      ? `${title} is live. Restart it ${from}? The running session stops.`
      : `${liveTitle()} is live. Stop it and start ${title} ${from}?`;
    if (!confirm(ask)) return;
  }
  const btn = $("#b-start");
  btn.disabled = true;
  btn.textContent = "Starting…";
  await sessionPost("start", body, `Following ${S.project?.title || S.selected}`);
  btn.disabled = false;
  btn.textContent = "Start";
});

const post = (path, body) => run(() => api(`/api/session/${path}`, { method: "POST", body: body ?? {} }));
// Go and Start at a cue name the cue by its line in the sheet shown here, with that sheet's rev: a console
// showing an older version of the file is refused (409, carrying the current sheet) instead of moving the
// show to whatever cue sits on that line now.
async function sessionPost(path, body, okText) {
  try {
    const res = await api(`/api/session/${path}`, { method: "POST", body });
    if (okText) notify(okText);
    return res;
  } catch (e) {
    const cur = e.status === 409 ? sheetOf(e.data) : null;
    if (cur && S.selected === body.project_id) applySheet(cur);
    notify(cur ? `${e.message} The cue list was reloaded: check the cue and try again.` : e.message, true);
    return null;
  }
}
$("#b-stop").addEventListener("click", () => post("stop"));
const offlineText = "Disconnected from the server: the session state is unknown. Wait for it to reconnect.";
function toggleHold() {
  if (S.offline) { notify(offlineText, true); return; }
  if (!sessionRunning()) { notify("No live session to hold.", true); return; }
  post("hold", { on: S.last.status !== "hold" });
}
$("#b-hold").addEventListener("click", toggleHold);
function nudge(dt) {
  if (S.offline) { notify(offlineText, true); return; }
  if (!sessionRunning()) { notify("No live session to nudge.", true); return; }
  post("nudge", { dt });
}
for (const b of $$("[data-nudge]")) b.addEventListener("click", () => nudge(Number(b.dataset.nudge)));

// tracking style of the selected project: saved to the project, a live session of it switches at once
const STYLE_TEXT = { classic: "Classical — flexible tempo", pop: "Pop & band — steady tempo" };
function showStyle(style) {
  const sel = fs.elements.style;
  sel.disabled = !S.selected;
  if (document.activeElement !== sel) sel.value = STYLE_TEXT[style] ? style : "classic";
}
fs.elements.style.addEventListener("change", async () => {
  const id = S.selected, style = fs.elements.style.value;
  if (!id) return;
  const res = await run(() => api(`/api/projects/${enc(id)}/style`, { method: "PUT", body: { style } }));
  if (!res) { showStyle(S.project?.style); return; }
  if (S.project && S.selected === id) S.project.style = style;
  const p = S.projects.find((x) => x.id === id);
  if (p) p.style = style;
  notify(`${STYLE_TEXT[style]}${res.session ? ", applied to the live session" : ""}.`);
});

// input latency: a setting (PUT /api/settings saves it and gives it to a running session at once)
const latencyMs = () => Math.max(-500, Math.min(2000, Number(fs.elements.input_latency.value) || 0));
fs.elements.input_latency.addEventListener("change", () => {
  const ms = latencyMs();
  fs.elements.input_latency.value = String(ms);
  putSettings({ input_latency_ms: ms }, `Input latency ${ms} ms`);
});

// The key panel shows the live project's key while a session runs (from its state messages), else the
// key prepared for the selected project. A key belongs to one project: switching to another one starts
// from its original key (0), so neither a detected key nor a hand-set one carries over.
const fmtKey = (k) => (k > 0 ? "+" : "") + k;
function showKey() {
  $("#o-transpose").value = fmtKey(S.transpose);
  const b = $("#b-autokey");
  b.classList.toggle("on", S.autoKey);
  b.setAttribute("aria-pressed", String(S.autoKey));
}
function keyFor(id) {
  if (S.keyProject === id) return;
  if (S.keyProject != null) S.transpose = 0;
  S.keyProject = id;
  $("#b-keysuggest").hidden = true;
  showKey();
}
const keyOwner = () => (sessionRunning() ? S.last.project_id : S.selected);
function setKey(k) {                       // operator key: turns auto detection off
  S.transpose = Math.max(-12, Math.min(12, k));
  S.autoKey = false;
  S.keyProject = keyOwner();
  showKey();
  $("#b-keysuggest").hidden = true;
  if (sessionRunning()) post("transpose", { semitones: S.transpose });
}
for (const b of $$("[data-transpose]")) b.addEventListener("click", () => setKey(S.transpose + Number(b.dataset.transpose)));
$("#b-autokey").addEventListener("click", () => {
  S.autoKey = !S.autoKey;
  S.keyProject = keyOwner();
  if (!S.autoKey) $("#b-keysuggest").hidden = true;
  showKey();
  if (sessionRunning()) post("autokey", { on: S.autoKey });
});
$("#b-keysuggest").addEventListener("click", (e) => {
  const k = Number(e.currentTarget.dataset.key);
  if (Number.isFinite(k)) setKey(k);
});

// ------------------------------------------------------------------ settings: ARM, targets, options
function settingsOf(res) {
  if (res && Array.isArray(res.targets)) return res;
  if (res?.settings && Array.isArray(res.settings.targets)) return res.settings;
  return null;
}
async function loadSettings() {
  const s = await run(() => api("/api/settings"));
  if (s && settingsOf(s)) applySettings(settingsOf(s));
}
// PUT a partial settings change; re-render from the response (or reload on failure). Returns success.
async function putSettings(patch, okText) {
  const res = await run(() => api("/api/settings", { method: "PUT", body: patch }), okText);
  if (!res) { await loadSettings(); return false; }
  const s = settingsOf(res);
  if (s) applySettings(s); else await loadSettings();
  return true;
}

let armHoldUntil = 0;              // ignore state.armed briefly after a click (snapshots taken before it)
function applySettings(s) {
  S.settings = s;
  if (s.rev !== undefined) S.settingsRev = s.rev;
  if (s.targets_status) S.targetsStatus = s.targets_status;
  if (typeof s.armed === "boolean") setArmed(s.armed);
  // what the server repaired when it loaded a damaged settings file: stays until the settings are saved again
  const warn = $("#settings-warn");
  warn.hidden = !s.warnings?.length;
  warn.textContent = s.warnings?.length ? `Settings file repaired at startup: ${s.warnings.join(" · ")}` : "";
  renderTargets();
  const def = $("#s-default");
  def.innerHTML = s.targets.map((t) => `<option value="${esc(t.name)}">${esc(t.name)}</option>`).join("");
  def.value = s.default_target;
  if (document.activeElement !== $("#s-skip")) $("#s-skip").value = s.skip_policy;
  if (document.activeElement !== $("#s-catchup")) $("#s-catchup").value = String(s.catchup_s);
  if (document.activeElement !== fs.elements.input_latency) fs.elements.input_latency.value = String(s.input_latency_ms ?? 0);
  if (rebuildColours() && S.sheet) { renderCues(); renderAddTargets(); }
  wave.dirty = true;
  wave.lastZoomKey = "";
}

function setArmed(on) {
  S.armed = on;
  const b = $("#b-arm");
  b.dataset.armed = on == null ? "unknown" : String(on);
  b.setAttribute("aria-pressed", String(!!on));
  $("b", b).textContent = on == null ? "OSC …" : on ? "OSC LIVE" : "SAFE — dry run";
  b.title = on ? "OSC LIVE: cues are sent to the targets. Click for SAFE (dry run: cues are only logged)."
    : "SAFE: dry run, cues are logged but not sent. Click to send OSC (OSC LIVE).";
}
$("#b-arm").addEventListener("click", async () => {
  if (S.armed == null) return;
  const on = !S.armed;
  const b = $("#b-arm");
  b.disabled = true;
  const res = await run(() => api("/api/session/arm", { method: "POST", body: { on } }));
  b.disabled = false;
  if (!res) { loadSettings(); return; }
  armHoldUntil = performance.now() + 800;
  const s = settingsOf(res);
  if (s) applySettings(s);
  else setArmed(typeof res.armed === "boolean" ? res.armed : on);
  notify(S.armed ? "OSC LIVE: cues are sent to the targets." : "SAFE: dry run, cues are logged but not sent.");
});

// Targets table: one <tbody> per target (settings fields + test message, Test, Measure). Any change
// PUTs the whole list. While the operator types in it, re-renders wait until the focus leaves it.
const tTargets = $("#t-targets");
let targetsDeferred = false;
function renderTargets() {
  if (tTargets.contains(document.activeElement) && document.activeElement !== document.body) { targetsDeferred = true; return; }
  targetsDeferred = false;
  for (const b of $$("tbody", tTargets)) b.remove();
  for (const [i, t] of (S.settings?.targets || []).entries()) {
    const tb = document.createElement("tbody");
    tb.dataset.i = String(i);
    tb.className = t.enabled ? "" : "off";
    tb.innerHTML = `<tr>
      <td class="c-name"><div class="name-cell"><span class="swatch" style="background:${colourOf(t.name)}"></span>
        <input data-f="name" value="${esc(t.name)}" pattern="[A-Za-z_][A-Za-z0-9_.\\-]*" aria-label="Name" spellcheck="false"></div></td>
      <td><input data-f="host" value="${esc(t.host)}" aria-label="Host" spellcheck="false"></td>
      <td class="c-port"><input data-f="port" type="number" min="1" max="65535" value="${t.port}" aria-label="Port"></td>
      <td class="c-lat"><input data-f="latency_ms" type="number" min="0" max="5000" step="1" value="${t.latency_ms ?? 0}" aria-label="Latency (ms)"></td>
      <td class="c-on"><input data-f="enabled" type="checkbox" ${t.enabled ? "checked" : ""} aria-label="Enabled"></td>
      <td class="c-x"><button class="btn ghost small danger" type="button" data-act="remove" title="Remove this target" aria-label="Remove ${esc(t.name)}">✕</button></td>
    </tr><tr class="sub"><td colspan="6"><div class="row">
      <input data-f="test_message" value="${esc(t.test_message || "")}" placeholder="/oscsender/test" aria-label="Test message"
        title="Test message: sent by Test and by the audio loopback measurement" spellcheck="false">
      <span class="status" data-status="${esc(t.name)}"></span>
      <button class="btn small" type="button" data-act="test" title="Send the test message now">Test</button>
      <button class="btn small" type="button" data-act="measure" title="Measure this receiver's latency">Measure</button>
    </div></td></tr>`;
    tTargets.append(tb);
  }
  renderTargetStatus();
}
tTargets.addEventListener("focusout", () => {
  setTimeout(() => { if (targetsDeferred && !tTargets.contains(document.activeElement)) renderTargets(); }, 0);
});
function renderTargetStatus() {
  const stats = S.last?.targets || S.targetsStatus || {};
  for (const el of $$("[data-status]", tTargets)) {
    const st = stats[el.dataset.status];
    let text = "", title = "", bad = false;
    if (st) {
      if (st.resolve_error) { text = "unresolved"; title = `Cannot resolve the host: ${st.resolve_error}`; bad = true; }
      else if (st.last_error && st.errors) { text = plural(st.errors, "error"); title = `Last error: ${st.last_error}`; bad = true; }
      else { text = `${st.sent || 0} sent`; title = `${st.resolved ? `Sends to ${st.resolved}. ` : ""}${plural(st.sent || 0, "message")} sent`; }
    }
    if (el.textContent !== text) el.textContent = text;
    if (el.title !== title) el.title = title;
    el.classList.toggle("bad", bad);
  }
}
function readTargets() {
  return $$("tbody", tTargets).map((tb) => {
    const f = (k) => $(`[data-f=${k}]`, tb);
    return {
      name: f("name").value.trim(), host: f("host").value.trim(), port: Number(f("port").value),
      latency_ms: Number(f("latency_ms").value) || 0, enabled: f("enabled").checked, test_message: f("test_message").value.trim(),
    };
  });
}
tTargets.addEventListener("change", (e) => {
  const tb = e.target.closest("tbody[data-i]");
  if (!tb || !S.settings) return;
  const targets = readTargets();
  const patch = { targets };
  // renaming the default target renames the default too (or the settings would not validate)
  const i = Number(tb.dataset.i), old = S.settings.targets[i]?.name;
  if (e.target.dataset.f === "name" && old === S.settings.default_target) patch.default_target = targets[i].name;
  putSettings(patch);
});
tTargets.addEventListener("click", async (e) => {
  const b = e.target.closest("button[data-act]");
  const tb = e.target.closest("tbody[data-i]");
  if (!b || !tb || !S.settings) return;
  const i = Number(tb.dataset.i), t = S.settings.targets[i];
  if (!t) return;
  if (b.dataset.act === "remove") {
    if (S.settings.targets.length <= 1) { notify("Keep at least one target.", true); return; }
    if (!confirm(`Remove the target ${t.name} (${t.host}:${t.port})? Cues for it will have no receiver.`)) return;
    const targets = S.settings.targets.filter((_, j) => j !== i);
    const patch = { targets };
    if (t.name === S.settings.default_target) patch.default_target = targets[0].name;
    putSettings(patch, `Removed ${t.name}`);
  } else if (b.dataset.act === "test") {
    const msg = $("[data-f=test_message]", tb).value.trim();
    const res = await run(() => api(`/api/targets/${enc(t.name)}/test`, { method: "POST", body: msg ? { message: msg } : {} }));
    if (res) notify(res.ok ? `Test sent to ${t.name} (${t.host}:${t.port})` : `Test to ${t.name} failed: ${res.error}`, !res.ok);
  } else if (b.dataset.act === "measure") {
    openMeasure(t.name);
  }
});
$("#b-add-target").addEventListener("click", () => {
  if (!S.settings) return;
  const names = new Set(S.settings.targets.map((t) => t.name));
  let n = S.settings.targets.length + 1;
  while (names.has(`target${n}`)) n++;
  const targets = [...S.settings.targets, { name: `target${n}`, host: "127.0.0.1", port: 9000, latency_ms: 0, enabled: true, test_message: "" }];
  putSettings({ targets }, `Added target${n}: set its host and port`);
});
// (both apply to a running session at once: say so)
$("#s-default").addEventListener("change", (e) => putSettings({ default_target: e.target.value }, `Default target: ${e.target.value}`));
$("#s-skip").addEventListener("change", (e) => {
  putSettings({ skip_policy: e.target.value }, `Skipped cues: ${e.target.selectedOptions[0]?.textContent || e.target.value}`);
});
$("#s-catchup").addEventListener("change", (e) => {
  const v = Number(e.target.value);
  if (!(v >= 0.1 && v <= 30)) { notify("The catch-up window is 0.1 to 30 s.", true); e.target.value = String(S.settings?.catchup_s ?? 1); return; }
  putSettings({ catchup_s: v });
});

// ------------------------------------------------------------------ measure dialog
const dMeasure = $("#d-measure");
const fMeasure = $("#f-measure");
const measure = { target: null, token: 0, report: null };
function openMeasure(name) {
  const t = settingsTarget(name);
  if (!t) return;
  measure.target = name;
  measure.report = null;
  measure.token++;
  const el = fMeasure.elements;
  $("#m-target").textContent = `${name} (${t.host}:${t.port})`;
  el.message.value = t.test_message || "";
  fillDevices(el.device);
  if (fs.elements.device.value) el.device.value = fs.elements.device.value;
  renderChannels(el.device, $("#m-chans"), "mch");
  $("#m-result").hidden = true;
  $("#b-m-use").disabled = true;
  $("#b-m-run").disabled = false;
  syncMeasure();
  dMeasure.showModal();
}
function syncMeasure() {
  const el = fMeasure.elements;
  const method = el.method.value, via = el.echo_via.value;
  for (const d of $$("[data-method]", fMeasure)) d.hidden = d.dataset.method !== method;
  for (const d of $$("[data-via]", fMeasure)) d.hidden = d.dataset.via !== via;
}
fMeasure.addEventListener("change", syncMeasure);
fMeasure.elements.device.addEventListener("change", () => renderChannels(fMeasure.elements.device, $("#m-chans"), "mch"));
$("#b-m-cancel").addEventListener("click", () => { measure.token++; dMeasure.close(); });
dMeasure.addEventListener("close", () => { measure.token++; });
$("#b-m-run").addEventListener("click", async () => {
  const t = settingsTarget(measure.target);
  if (!t) return;
  const el = fMeasure.elements;
  let path, body, seconds;
  if (el.method.value === "echo") {
    const count = Number(el.echo_count.value) || 20, timeout = Number(el.timeout_ms.value) || 500;
    body = { count, timeout_ms: timeout };
    if (el.echo_via.value === "responder") {
      Object.assign(body, { host: t.host, port: Number(el.echo_port.value) || 9001 });
    } else {
      const msg = el.echo_message.value.trim();
      if (!msg) { notify("Write the message the receiver answers.", true); el.echo_message.focus(); return; }
      Object.assign(body, { target: t.name, message: msg });
      if (el.listen_port.value) body.listen_port = Number(el.listen_port.value);
    }
    path = "/api/latency/echo";
    seconds = count * (0.05 + 0.02);
  } else {
    const channels = checkedChannels($("#m-chans"), "mch");
    if (!channels.length) { notify("Select at least one input channel.", true); return; }
    const count = Number(el.lb_count.value) || 8;
    body = {
      target: t.name, device: el.device.value === "" ? null : Number(el.device.value), channels, count,
      threshold_db: Number(el.threshold_db.value) || 20,
    };
    if (el.message.value.trim()) body.message = el.message.value.trim();
    path = "/api/latency/loopback";
    seconds = count * 1.5;
  }
  const token = ++measure.token;
  const box = $("#m-result");
  box.hidden = false;
  box.innerHTML = `Measuring… (about ${Math.max(1, Math.round(seconds))} s)`;
  $("#b-m-run").disabled = true;
  $("#b-m-use").disabled = true;
  let rep = null, err = null;
  try { rep = await api(path, { method: "POST", body }); } catch (e) { err = e.message; }
  if (token !== measure.token) return;           // cancelled / closed meanwhile
  $("#b-m-run").disabled = false;
  measure.report = rep;
  box.innerHTML = err ? `<p class="error">${esc(err)}</p>` : reportHtml(rep);
  $("#b-m-use").disabled = !(rep && Number.isFinite(rep.estimate_ms));
});
const ms = (v) => (v == null || !Number.isFinite(v) ? "–" : `${v.toFixed(1)} ms`);
function reportHtml(r) {
  const est = Number.isFinite(r.estimate_ms)
    ? `${r.estimate_ms.toFixed(1)} ms <small>(Use as latency sets ${Math.round(r.estimate_ms)} ms)</small>` : "no estimate";
  return `<div class="est">${est}</div>
    ${r.error ? `<p class="error">${esc(r.error)}</p>` : ""}
    <dl>
      <dt>Median</dt><dd>${ms(r.median_ms)}${r.method === "echo" ? " round trip (estimate: half)" : ""}</dd>
      <dt>Min / p95 / max</dt><dd>${ms(r.min_ms)} / ${ms(r.p95_ms)} / ${ms(r.max_ms)}</dd>
      <dt>Jitter (IQR)</dt><dd>${ms(r.jitter_ms)}</dd>
      <dt>Received</dt><dd>${r.received ?? 0} of ${r.sent ?? 0}</dd>
      ${r.note ? `<dt>Note</dt><dd>${esc(r.note)}</dd>` : ""}
    </dl>`;
}
$("#b-m-use").addEventListener("click", async () => {
  const r = measure.report, name = measure.target;
  if (!r || !Number.isFinite(r.estimate_ms) || !S.settings) return;
  const value = Math.max(0, Math.min(5000, Math.round(r.estimate_ms)));
  const targets = S.settings.targets.map((t) => (t.name === name ? { ...t, latency_ms: value } : t));
  const loopMsg = fMeasure.elements.message.value.trim();
  if (fMeasure.elements.method.value === "loopback") {
    for (const t of targets) if (t.name === name) t.test_message = loopMsg;
  }
  if (await putSettings({ targets }, `${name}: latency set to ${value} ms`)) dMeasure.close();
});

// ------------------------------------------------------------------ live state
const lvl = { conf: $("#m-conf"), level: $("#m-level"), peak: $("#m-peak") };
const pct = (db) => `${Math.max(0, Math.min(100, ((db ?? -120) + 60) / 60 * 100))}%`;
let statusTimer = 0;

function onState(msg) {
  S.last = msg;
  S.lastAt = performance.now();
  clock.update(msg);
  if (S.offline) {                         // connected again: everything below shows the server's state
    S.offline = false;
    $("#live").classList.remove("offline");
  }
  const running = sessionRunning();
  const st = $("#st");
  const shown = shownStatus(msg);
  if (st.dataset.status !== shown) {
    st.dataset.status = shown;
    $("b", st).textContent = STATUS_TEXT[shown] || shown;
  }
  if (running && !S.wasRunning && $("#c-osc").checked) {
    // a show started: the preview must not send to its targets unless asked again (with a confirmation)
    $("#c-osc").checked = false;
    preview.rearm();
    notify("A live session started: Send OSC while previewing is now off.", true);
  }
  S.wasRunning = running;
  const liveProject = running ? msg.project_id : null;
  if (liveProject !== S.liveProject) {
    S.liveProject = liveProject;
    renderProjects();
    goBlocked = null;
  }
  const lp = S.projects.find((p) => p.id === msg.project_id);
  $("#live-project").textContent = msg.mode ? `${lp?.title || msg.project_id || ""} · ${msg.mode === "vocal" ? "vocal" : "mix"}` : "";
  $("#r-live").textContent = msg.mode ? fmtTime(msg.live_time, 1) : "–";
  $("#r-tempo").textContent = msg.mode && msg.tempo ? `${Math.round(msg.tempo * 100)}%` : "–";
  const conf = msg.confidence || 0;
  lvl.conf.style.width = `${Math.round(conf * 100)}%`;
  lvl.conf.style.background = conf < 0.35 ? "var(--warn)" : conf < 0.6 ? "var(--gel)" : "var(--ok)";
  const level = msg.level_db, peak = msg.source?.peak_db;
  lvl.level.style.width = pct(level);
  lvl.level.style.background = (peak ?? -120) > -3 ? "var(--warn)" : (level ?? -120) > -12 ? "var(--gel)" : "var(--ok)";
  lvl.peak.style.left = pct(peak);
  $("#r-level").textContent = level != null && msg.mode ? `${Math.round(level)} dBFS` : "";
  if (msg.mode && msg.style && msg.project_id === S.selected && S.project) {
    S.project.style = msg.style;           // (switched from another console)
    showStyle(msg.style);
  }
  const hb = $("#b-hold");
  hb.textContent = msg.status === "hold" ? "Resume" : "Hold";
  hb.classList.toggle("on", msg.status === "hold");
  for (const b of [hb, ...$$("[data-nudge]")]) b.disabled = !running;
  // a session whose recording has ended still holds its project (delete, device rescan): Stop clears it
  $("#b-stop").disabled = !msg.mode;
  if (typeof msg.armed === "boolean" && performance.now() > armHoldUntil && msg.armed !== S.armed) setArmed(msg.armed);
  // key panel
  if (typeof msg.transpose === "number" && msg.mode) {
    S.transpose = msg.transpose;
    S.keyProject = msg.project_id;
    if (typeof msg.key_auto === "boolean") S.autoKey = msg.key_auto;
    showKey();
  } else if (!msg.mode && S.selected) {
    keyFor(S.selected);                    // no session (any more): the selected project's own key
  }
  // a suggestion only exists for a running session (a stale one would apply an old project's key)
  const sug = $("#b-keysuggest"), k = msg.key_suggest;
  sug.hidden = !(running && S.autoKey && typeof k === "number" && k !== S.transpose);
  if (!sug.hidden) { sug.dataset.key = String(k); sug.textContent = `${fmtKey(k)}?`; }
  const rep = msg.input_latency_reported_ms;
  const hint = $("#lat-reported");
  hint.hidden = !(running && Number.isFinite(rep));
  if (!hint.hidden) hint.textContent = `The audio interface reports ${Math.round(rep)} ms input latency.`;
  addFired(msg);
  syncGo();
  markLive();
  // warnings
  const warn = [];
  const src = msg.source || {};
  if (msg.error) warn.push(`Tracking error: ${msg.error}`);
  if (msg.warning) warn.push(msg.warning);
  if (src.last_error) warn.push(`Input error: ${src.last_error}`);
  if (running && src.running && src.silent_s > 1.5 && !(src.stalled_s > 1)) {
    warn.push("No signal on the input. Check the channel selection and that this terminal app may use the microphone.");
  }
  if (shown === "stale") warn.push(`No audio has reached the tracker for ${(msg.age || 0).toFixed(1)} s. Check the input.`);
  if (src.overflows) warn.push(`Audio processing fell behind; ${src.overflows} blocks dropped.`);
  if (running) {
    for (const [name, t] of Object.entries(msg.targets || {})) {
      if (t.enabled !== false && t.resolve_error) warn.push(`Target ${name}: ${t.resolve_error}`);
    }
  }
  if (msg.ended) warn.push("The recording has finished.");
  const w = $("#live-warn");
  const text = warn.join(" ");
  w.hidden = !warn.length;
  if (w.textContent !== text) w.textContent = text;
  const c = msg.counts;
  const counts = c && msg.mode ? `<b>${c.sent ?? 0}</b> sent · ${c.dry ?? 0} dry · ${c.skipped ?? 0} skipped${c.errors ? ` · <b class="bad">${plural(c.errors, "error")}</b>` : ""}` : "";
  const ce = $("#counts");
  if (ce.dataset.html !== counts) { ce.dataset.html = counts; ce.innerHTML = counts; }
  if (performance.now() - statusTimer > 500) { statusTimer = performance.now(); renderTargetStatus(); }
}

// Fired log: records arrive repeatedly (the last 30 in every state message): dedupe by seq, newest first.
// seq restarts with each session (runner), so a new run shows as a seq going back, another project, or
// another record under the newest seq seen: a restart that fired past that seq at once (start chase,
// skipped cues, catch-up) before this console saw its first state.
const firedSig = (r) => `${r.wall}|${r.cue_id}|${r.time}|${r.reason}`;
function addFired(msg) {
  const seq = msg.fired_seq;
  if (seq == null || !Array.isArray(msg.fired)) return;
  const known = S.lastSig != null ? msg.fired.find((r) => r.seq === S.lastSeq) : null;
  if (seq < S.lastSeq || msg.project_id !== S.firedProject || (known && firedSig(known) !== S.lastSig)) {
    S.firedRun++;
    S.lastSeq = 0;
    S.lastSig = null;
    S.firedProject = msg.project_id;
  }
  const fresh = msg.fired.filter((r) => r.seq > S.lastSeq).sort((a, b) => a.seq - b.seq);
  if (!fresh.length) return;
  S.lastSeq = fresh[fresh.length - 1].seq;
  S.lastSig = firedSig(fresh[fresh.length - 1]);
  const tb = $("#t-fired tbody");
  const rows = [];
  for (const r of fresh) {
    const key = `${S.firedRun}:${r.seq}`;
    if (S.firedKeys.has(key)) continue;
    S.firedKeys.add(key);
    S.fired.unshift({ ...r, key });
    rows.unshift(firedRow(r));
    if (r.reason !== "skipped") S.liveFired = { project: msg.project_id, id: r.cue_id };
  }
  if (!rows.length) return;
  tb.insertAdjacentHTML("afterbegin", rows.join(""));
  while (S.fired.length > FIRED_MAX) S.firedKeys.delete(S.fired.pop().key);
  while (tb.rows.length > FIRED_MAX) tb.deleteRow(-1);
  $("#fired-empty").hidden = true;
}
function firedResult(r) {
  if (r.error) return "error";
  if (r.reason === "skipped") return "skipped";
  if (r.dry_run) return "dry";
  return r.sent ? "sent" : "skipped";
}
const RESULT_TEXT = { sent: "sent", dry: "dry", skipped: "skipped", error: "error" };
function firedRow(r) {
  const res = firedResult(r);
  const chase = r.reason === "chase" || r.reason === "start" ? " chase" : "";
  const late = res === "skipped" || !Number.isFinite(r.ref_late_ms) ? "–" : `${Math.round(r.ref_late_ms)}`;
  const why = r.reason && r.reason !== "track" && r.reason !== "skipped" ? `<small class="why">${esc(r.reason)}</small>` : "";
  return `<tr class="${res}${chase} fresh" title="${esc(r.error || `${r.reason}: ${r.message}`)}">
    <td>${fmtClock(r.wall)}</td><td>${r.cue_id}</td><td>${fmtTime(r.time, 3)}</td><td>${esc(r.target)}</td>
    <td class="msg">${esc(r.message)}</td><td class="res">${RESULT_TEXT[res]}${why}</td><td class="num">${late}</td></tr>`;
}
$("#b-clear-log").addEventListener("click", () => {
  S.fired = [];
  S.firedKeys.clear();
  $("#t-fired tbody").textContent = "";
  $("#fired-empty").hidden = false;
});

// The WebSocket dropped (server gone, or the link stalled): the live panel must not go on looking like a
// tracking session. Its status, countdown and live controls show that the state is unknown until the
// next state message (onState) shows the server's state again.
function goOffline() {
  clock.moving = false;
  if (S.offline) return;
  S.offline = true;
  S.wasRunning = false;                    // a session found running on reconnect turns preview OSC off
  const st = $("#st");
  st.dataset.status = "offline";
  $("b", st).textContent = STATUS_TEXT.offline;
  $("#live").classList.add("offline");
  for (const b of [$("#b-hold"), ...$$("[data-nudge]")]) b.disabled = true;
  const w = $("#live-warn");
  w.textContent = S.last?.mode
    ? "Lost the connection to the server: the session state is unknown (it may still be running, or have "
      + "stopped with the server). Reconnecting…"
    : "Lost the connection to the server. Reconnecting…";
  w.hidden = false;
  markLive();
}

// Next cue with a countdown extrapolated from the state message's eta.
let nextKey = "";
function drawNext(now) {
  const up = sessionRunning() && !S.offline ? S.last.upcoming || [] : [];
  const box = $("#next");
  let eta = "–", what = "", more = "", soon = false;
  if (S.offline) {
    what = S.last?.mode ? "Disconnected: the next cue is unknown" : "";
  } else if (up.length) {
    const u = up[0];
    if (u.eta != null && Number.isFinite(u.eta)) {
      const left = Math.max(0, u.eta - (now - S.lastAt) / 1000);
      eta = left >= 60 ? fmtTime(left, 0) : `${left.toFixed(1)} s`;
      soon = left < 2;
    }
    what = `#${u.id}  ${fmtTime(u.time, 3)}  ${u.target || ""}  ${u.message || ""}`;
    more = up.slice(1, 3).map((x) => `#${x.id} ${fmtTime(x.time, 3)} ${x.target || ""} ${x.message || ""}`).join("   ·   ");
  } else if (sessionRunning()) {
    what = "No more cues";
  }
  const key = `${eta}|${what}|${more}|${soon}`;
  if (key === nextKey) return;
  nextKey = key;
  box.classList.toggle("idle", !up.length);
  const e = $("#next-eta");
  e.textContent = eta;
  e.classList.toggle("soon", soon);
  $("#next-what").textContent = what;
  $("#next-what").title = what;
  $("#next-more").textContent = more;
}

// ------------------------------------------------------------------ websocket
function onHello(msg) {
  if (msg.settings_rev === undefined || msg.settings_rev !== S.settingsRev) {
    S.settingsRev = msg.settings_rev;
    loadSettings();
  }
  if (!S.selected && msg.project_id) selectProject(msg.project_id);
}
// A project changed: {id, rev (a counter per project), sheet_rev}: its cue file (any console's edit, or
// cues.txt edited by hand), style, stems, or it was deleted. Checked through the edit queue, so the
// response of an edit in flight lands first and the announcement of our own edit is a no-op.
function onProjectMsg(msg) {
  refreshProjectsSoon();
  if (msg.id !== S.selected) return;
  queueEdit(async () => {
    const id = msg.id;
    if (id !== S.selected || !S.sheet || msg.rev === S.projectRev) return;
    const before = S.sheet.rev;
    let detail;
    try {
      detail = await api(`/api/projects/${enc(id)}`);
    } catch (e) {
      if (e.status === 404 && S.selected === id) { clearSelection(); renderProjects(); notify("This project was deleted.", true); }
      return;
    }
    if (id !== S.selected) return;
    S.project = detail.project || S.project;
    S.projectRev = detail.rev ?? msg.rev;
    syncStemOptions();
    showStyle(S.project.style);
    renderLogs(detail.logs || []);
    const sheet = sheetOf(detail);
    if (sheet && sheet.rev !== before) {
      applySheet(sheet);
      notify("The cue file changed elsewhere and was reloaded.");
    }
  });
}
function onSettingsMsg(msg) {
  if (typeof msg.armed === "boolean" && performance.now() > armHoldUntil) setArmed(msg.armed);
  if (msg.rev !== undefined && msg.rev === S.settingsRev) return;
  S.settingsRev = msg.rev;
  loadSettings();
}

connectWS((msg) => {
  if (msg.type === "state") onState(msg);
  else if (msg.type === "hello") onHello(msg);
  else if (msg.type === "project") onProjectMsg(msg);
  else if (msg.type === "settings") onSettingsMsg(msg);
}, (ok) => {
  const c = $("#conn");
  c.dataset.status = ok ? "online" : "offline";
  $("b", c).textContent = ok ? "Connected" : "Disconnected";
  if (!ok) { goOffline(); return; }
  // (re)connected, possibly to a restarted server: reload what it owns
  S.settingsRev = undefined;
  refreshProjects();
  if (S.selected) reloadSheet();
});

// ------------------------------------------------------------------ keyboard
// Space play / pause, T tap, ⌘Z / Ctrl+Z undo, H hold, [ / ] nudge, Esc cancels a drag / leaves Tap
// mode / closes the add form. Typing in a field (and any open dialog) is left alone. A focused select
// (it keeps the focus after use) is not typing: the shortcut keys act there and are kept from its
// type-ahead, which would pick another option and apply it at once (default target, style …).
const TYPING = "textarea, [contenteditable], input:not([type=checkbox]):not([type=radio]):not([type=range]):not([type=button])";
const SHORTCUT_KEYS = new Set([" ", "t", "h", "[", "]"]);
document.addEventListener("keydown", (e) => {
  if (dFile.open || dMeasure.open) return;
  const typing = e.target.closest?.(TYPING);
  if (e.key === "Escape") {
    if (cancelZoomDrag()) { e.preventDefault(); return; }
    if (typing) return;
    if (!fAdd.hidden) closeAdd();
    else if (tap.on) setTap(false);
    return;
  }
  if (typing) return;
  const k = (e.key || "").toLowerCase();
  if (e.repeat) {
    if (SHORTCUT_KEYS.has(k) && !e.metaKey && !e.ctrlKey && !e.altKey) e.preventDefault();
    return;
  }
  if (k === "z" && (e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey) { e.preventDefault(); undoEdit(); return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === " ") { e.preventDefault(); if (S.selected) preview.toggle(); }
  else if (k === "t") {
    e.preventDefault();
    if (tap.on) tapNow(); else notify("Turn on Tap mode first (Tap button), then press T at each cue.");
  }
  else if (k === "h") { e.preventDefault(); toggleHold(); }
  else if (e.key === "[") { e.preventDefault(); nudge(-0.5); }
  else if (e.key === "]") { e.preventDefault(); nudge(0.5); }
});
// a focused button / checkbox activates on Space key-up: Space is play / pause here, not a click
document.addEventListener("keyup", (e) => {
  if (e.key === " " && !dFile.open && !dMeasure.open && !e.target.closest?.(TYPING)) e.preventDefault();
});

// ------------------------------------------------------------------ frame loop
let posText = "", playText = "", refText = "", durText = "";
function frame(now) {
  requestAnimationFrame(frame);
  const pos = preview.heard();
  const p = fmtTime(pos, 3);
  if (p !== posText) { posText = p; $("#p-pos").textContent = p; }
  const pl = preview.playing ? "Pause" : "Play";
  if (pl !== playText) { playText = pl; $("#b-play").textContent = pl; $("#b-play").classList.toggle("on", preview.playing); }
  const d = songDur();
  const dt = d ? fmtTime(d, 3) : "–";
  if (dt !== durText) { durText = dt; $("#p-dur").textContent = dt; }
  let livePos = null, refT = "–";
  if (S.last?.mode) {
    const t = clock.tick(now);
    refT = fmtTime(t, 2);
    if (liveHere()) livePos = t;
  }
  if (refT !== refText) { refText = refT; $("#r-ref").textContent = refT; }
  updatePassed(pos);
  drawNext(now);
  if (S.selected) {
    lastCenter = zoomCenter(pos, livePos);
    drawWave(pos, livePos);
    drawZoom(pos, livePos);
  } else if (wave.lastKey !== "none") {
    wave.lastKey = "none";
    wave.lastZoomKey = "";
    renderWaveStatic();
    waveCv.getContext("2d").drawImage(wave.staticCv, 0, 0);
    drawZoom(0, null);
  }
}

// ------------------------------------------------------------------ boot
fitCanvas(waveCv);
fitCanvas(zoomCv);
setArmed(null);
syncFormVisibility();
syncStemOptions();
syncEditorTools();
renderCues();
refreshProjects();
loadSettings();
loadDevices();
requestAnimationFrame(frame);
