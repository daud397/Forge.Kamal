const $ = (id) => document.getElementById(id);

let config = null;
let jobId = null;
let poll = null;
let attached = [];
let busy = false;
let lastRender = "";      // so polling doesn't rebuild an identical thread

init();

async function init() {
  config = await (await fetch("/api/status")).json();

  const mode = $("mode");
  mode.textContent = config.mode === "live" ? "live" : "mock mode";
  mode.className = "meter " + config.mode;
  mode.title = "Build " + (config.build || "unknown") +
               " · " + config.draft_model;

  if (config.auth) $("logoutform").hidden = false;
  drawSpend(config.spend);
  setInterval(refreshSpend, 30000);

  wire();
  $("thread").appendChild(introBlock());
  loadJobs();
}

function wire() {
  $("send").onclick = submit;
  $("attach").onclick = () => $("files").click();
  $("files").onchange = showAttached;
  $("newrun").onclick = startFresh;
  $("toggleside").onclick = () => $("sidebar").classList.toggle("hidden");
  $("lbclose").onclick = () => { $("lightbox").hidden = true; };
  $("fileunder").onclick = fileUnderProject;

  document.querySelectorAll(".tab").forEach(t => {
    t.onclick = () => showTab(t.dataset.tab);
  });
  wireEditor();
  $("lightbox").onclick = (ev) => {
    if (ev.target.id !== "lbimg") $("lightbox").hidden = true;
  };

  const box = $("box");
  box.addEventListener("keydown", ev => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); submit(); }
  });
  box.addEventListener("input", () => {
    box.style.height = "auto";
    box.style.height = Math.min(box.scrollHeight, 190) + "px";
  });

  document.querySelectorAll(".chip").forEach(c => {
    c.onclick = () => { $("box").value = c.dataset.say; submit(); };
  });
}

function startFresh() {
  jobId = null;
  attached = [];
  clearInterval(poll);
  lastRender = "";
  $("files").value = "";
  $("attached").hidden = true;
  $("threadtitle").textContent = "Kamal Forge";
  $("fileunder").hidden = true;
  $("thread").innerHTML = "";
  $("thread").appendChild(introBlock());
  loadJobs();
}

function introBlock() {
  const d = document.createElement("div");
  d.className = "intro";
  d.innerHTML = `
    <h1>What are we photographing?</h1>
    <p>Describe a product and it gets made from nothing, or attach a photo or
       CAD file to work from the real thing.</p>
    <div class="chips">
      <button class="chip" data-say="a matte charcoal linen cushion cover, 45cm square, hidden zip">A charcoal linen cushion cover</button>
      <button class="chip" data-say="a stonewashed sand cotton duvet set on a made bed, UK bedroom">A sand cotton duvet set</button>
      <button class="chip" data-say="a grey cheetah print duvet set on a made bed in a modern UK bedroom">A grey cheetah print duvet set</button>
    </div>`;
  setTimeout(() => d.querySelectorAll(".chip").forEach(c => {
    c.onclick = () => { $("box").value = c.dataset.say; submit(); };
  }), 0);
  return d;
}

// ---------------------------------------------------------------- sending

function showAttached() {
  attached = [...$("files").files];
  const note = $("attached");
  if (!attached.length) { note.hidden = true; return; }
  note.hidden = false;
  note.textContent = attached.map(f => f.name).join(", ") +
    " — add a note if useful, then send.";
}

async function submit() {
  if (busy) return;
  const box = $("box");
  const text = box.value.trim();

  if (!text && !attached.length) return;

  busy = true;
  $("send").disabled = true;
  box.value = "";
  box.style.height = "auto";

  const intro = document.querySelector(".intro");
  if (intro) intro.remove();

  if (text) say("user", text);

  try {
    if (attached.length) {
      await startFromFiles(text);
    } else if (!jobId) {
      await startFromDescription(text);
    } else {
      await sendToChat(text);
    }
  } catch (err) {
    say("bot", "That didn't go through: " + err.message);
  }

  busy = false;
  $("send").disabled = false;
  box.focus();
}

async function startFromFiles(note) {
  const body = new FormData();
  attached.forEach(f => body.append("files", f));
  body.append("note", note);

  const pend = say("bot", "Reading the file…", true);
  const res = await fetch("/api/jobs", { method: "POST", body });
  if (!res.ok) throw new Error(await res.text());

  attached = [];
  $("files").value = "";
  $("attached").hidden = true;
  pend.remove();

  jobId = (await res.json()).job_id;
  lastRender = "";
  watch();
}

async function startFromDescription(text) {
  const body = new FormData();
  body.append("description", text);

  const pend = say("bot", "Writing the brief…", true);
  const res = await fetch("/api/jobs/describe", { method: "POST", body });
  if (!res.ok) throw new Error(await res.text());

  pend.remove();
  jobId = (await res.json()).job_id;
  lastRender = "";
  watch();
}

async function sendToChat(text) {
  const body = new FormData();
  body.append("message", text);

  const pend = say("bot", "Working…", true);
  const res = await fetch(`/api/jobs/${jobId}/chat`, { method: "POST", body });
  const data = await res.json();

  pend.remove();
  say("bot", data.reply);
  lastRender = "";
  watch();
}

function say(who, text, thinking) {
  const t = document.createElement("div");
  t.className = "turn " + (who === "user" ? "user" : "bot");
  const b = document.createElement("div");
  b.className = "bubble" + (thinking ? " thinking" : "");
  b.textContent = text;
  t.appendChild(b);

  const thread = $("thread");
  thread.appendChild(t);
  thread.scrollTop = thread.scrollHeight;
  return t;
}

// ---------------------------------------------------------------- polling

function watch() {
  clearInterval(poll);
  poll = setInterval(refresh, 1300);
  refresh();
}

async function refresh() {
  if (!jobId) return;
  const state = await (await fetch(`/api/jobs/${jobId}`)).json();

  render(state);

  if (["drafted", "finalised", "exported", "failed"].includes(state.stage)) {
    clearInterval(poll);
    loadJobs();
  }
}

/* The thread is rebuilt from job state rather than appended to, so reopening an
   old run shows exactly what that run produced. The fingerprint stops the
   1.3s poll from redrawing an unchanged thread and stealing the scroll. */
function render(state) {
  const stamp = JSON.stringify([
    state.stage, state.chosen_draft, state.final_file,
    (state.drafts || []).length, (state.versions || []).length,
    (state.exports || []).length, (state.chat || []).length,
    (state.log || []).length,
  ]);
  if (stamp === lastRender) return;
  lastRender = stamp;

  const brief = state.brief || {};
  $("threadtitle").textContent = brief.product_name || state.note || "New run";
  $("fileunder").hidden = false;
  $("fileunder").textContent = state.project
    ? `Project: ${state.project}` : "File under a project";

  const thread = $("thread");
  thread.innerHTML = "";

  if (state.note) say("user", state.note);

  (state.chat || []).forEach(m => say(m.role === "user" ? "user" : "bot", m.content));

  if (state.stage === "failed") {
    status(friendlyError(state.error), true);
    return;
  }

  if (!state.drafts && !state.final_file) {
    status(working(state.stage));
  }

  if (state.renders && Object.keys(state.renders).length) {
    imageCard("Rendered from your file", Object.entries(state.renders)
      .map(([name, file]) => ({ file, label: name })), state, "three");
  }

  if (state.drafts && !state.final_file) {
    imageCard(state.mode === "scratch" ? "Pick one" : "Pick a scene",
      state.drafts, state, state.drafts.length > 2 ? "three" : "two", true);
  }

  if (state.final_file) finalCard(state);

  thread.scrollTop = thread.scrollHeight;
}

function working(stage) {
  return {
    ingesting: "Reading your file…",
    ingested: "Read it.",
    briefing: "Writing the brief…",
    analysing: "Looking at the product…",
    briefed: "Brief written.",
    drafting: "Generating options…",
    finalising: "Making the final image…",
    exporting: "Exporting…",
  }[stage] || "Working…";
}

function friendlyError(raw) {
  const t = (raw || "").toLowerCase();
  if (t.includes("patches") || t.includes("resize the image"))
    return "That image was too large to read. Try one under about 3000px.";
  if (t.includes("cap"))
    return "The daily spend cap stopped this run. It resets at midnight UTC.";
  if (t.includes("verif"))
    return "OpenAI hasn't verified this organisation yet, so images can't be generated.";
  if (t.includes("api key") || t.includes("authentication"))
    return "The API key was rejected.";
  return "That run failed: " + (raw || "no reason given");
}

function status(text, bad) {
  const d = document.createElement("div");
  d.className = "status" + (bad ? " bad" : "");
  d.innerHTML = `<span class="dot"></span><span></span>`;
  d.lastChild.textContent = text;
  $("thread").appendChild(d);
}

// ---------------------------------------------------------------- cards

function imageCard(heading, items, state, cols, pickable) {
  const card = document.createElement("div");
  card.className = "card";
  card.innerHTML = `<h3>${esc(heading)}</h3>
    <div class="grid ${cols}"></div>`;
  const grid = card.querySelector(".grid");

  items.forEach(it => {
    const b = document.createElement("button");
    b.className = "shot" + (pickable ? " pick" : "") +
      (state.chosen_draft === it.index ? " chosen" : "");
    b.innerHTML = `<span class="frame">
        <img src="/api/jobs/${jobId}/file/${it.file}" alt="${esc(it.label)}">
      </span><span class="cap">${esc(it.label)}</span>`;

    b.onclick = pickable
      ? () => choose(it.index, it.label)
      : () => zoom(`/api/jobs/${jobId}/file/${it.file}`);
    grid.appendChild(b);
  });

  $("thread").appendChild(card);
}

function finalCard(state) {
  const card = document.createElement("div");
  card.className = "card single";
  card.innerHTML = `
    <h3>Final image</h3>
    <div class="frame">
      <img id="finalimg" src="/api/jobs/${jobId}/file/${state.final_file}?v=${(state.versions || []).length}" alt="Final image">
    </div>`;

  const qa = state.qa;
  if (qa) {
    const words = { pass: "Colour and shape held", warn: "Worth a look",
                    fail: "The model drifted", info: "Judge this by eye" };
    const v = document.createElement("p");
    v.className = "verdict";
    v.innerHTML = `<span class="tag ${qa.level}">${words[qa.level] || qa.level}</span>`;
    if (qa.notes && qa.notes[0]) {
      v.appendChild(document.createTextNode(" — " + qa.notes[0]));
    }
    card.appendChild(v);
  }

  const acts = document.createElement("div");
  acts.className = "actions";

  add(acts, "Edit image", () =>
    openEditor(`/api/jobs/${jobId}/file/${state.final_file}?v=${(state.versions || []).length}`));
  add(acts, "Enlarge 2×", () => enhance("upscale"));
  add(acts, "Run again", repeat);

  (state.exports || []).forEach(e => {
    const a = document.createElement("a");
    a.href = `/api/jobs/${jobId}/file/${e.file}`;
    a.download = "";
    a.textContent = "↓ " + e.label;
    if (!e.generative_allowed) {
      a.className = "warnlink";
      a.title = "This slot expects a real photograph.";
    }
    acts.appendChild(a);
  });

  if (!(state.exports || []).length) {
    add(acts, "Export all sizes", exportAll);
  }

  card.appendChild(acts);
  $("thread").appendChild(card);

  card.querySelector("#finalimg").onclick = () =>
    zoom(`/api/jobs/${jobId}/file/${state.final_file}`);
}

function add(parent, label, fn) {
  const b = document.createElement("button");
  b.textContent = label;
  b.onclick = fn;
  parent.appendChild(b);
}

// ---------------------------------------------------------------- actions

async function choose(index, label) {
  say("user", `Use ${label}`);
  const body = new FormData();
  body.append("draft_index", index);
  body.append("instruction", "");
  body.append("presets", "");
  await fetch(`/api/jobs/${jobId}/finalise`, { method: "POST", body });
  lastRender = "";
  watch();
}

async function enhance(action) {
  const body = new FormData();
  body.append("instruction", "");
  body.append("action", action);
  body.append("scale", "2");
  await fetch(`/api/jobs/${jobId}/enhance`, { method: "POST", body });
  lastRender = "";
  watch();
}

async function exportAll() {
  const wanted = Object.entries(config.presets)
    .filter(([, p]) => p.generative).map(([k]) => k);
  const body = new FormData();
  body.append("presets", wanted.join(","));
  await fetch(`/api/jobs/${jobId}/export`, { method: "POST", body });
  lastRender = "";
  watch();
}

async function repeat() {
  const res = await fetch(`/api/jobs/${jobId}/rerun`, { method: "POST" });
  if (!res.ok) return;
  jobId = (await res.json()).job_id;
  lastRender = "";
  $("thread").innerHTML = "";
  watch();
}

function zoom(src) {
  $("lbimg").src = src;
  $("lightbox").hidden = false;
}

// ---------------------------------------------------------------- sidebar

async function loadJobs() {
  const url = projectFilter === null
    ? "/api/jobs"
    : `/api/jobs?project=${encodeURIComponent(projectFilter)}`;
  const { jobs } = await (await fetch(url)).json();
  const box = $("jobs");

  if (!jobs.length) {
    box.innerHTML = `<p class="empty">No runs yet</p>`;
    return;
  }

  box.innerHTML = jobs.map(j => {
    const thumb = j.thumb
      ? `<img class="run-thumb" src="/api/jobs/${j.id}/file/${j.thumb}" alt="">`
      : `<span class="run-thumb blank">${j.stage === "failed" ? "—" : "…"}</span>`;
    const meta = j.stage === "failed"
      ? `<span class="bad">failed</span>`
      : (j.exported ? `${j.exported} exported` : esc(j.stage));
    return `<button class="run ${j.id === jobId ? "current" : ""}" data-id="${j.id}">
      ${thumb}
      <span class="run-text">
        <span class="run-name">${esc(j.product || "Untitled run")}</span>
        <span class="run-meta">${meta}</span>
      </span></button>`;
  }).join("");

  box.querySelectorAll(".run").forEach(b => {
    b.onclick = () => {
      jobId = b.dataset.id;
      lastRender = "";
      $("thread").innerHTML = "";
      watch();
      if (window.innerWidth < 860) $("sidebar").classList.add("hidden");
    };
  });
}

// ---------------------------------------------------------------- spend

function drawSpend(spend) {
  if (!spend || !spend.enforced) return;
  const chip = $("spend");
  chip.hidden = false;
  const ratio = spend.cap > 0 ? spend.today / spend.cap : 0;
  chip.textContent = `$${spend.today.toFixed(2)} / $${spend.cap.toFixed(2)}`;
  chip.className = "meter" + (ratio >= 1 ? " over" : ratio >= 0.8 ? " near" : "");
  chip.title = spend.note + " Resets at midnight UTC.";
}

async function refreshSpend() {
  try { drawSpend(await (await fetch("/api/spend")).json()); } catch { /* not critical */ }
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}


// ---------------------------------------------------------------- sidebar tabs

let projectFilter = null;

function showTab(name) {
  document.querySelectorAll(".tab").forEach(t =>
    t.classList.toggle("on", t.dataset.tab === name));
  ["runs", "projects", "usage"].forEach(p =>
    $("pane-" + p).hidden = p !== name);

  if (name === "projects") loadProjects();
  if (name === "usage") loadUsage();
}

async function loadProjects() {
  const { projects } = await (await fetch("/api/projects")).json();
  const box = $("projects");

  const all = `<button class="proj ${projectFilter === null ? "on" : ""}" data-name="">
      <span>All runs</span></button>`;

  box.innerHTML = all + (projects.length
    ? projects.map(p => `<button class="proj ${projectFilter === p.name ? "on" : ""}"
        data-name="${esc(p.name)}">
        <span>${esc(p.name)}</span><span class="count">${p.runs}</span></button>`).join("")
    : `<p class="empty">No projects yet. Open a run and use “File under a project”.</p>`);

  box.querySelectorAll(".proj").forEach(b => {
    b.onclick = () => {
      projectFilter = b.dataset.name || null;
      loadProjects();
      loadJobs();
      showTab("runs");
    };
  });
}

async function loadUsage() {
  const data = await (await fetch("/api/usage")).json();
  const s = data.summary;

  const rows = (data.days || []).map(d => `
    <tr><td class="day">${esc(d.day)}</td>
        <td>${d.images} images</td>
        <td>$${d.estimate.toFixed(2)}</td></tr>`).join("");

  $("usage").innerHTML = `
    <div class="big">$${s.today.toFixed(2)}</div>
    <div class="cap">of $${s.cap.toFixed(2)} today${s.enforced ? "" : " (no cap set)"}</div>
    <table>${rows || `<tr><td class="day">Nothing yet</td></tr>`}</table>
    <p class="note">Estimated from the per-call figures in settings, not billed
    amounts. Compare against your OpenAI usage page and correct
    COST_PER_IMAGE if they drift apart.</p>`;
}

async function fileUnderProject() {
  if (!jobId) return;
  const name = prompt("File this run under which project?", projectFilter || "");
  if (name === null) return;

  const body = new FormData();
  body.append("name", name);
  await fetch(`/api/jobs/${jobId}/project`, { method: "POST", body });
  loadProjects();
  loadJobs();
}

// ---------------------------------------------------------------- editor

let edCtx = null, edPainting = false, edSource = null;

function wireEditor() {
  const canvas = $("edpaint");
  edCtx = canvas.getContext("2d");

  const at = ev => {
    const r = canvas.getBoundingClientRect();
    return [(ev.clientX - r.left) * canvas.width / r.width,
            (ev.clientY - r.top) * canvas.height / r.height];
  };

  const stroke = ev => {
    if (!edPainting) return;
    const [x, y] = at(ev);
    const size = Number($("edbrush").value) *
                 canvas.width / canvas.getBoundingClientRect().width;
    edCtx.fillStyle = "rgba(91,127,185,0.45)";
    edCtx.beginPath();
    edCtx.arc(x, y, size / 2, 0, Math.PI * 2);
    edCtx.fill();
    markPainted();
  };

  canvas.addEventListener("pointerdown", ev => {
    edPainting = true; canvas.setPointerCapture(ev.pointerId); stroke(ev);
  });
  canvas.addEventListener("pointermove", stroke);
  canvas.addEventListener("pointerup", () => { edPainting = false; });
  canvas.addEventListener("pointercancel", () => { edPainting = false; });

  $("edclear").onclick = () => {
    edCtx.clearRect(0, 0, canvas.width, canvas.height);
    $("edstate").textContent = "Whole frame";
  };
  $("edclose").onclick = () => { $("editor").hidden = true; };
  $("edapply").onclick = applyEdit;
  $("edtext").addEventListener("keydown", ev => {
    if (ev.key === "Enter") { ev.preventDefault(); applyEdit(); }
  });
}

function markPainted() {
  $("edstate").textContent = "Painted area only";
}

function openEditor(src) {
  const img = $("edimg");
  const canvas = $("edpaint");
  edSource = src;

  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.style.width = img.clientWidth + "px";
    canvas.style.height = img.clientHeight + "px";
    edCtx.clearRect(0, 0, canvas.width, canvas.height);
    $("edstate").textContent = "Whole frame";
  };
  img.src = src;

  $("edtext").value = "";
  $("editor").hidden = false;
  setTimeout(() => $("edtext").focus(), 60);
}

function paintedRegion() {
  const canvas = $("edpaint");
  const data = edCtx.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] > 0) return new Promise(r => canvas.toBlob(r, "image/png"));
  }
  return Promise.resolve(null);
}

async function applyEdit() {
  const text = $("edtext").value.trim();
  if (!text) { $("edtext").focus(); return; }

  const body = new FormData();
  body.append("instruction", text);
  body.append("action", "edit");
  body.append("scale", "2");

  const region = await paintedRegion();
  if (region) body.append("region", region, "region.png");

  $("edapply").disabled = true;
  await fetch(`/api/jobs/${jobId}/enhance`, { method: "POST", body });
  $("edapply").disabled = false;
  $("editor").hidden = true;

  say("user", (region ? "Edit the painted area: " : "Edit: ") + text);
  lastRender = "";
  watch();
}
