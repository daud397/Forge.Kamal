const $ = (id) => document.getElementById(id);

const STAGES = ["ingesting", "ingested", "briefing", "analysing", "briefed",
                "drafting", "drafted", "finalising", "finalised",
                "exporting", "exported"];

let jobId = null;
let picked = null;
let poll = null;
let config = null;
let pending = [];
let sending = false;

init();

async function init() {
  config = await (await fetch("/api/status")).json();

  const mode = $("mode");
  mode.textContent = config.mode === "live"
    ? `live, ${config.draft_model}`
    : "mock mode, no API calls";
  mode.className = "meter " + config.mode;
  mode.title = "Build " + (config.build || "unknown");

  if (config.auth) $("logoutform").hidden = false;
  drawSpend(config.spend);
  setInterval(refreshSpend, 30000);

  renderPresets();
  loadJobs();
  wireIntake();
  wirePaint();
  wireChat();

  $("makefinal").onclick = makeFinal;
  $("doexport").onclick = runExport;
  $("jumptochat").onclick = () => $("chatbox").focus();
  $("doedit").onclick = () => applyWork("edit");
  $("doupscale").onclick = () => applyWork("upscale");
  $("dorerun").onclick = repeatRun;
  $("workbox").addEventListener("keydown", ev => {
    if (ev.key === "Enter") { ev.preventDefault(); applyWork("edit"); }
  });
}

// ---------------------------------------------------------------- intake

function wireIntake() {
  const drop = $("drop"), input = $("files");

  ["dragenter", "dragover"].forEach(e =>
    drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("hot"); }));
  ["dragleave", "drop"].forEach(e =>
    drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("hot"); }));

  drop.addEventListener("drop", ev => { input.files = ev.dataTransfer.files; showFiles(); });
  input.addEventListener("change", showFiles);
  $("start").onclick = startRun;
}

const CAD = ["stl","obj","ply","glb","gltf","off","3mf","step","stp","iges","igs"];

function showFiles() {
  pending = [...$("files").files];
  $("filelist").innerHTML = pending.map(f => {
    const ext = f.name.split(".").pop().toLowerCase();
    return `<li><span>${esc(f.name)}</span>
            <span class="kind">${CAD.includes(ext) ? "CAD" : "photo"}</span></li>`;
  }).join("");
  $("start").disabled = pending.length === 0;
}

async function startRun() {
  const body = new FormData();
  pending.forEach(f => body.append("files", f));
  body.append("note", $("note").value);

  $("start").disabled = true;
  $("start").textContent = "Starting";

  const res = await fetch("/api/jobs", { method: "POST", body });
  $("start").textContent = "Start run";

  if (!res.ok) {
    $("start").disabled = false;
    return alert("Upload failed: " + await res.text());
  }

  beginWatching((await res.json()).job_id);
}

function beginWatching(id) {
  jobId = id;
  picked = null;

  // Clear the previous run's panels. Without this, opening an older job leaves
  // the last run's drafts and verdict on screen next to the new one's, which
  // reads as the portal showing you the wrong images.
  ["renders", "paint", "sheet", "proof", "work", "briefblock", "qablock"]
    .forEach(p => { $(p).hidden = true; });
  $("exports").innerHTML = "";
  $("doexport").disabled = true;
  $("makefinal").disabled = true;

  $("logblock").hidden = false;
  $("placeholder").hidden = true;
  enableChatHistory();
  loadJobs();
  watch();
}

function enableChatHistory() {
  const starters = $("chat").querySelector(".starters");
  if (starters) starters.remove();
}

// ---------------------------------------------------------------- polling

function watch() {
  clearInterval(poll);
  poll = setInterval(refresh, 1200);
  refresh();
}

async function refresh() {
  if (!jobId) return;
  const state = await (await fetch(`/api/jobs/${jobId}`)).json();
  const scratch = state.mode === "scratch";

  drawStages(state.stage, scratch);
  drawLog(state.log || []);

  if (state.renders && Object.keys(state.renders).length) drawRenders(state);
  if (state.mask_source === "needs_paint" && state.base_image) setupPaint(state);
  if (state.brief) drawBrief(state.brief);
  if (state.drafts) drawDrafts(state, scratch);
  if (state.qa) drawQA(state, scratch);
  if (state.final_file) drawProof(state, scratch);
  if (state.drafts || state.final_file) drawWork(state);
  if (state.exports) drawExports(state);
  if (state.chat && !sending) drawChat(state.chat);

  if (["drafted", "finalised", "exported", "failed"].includes(state.stage)) {
    clearInterval(poll);
    loadJobs();
  }
}

function drawStages(current, scratch) {
  const steps = scratch
    ? [["brief", 2, 4], ["generate", 5, 6], ["refine", 7, 8], ["export", 9, 10]]
    : [["ingest", 0, 1], ["brief", 2, 4], ["draft", 5, 6], ["final", 7, 8], ["export", 9, 10]];

  const idx = STAGES.indexOf(current);
  $("stages").innerHTML = steps.map(([name, from, to]) => {
    let cls = "";
    if (current !== "failed") {
      if (idx > to) cls = "done";
      else if (idx >= from) cls = "active";
    } else if (idx >= from) cls = "";
    else cls = "done";
    return `<span class="${cls}">${name}</span>`;
  }).join("") + (current === "failed" ? `<span class="failed">failed</span>` : "");
}

function drawLog(entries) {
  const log = $("log");
  log.innerHTML = entries.map(e => `<li class="${e.level}">${esc(e.message)}</li>`).join("");
  log.scrollTop = log.scrollHeight;
}

// ---------------------------------------------------------------- panels

function drawRenders(state) {
  $("renders").hidden = false;
  $("render-row").innerHTML = Object.entries(state.renders).map(([name, file]) =>
    `<div class="well"><img src="/api/jobs/${jobId}/file/${file}"
       alt="${esc(name)} render" title="${esc(name)}"></div>`).join("");
}

function drawBrief(brief) {
  $("briefblock").hidden = false;
  $("brief").innerHTML = `
    <dl>
      <dt>Product</dt><dd>${esc(brief.product_name || "—")}</dd>
      <dt>Category</dt><dd>${esc(brief.category || "—")}</dd>
      <dt>Colours</dt>
      <dd><div class="swatches">${(brief.dominant_colours || []).map(c =>
        `<span class="swatch" style="background:${esc(c.hex)}"
           title="${esc(c.name)} ${esc(c.hex)}"></span>`).join("")}</div></dd>
      <dt>Surface</dt><dd>${esc(brief.surface_pattern || "—")}</dd>
      <dt>Must survive editing</dt>
      <dd><div class="tags">${(brief.must_preserve || []).map(m =>
        `<span>${esc(m)}</span>`).join("")}</div></dd>
      ${(brief.risks || []).length
        ? `<dt>Known risks</dt><dd>${brief.risks.map(esc).join("<br>")}</dd>` : ""}
    </dl>`;
}

function drawDrafts(state, scratch) {
  $("sheet").hidden = false;
  $("sheet-title").textContent = scratch ? "Generated options" : "Drafts";

  $("sheet-row").innerHTML = state.drafts.map(d => `
    <button class="card ${picked === d.index ? "selected" : ""}" data-i="${d.index}">
      <span class="well"><img src="/api/jobs/${jobId}/file/${d.file}" alt="${esc(d.label)}"></span>
      <figcaption>${esc(d.label)}</figcaption>
    </button>`).join("");

  document.querySelectorAll(".card").forEach(card => {
    card.onclick = () => {
      picked = Number(card.dataset.i);
      document.querySelectorAll(".card").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      $("makefinal").disabled = false;
    };
  });
}

function drawQA(state, scratch) {
  const { stats, level, notes, review } = state.qa;
  $("qablock").hidden = false;

  const label = {
    pass: "within tolerance", warn: "worth a look",
    fail: "out of tolerance", info: "no source to compare",
  }[level] || level;

  const t = config.tolerances;
  $("qa").innerHTML = `
    <span class="verdict ${level}">${label}</span>
    <table class="metrics">
      <tr><td>Colour shift, mean</td><td>${stats.delta_e_mean} dE</td></tr>
      <tr><td>Colour shift, 95th</td><td>${stats.delta_e_p95} dE</td></tr>
      <tr><td>Structure kept</td><td>${stats.ssim_product}</td></tr>
      <tr><td>Product area</td><td>${(stats.product_coverage * 100).toFixed(1)}%</td></tr>
    </table>
    <ul class="notes">
      ${notes.map(n => `<li>${esc(n)}</li>`).join("")}
      ${(review?.issues || []).map(i => `<li>${esc(i)}</li>`).join("")}
    </ul>
    ${scratch ? "" : `<p class="threshold">Passes under ${t.delta_e_pass} dE,
      fails over ${t.delta_e_warn} dE, structure floor ${t.ssim_pass}.</p>`}`;
}

function drawProof(state, scratch) {
  $("proof").hidden = false;
  $("before-label").textContent = scratch ? "Option you chose" : "Source";
  const before = scratch
    ? (state.drafts.find(d => d.index === state.chosen_draft) || {}).file
    : state.base_image;
  if (before) $("proof-before").src = `/api/jobs/${jobId}/file/${before}`;
  $("proof-after").src = `/api/jobs/${jobId}/file/${state.final_file}?t=${Date.now()}`;
  $("doexport").disabled = false;
}

function renderPresets() {
  $("presets").innerHTML = Object.entries(config.presets).map(([key, p]) => `
    <li><label>
      <input type="checkbox" value="${key}" ${p.generative ? "checked" : ""}>
      <span>${esc(p.label)}
        <span class="dims">${p.size} ${p.format}</span>
        ${p.generative ? "" :
          `<span class="flag">Needs a real photograph. Generated imagery risks rejection here.</span>`}
      </span>
    </label></li>`).join("");
}

function drawExports(state) {
  $("exports").innerHTML = state.exports.map(e => `
    <div class="row">
      <a href="/api/jobs/${jobId}/file/${e.file}" download>${esc(e.label)}</a>
      <span class="meta">${e.size}, ${e.scale}</span>
    </div>`).join("") + (state.zip_file
      ? `<div class="row"><a href="/api/jobs/${jobId}/file/${state.zip_file}" download>Everything as a zip</a></div>`
      : "");
}

// ---------------------------------------------------------------- actions

async function makeFinal() {
  if (picked === null) return;

  const body = new FormData();
  body.append("draft_index", picked);
  body.append("instruction", $("instruction").value);
  body.append("presets", selectedPresets().join(","));

  const mask = await paintedMask();
  if (mask) body.append("mask", mask, "mask.png");

  $("makefinal").disabled = true;
  await fetch(`/api/jobs/${jobId}/finalise`, { method: "POST", body });
  watch();
}

async function runExport() {
  const body = new FormData();
  body.append("presets", selectedPresets().join(","));
  $("doexport").disabled = true;
  await fetch(`/api/jobs/${jobId}/export`, { method: "POST", body });
  watch();
}

function selectedPresets() {
  return [...document.querySelectorAll("#presets input:checked")].map(i => i.value);
}

// ---------------------------------------------------------------- mask painting

let painting = false, ctx = null;

function wirePaint() {
  const canvas = $("paint-canvas");
  ctx = canvas.getContext("2d");

  const pos = ev => {
    const r = canvas.getBoundingClientRect();
    return [(ev.clientX - r.left) * canvas.width / r.width,
            (ev.clientY - r.top) * canvas.height / r.height];
  };

  const stroke = ev => {
    if (!painting) return;
    const [x, y] = pos(ev);
    const size = Number($("brush").value) * canvas.width / canvas.getBoundingClientRect().width;
    ctx.fillStyle = "rgba(91,127,185,0.55)";
    ctx.beginPath();
    ctx.arc(x, y, size / 2, 0, Math.PI * 2);
    ctx.fill();
  };

  canvas.addEventListener("pointerdown", ev => {
    painting = true; canvas.setPointerCapture(ev.pointerId); stroke(ev);
  });
  canvas.addEventListener("pointermove", stroke);
  canvas.addEventListener("pointerup", () => { painting = false; });
  canvas.addEventListener("pointercancel", () => { painting = false; });

  $("clearpaint").onclick = () => ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function setupPaint(state) {
  if (!$("paint").hidden) return;
  $("paint").hidden = false;

  const img = $("paint-base");
  img.onload = () => {
    const canvas = $("paint-canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.style.width = img.clientWidth + "px";
    canvas.style.height = img.clientHeight + "px";
  };
  img.src = `/api/jobs/${jobId}/file/${state.base_image}`;
}

function paintedMask() {
  const canvas = $("paint-canvas");
  if ($("paint").hidden) return Promise.resolve(null);

  const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] > 0) return new Promise(r => canvas.toBlob(r, "image/png"));
  }
  return Promise.resolve(null);
}

// ---------------------------------------------------------------- chat

function wireChat() {
  $("send").onclick = sendMessage;
  $("chatbox").addEventListener("keydown", ev => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); sendMessage(); }
  });
  document.querySelectorAll(".starter").forEach(b => {
    b.onclick = () => { $("chatbox").value = b.dataset.say; sendMessage(); };
  });
}

function drawChat(history) {
  const chat = $("chat");
  chat.innerHTML = history.map(m =>
    `<div class="msg ${m.role === "user" ? "you" : "forge"}">${esc(m.content)}</div>`).join("");
  chat.scrollTop = chat.scrollHeight;
}

async function sendMessage() {
  const box = $("chatbox");
  const text = box.value.trim();
  if (!text || sending) return;

  sending = true;
  box.value = "";
  $("send").disabled = true;

  const chat = $("chat");
  const starters = chat.querySelector(".starters");
  if (starters) starters.remove();

  chat.insertAdjacentHTML("beforeend", `<div class="msg you">${esc(text)}</div>`);
  chat.insertAdjacentHTML("beforeend",
    `<div class="msg forge thinking" id="pending">working</div>`);
  chat.scrollTop = chat.scrollHeight;

  try {
    // With no run open, the first message describes a product to create.
    if (!jobId) {
      const body = new FormData();
      body.append("description", text);
      const res = await fetch("/api/jobs/describe", { method: "POST", body });
      if (!res.ok) throw new Error(await res.text());

      settle("Writing the brief, then generating three treatments. They'll appear "
             + "in the middle as they finish.", true);
      beginWatching((await res.json()).job_id);
    } else {
      const body = new FormData();
      body.append("message", text);
      const res = await fetch(`/api/jobs/${jobId}/chat`, { method: "POST", body });
      const data = await res.json();
      settle(data.reply, data.actions?.length > 0);
      watch();
    }
  } catch (err) {
    settle("That didn't go through: " + err.message, false);
  }

  chat.scrollTop = chat.scrollHeight;
  sending = false;
  $("send").disabled = false;
  box.focus();
}

function settle(text, acted) {
  const el = $("pending");
  if (!el) return;
  el.className = "msg forge" + (acted ? " acted" : "");
  el.id = "";
  el.textContent = text;
}

// ---------------------------------------------------------------- spend

function drawSpend(spend) {
  if (!spend || !spend.enforced) return;
  const chip = $("spend");
  chip.hidden = false;

  const ratio = spend.cap > 0 ? spend.today / spend.cap : 0;
  chip.textContent = `$${spend.today.toFixed(2)} of $${spend.cap.toFixed(2)} today`;
  chip.className = "meter" + (ratio >= 1 ? " over" : ratio >= 0.8 ? " near" : "");
  chip.title = spend.note + " Resets at midnight UTC.";
}

async function refreshSpend() {
  try { drawSpend(await (await fetch("/api/spend")).json()); } catch { /* non-critical */ }
}

// ---------------------------------------------------------------- misc

const VERDICT_WORD = {
  pass: ["checked", "ok"], warn: ["worth a look", "mid"],
  fail: ["drifted", "bad"], info: ["made up", ""],
};

async function loadJobs() {
  const { jobs } = await (await fetch("/api/jobs")).json();
  const box = $("jobs");

  if (!jobs.length) {
    box.innerHTML = `<p class="nothing">Nothing yet</p>`;
    return;
  }

  box.innerHTML = jobs.map(j => {
    const thumb = j.thumb
      ? `<img class="run-thumb" src="/api/jobs/${j.id}/file/${j.thumb}" alt="">`
      : `<span class="run-thumb blank">${j.stage === "failed" ? "—" : "…"}</span>`;

    const bits = [];
    if (j.created) bits.push(esc(j.created));
    if (j.stage === "failed") bits.push(`<span class="bad">failed</span>`);
    else if (j.exported) bits.push(`${j.exported} exported`);
    else bits.push(esc(j.stage));

    const v = VERDICT_WORD[j.verdict];
    if (v && j.stage !== "failed") bits.push(`<span class="${v[1]}">${v[0]}</span>`);

    return `<button class="run ${j.id === jobId ? "current" : ""}" data-id="${j.id}">
      ${thumb}
      <span class="run-text">
        <span class="run-name">${esc(j.product || "Untitled run")}</span>
        <span class="run-meta">${bits.join(" · ")}</span>
      </span>
    </button>`;
  }).join("");

  document.querySelectorAll(".run").forEach(b => {
    b.onclick = () => beginWatching(b.dataset.id);
  });
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}


// ---------------------------------------------------------------- working on an image

function drawWork(state) {
  $("work").hidden = false;

  const versions = state.versions || [];
  $("versions").innerHTML = versions.map(v => `
    <li>
      <span class="step">${v.step}</span>
      <span class="what">${esc(v.action === "upscale" ? "Enlarged " + v.instruction : v.instruction)}</span>
      <span class="dim">${esc(v.size)}</span>
    </li>`).join("");

  const latest = versions[versions.length - 1];
  $("worksize").textContent = latest
    ? `Now ${latest.size}`
    : "Each change stacks on the last";
}

async function applyWork(action) {
  if (!jobId) return;
  const box = $("workbox");

  if (action === "edit" && !box.value.trim()) {
    box.focus();
    return;
  }

  const body = new FormData();
  body.append("instruction", box.value.trim());
  body.append("action", action);
  body.append("scale", "2");

  $("doedit").disabled = true;
  $("doupscale").disabled = true;
  await fetch(`/api/jobs/${jobId}/enhance`, { method: "POST", body });

  if (action === "edit") box.value = "";
  setTimeout(() => {
    $("doedit").disabled = false;
    $("doupscale").disabled = false;
  }, 1500);
  watch();
}

async function repeatRun() {
  if (!jobId) return;
  $("dorerun").disabled = true;
  const res = await fetch(`/api/jobs/${jobId}/rerun`, { method: "POST" });
  $("dorerun").disabled = false;
  if (!res.ok) return;
  beginWatching((await res.json()).job_id);
}
