const $ = (id) => document.getElementById(id);

const STAGES = ["ingesting", "ingested", "analysing", "briefed", "drafting",
                "drafted", "finalising", "finalised", "exporting", "exported"];

let jobId = null;
let picked = null;
let poll = null;
let config = null;
let pending = [];

// ---------------------------------------------------------------- boot

init();

async function init() {
  config = await (await fetch("/api/status")).json();

  const chip = $("mode");
  chip.textContent = config.mode === "live"
    ? `live · ${config.draft_model} / ${config.final_model}`
    : "mock mode · no API calls";
  chip.className = "chip " + config.mode;

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

function showFiles() {
  pending = [...$("files").files];
  const list = $("filelist");
  list.innerHTML = pending.map(f => {
    const ext = f.name.split(".").pop().toLowerCase();
    const kind = ["stl","obj","ply","glb","gltf","off","3mf","step","stp","iges","igs"].includes(ext)
      ? "CAD" : "photo";
    return `<li><span>${esc(f.name)}</span><span class="kind">${kind}</span></li>`;
  }).join("");
  $("start").disabled = pending.length === 0;
}

async function startRun() {
  const body = new FormData();
  pending.forEach(f => body.append("files", f));
  body.append("note", $("note").value);

  $("start").disabled = true;
  $("start").textContent = "Starting…";

  const res = await fetch("/api/jobs", { method: "POST", body });
  if (!res.ok) {
    $("start").textContent = "Start run";
    $("start").disabled = false;
    return alert("Upload failed: " + await res.text());
  }

  jobId = (await res.json()).job_id;
  $("start").textContent = "Start run";
  $("logblock").hidden = false;
  $("placeholder").hidden = true;
  enableChat();
  watch();
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

  drawStages(state.stage);
  drawLog(state.log || []);

  if (state.renders && Object.keys(state.renders).length) drawRenders(state);
  if (state.mask_source === "needs_paint" && state.base_image) setupPaint(state);
  if (state.brief) drawBrief(state.brief);
  if (state.drafts) drawDrafts(state);
  if (state.qa) drawQA(state);
  if (state.final_file) drawProof(state);
  if (state.exports) drawExports(state);
  if (state.chat && !sending) drawChat(state.chat);

  if (["drafted", "finalised", "exported", "failed"].includes(state.stage)) {
    clearInterval(poll);
    loadJobs();
  }
}

function drawStages(current) {
  const idx = STAGES.indexOf(current);
  $("stages").innerHTML = ["ingest", "brief", "draft", "final", "export"].map((name, i) => {
    const bounds = [[0,1],[2,3],[4,5],[6,7],[8,9]][i];
    let cls = "";
    if (current === "failed") cls = idx >= bounds[0] ? "" : "done";
    else if (idx > bounds[1]) cls = "done";
    else if (idx >= bounds[0]) cls = "active";
    return `<span class="${cls}">${name}</span>`;
  }).join("") + (current === "failed" ? `<span class="failed">failed</span>` : "");
}

function drawLog(entries) {
  $("log").innerHTML = entries.map(e =>
    `<li class="${e.level}">${esc(e.message)}</li>`).join("");
  const log = $("log");
  log.scrollTop = log.scrollHeight;
}

// ---------------------------------------------------------------- panels

function drawRenders(state) {
  $("renders").hidden = false;
  $("render-row").innerHTML = Object.entries(state.renders).map(([name, file]) =>
    `<img src="/api/jobs/${jobId}/file/${file}" alt="${esc(name)} render" title="${esc(name)}">`
  ).join("");
}

function drawBrief(brief) {
  $("briefblock").hidden = false;
  $("brief").innerHTML = `
    <dl>
      <dt>Product</dt><dd>${esc(brief.product_name || "—")}</dd>
      <dt>Category</dt><dd>${esc(brief.category || "—")}</dd>
      <dt>Sampled colours</dt>
      <dd><div class="swatches">${(brief.dominant_colours || []).map(c =>
        `<span class="swatch" style="background:${esc(c.hex)}" title="${esc(c.name)} ${esc(c.hex)}"></span>`
      ).join("")}</div></dd>
      <dt>Surface</dt><dd>${esc(brief.surface_pattern || "—")}</dd>
      <dt>Must survive editing</dt>
      <dd><div class="tags">${(brief.must_preserve || []).map(m =>
        `<span>${esc(m)}</span>`).join("")}</div></dd>
      ${(brief.risks || []).length ? `<dt>Known risks</dt><dd>${
        brief.risks.map(r => esc(r)).join("<br>")}</dd>` : ""}
    </dl>`;
}

function drawDrafts(state) {
  $("sheet").hidden = false;
  $("sheet-row").innerHTML = state.drafts.map(d => `
    <figure class="card ${picked === d.index ? "selected" : ""}" data-i="${d.index}">
      <img src="/api/jobs/${jobId}/file/${d.file}" alt="${esc(d.label)}">
      <figcaption>${esc(d.label)}</figcaption>
    </figure>`).join("");

  [...document.querySelectorAll(".card")].forEach(card => {
    card.onclick = () => {
      picked = Number(card.dataset.i);
      document.querySelectorAll(".card").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      $("makefinal").disabled = false;
    };
  });
}

function drawQA(state) {
  const { stats, level, notes, review } = state.qa;
  $("qablock").hidden = false;

  const label = { pass: "within tolerance", warn: "marginal", fail: "out of tolerance" }[level];
  const tol = config.tolerances;

  $("qa").innerHTML = `
    <span class="verdict ${level}">${label}</span>
    <table class="metrics">
      <tr><td>dE2000 mean</td><td>${stats.delta_e_mean}</td></tr>
      <tr><td>dE2000 p95</td><td>${stats.delta_e_p95}</td></tr>
      <tr><td>dE2000 max</td><td>${stats.delta_e_max}</td></tr>
      <tr><td>SSIM, product</td><td>${stats.ssim_product}</td></tr>
      <tr><td>product area</td><td>${(stats.product_coverage * 100).toFixed(1)}%</td></tr>
    </table>
    <ul class="qa-notes">
      ${notes.map(n => `<li>${esc(n)}</li>`).join("")}
      ${(review?.issues || []).map(i => `<li>${esc(i)}</li>`).join("")}
    </ul>
    <p class="qa-notes" style="padding-left:0">Thresholds: pass under ${tol.delta_e_pass} dE,
    fail over ${tol.delta_e_warn} dE, SSIM floor ${tol.ssim_pass}.</p>`;
}

function drawProof(state) {
  $("proof").hidden = false;
  $("proof-before").src = `/api/jobs/${jobId}/file/${state.base_image}`;
  $("proof-after").src = `/api/jobs/${jobId}/file/${state.final_file}?t=${Date.now()}`;
  $("doexport").disabled = false;
}

function renderPresets() {
  $("presets").innerHTML = Object.entries(config.presets).map(([key, p]) => `
    <li><label>
      <input type="checkbox" value="${key}" ${p.generative ? "checked" : ""}>
      <span>${esc(p.label)}
        <span class="dims">${p.size} · ${p.format}</span>
        ${p.generative ? "" : `<span class="flag">Photograph required — generated imagery may be rejected here.</span>`}
      </span>
    </label></li>`).join("");
}

function drawExports(state) {
  $("exports").innerHTML = state.exports.map(e => `
    <div class="row">
      <a href="/api/jobs/${jobId}/file/${e.file}" download>${esc(e.label)}</a>
      <span class="meta">${e.size} · ${e.scale}</span>
    </div>`).join("") + (state.zip_file
      ? `<div class="row"><a href="/api/jobs/${jobId}/file/${state.zip_file}" download>All files (zip)</a></div>`
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

  const pos = (ev) => {
    const r = canvas.getBoundingClientRect();
    return [(ev.clientX - r.left) * canvas.width / r.width,
            (ev.clientY - r.top) * canvas.height / r.height];
  };

  const stroke = (ev) => {
    if (!painting) return;
    const [x, y] = pos(ev);
    const size = Number($("brush").value) * canvas.width / canvas.getBoundingClientRect().width;
    ctx.fillStyle = "rgba(23,162,201,0.55)";
    ctx.beginPath();
    ctx.arc(x, y, size / 2, 0, Math.PI * 2);
    ctx.fill();
  };

  canvas.addEventListener("pointerdown", ev => { painting = true; canvas.setPointerCapture(ev.pointerId); stroke(ev); });
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
  let painted = false;
  for (let i = 3; i < data.length; i += 4) { if (data[i] > 0) { painted = true; break; } }
  if (!painted) return Promise.resolve(null);

  return new Promise(resolve => canvas.toBlob(resolve, "image/png"));
}

// ---------------------------------------------------------------- misc

async function loadJobs() {
  const { jobs } = await (await fetch("/api/jobs")).json();
  $("jobs").innerHTML = jobs.length
    ? jobs.map(j => `<li><a href="#" data-id="${j.id}">${esc(j.product || j.id)}</a> · ${esc(j.stage)}</li>`).join("")
    : `<li class="empty">Nothing yet.</li>`;

  [...document.querySelectorAll("#jobs a")].forEach(a => {
    a.onclick = (ev) => {
      ev.preventDefault();
      jobId = a.dataset.id;
      picked = null;
      $("logblock").hidden = false;
      $("placeholder").hidden = true;
      enableChat();
      watch();
    };
  });
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}


// ---------------------------------------------------------------- chat

let sending = false;

function wireChat() {
  const box = $("chatbox");
  $("send").onclick = sendMessage;
  box.addEventListener("keydown", ev => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); sendMessage(); }
  });
}

function enableChat() {
  $("chatbox").disabled = false;
  $("send").disabled = false;
}

function drawChat(history) {
  const chat = $("chat");
  chat.innerHTML = history.map(m =>
    `<div class="msg ${m.role === "user" ? "you" : "forge"}">${esc(m.content)}</div>`
  ).join("");
  chat.scrollTop = chat.scrollHeight;
}

async function sendMessage() {
  const box = $("chatbox");
  const text = box.value.trim();
  if (!text || !jobId || sending) return;

  sending = true;
  box.value = "";
  $("send").disabled = true;

  const chat = $("chat");
  const empty = chat.querySelector(".chat-empty");
  if (empty) empty.remove();

  chat.insertAdjacentHTML("beforeend", `<div class="msg you">${esc(text)}</div>`);
  chat.insertAdjacentHTML("beforeend", `<div class="msg forge thinking" id="pending">working…</div>`);
  chat.scrollTop = chat.scrollHeight;

  const body = new FormData();
  body.append("message", text);

  try {
    const res = await fetch(`/api/jobs/${jobId}/chat`, { method: "POST", body });
    const data = await res.json();
    const pending = $("pending");
    pending.className = "msg forge" + (data.actions?.length ? " acted" : "");
    pending.id = "";
    pending.textContent = data.reply;
  } catch (err) {
    const pending = $("pending");
    if (pending) { pending.className = "msg forge"; pending.id = ""; pending.textContent = "Request failed: " + err; }
  }

  chat.scrollTop = chat.scrollHeight;
  sending = false;
  $("send").disabled = false;
  box.focus();
  watch();
}


// ---------------------------------------------------------------- spend

function drawSpend(spend) {
  if (!spend || !spend.enforced) return;
  const chip = $("spend");
  chip.hidden = false;

  const used = spend.today, cap = spend.cap;
  const ratio = cap > 0 ? used / cap : 0;
  chip.textContent = `$${used.toFixed(2)} / $${cap.toFixed(2)} today`;
  chip.className = "chip" + (ratio >= 1 ? " over" : ratio >= 0.8 ? " near" : "");
  chip.title = spend.note + " Resets 00:00 UTC.";
}

async function refreshSpend() {
  try {
    drawSpend(await (await fetch("/api/spend")).json());
  } catch { /* the page is still usable without it */ }
}
