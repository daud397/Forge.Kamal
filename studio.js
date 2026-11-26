/* Design studio: textile CAD inside Forge.
 *
 * Everything here runs in the browser - no model call, no spend. A motif goes
 * in (scanned artwork, a photo of a print, or an image from a run); out comes
 * a seamless repeat in a chosen layout, separated into a fixed number of
 * colours that can be recoloured into colourways, and either downloaded or
 * attached straight to the chat so Forge can put it on a bed.
 *
 * Shares app.js's globals: $, attached, renderAttached.
 */

const ST = {
  img: null,          // full motif as loaded (canvas)
  crop: null,         // {x,y,w,h} in motif pixels, or null
  cropping: false,
  drag: null,
  repeat: "straight",
  scale: 1,
  gap: 0,
  rot: 0,
  ground: "#f4efe6",
  k: 6,
  palette: null,      // [[r,g,b], ...] after separation
  original: null,     // palette as separated, for comparison
  index: null,        // Uint8Array: palette index per motif pixel (255 = transparent)
  colourways: [],
  mode: "repeat",
  zoom: 0.6,
  outline: false,
  tile: null,         // last built repeat tile (canvas)
};

const WORK_EDGE = 1500;     // motif working size; keeps separation instant

function stWire() {
  $("studiobtn") && ($("studiobtn").onclick = () => openStudio());
  $("stclose").onclick = () => { $("studio").hidden = true; };
  $("stopen").onclick = $("stopen2").onclick = () => $("stfile").click();
  $("stfile").onchange = () => {
    const f = $("stfile").files[0];
    if (f) loadMotif(URL.createObjectURL(f), f.name);
    $("stfile").value = "";
  };

  $("stcrop").onclick = () => {
    ST.cropping = !ST.cropping;
    ST.mode = "motif"; segSet("stmode", "motif");
    $("stcrop").classList.toggle("on", ST.cropping);
    $("sthint").textContent = ST.cropping
      ? "Drag a box around the part of the image that is one motif."
      : "Build the repeat, then separate colours for colourways.";
    draw();
  };
  $("stuncrop").onclick = () => { ST.crop = null; resetColours(); rebuild(); };

  seg("strepeat", v => { ST.repeat = v; rebuild(); });
  seg("strot", v => { ST.rot = Number(v); rebuild(); });
  seg("stmode", v => { ST.mode = v; ST.cropping = false; $("stcrop").classList.remove("on"); draw(); });

  $("stscale").oninput = () => {
    ST.scale = $("stscale").value / 100;
    $("stscalev").textContent = $("stscale").value + "%";
    rebuild();
  };
  $("stgap").oninput = () => {
    ST.gap = $("stgap").value / 100;
    $("stgapv").textContent = $("stgap").value + "%";
    rebuild();
  };
  $("stground").oninput = () => { ST.ground = $("stground").value; rebuild(); };
  $("stzoom").oninput = () => { ST.zoom = $("stzoom").value / 100; draw(); };
  $("stoutline").onchange = () => { ST.outline = $("stoutline").checked; draw(); };
  $("stdpi").onchange = sizeNote;

  $("stk").oninput = () => { $("stkv").textContent = $("stk").value; };
  $("stseparate").onclick = separate;
  $("storig").onclick = () => { resetColours(); rebuild(); };
  $("stsavecw").onclick = saveColourway;

  $("sttile").onclick = () => download(ST.tile, "repeat-tile.png");
  $("stsheet").onclick = () => download(sheet(3000), "fabric-sheet.png");
  $("stsend").onclick = sendToForge;

  const cv = $("stcanvas");
  cv.addEventListener("pointerdown", cropStart);
  cv.addEventListener("pointermove", cropMove);
  cv.addEventListener("pointerup", cropEnd);
  window.addEventListener("resize", () => { if (!$("studio").hidden) draw(); });
}

function seg(id, fn) {
  $(id).querySelectorAll("button").forEach(b => {
    b.onclick = () => { segSet(id, b.dataset.v); fn(b.dataset.v); };
  });
}
function segSet(id, v) {
  $(id).querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.v === v));
}

function openStudio(src, name) {
  $("studio").hidden = false;
  if (src) loadMotif(src, name);
  else draw();
}

function loadMotif(src, name) {
  const im = new Image();
  im.onload = () => {
    const k = Math.min(1, WORK_EDGE / Math.max(im.naturalWidth, im.naturalHeight));
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(im.naturalWidth * k));
    c.height = Math.max(1, Math.round(im.naturalHeight * k));
    c.getContext("2d").drawImage(im, 0, 0, c.width, c.height);
    ST.img = c;
    ST.crop = null;
    ST.colourways = [];
    resetColours();
    $("stsrcinfo").textContent = `${name || "Image"} · ${im.naturalWidth}×${im.naturalHeight}px`;
    ["stcrop", "stseparate", "sttile", "stsheet", "stsend"].forEach(id => $(id).disabled = false);
    $("stempty").hidden = true;
    $("sthint").textContent = "Crop to one motif if needed, pick a repeat, then separate colours for colourways.";
    renderColourways();
    rebuild();
  };
  im.onerror = () => { $("stsrcinfo").textContent = "That file could not be opened as an image."; };
  im.crossOrigin = "anonymous";
  im.src = src;
}

// ------------------------------------------------------------------ motif

function cropped() {
  if (!ST.crop) return ST.img;
  const { x, y, w, h } = ST.crop;
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").drawImage(ST.img, x, y, w, h, 0, 0, w, h);
  return c;
}

// The motif as it will be repeated: cropped, recoloured if separated.
function motif() {
  const base = cropped();
  if (!ST.palette || !ST.index) return base;
  const c = document.createElement("canvas");
  c.width = base.width; c.height = base.height;
  const ctx = c.getContext("2d");
  const data = ctx.createImageData(c.width, c.height);
  const d = data.data, idx = ST.index;
  for (let i = 0, p = 0; i < idx.length; i++, p += 4) {
    const k = idx[i];
    if (k === 255) { d[p + 3] = 0; continue; }
    const col = ST.palette[k];
    d[p] = col[0]; d[p + 1] = col[1]; d[p + 2] = col[2]; d[p + 3] = 255;
  }
  ctx.putImageData(data, 0, 0);
  return c;
}

// One repeat cell: motif scaled, rotated, with spacing on the ground colour.
function cell() {
  const m = motif();
  const rot = ST.rot % 180 !== 0;
  const mw = Math.max(1, Math.round((rot ? m.height : m.width) * ST.scale));
  const mh = Math.max(1, Math.round((rot ? m.width : m.height) * ST.scale));
  const gw = Math.round(mw * ST.gap), gh = Math.round(mh * ST.gap);
  const c = document.createElement("canvas");
  c.width = mw + gw; c.height = mh + gh;
  const ctx = c.getContext("2d");
  ctx.fillStyle = ST.ground;
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.imageSmoothingQuality = "high";
  ctx.translate(c.width / 2, c.height / 2);
  ctx.rotate(ST.rot * Math.PI / 180);
  const dw = Math.round(m.width * ST.scale), dh = Math.round(m.height * ST.scale);
  ctx.drawImage(m, -dw / 2, -dh / 2, dw, dh);
  return c;
}

// The smallest rectangle that tiles seamlessly in the chosen layout.
function buildTile() {
  const c0 = cell();
  const w = c0.width, h = c0.height;
  const t = document.createElement("canvas");
  const ctx = t.getContext("2d");

  if (ST.repeat === "halfdrop") {
    t.width = w * 2; t.height = h;
    ctx.drawImage(c0, 0, 0);
    ctx.drawImage(c0, w, h / 2);
    ctx.drawImage(c0, w, -h / 2);
  } else if (ST.repeat === "brick") {
    t.width = w; t.height = h * 2;
    ctx.drawImage(c0, 0, 0);
    ctx.drawImage(c0, w / 2, h);
    ctx.drawImage(c0, -w / 2, h);
  } else if (ST.repeat === "mirror") {
    t.width = w * 2; t.height = h * 2;
    const flip = (sx, sy, x, y) => {
      ctx.save(); ctx.translate(x + (sx < 0 ? w : 0), y + (sy < 0 ? h : 0));
      ctx.scale(sx, sy); ctx.drawImage(c0, 0, 0); ctx.restore();
    };
    flip(1, 1, 0, 0); flip(-1, 1, w, 0); flip(1, -1, 0, h); flip(-1, -1, w, h);
  } else {
    t.width = w; t.height = h;
    ctx.drawImage(c0, 0, 0);
  }
  return t;
}

function rebuild() {
  if (!ST.img) return;
  ST.tile = buildTile();
  $("stuncrop").hidden = !ST.crop;
  sizeNote();
  draw();
}

function sizeNote() {
  if (!ST.tile) return;
  const dpi = Number($("stdpi").value);
  const cm = px => (px / dpi * 2.54).toFixed(1);
  $("stsize").textContent =
    `Repeat tile ${ST.tile.width}×${ST.tile.height}px = ${cm(ST.tile.width)}×${cm(ST.tile.height)} cm at ${dpi} dpi.`;
}

// ------------------------------------------------------------------ drawing

function draw() {
  const wrap = $("stwrap"), cv = $("stcanvas");
  const W = wrap.clientWidth, H = wrap.clientHeight;
  if (!W || !H) return;
  const dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  cv.style.width = W + "px"; cv.style.height = H + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#7e7e7e";
  ctx.fillRect(0, 0, W, H);
  if (!ST.img) return;

  if (ST.mode === "motif") return drawMotif(ctx, W, H);

  const z = ST.zoom;
  const pat = ctx.createPattern(ST.tile, "repeat");
  ctx.save();
  ctx.scale(z, z);
  ctx.fillStyle = pat;
  ctx.fillRect(0, 0, W / z, H / z);
  if (ST.outline) {
    ctx.strokeStyle = "rgba(229,57,53,.9)";
    ctx.lineWidth = 2 / z;
    ctx.setLineDash([8 / z, 6 / z]);
    ctx.strokeRect(ST.tile.width, ST.tile.height, ST.tile.width, ST.tile.height);
  }
  ctx.restore();

  if (ST.mode === "fabric") drape(ctx, W, H);
}

// Soft folds and a little falloff, so a print can be judged as cloth rather
// than as a flat swatch. Light only - it never changes the design's colours
// in the files that get downloaded or sent.
function drape(ctx, W, H) {
  ctx.save();
  ctx.globalCompositeOperation = "multiply";
  const g = ctx.createLinearGradient(0, 0, W, 0);
  const folds = 5;
  for (let i = 0; i <= 40; i++) {
    const t = i / 40;
    const v = 0.5 + 0.5 * Math.sin(t * Math.PI * 2 * folds + Math.sin(t * 7) * 0.8);
    const s = Math.round(255 - 70 * Math.pow(1 - v, 2));
    g.addColorStop(t, `rgb(${s},${s},${s})`);
  }
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);
  const r = ctx.createRadialGradient(W * 0.4, H * 0.3, Math.min(W, H) * 0.2,
                                     W * 0.5, H * 0.5, Math.max(W, H) * 0.8);
  r.addColorStop(0, "rgb(255,255,255)");
  r.addColorStop(1, "rgb(175,175,175)");
  ctx.fillStyle = r;
  ctx.fillRect(0, 0, W, H);
  ctx.globalCompositeOperation = "screen";
  const hl = ctx.createLinearGradient(0, 0, W, 0);
  for (let i = 0; i <= 40; i++) {
    const t = i / 40;
    const v = Math.max(0, Math.sin(t * Math.PI * 2 * 5 + Math.sin(t * 7) * 0.8 - 0.6));
    hl.addColorStop(t, `rgba(255,255,255,${(v * 0.14).toFixed(3)})`);
  }
  ctx.fillStyle = hl;
  ctx.fillRect(0, 0, W, H);
  ctx.restore();
}

let motifBox = null;   // where the motif sits on screen in motif mode

function drawMotif(ctx, W, H) {
  const im = ST.img;
  const k = Math.min((W - 40) / im.width, (H - 40) / im.height);
  const w = im.width * k, h = im.height * k;
  const x = (W - w) / 2, y = (H - h) / 2;
  motifBox = { x, y, k };
  ctx.drawImage(im, x, y, w, h);
  const c = ST.drag || ST.crop;
  if (c) {
    ctx.fillStyle = "rgba(0,0,0,.45)";
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.rect(x + c.x * k, y + c.y * k, c.w * k, c.h * k);
    ctx.fill("evenodd");
    ctx.strokeStyle = "#5b7fb9"; ctx.lineWidth = 2; ctx.setLineDash([]);
    ctx.strokeRect(x + c.x * k, y + c.y * k, c.w * k, c.h * k);
  }
}

// ------------------------------------------------------------------ crop

function toMotif(ev) {
  const r = $("stcanvas").getBoundingClientRect();
  const { x, y, k } = motifBox;
  return [
    Math.max(0, Math.min(ST.img.width, (ev.clientX - r.left - x) / k)),
    Math.max(0, Math.min(ST.img.height, (ev.clientY - r.top - y) / k)),
  ];
}
function cropStart(ev) {
  if (!ST.cropping || !motifBox) return;
  $("stcanvas").setPointerCapture(ev.pointerId);
  const [x, y] = toMotif(ev);
  ST.drag = { x0: x, y0: y, x, y, w: 0, h: 0 };
}
function cropMove(ev) {
  if (!ST.drag) return;
  const [x, y] = toMotif(ev);
  const d = ST.drag;
  d.x = Math.min(d.x0, x); d.y = Math.min(d.y0, y);
  d.w = Math.abs(x - d.x0); d.h = Math.abs(y - d.y0);
  draw();
}
function cropEnd() {
  const d = ST.drag;
  ST.drag = null;
  if (!d) return;
  if (d.w > 8 && d.h > 8) {
    ST.crop = { x: Math.round(d.x), y: Math.round(d.y), w: Math.round(d.w), h: Math.round(d.h) };
    resetColours();
  }
  ST.cropping = false;
  $("stcrop").classList.remove("on");
  ST.mode = "repeat"; segSet("stmode", "repeat");
  $("sthint").textContent = "Build the repeat, then separate colours for colourways.";
  rebuild();
}

// ------------------------------------------------------------------ colours

function resetColours() {
  ST.palette = ST.original = ST.index = null;
  $("stpalette").innerHTML = "";
  $("stpalnote").hidden = true;
  $("stcwrow").hidden = true;
  $("storig").hidden = true;
}

// Colour separation: k-means on a sample of the motif, then every pixel goes
// to its nearest colour. This is what a print room calls "reducing to N
// screens" - each colour becomes one recolourable layer.
function separate() {
  const src = cropped();
  const ctx = src.getContext("2d");
  const px = ctx.getImageData(0, 0, src.width, src.height).data;
  const n = src.width * src.height;
  const k = Number($("stk").value);

  const step = Math.max(1, Math.floor(n / 40000));
  const sample = [];
  for (let i = 0; i < n; i += step) {
    if (px[i * 4 + 3] < 128) continue;
    sample.push([px[i * 4], px[i * 4 + 1], px[i * 4 + 2]]);
  }
  if (!sample.length) return;

  // k-means++ seeding, then a dozen rounds of refinement.
  const cents = [sample[Math.floor(Math.random() * sample.length)].slice()];
  const dist = new Float64Array(sample.length);
  while (cents.length < k) {
    let sum = 0;
    for (let i = 0; i < sample.length; i++) {
      let best = Infinity;
      for (const c of cents) best = Math.min(best, d2(sample[i], c));
      dist[i] = best; sum += best;
    }
    if (sum === 0) break;
    let r = Math.random() * sum, j = 0;
    while (r > dist[j] && j < sample.length - 1) { r -= dist[j]; j++; }
    cents.push(sample[j].slice());
  }
  for (let it = 0; it < 12; it++) {
    const acc = cents.map(() => [0, 0, 0, 0]);
    for (const s of sample) {
      const a = acc[nearest(s, cents)];
      a[0] += s[0]; a[1] += s[1]; a[2] += s[2]; a[3]++;
    }
    acc.forEach((a, i) => { if (a[3]) cents[i] = [a[0] / a[3], a[1] / a[3], a[2] / a[3]]; });
  }

  // Largest area first, so the ground is swatch one.
  const counts = new Array(cents.length).fill(0);
  const index = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (px[i * 4 + 3] < 128) { index[i] = 255; continue; }
    const c = nearest([px[i * 4], px[i * 4 + 1], px[i * 4 + 2]], cents);
    index[i] = c; counts[c]++;
  }
  const order = cents.map((_, i) => i).sort((a, b) => counts[b] - counts[a]);
  const remap = new Uint8Array(256).fill(255);
  order.forEach((old, i) => { remap[old] = i; });
  for (let i = 0; i < n; i++) index[i] = remap[index[i]];

  ST.palette = order.map(i => cents[i].map(Math.round));
  ST.original = ST.palette.map(c => c.slice());
  ST.index = index;
  ST.shares = order.map(i => counts[i] / n);
  drawPalette();
  $("storig").hidden = false;
  $("stpalnote").hidden = false;
  $("stcwrow").hidden = false;
  rebuild();
}

function d2(a, b) {
  const r = (a[0] + b[0]) / 2;   // weighted RGB: closer to how the eye groups colour
  const dr = a[0] - b[0], dg = a[1] - b[1], db = a[2] - b[2];
  return (2 + r / 256) * dr * dr + 4 * dg * dg + (2 + (255 - r) / 256) * db * db;
}
function nearest(p, cents) {
  let best = 0, bd = Infinity;
  for (let i = 0; i < cents.length; i++) {
    const d = d2(p, cents[i]);
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function hex(c) {
  return "#" + c.map(v => v.toString(16).padStart(2, "0")).join("");
}

function drawPalette() {
  const box = $("stpalette");
  box.innerHTML = "";
  ST.palette.forEach((c, i) => {
    const lab = document.createElement("label");
    lab.className = "st-swatch";
    lab.title = `Colour ${i + 1} · ${Math.round((ST.shares[i] || 0) * 100)}% of the motif · click to change`;
    lab.style.background = hex(c);
    const inp = document.createElement("input");
    inp.type = "color";
    inp.value = hex(c);
    inp.oninput = () => {
      const v = inp.value;
      ST.palette[i] = [1, 3, 5].map(o => parseInt(v.slice(o, o + 2), 16));
      lab.style.background = v;
      tag.textContent = v.toUpperCase();
      rebuild();
    };
    const tag = document.createElement("span");
    tag.textContent = hex(c).toUpperCase();
    lab.appendChild(inp);
    lab.appendChild(tag);
    box.appendChild(lab);
  });
}

function saveColourway() {
  if (!ST.palette) return;
  ST.colourways.push({ palette: ST.palette.map(c => c.slice()) });
  renderColourways();
}

function renderColourways() {
  const box = $("stcolourways");
  box.innerHTML = "";
  ST.colourways.forEach((cw, i) => {
    const b = document.createElement("button");
    b.className = "st-cw";
    b.title = "Switch to this colourway";
    b.innerHTML = `<span class="st-cw-name">${String.fromCharCode(65 + i)}</span>` +
      cw.palette.slice(0, 8).map(c => `<i style="background:${hex(c)}"></i>`).join("");
    b.onclick = () => {
      ST.palette = cw.palette.map(c => c.slice());
      drawPalette();
      rebuild();
    };
    box.appendChild(b);
  });
}

// ------------------------------------------------------------------ output

// A square of fabric big enough for the image model to read the motif from.
function sheet(edge) {
  const c = document.createElement("canvas");
  c.width = c.height = edge;
  const ctx = c.getContext("2d");
  // At least four repeats across, so the model sees it as a repeating print.
  const k = Math.min(1, edge / 4 / Math.max(ST.tile.width, ST.tile.height));
  ctx.scale(k, k);
  ctx.fillStyle = ctx.createPattern(ST.tile, "repeat");
  ctx.fillRect(0, 0, edge / k, edge / k);
  return c;
}

function download(canvas, name) {
  if (!canvas) return;
  canvas.toBlob(b => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }, "image/png");
}

function sendToForge() {
  const c = sheet(2048);
  c.toBlob(b => {
    const n = ST.colourways.length ? "colourway" : "design";
    const file = new File([b], `studio-${n}-${Date.now() % 100000}.png`, { type: "image/png" });
    attached.push(file);
    renderAttached();
    $("studio").hidden = true;
    const box = $("box");
    if (!box.value.trim()) {
      box.value = "Apply this print as the duvet cover and pillowcase design on a " +
        "dressed bed in a UK-style bedroom. Match the colours, motif, motif size " +
        "and repeat exactly. Full bed front view. 1:1 aspect ratio.";
      box.dispatchEvent(new Event("input"));
    }
    box.focus();
  }, "image/png");
}

stWire();
