/* avsim test platform -- browser client.
 *
 * The server simulates and scores; this file draws, edits and asks.  Three
 * pieces of state matter: `boot` (what the server offers), `params` (the user's
 * vehicle), and `score` (the user's reporting policy).  Everything else is
 * derived from a frame stream or a results list.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const api = {
  async get(path) {
    const r = await fetch(path);
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    return j;
  },
  async post(path, body) {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const j = await r.json();
    if (j.error) throw new Error(j.error);
    return j;
  },
};

let boot = null;         // /api/bootstrap
let params = {};         // current parameter values
let score = null;        // current ScoreConfig as a plain object
let scene = null;        // static geometry for the canvas
let runId = null;
let frameIndex = 0;      // how many frames already consumed
let timer = null;
let lastFrame = null;
let results = [];

/* ============================ bootstrap ============================ */

async function init() {
  boot = await api.get("/api/bootstrap");
  params = { ...boot.defaults };
  score = boot.score;
  $("contract").textContent = "contract " + boot.contract_version;

  buildPresets();
  buildControllers();
  buildParams();
  buildScorePanel();
  renderDerived(boot.derived);
  wire();
  drawEmpty();
  await refreshResults();
}

function buildPresets() {
  const sel = $("preset");
  sel.innerHTML = "";
  boot.presets.forEach((p) => {
    const o = document.createElement("option");
    o.value = p.key;
    o.textContent = `${p.name}`;
    sel.appendChild(o);
  });
  sel.onchange = applyPreset;
  applyPreset();
}

function currentPreset() {
  return boot.presets.find((p) => p.key === $("preset").value);
}

function applyPreset() {
  const p = currentPreset();
  $("preset-desc").innerHTML =
    `${p.description}<br><span style="color:var(--accent)">${p.focus}</span>` +
    ` · ${p.rows}×${p.cols} grid · ${p.n_lanes} lane(s)`;
  $("n_vehicles").value = p.n_vehicles;
  $("n_vehicles_slider").value = p.n_vehicles;
  $("duration").value = p.duration;
}

function buildControllers() {
  const sel = $("controller");
  const keep = sel.value;
  sel.innerHTML = "";
  boot.controllers.forEach((c) => {
    const o = document.createElement("option");
    o.value = c.value;
    o.textContent = c.label;
    o.dataset.kind = c.kind;
    sel.appendChild(o);
  });
  if (keep) sel.value = keep;
  sel.onchange = () => {
    const kind = sel.selectedOptions[0]?.dataset.kind;
    $("controller-desc").textContent =
      kind === "plugin" ? "외부 플러그인 — " + sel.value : "";
  };
  sel.onchange();
}

/* ============================ parameters ============================ */

function buildParams() {
  const host = $("params");
  host.innerHTML = "";
  boot.groups.forEach((g) => {
    const specs = boot.parameters.filter((s) => s.group === g.key);
    if (!specs.length) return;
    const box = document.createElement("div");
    box.className = "pgroup";
    box.innerHTML = `<h4>${g.label}</h4>`;
    specs.forEach((s) => box.appendChild(paramRow(s)));
    host.appendChild(box);
  });
}

function paramRow(spec) {
  const row = document.createElement("div");
  row.className = "param";
  row.dataset.key = spec.key;

  const top = document.createElement("div");
  top.className = "top";
  top.innerHTML = `<span class="name">${spec.label}</span>` +
                  `<span class="unit">${spec.unit || ""}</span>`;
  row.appendChild(top);

  const ctl = document.createElement("div");
  ctl.className = "ctl";

  if (spec.kind === "choice") {
    const sel = document.createElement("select");
    spec.choices.forEach((c) => {
      const o = document.createElement("option");
      o.value = String(c.value);
      o.textContent = c.label;
      sel.appendChild(o);
    });
    sel.value = String(spec.default);
    sel.oninput = () => {
      const raw = sel.value;
      const num = Number(raw);
      setParam(spec, Number.isNaN(num) || raw === "" ? raw : num, row);
    };
    ctl.appendChild(sel);
  } else if (spec.kind === "bool") {
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !!spec.default;
    cb.oninput = () => setParam(spec, cb.checked, row);
    ctl.appendChild(cb);
  } else {
    // A slider for feel, a number box for a value you actually know.  Both
    // edit the same parameter, which is the whole point of having both.
    const range = document.createElement("input");
    range.type = "range";
    range.min = spec.min; range.max = spec.max;
    range.step = spec.step || (spec.kind === "int" ? 1 : 0.001);
    range.value = spec.default;

    const num = document.createElement("input");
    num.type = "number";
    num.min = spec.min; num.max = spec.max;
    num.step = range.step;
    num.value = spec.default;

    range.oninput = () => { num.value = range.value; setParam(spec, Number(range.value), row); };
    num.oninput = () => { range.value = num.value; setParam(spec, Number(num.value), row); };
    ctl.appendChild(range);
    ctl.appendChild(num);
  }
  row.appendChild(ctl);
  if (spec.help) {
    const h = document.createElement("div");
    h.className = "help";
    h.textContent = spec.help;
    row.appendChild(h);
  }
  return row;
}

let derivedTimer = null;
function setParam(spec, value, row) {
  params[spec.key] = value;
  row.classList.toggle("changed", String(value) !== String(spec.default));
  clearTimeout(derivedTimer);
  derivedTimer = setTimeout(refreshDerived, 180);
}

async function refreshDerived() {
  try {
    renderDerived(await api.post("/api/derived", { parameters: params }));
  } catch (e) {
    $("derived").innerHTML = `<span class="err">${e.message}</span>`;
  }
}

const DERIVED_LABELS = {
  wheelbase: ["축거 L", "m", 3],
  understeer_gradient: ["언더스티어 K_us", "s²/m", 5],
  characteristic_speed: ["특성속도 v_ch", "m/s", 1],
  critical_speed: ["임계속도 v_cr", "m/s", 1],
  max_lateral_accel: ["횡가속 한계", "m/s²", 2],
  planning_lateral_accel: ["계획용 횡가속", "m/s²", 2],
  min_turn_radius: ["최소회전반경", "m", 2],
  static_load_front: ["전륜 정하중", "N", 0],
  static_load_rear: ["후륜 정하중", "N", 0],
};

function renderDerived(d) {
  const host = $("derived");
  host.innerHTML = "";
  Object.entries(DERIVED_LABELS).forEach(([k, [label, unit, digits]]) => {
    if (d[k] === undefined || d[k] === null) return;
    const cell = document.createElement("div");
    cell.innerHTML = `<span>${label}</span> <b>${Number(d[k]).toFixed(digits)} ${unit}</b>`;
    host.appendChild(cell);
  });
  const bal = document.createElement("div");
  bal.innerHTML = `<span>거동</span> <b style="color:${d.balance === "understeer" ? "var(--good)" : "var(--bad)"}">${d.balance}</b>`;
  host.appendChild(bal);
}

function resetParams() {
  params = { ...boot.defaults };
  buildParams();
  refreshDerived();
}

/* ============================ running ============================ */

function runPayload() {
  let options = {};
  const raw = $("controller-options").value.trim();
  if (raw) {
    try { options = JSON.parse(raw); }
    catch (e) { throw new Error("controller options is not valid JSON: " + e.message); }
  }
  return {
    preset: $("preset").value,
    controller: $("controller").value,
    controller_options: options,
    parameters: params,
    seed: Number($("seed").value || 0),
    n_vehicles: $("n_vehicles").value,
    duration: $("duration").value,
  };
}

async function startRun() {
  message("run-msg", "");
  try {
    const payload = runPayload();
    $("btn-run").disabled = true;
    const r = await api.post("/api/run", payload);
    runId = r.id;
    frameIndex = 0;
    scene = await api.get("/api/scene?id=" + runId);
    $("btn-stop").disabled = false;
    timer = setInterval(poll, 120);
    message("run-msg", "running…");
  } catch (e) {
    $("btn-run").disabled = false;
    message("run-msg", e.message, "err");
  }
}

async function poll() {
  if (!runId) return;
  let s;
  try {
    s = await api.get(`/api/stream?id=${runId}&from=${frameIndex}`);
  } catch (e) {
    return stopPolling(e.message, "err");
  }
  if (s.frames.length) {
    frameIndex += s.frames.length;
    lastFrame = s.frames[s.frames.length - 1];
    draw(lastFrame);
    renderReadouts(lastFrame);
  }
  if (s.status === "error") return stopPolling(s.error, "err");
  if (s.status === "finished" || s.status === "stopped") {
    if (s.result) {
      renderScore(s.result.score);
      message("run-msg",
        `${s.status} — ${s.result.finish_reason} · 종합 ${s.result.score.total.toFixed(1)}점`,
        s.result.score.collided ? "err" : "ok");
      await refreshResults();
    }
    stopPolling();
  }
}

function stopPolling(msg, cls) {
  clearInterval(timer);
  timer = null;
  runId = null;
  $("btn-run").disabled = false;
  $("btn-stop").disabled = true;
  if (msg) message("run-msg", msg, cls);
}

async function stopRun() {
  if (runId) await api.post("/api/stop", { id: runId });
}

/* ============================ canvas ============================ */

const cv = $("view");
const ctx = cv.getContext("2d");
let view = { scale: 1, ox: 0, oy: 0 };

/* Two views of the same world.  Whole-network is where the run is; ego-centred
 * is where the *driving* is -- at 760 m across, a 4.5 m car is four pixels and
 * a lane change is invisible. */
function fitBox(xmin, xmax, ymin, ymax) {
  const w = xmax - xmin, h = ymax - ymin;
  const scale = Math.min(cv.width / w, cv.height / h);
  view = {
    scale,
    ox: -xmin * scale + (cv.width - w * scale) / 2,
    oy: ymax * scale + (cv.height - h * scale) / 2,
  };
}

function fitView(frame) {
  if ($("follow").checked && frame) {
    const span = Number($("zoom").value);
    const half = span / 2;
    fitBox(frame.ego.x - half, frame.ego.x + half, frame.ego.y - half, frame.ego.y + half);
  } else {
    const b = scene.bounds;
    fitBox(b.xmin, b.xmax, b.ymin, b.ymax);
  }
}
const X = (x) => x * view.scale + view.ox;
const Y = (y) => -y * view.scale + view.oy;

function drawEmpty() {
  ctx.fillStyle = "#0c0f15";
  ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = "#4a5568";
  ctx.font = "13px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("실행을 누르면 시뮬레이션이 표시됩니다 — press Run", cv.width / 2, cv.height / 2);
  ctx.textAlign = "left";
}

function draw(frame) {
  if (!scene) return;
  fitView(frame);
  ctx.fillStyle = "#0c0f15";
  ctx.fillRect(0, 0, cv.width, cv.height);

  // --- lanes: stroke each centreline at its own width ---------------------
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  scene.lanes.forEach((lane) => {
    ctx.strokeStyle = lane.kind.startsWith("connector") ? "#242a38" : "#2a3040";
    ctx.lineWidth = Math.max(lane.width * view.scale, 1.5);
    ctx.beginPath();
    lane.pts.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
    ctx.stroke();
  });

  // --- intersection boxes and their signals ------------------------------
  scene.nodes.forEach((n) => {
    const h = n.half * view.scale;
    ctx.strokeStyle = "#333c4f";
    ctx.lineWidth = 1;
    ctx.strokeRect(X(n.x) - h, Y(n.y) - h, 2 * h, 2 * h);
    drawSignals(n, frame.signals);
  });

  // --- the ego's route ----------------------------------------------------
  ctx.strokeStyle = "rgba(90,169,255,.55)";
  ctx.lineWidth = 1.6;
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  scene.route.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
  ctx.stroke();
  ctx.setLineDash([]);

  // --- goal ---------------------------------------------------------------
  ctx.fillStyle = "#46c98b";
  ctx.beginPath();
  ctx.arc(X(scene.goal[0]), Y(scene.goal[1]), 5, 0, 6.284);
  ctx.fill();

  // --- traffic ------------------------------------------------------------
  frame.actors.forEach((a) => carBox(a, "#7d879b"));
  carBox(frame.ego, "#5aa9ff", true);
}

function carBox(a, colour, isEgo) {
  const c = Math.cos(a.psi), s = Math.sin(a.psi);
  const L = a.length / 2, W = a.width / 2;
  const pts = [[L, W], [L, -W], [-L, -W], [-L, W]].map(([u, v]) => [
    X(a.x + u * c - v * s), Y(a.y + u * s + v * c),
  ]);
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
  ctx.closePath();
  ctx.fillStyle = colour;
  ctx.fill();
  if (isEgo) {
    ctx.strokeStyle = "#dfe4ee";
    ctx.lineWidth = 1.2;
    ctx.stroke();
    // a nose mark, so heading is readable at any zoom
    ctx.beginPath();
    ctx.moveTo(X(a.x + L * c), Y(a.y + L * s));
    ctx.lineTo(X(a.x + (L + 2.2) * c), Y(a.y + (L + 2.2) * s));
    ctx.stroke();
  }
}

const SIGNAL_COLOUR = { green: "#46c98b", yellow: "#e8b23a", red: "#e2564b" };

function drawSignals(node, signals) {
  const h = node.half;
  const spots = {
    NS: [[0, h + 2], [0, -h - 2]],
    EW: [[h + 2, 0], [-h - 2, 0]],
  };
  Object.entries(spots).forEach(([phase, offs]) => {
    const state = signals[`${node.id}:${phase}`];
    if (!state) return;
    ctx.fillStyle = SIGNAL_COLOUR[state] || "#666";
    offs.forEach(([dx, dy]) => {
      ctx.beginPath();
      ctx.arc(X(node.x + dx), Y(node.y + dy), 3, 0, 6.284);
      ctx.fill();
    });
  });
}

/* ============================ readouts ============================ */

const READOUT_CELLS = [
  ["t", "time [s]", (f) => f.t.toFixed(1)],
  ["speed", "speed [m/s]", (f) => f.readout.speed.toFixed(1)],
  ["progress", "progress", (f) => (100 * f.readout.progress).toFixed(0) + "%"],
  ["e_y", "path error [m]", (f) => f.readout.e_y.toFixed(2)],
  ["a_y", "lat accel [m/s²]", (f) => f.readout.a_y.toFixed(2)],
  ["friction", "friction use", (f) => f.readout.friction.toFixed(2), (f) => f.readout.friction > 0.95],
  ["min_clearance", "min clear [m]", (f) => fmt(f.readout.min_clearance), (f) => f.readout.min_clearance < 1.0],
  ["min_ttc", "min TTC [s]", (f) => fmt(f.readout.min_ttc), (f) => f.readout.min_ttc < 1.5],
  ["tracks", "tracked cars", (f) => f.readout.tracks],
  ["signal", "signal", (f) => f.readout.signal + (f.readout.signal_distance != null ? ` ${f.readout.signal_distance.toFixed(0)}m` : "")],
  ["compute_ms", "compute [ms]", (f) => f.readout.compute_ms.toFixed(1), (f) => f.readout.compute_ms > 100],
];

const fmt = (v) => (v === null || v === undefined ? "–" : Number(v).toFixed(2));

function renderReadouts(f) {
  const host = $("readouts");
  if (host.children.length !== READOUT_CELLS.length) {
    host.innerHTML = "";
    READOUT_CELLS.forEach(([key, label]) => {
      const d = document.createElement("div");
      d.className = "cell";
      d.id = "ro-" + key;
      d.innerHTML = `<b>–</b><span>${label}</span>`;
      host.appendChild(d);
    });
  }
  READOUT_CELLS.forEach(([key, , get, alert]) => {
    const cell = $("ro-" + key);
    cell.firstChild.textContent = get(f);
    cell.classList.toggle("alert", alert ? !!alert(f) : false);
  });

  bar("steer", f.command.steer, true);
  bar("throttle", f.command.throttle, false);
  bar("brake", f.command.brake, false);

  const rows = Object.entries(f.diagnostics || {});
  $("diag").innerHTML = rows.length
    ? rows.map(([k, v]) => `<tr><td>${k}</td><td>${typeof v === "number" ? v.toFixed(3) : v}</td></tr>`).join("")
    : "<tr><td>—</td><td>제어기가 diagnostics()를 제공하지 않음</td></tr>";
}

function bar(name, value, signed) {
  const el = $("bar-" + name);
  $("val-" + name).textContent = value.toFixed(2);
  if (signed) {
    const w = Math.abs(value) * 50;
    el.style.width = w + "%";
    el.style.left = value >= 0 ? "50%" : 50 - w + "%";
  } else {
    el.style.width = Math.max(0, Math.min(1, value)) * 100 + "%";
  }
}

/* ============================ scoring ============================ */

function buildScorePanel() {
  $("collision-policy").value = score.collision_policy;
  $("collision-penalty").value = score.collision_penalty;
  $("ttc-threshold").value = score.ttc_threshold;

  const host = $("cat-weights");
  host.innerHTML = "";
  Object.entries(boot.categories).forEach(([key, label]) => {
    const l = document.createElement("label");
    l.innerHTML = `<span>${label}</span>`;
    const input = document.createElement("input");
    input.type = "number";
    input.step = "0.1";
    input.min = "0";
    input.value = score.category_weights[key] ?? 1;
    input.oninput = () => (score.category_weights[key] = Number(input.value));
    l.appendChild(input);
    host.appendChild(l);
  });

  renderKpiTable(null);
}

/* The KPI table is the platform's argument surface: every threshold and weight
 * is an input, because "how much is 1 m of clearance worth against 5 s of
 * mission time" has no answer the platform is entitled to fix. */
function renderKpiTable(breakdown) {
  const rows = breakdown ? breakdown.metrics : null;
  const byKey = {};
  (rows || []).forEach((r) => (byKey[r.key] = r));

  const table = $("kpi");
  table.innerHTML = "<tr><th>지표 metric</th><th class='num'>값</th><th class='num'>점수</th>" +
                    "<th class='num'>good</th><th class='num'>bad</th><th class='num'>weight</th></tr>";

  const cats = {};
  score.metrics.forEach((m) => (cats[m.category] = cats[m.category] || []).push(m));

  Object.entries(cats).forEach(([cat, metrics]) => {
    const head = table.insertRow();
    head.className = "catrow";
    head.insertCell().colSpan = 6;
    head.cells[0].textContent = boot.categories[cat] || cat;

    metrics.forEach((m) => {
      const r = byKey[m.key];
      const tr = table.insertRow();
      tr.insertCell().innerHTML = `<span title="${m.help || ""}">${m.label}</span>` +
                                  (m.unit ? ` <span class="unit">[${m.unit}]</span>` : "");
      const v = tr.insertCell();
      v.className = "num";
      v.textContent = r ? (r.value === null ? "∞" : Number(r.value).toFixed(2)) : "–";
      const s = tr.insertCell();
      s.className = "num";
      s.innerHTML = r ? pill(r.score) : "–";
      ["good", "bad", "weight"].forEach((field) => {
        const cell = tr.insertCell();
        cell.className = "num";
        const input = document.createElement("input");
        input.type = "number";
        input.step = "0.1";
        input.value = m[field];
        input.oninput = () => (m[field] = Number(input.value));
        cell.appendChild(input);
      });
    });
  });
}

function pill(v) {
  const cls = v >= 70 ? "hi" : v >= 40 ? "mid" : "lo";
  return `<span class="pill ${cls}">${v.toFixed(0)}</span>`;
}

function renderScore(breakdown) {
  $("score-total").textContent = breakdown.total.toFixed(1);
  $("score-total").style.color =
    breakdown.collided ? "var(--bad)" : breakdown.total >= 70 ? "var(--good)" : "var(--ink)";
  const host = $("score-cats");
  host.innerHTML = "";
  Object.entries(boot.categories).forEach(([key, label]) => {
    const v = breakdown.categories[key];
    if (v === undefined) return;
    const d = document.createElement("div");
    d.className = "cat";
    d.innerHTML = `<span>${label}</span><span class="track"><i style="width:${v}%;background:${
      v >= 70 ? "var(--good)" : v >= 40 ? "var(--warn)" : "var(--bad)"
    }"></i></span><b>${v.toFixed(0)}</b>`;
    host.appendChild(d);
  });
  $("score-notes").textContent = (breakdown.notes || []).join(" ");
  $("score-notes").className = "msg" + (breakdown.collided ? " err" : "");
  renderKpiTable(breakdown);
}

async function rescore() {
  score.collision_policy = $("collision-policy").value;
  score.collision_penalty = Number($("collision-penalty").value);
  score.ttc_threshold = Number($("ttc-threshold").value);
  const r = await api.post("/api/rescore", { score });
  score = r.score;
  results = r.results;
  renderResults();
  if (results.length) renderScore(results[results.length - 1].score);
  message("run-msg", `${results.length}개 결과를 새 가중치로 재채점했습니다.`, "ok");
}

/* ============================ results ============================ */

async function refreshResults() {
  const r = await api.get("/api/results");
  results = r.results;
  renderResults();
}

function renderResults() {
  const table = $("results");
  if (!results.length) {
    table.innerHTML = "<tr><td class='hint'>아직 결과가 없습니다.</td></tr>";
    return;
  }
  // The server ranks by total as well; sorting the same list locally keeps the
  // table right after a rescore without a second round trip.
  table.innerHTML = "<tr><th>#</th><th>실행 run</th><th class='num'>총점</th>" +
                    "<th class='num'>임무</th><th class='num'>안전</th><th class='num'>에너지</th><th>종료</th></tr>";
  const sorted = [...results].sort((a, b) => (b.score?.total ?? -1) - (a.score?.total ?? -1));
  sorted.forEach((r, i) => {
    const tr = table.insertRow();
    if (r.score?.collided) tr.className = "crashed";
    tr.insertCell().textContent = i + 1;
    tr.insertCell().innerHTML = `${r.label}<br><span class="unit">${r.controller || ""}</span>`;
    const cells = [
      r.score ? r.score.total : null,
      r.score?.categories.mission,
      r.score?.categories.safety,
      r.score?.categories.energy,
    ];
    cells.forEach((v, k) => {
      const c = tr.insertCell();
      c.className = "num";
      c.innerHTML = v === undefined || v === null ? "–" : (k === 0 ? `<b>${v.toFixed(1)}</b>` : v.toFixed(0));
    });
    tr.insertCell().textContent = r.finish_reason || r.error || "";
    tr.onclick = () => r.score && renderScore(r.score);
    tr.style.cursor = "pointer";
  });
}

function exportResults() {
  const blob = new Blob([JSON.stringify({ score, results }, null, 2)],
                        { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "avsim_results.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

/* ============================ batch ============================ */

function openBatch() {
  const p = $("batch-presets"), c = $("batch-controllers");
  p.innerHTML = ""; c.innerHTML = "";
  boot.presets.forEach((x) => p.appendChild(check("bp", x.key, x.name, x.key === $("preset").value)));
  boot.controllers.forEach((x) => c.appendChild(check("bc", x.value, x.label, x.value === $("controller").value)));
  $("batch-modal").classList.remove("hidden");
}

function check(name, value, label, on) {
  const l = document.createElement("label");
  const i = document.createElement("input");
  i.type = "checkbox"; i.name = name; i.value = value; i.checked = on;
  l.appendChild(i);
  l.appendChild(document.createTextNode(label));
  return l;
}

function picked(name) {
  return [...document.querySelectorAll(`input[name=${name}]:checked`)].map((i) => i.value);
}

async function runBatch() {
  const presets = picked("bp"), controllers = picked("bc");
  const seeds = $("batch-seeds").value.split(",").map((s) => Number(s.trim())).filter((n) => !Number.isNaN(n));
  if (!presets.length || !controllers.length) return message("batch-msg", "프리셋과 제어기를 선택하세요", "err");
  const n = presets.length * controllers.length * (seeds.length || 1);
  message("batch-msg", `${n}개 실행 중… 화면 갱신 없이 계산합니다.`);
  $("batch-run").disabled = true;
  try {
    const r = await api.post("/api/batch", {
      presets, controllers, seeds,
      parameters: params,
      n_vehicles: $("n_vehicles").value,
      duration: $("duration").value,
    });
    results = r.all;
    renderResults();
    message("batch-msg", `${r.results.length}개 완료.`, "ok");
  } catch (e) {
    message("batch-msg", e.message, "err");
  } finally {
    $("batch-run").disabled = false;
  }
}

/* ============================ report modal ============================ */

async function showReport() {
  const r = await api.get("/api/report");
  $("modal-title").textContent = `제어기 규격서 — contract ${r.version}`;
  $("modal-body").innerHTML = markdown(r.markdown);
  $("modal").classList.remove("hidden");
}

/* A deliberately small markdown renderer: headings, tables, fences, lists,
 * inline code/bold.  The report is ours, so the input is known. */
function markdown(src) {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const inline = (s) =>
    esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
      .replace(/(^|[\s(])\*([^*]+)\*/g, "$1<i>$2</i>");

  const out = [];
  const lines = src.split("\n");
  let i = 0, inFence = false, fence = [];
  let table = null, list = null;

  const closeList = () => { if (list) { out.push("</ul>"); list = null; } };
  const closeTable = () => { if (table) { out.push("</table>"); table = null; } };

  while (i < lines.length) {
    const line = lines[i++];
    if (line.startsWith("```")) {
      if (inFence) { out.push(`<pre><code>${esc(fence.join("\n"))}</code></pre>`); fence = []; inFence = false; }
      else { closeList(); closeTable(); inFence = true; }
      continue;
    }
    if (inFence) { fence.push(line); continue; }

    if (/^\|/.test(line)) {
      const cells = line.split("|").slice(1, -1).map((c) => c.trim());
      if (/^[\s|:-]+$/.test(line)) continue;              // the --- separator row
      if (!table) { closeList(); out.push("<table>"); table = "head"; }
      const tag = table === "head" ? "th" : "td";
      out.push("<tr>" + cells.map((c) => `<${tag}>${inline(c)}</${tag}>`).join("") + "</tr>");
      if (table === "head") table = "body";
      continue;
    }
    closeTable();

    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) { closeList(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); continue; }

    const li = /^\s*[-*]\s+(.*)$/.exec(line);
    if (li) {
      if (!list) { out.push("<ul>"); list = true; }
      out.push(`<li>${inline(li[1].replace(/^\[[ x]\]\s*/, ""))}</li>`);
      continue;
    }
    closeList();

    if (/^>\s?/.test(line)) { out.push(`<blockquote>${inline(line.replace(/^>\s?/, ""))}</blockquote>`); continue; }
    if (/^---+$/.test(line)) { out.push("<hr>"); continue; }
    if (line.trim() === "") continue;
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList(); closeTable();
  return out.join("\n");
}

/* ============================ wiring ============================ */

function message(id, text, cls) {
  const el = $(id);
  el.textContent = text || "";
  el.className = "msg" + (cls ? " " + cls : "");
}

function wire() {
  $("btn-run").onclick = startRun;
  $("btn-stop").onclick = stopRun;
  $("btn-defaults").onclick = resetParams;
  $("btn-report").onclick = () => showReport().catch((e) => alert(e.message));
  $("modal-close").onclick = () => $("modal").classList.add("hidden");
  $("btn-rescore").onclick = () => rescore().catch((e) => message("run-msg", e.message, "err"));
  $("btn-score-reset").onclick = () => { score = JSON.parse(JSON.stringify(boot.score)); buildScorePanel(); };
  $("btn-export").onclick = exportResults;
  $("btn-clear").onclick = () => { results = []; renderResults(); };
  $("btn-batch").onclick = openBatch;
  $("batch-close").onclick = () => $("batch-modal").classList.add("hidden");
  $("batch-run").onclick = runBatch;

  const zoom = $("zoom");
  const redraw = () => {
    $("zoom-label").textContent = zoom.value + " m";
    zoom.disabled = !$("follow").checked;
    if (lastFrame) draw(lastFrame);
  };
  zoom.oninput = redraw;
  $("follow").onchange = redraw;
  redraw();

  const n = $("n_vehicles"), slider = $("n_vehicles_slider");
  n.oninput = () => (slider.value = n.value);
  slider.oninput = () => (n.value = slider.value);

  $("btn-plugin").onclick = async () => {
    const path = $("plugin-path").value.trim();
    if (!path) return;
    try {
      const r = await api.post("/api/controller", { path });
      boot.controllers = (await api.get("/api/bootstrap")).controllers;
      buildControllers();
      $("controller").value = r.path;
      $("controller").onchange();
      $("controller-desc").textContent = r.description || "";
      message("plugin-msg", `불러왔습니다: ${r.label}`, "ok");
    } catch (e) {
      message("plugin-msg", e.message, "err");
    }
  };
}

init().catch((e) => {
  document.body.innerHTML = `<pre style="padding:20px;color:#e2564b">${e.message}</pre>`;
});
