"use strict";
/* Aurelia UI – vanilla JS, no build step. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const api = async (path, opts) => {
  const r = await fetch("/api" + path, opts);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};
const toast = msg => { const t = document.createElement("div"); t.className = "toast"; t.textContent = msg; $("#toasts").append(t); setTimeout(() => t.remove(), 3200); };
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------- 3D particle globe + orbiting rings (canvas, perspective projection) ---------- */
(function globe() {
  const cv = $("#bg"), cx = cv.getContext("2d"); let W, H, pts = [], mx = 0, my = 0, rot = 0;
  const N = 520, golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < N; i++) { const y = 1 - (i / (N - 1)) * 2, r = Math.sqrt(1 - y * y), t = golden * i; pts.push([Math.cos(t) * r, y, Math.sin(t) * r]); }
  const stars = Array.from({ length: 90 }, () => [Math.random(), Math.random(), Math.random() * 1.2 + .2, Math.random() * 6.28]);
  const resize = () => { const d = devicePixelRatio || 1; W = cv.width = innerWidth * d; H = cv.height = innerHeight * d; cx.setTransform(d, 0, 0, d, 0, 0); };
  addEventListener("resize", resize); resize();
  addEventListener("pointermove", e => { mx = e.clientX / innerWidth - .5; my = e.clientY / innerHeight - .5; });
  const gold = () => document.documentElement.dataset.theme === "light" ? "154,107,20" : "232,201,135";
  function frame(t) {
    const w = innerWidth, h = innerHeight; cx.clearRect(0, 0, w, h);
    const c = gold(), R = Math.min(w, h) * .30, ox = w * .74, oy = h * .17, f = 700;
    for (const s of stars) { cx.fillStyle = `rgba(${c},${.15 + .25 * Math.abs(Math.sin(t / 1500 + s[3]))})`; cx.fillRect(s[0] * w, s[1] * h, s[2], s[2]); }
    rot += reduce ? 0 : .0035;
    const ay = rot + mx * .9, ax = -.35 + my * .5, cyaw = Math.cos(ay), syaw = Math.sin(ay), cp = Math.cos(ax), sp = Math.sin(ax);
    const proj = (x, y, z, scale = R) => { let X = x * cyaw + z * syaw, Z = -x * syaw + z * cyaw, Y = y * cp - Z * sp; Z = y * sp + Z * cp; const k = f / (f + Z * scale + 400); return [ox + X * scale * k, oy + Y * scale * k, Z, k]; };
    const drawn = pts.map(p => proj(p[0], p[1], p[2])).sort((a, b) => a[2] - b[2]);
    for (const [x, y, z, k] of drawn) { const a = Math.min(1, (.25 + (z + 1) * .5) * k); cx.fillStyle = `rgba(${c},${a.toFixed(3)})`; cx.beginPath(); cx.arc(x, y, 1.4 + k * 1.6, 0, 6.283); cx.fill(); }
    // links between near neighbours for a "neural" mesh feel
    cx.lineWidth = .5; for (let i = 0; i < drawn.length; i += 7) { const a = drawn[i], b = drawn[(i + 19) % drawn.length]; const d = Math.hypot(a[0] - b[0], a[1] - b[1]); if (d < R * .45) { cx.strokeStyle = `rgba(${c},${(.16 * a[3]).toFixed(3)})`; cx.beginPath(); cx.moveTo(a[0], a[1]); cx.lineTo(b[0], b[1]); cx.stroke(); } }
    // orbit rings
    for (const [tilt, rad, col] of [[.9, 1.28, c], [-.5, 1.5, "88,214,201"]]) {
      cx.beginPath(); for (let i = 0; i <= 120; i++) { const a = i / 120 * 6.283 + rot * (tilt > 0 ? 1.4 : -1); const p = proj(Math.cos(a) * rad, Math.sin(a) * rad * Math.sin(tilt) * .5, Math.sin(a) * rad * Math.cos(tilt) * .5); i ? cx.lineTo(p[0], p[1]) : cx.moveTo(p[0], p[1]); }
      cx.strokeStyle = `rgba(${col},.34)`; cx.lineWidth = 1; cx.stroke();
    }
    if (!reduce) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();

/* ---------- spotlight + 3D tilt ---------- */
addEventListener("pointermove", e => { const s = $("#spot"); s.style.left = e.clientX + "px"; s.style.top = e.clientY + "px"; });
document.addEventListener("pointermove", e => {
  const el = e.target.closest?.(".tilt"); $$(".tilt").forEach(t => { if (t !== el) { t.style.setProperty("--rx", "0deg"); t.style.setProperty("--ry", "0deg"); } });
  if (!el || reduce) return; const r = el.getBoundingClientRect();
  el.style.setProperty("--ry", ((e.clientX - r.left) / r.width - .5) * 8 + "deg"); el.style.setProperty("--rx", -((e.clientY - r.top) / r.height - .5) * 8 + "deg");
});

/* ---------- theme ---------- */
const setTheme = t => { document.documentElement.dataset.theme = t; $("#theme-label").textContent = t === "dark" ? "Light mode" : "Dark mode"; try { localStorage.setItem("aurelia-theme", t); } catch {} };
try { const t = localStorage.getItem("aurelia-theme"); if (t) setTheme(t); } catch {}
$("#theme-btn").onclick = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

/* ---------- router ---------- */
const TITLES = { overview: ["Command Center", "A live view of the synthetic cohort."], patients: ["Patients", "Ranked by an explainable complexity score."],
  ask: ["Ask Aurelia", "Grounded answers with cited evidence – never invented."], map: ["Cohort Map", "Patients as a 3D similarity space, grouped by clinical profile."], lab: ["RAG Lab", "Thirteen retrieval strategies, compared honestly."], audit: ["Audit Trail", "Every access is recorded in a tamper-evident chain."] };
let current = "overview"; const loaded = {};
function go(v) {
  current = v; $$(".view").forEach(s => s.classList.toggle("on", s.id === "v-" + v)); $$("#nav button").forEach(b => b.classList.toggle("on", b.dataset.view === v));
  $("#title").textContent = TITLES[v][0]; $("#subtitle").textContent = TITLES[v][1]; history.replaceState(null, "", "#" + v);
  if (v === "audit") loadAudit(); if (v === "patients" && !loaded.patients) loadPatients(); if (v === "ask") $("#q").focus(); if (v === "map") loadMap(); if (v === "lab") loadLab();
}
$$("#nav button").forEach(b => b.onclick = () => go(b.dataset.view));

/* ---------- charts ---------- */
function countUp(el, to, dec = 0) { if (reduce) { el.textContent = to.toFixed(dec); return; } const t0 = performance.now(); (function s(t) { const p = Math.min(1, (t - t0) / 1100), e = 1 - Math.pow(1 - p, 4); el.textContent = (to * e).toFixed(dec); if (p < 1) requestAnimationFrame(s); })(t0); }
function spark(seed) { let y = 18, d = ""; for (let i = 0; i < 12; i++) { y = Math.max(4, Math.min(30, y + Math.sin(seed * (i + 1)) * 7 - 1)); d += (i ? "L" : "M") + i * 8.2 + " " + y.toFixed(1); } return `<svg class="spark" viewBox="0 0 90 34"><path d="${d}" fill="none" stroke="var(--gold)" stroke-width="1.8" stroke-linecap="round"/></svg>`; }
function donut(data, colors) {
  const total = data.reduce((a, d) => a + d.v, 0) || 1; let acc = 0; const R = 54, C = 2 * Math.PI * R;
  const arcs = data.map((d, i) => { const len = d.v / total * C, o = `<circle r="${R}" cx="70" cy="70" fill="none" stroke="${colors[i]}" stroke-width="16" stroke-dasharray="${len} ${C - len}" stroke-dashoffset="${-acc}" transform="rotate(-90 70 70)" stroke-linecap="butt"/>`; acc += len; return o; }).join("");
  return `<svg width="150" height="150" viewBox="0 0 140 140"><circle r="${R}" cx="70" cy="70" fill="none" stroke="rgba(255,255,255,.06)" stroke-width="16"/>${arcs}<text x="70" y="68" text-anchor="middle" style="font:600 28px 'Cormorant Garamond',serif;fill:var(--text)">${total}</text><text x="70" y="84" text-anchor="middle">patients</text></svg>`;
}
function sparkline(points, lo, hi, w = 180, h = 44) {
  const vals = points.map(p => p.value), mn = Math.min(lo, ...vals), mx = Math.max(hi, ...vals), sx = i => points.length < 2 ? w / 2 : 6 + i * (w - 12) / (points.length - 1), sy = v => h - 5 - (v - mn) / ((mx - mn) || 1) * (h - 10);
  const band = `<rect x="0" y="${sy(hi)}" width="${w}" height="${Math.max(2, sy(lo) - sy(hi))}" fill="rgba(95,212,160,.12)"/>`;
  const line = points.map((p, i) => (i ? "L" : "M") + sx(i) + " " + sy(p.value)).join(" ");
  const dots = points.map((p, i) => `<circle cx="${sx(i)}" cy="${sy(p.value)}" r="3" fill="${p.flag === "NORMAL" ? "var(--ok)" : "var(--rose)"}"><title>${esc(p.date)}: ${p.value}</title></circle>`).join("");
  return `<svg width="100%" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${band}<path d="${line}" fill="none" stroke="var(--gold)" stroke-width="1.6"/>${dots}</svg>`;
}
const ringHTML = (score, band) => `<div class="ring ${band}" style="--p:${score}"><b>${score}</b></div>`;

/* ---------- overview ---------- */
async function loadOverview() {
  const [c, pts] = await Promise.all([api("/cohort"), api("/patients?sort=risk")]);
  const total = c.alerts.critical + c.alerts.high + c.alerts.medium;
  const kp = [["Patients", c.patients, 0, "synthetic records"], ["Mean age", c.mean_age, 1, "years"], ["High-risk", c.risk_bands.high, 0, "need review"], ["Active alerts", total, 0, `${c.alerts.critical} critical`]];
  $("#kpis").innerHTML = kp.map((k, i) => `<div class="glass tilt kpi"><div class="lbl">${k[0]}</div><div class="val" data-to="${k[1]}" data-d="${k[2]}">0</div><div class="sub">${k[3]}</div>${spark(i + 1.7)}</div>`).join("");
  $$("#kpis .val").forEach(v => countUp(v, +v.dataset.to, +v.dataset.d));
  const max = Math.max(...c.conditions.map(x => x.count));
  $("#c-cond").innerHTML = c.conditions.slice(0, 8).map(x => `<div class="bar-row"><span title="${esc(x.name)}">${esc(x.name)}</span><div class="bar"><i data-w="${x.count / max * 100}"></i></div><b>${x.count}</b></div>`).join("");
  requestAnimationFrame(() => setTimeout(() => $$("#c-cond .bar i, #c-age .bar i").forEach(i => i.style.width = i.dataset.w + "%"), 60));
  const rb = c.risk_bands, cols = ["#5fd4a0", "#ffb454", "#ff7a90"];
  $("#c-risk").innerHTML = donut([{ v: rb.low }, { v: rb.moderate }, { v: rb.high }], cols) + `<div class="legend">${["Low", "Moderate", "High"].map((n, i) => `<div><i style="background:${cols[i]}"></i>${n} <b>${Object.values(rb)[i]}</b></div>`).join("")}</div>`;
  const am = Math.max(...c.age_bins.map(x => x.count));
  $("#c-age").innerHTML = c.age_bins.map(x => `<div class="bar-row" style="grid-template-columns:60px 1fr 34px"><span>${x.bin}</span><div class="bar"><i data-w="${x.count / am * 100}"></i></div><b>${x.count}</b></div>`).join("");
  $("#c-alerts").innerHTML = `<div class="alerts-big">${["critical", "high", "medium"].map(s => `<div class="alert-chip sev-${s}"><span style="text-transform:capitalize">${s}</span><b>${c.alerts[s]}</b></div>`).join("")}</div>`;
  $("#c-top").innerHTML = pts.slice(0, 5).map(p => `<div class="top-item" data-id="${p.patient_id}">${ringHTML(p.risk, p.band)}<div><b>${esc(p.name)}</b><small>${esc(p.conditions[0] || "")}</small></div></div>`).join("");
  $$("#c-top .top-item").forEach(e => e.onclick = () => openPatient(e.dataset.id));
  requestAnimationFrame(() => setTimeout(() => $$("#c-age .bar i").forEach(i => i.style.width = i.dataset.w + "%"), 60));
}

/* ---------- patients ---------- */
let sortMode = "risk", plistData = [];
async function loadPatients() {
  const q = $("#p-search").value; plistData = await api(`/patients?sort=${sortMode}&q=${encodeURIComponent(q)}`); loaded.patients = true;
  $("#plist").innerHTML = plistData.length ? plistData.map((p, i) => `<div class="glass tilt pc" data-id="${p.patient_id}" style="animation-delay:${Math.min(i, 14) * 30}ms">${ringHTML(p.risk, p.band)}<div><h4>${esc(p.name)}</h4><small>${esc(p.patient_id)} · ${p.age}${p.sex} · ${p.alerts} alert${p.alerts === 1 ? "" : "s"}</small><div class="tags">${p.conditions.slice(0, 2).map(c => `<span class="tag">${esc(c)}</span>`).join("")}</div></div></div>`).join("") : `<div class="empty">No patients match.</div>`;
  $$("#plist .pc").forEach(e => e.onclick = () => openPatient(e.dataset.id));
}
let deb; $("#p-search").oninput = () => { clearTimeout(deb); deb = setTimeout(loadPatients, 180); };
$$("#p-sort button").forEach(b => b.onclick = () => { sortMode = b.dataset.s; $$("#p-sort button").forEach(x => x.classList.toggle("on", x === b)); loadPatients(); });

/* ---------- patient drawer ---------- */
async function openPatient(id) {
  $("#scrim").classList.add("on"); $("#drawer").classList.add("on"); $("#drawer").setAttribute("aria-hidden", "false");
  $("#drawer-body").innerHTML = `<div class="skel" style="width:60%;height:36px"></div><div class="skel"></div><div class="skel" style="width:80%"></div>`;
  try {
    const v = await api("/patients/" + id), p = v.patient, r = v.risk;
    $("#drawer-body").innerHTML = `
      <div class="dh">${ringHTML(r.score, r.band)}<div><h2>${esc(p.name)}</h2><span class="muted">${esc(p.patient_id)} · ${p.age} y · ${p.sex === "F" ? "Female" : "Male"} · Allergies: ${esc(p.allergies)}</span></div></div>
      <div class="sec">Safety alerts</div>${v.alerts.length ? v.alerts.map(a => `<div class="alert ${a.severity}"><i></i><div><b>${esc(a.title)}</b><span>${esc(a.detail)}</span></div></div>`).join("") : `<div class="muted small">No rule-based alerts.</div>`}
      <div class="sec">Why this complexity score (${r.score}/100 · ${r.band})</div>
      <div class="factors">${r.factors.map(f => `<div><div class="factor"><span>${esc(f.factor)}</span><b>+${f.points}</b></div><div class="fbar"><i style="width:${Math.min(100, f.points * 4)}%"></i></div></div>`).join("")}</div>
      <div class="sec">Diagnoses</div><div class="meds">${v.diagnoses.map(d => `<span class="tag">${esc(d.condition)} · ${esc(d.status)}</span>`).join("")}</div>
      <div class="sec">Medications</div><div class="meds">${v.medications.map(m => `<span class="med ${m.active ? "" : "off"}">${esc(m.name)}</span>`).join("")}</div>
      <div class="sec">Laboratory trends</div><div class="labgrid">${Object.entries(v.labs).map(([t, l]) => { const last = l.points[l.points.length - 1]; return `<div class="lab"><div class="t"><span>${esc(t)}</span><span class="tr-${l.trend.split(" ")[0]}">${esc(l.trend)}</span></div><div class="v">${last.value} <small>${esc(l.unit)}</small></div>${sparkline(l.points, l.ref_low, l.ref_high)}${l.forecast ? `<div class="fc" title="OLS projection with prediction interval (R² ${l.forecast.r2})">→ ${l.forecast.predicted} <small>(${l.forecast.low}–${l.forecast.high}) by ${esc(l.forecast.target_date)}</small></div>` : ""}${l.anomaly && l.anomaly.flag ? `<div class="an">⚠ unusual latest reading (z ${l.anomaly.z})</div>` : ""}</div>`; }).join("")}</div>
      <div class="sec">Clinical notes · extracted findings</div><div id="ents" class="meds"></div>
      <div class="sec">Similar patients</div><div id="sim" class="sim"></div>
      <div class="sec">Timeline</div><div class="tl">${v.timeline.slice(0, 14).map(e => `<div class="tle"><small>${esc(e.date)}</small><b>${esc(e.title)}</b><span>${esc(e.detail)}</span></div>`).join("")}</div>
      <div style="margin-top:22px"><button class="btn" id="ask-this">Ask about this patient</button></div>`;
    $("#ents").innerHTML = (v.entities.present.map(e => `<span class="tag">${esc(e.name)} ×${e.mentions}</span>`).join("") + v.entities.denied.map(e => `<span class="med off" title="explicitly denied">no ${esc(e.name)}</span>`).join("")) || `<span class="muted small">No symptoms extracted.</span>`;
    api(`/patients/${id}/similar`).then(sim => { $("#sim").innerHTML = sim.map(x => `<div class="top-item" data-id="${esc(x.patient_id)}"><div class="simv">${Math.round(x.similarity * 100)}%</div><div><b>${esc(x.name)}</b><small>${esc((x.shared_conditions || []).join(", ") || "shared profile")}</small></div></div>`).join("") || `<div class="muted small">No close matches.</div>`; $$("#sim .top-item").forEach(e => e.onclick = () => openPatient(e.dataset.id)); }).catch(() => {});
    $("#ask-this").onclick = () => { closeDrawer(); go("ask"); $("#scope").value = id; };
  } catch (e) { $("#drawer-body").innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
}
const closeDrawer = () => { $("#scrim").classList.remove("on"); $("#drawer").classList.remove("on"); $("#drawer").setAttribute("aria-hidden", "true"); };
$("#scrim").onclick = closeDrawer; $("#drawer-x").onclick = closeDrawer;

/* ---------- ask ---------- */
let STRATS = [], chat = [];
const SUGGESTIONS = ["Which patients on metformin have low eGFR?", "What is the latest HbA1c for Kabir Menon?", "How has creatinine trended?", "Who is allergic to NSAIDs?", "Any labs in the last 6 months that were abnormal?", "Does anyone have a history of cancer?"];
$("#suggest").innerHTML = SUGGESTIONS.map(s => `<button type="button">${esc(s)}</button>`).join("");
$$("#suggest button").forEach(b => b.onclick = () => { $("#q").value = b.textContent; $("#ask-form").requestSubmit(); });
const bar = (v, cls = "") => `<div class="meter ${cls}"><i style="width:${Math.round(v * 100)}%"></i></div>`;
const confCls = v => v >= .6 ? "hi" : v >= .34 ? "mid" : "lo";
function showHint() { const s = STRATS.find(x => x.id === $("#strategy").value); $("#strategy-hint").textContent = s ? `${s.family} · ${s.when}` : ""; }
$("#strategy").onchange = showHint;
$("#reset-chat").onclick = () => { chat = []; $("#answer").innerHTML = ""; $("#compare").innerHTML = ""; toast("Started a new conversation"); };
const TRACE_ICON = { route: "◈", decision: "◆", rewrite: "✎", retrieve: "⌕", rerank: "⇅", grade: "✓", graph: "⬡", tool: "⚙", decide: "◆", expand: "⤢", filter: "▽", fuse: "⊕" };
function traceHTML(t) {
  if (!t.length) return "";
  return `<details class="trace" open><summary>Reasoning trace · ${t.length} step${t.length === 1 ? "" : "s"}</summary><ol>${t.map(s => `<li><span class="ti">${TRACE_ICON[s.kind] || "•"}</span><div><b>${esc(s.name)}</b><span>${esc(s.detail)}</span></div><small>${s.ms ? s.ms + " ms" : ""}</small></li>`).join("")}</ol></details>`;
}
function cohortHTML(s) {
  if (!s) return "";
  return `<div class="cohort-res"><div class="muted small">${s.count} patient${s.count === 1 ? "" : "s"} · ${esc(s.query)}</div>${s.patients.map(p => `<div class="cp" data-id="${esc(p.patient_id)}"><div><b>${esc(p.name)}</b><small>${esc(p.patient_id)} · ${p.age}${esc(p.sex)}</small></div><div class="tags">${p.why.map(w => `<span class="tag">${esc(w)}</span>`).join("")}</div></div>`).join("")}</div>`;
}
$("#ask-form").onsubmit = async e => {
  e.preventDefault(); const question = $("#q").value.trim(); if (question.length < 2) return;
  $("#compare").innerHTML = "";
  $("#answer").innerHTML = `<div class="skel" style="width:40%"></div><div class="skel"></div><div class="skel" style="width:75%"></div>`;
  try {
    const r = await api("/query", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, patient_id: $("#scope").value || null, strategy: $("#strategy").value, history: chat.slice(-3), redact: $("#redact").checked, k: 6 }) });
    chat.push(question);
    const html = esc(r.answer).replace(/\[(\d+)\]/g, '<span class="cite" data-n="$1">[$1]</span>');
    const f = r.verification.faithfulness, routed = r.requested_strategy !== r.strategy;
    $("#answer").innerHTML = `<div class="answer-box ${r.abstained ? "abstain" : ""}"><div class="meta"><span class="pill gold">${esc((STRATS.find(s => s.id === r.strategy) || {}).name || r.strategy)}${routed ? " (routed)" : ""}</span><span class="pill">${r.latency_ms} ms</span>${r.abstained ? '<span class="pill bad">Abstained</span>' : ""}${r.redacted ? '<span class="pill ok">PHI redacted</span>' : ""}</div>
      <p>${html}</p>${cohortHTML(r.structured)}
      <div class="gauges"><div><span>Retrieval confidence <b>${Math.round(r.confidence * 100)}%</b></span>${bar(r.confidence, confCls(r.confidence))}</div><div><span>Citation faithfulness <b>${Math.round(f * 100)}%</b></span>${bar(f, f >= .99 ? "hi" : f >= .7 ? "mid" : "lo")}</div></div>
      ${traceHTML(r.trace)}</div>`;
    $$("#answer .cp").forEach(c => c.onclick = () => openPatient(c.dataset.id));
    $("#evidence").innerHTML = r.evidence.length ? r.evidence.map((x, i) => `<div class="ev" id="ev-${x.n}" style="animation-delay:${i * 50}ms"><div class="h"><span><span class="n">${x.n}</span> <span class="kind">${esc(x.kind)}</span> · ${esc(x.date || "n/a")}</span><span class="rk">${x.bm25_rank != null || x.dense_rank != null ? `BM25 #${x.bm25_rank ?? "–"} · Dense #${x.dense_rank ?? "–"}` : ""}${x.via && x.via !== "retrieved" ? (x.bm25_rank != null || x.dense_rank != null ? " · " : "") + esc(x.via) : ""}</span></div><p>${esc(x.text)}</p><div class="muted small">${esc(x.patient_id)}</div></div>`).join("") : `<div class="empty">No evidence found.</div>`;
    $$(".cite").forEach(c => c.onclick = () => { const t = $("#ev-" + c.dataset.n); if (!t) return; t.scrollIntoView({ behavior: "smooth", block: "center" }); t.classList.add("hl"); setTimeout(() => t.classList.remove("hl"), 1600); });
  } catch (err) { $("#answer").innerHTML = `<div class="empty">${esc(err.message)}</div>`; }
};
$("#compare-btn").onclick = async () => {
  const question = $("#q").value.trim(); if (question.length < 2) { toast("Type a question first"); return; }
  $("#compare").innerHTML = `<div class="skel"></div><div class="skel" style="width:70%"></div>`;
  try {
    const rows = await api("/compare", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, patient_id: $("#scope").value || null, k: 6 }) });
    $("#compare").innerHTML = `<div class="cmp glass"><h3>Same question, ${rows.length} strategies</h3><div class="cmp-table">${rows.map(r => `<div class="cmp-row ${r.abstained ? "abst" : ""}"><div><b>${esc(r.name)}</b><small>${esc(r.family)}</small></div><div class="cmp-top">${r.abstained ? "<em>abstained – evidence insufficient</em>" : r.structured_count != null ? `<em>${r.structured_count} patients matched structurally</em>` : esc(r.top[0] || "–")}</div><div>${bar(r.confidence, confCls(r.confidence))}<small>${Math.round(r.confidence * 100)}% conf</small></div><small>${r.latency_ms} ms</small></div>`).join("")}</div></div>`;
  } catch (err) { $("#compare").innerHTML = `<div class="empty">${esc(err.message)}</div>`; }
};

/* ---------- cohort map (3D scatter, drag to rotate) ---------- */
const CL = ["#e8c987", "#58d6c9", "#8b7bff", "#ff7a90", "#5fd4a0", "#ffb454", "#6ea8ff"];
let mapData = null, mrot = { y: .6, x: -.25 }, mapDrag = null, mapHit = [], mapHover = -1, mapRaf = 0;
async function loadMap() {
  if (!mapData) {
    const [m, g] = await Promise.all([api("/cohort/map"), api("/graph")]); mapData = m;
    $("#map-info").textContent = `${m.points.length} patients · ${m.clusters.length} clusters`;
    $("#map-legend").innerHTML = m.clusters.map(c => `<div class="cl" style="--c:${CL[c.id % CL.length]}"><i></i><div><b>${esc(c.name)}</b><small>${c.size} patients · mean age ${c.mean_age} · mean risk ${c.mean_risk}</small></div></div>`).join("");
    $("#graph-stats").innerHTML = `<div class="gs">${Object.entries(g.nodes).map(([k, v]) => `<div><b>${v}</b><span>${esc(k)}</span></div>`).join("")}</div><p class="muted small">${g.total_nodes ?? ""} nodes, ${Object.values(g.edges).reduce((a, b) => a + b, 0)} edges power cohort questions such as “who is on metformin with low eGFR?”.</p>`;
  }
  drawMap();
}
function drawMap() {
  if (current !== "map" || !mapData) return; cancelAnimationFrame(mapRaf);
  const cv = $("#map3d"), cx = cv.getContext("2d"), r = cv.getBoundingClientRect(), d = devicePixelRatio || 1;
  if (cv.width !== Math.round(r.width * d)) { cv.width = Math.round(r.width * d); cv.height = Math.round(r.width * .62 * d); cv.style.height = r.width * .62 + "px"; }
  const w = cv.width / d, h = cv.height / d; cx.setTransform(d, 0, 0, d, 0, 0); cx.clearRect(0, 0, w, h);
  if (!mapDrag && !reduce) mrot.y += .004;
  const cy = Math.cos(mrot.y), sy = Math.sin(mrot.y), cxr = Math.cos(mrot.x), sxr = Math.sin(mrot.x), S = Math.min(w, h) * .36, f = 3.2;
  const P = mapData.points.map((p, i) => { let X = p.x * cy + p.z * sy, Z = -p.x * sy + p.z * cy, Y = p.y * cxr - Z * sxr; Z = p.y * sxr + Z * cxr; const k = f / (f + Z); return { i, p, sx: w / 2 + X * S * k, sy: h / 2 + Y * S * k, z: Z, k }; }).sort((a, b) => a.z - b.z);
  cx.strokeStyle = "rgba(232,201,135,.10)"; cx.lineWidth = 1;
  for (const a of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) { const pr = s => { let X = a[0] * s * cy + a[2] * s * sy, Z = -a[0] * s * sy + a[2] * s * cy, Y = a[1] * s * cxr - Z * sxr; Z = a[1] * s * sxr + Z * cxr; const k = f / (f + Z); return [w / 2 + X * S * k, h / 2 + Y * S * k]; }; const A = pr(-1.1), B = pr(1.1); cx.beginPath(); cx.moveTo(...A); cx.lineTo(...B); cx.stroke(); }
  mapHit = [];
  for (const q of P) { const col = CL[q.p.cluster % CL.length], rad = (3 + q.p.risk / 16) * q.k, hov = q.i === mapHover; mapHit.push({ x: q.sx, y: q.sy, r: rad + 5, id: q.p.patient_id, i: q.i });
    cx.globalAlpha = Math.min(1, .35 + q.k * .45); const g = cx.createRadialGradient(q.sx - rad * .3, q.sy - rad * .3, 1, q.sx, q.sy, rad * 1.6); g.addColorStop(0, "#fff"); g.addColorStop(.35, col); g.addColorStop(1, col + "00");
    cx.fillStyle = g; cx.beginPath(); cx.arc(q.sx, q.sy, rad * (hov ? 2 : 1.4), 0, 6.283); cx.fill(); }
  cx.globalAlpha = 1;
  if (mapHover >= 0) { const q = P.find(x => x.i === mapHover); if (q) { const t = `${q.p.name} · risk ${q.p.risk}`; cx.font = "12px Inter,sans-serif"; const tw = cx.measureText(t).width + 16; cx.fillStyle = "rgba(10,14,28,.85)"; cx.fillRect(q.sx + 12, q.sy - 26, tw, 22); cx.fillStyle = "#ecebe6"; cx.fillText(t, q.sx + 20, q.sy - 11); } }
  mapRaf = requestAnimationFrame(drawMap);
}
(function mapEvents() {
  const cv = $("#map3d"); let moved = 0;
  cv.addEventListener("pointerdown", e => { mapDrag = { x: e.clientX, y: e.clientY }; moved = 0; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", e => {
    const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    if (mapDrag) { mrot.y += (e.clientX - mapDrag.x) * .008; mrot.x = Math.max(-1.2, Math.min(1.2, mrot.x + (e.clientY - mapDrag.y) * .006)); moved += Math.abs(e.clientX - mapDrag.x) + Math.abs(e.clientY - mapDrag.y); mapDrag = { x: e.clientX, y: e.clientY }; }
    const h = [...mapHit].reverse().find(q => Math.hypot(q.x - px, q.y - py) < q.r); mapHover = h ? h.i : -1; cv.style.cursor = h ? "pointer" : "grab";
  });
  cv.addEventListener("pointerup", e => { mapDrag = null; if (moved < 6 && mapHover >= 0) openPatient(mapData.points[mapHover].patient_id); });
})();

/* ---------- RAG lab ---------- */
const heat = (v, lo = 0) => { const t = Math.max(0, Math.min(1, (v - lo) / (1 - lo))); return `hsla(${40 + t * 100}, 70%, ${38 + t * 10}%, ${.25 + t * .55})`; };
async function loadLab(refresh = false) {
  if (!STRATS.length) STRATS = await api("/strategies");
  $("#strat-grid").innerHTML = STRATS.map(s => `<div class="sc tilt"><span class="tag">${esc(s.family)}</span><h4>${esc(s.name)}</h4><p>${esc(s.summary)}</p><small><b>Use when:</b> ${esc(s.when)}</small></div>`).join("");
  $("#bench").innerHTML = `<div class="skel"></div><div class="skel" style="width:60%"></div>`;
  try {
    const b = await api("/benchmark" + (refresh ? "?refresh=true" : ""));
    const names = Object.fromEntries(b.strategies.map(s => [s.id, s.name]));
    $("#bench").innerHTML = `<div class="heat-wrap"><table class="heat"><thead><tr><th>Strategy</th>${b.suites.map(s => `<th title="${esc(s.title)} – ${esc(s.metric)} (n=${s.n})">${esc(s.title.replace(/ questions?/i, ""))}<small>${esc(s.metric)}</small></th>`).join("")}</tr></thead><tbody>${b.strategies.map(st => `<tr><th>${esc(st.name)}</th>${b.suites.map(su => { const row = su.rows.find(r => r.strategy === st.id); const v = row ? row.value : null; return v == null ? "<td>–</td>" : `<td style="background:${heat(v)}">${v.toFixed(2)}</td>`; }).join("")}</tr>`).join("")}</tbody></table></div><p class="muted small">${esc(b.note)} ${b.patients} patients.</p>`;
  } catch (err) { $("#bench").innerHTML = `<div class="empty">${esc(err.message)}</div>`; }
}
$("#bench-run").onclick = () => loadLab(true);

/* ---------- audit ---------- */
async function loadAudit() {
  const a = await api("/audit?limit=60"), v = a.verify;
  $("#chain").className = "pill " + (v.valid ? "ok" : "bad"); $("#chain").textContent = v.valid ? `Chain intact · ${v.entries} entries` : `Broken at entry ${v.broken_at}`;
  $("#audit-list").innerHTML = a.entries.map(e => `<div class="arow"><span class="muted">${new Date(e.ts * 1000).toLocaleString()}</span><span class="act">${esc(e.action)}</span><span class="muted">${esc(Object.entries(e.details).filter(([, x]) => x !== null).map(([k, x]) => k + "=" + x).join("  "))}</span><code title="${e.hash}">${e.hash.slice(0, 14)}…</code></div>`).join("") || `<div class="empty">No activity yet.</div>`;
}

/* ---------- command palette ---------- */
let pItems = [], pSel = 0;
async function openPalette() {
  $("#palette-wrap").classList.add("on"); $("#palette-in").value = ""; $("#palette-in").focus();
  if (!plistData.length) plistData = await api("/patients");
  renderPalette("");
}
function renderPalette(q) {
  const nav = Object.entries(TITLES).map(([k, v]) => ({ t: v[0], s: "View", run: () => go(k) }));
  const pts = plistData.map(p => ({ t: p.name, s: p.patient_id, run: () => { go("patients"); openPatient(p.patient_id); } }));
  pItems = [...nav, ...pts].filter(i => (i.t + i.s).toLowerCase().includes(q.toLowerCase())).slice(0, 9); pSel = 0; paintPalette();
}
const paintPalette = () => { $("#palette-list").innerHTML = pItems.map((i, n) => `<div class="pi ${n === pSel ? "sel" : ""}" data-n="${n}"><span>${esc(i.t)}</span><small>${esc(i.s)}</small></div>`).join(""); $$(".pi").forEach(e => e.onclick = () => { closePalette(); pItems[+e.dataset.n].run(); }); };
const closePalette = () => $("#palette-wrap").classList.remove("on");
$("#palette-btn").onclick = openPalette; $("#palette-wrap").onclick = e => { if (e.target.id === "palette-wrap") closePalette(); };
$("#palette-in").oninput = e => renderPalette(e.target.value);
$("#palette-in").onkeydown = e => { if (e.key === "ArrowDown") { pSel = Math.min(pItems.length - 1, pSel + 1); paintPalette(); e.preventDefault(); } else if (e.key === "ArrowUp") { pSel = Math.max(0, pSel - 1); paintPalette(); e.preventDefault(); } else if (e.key === "Enter" && pItems[pSel]) { closePalette(); pItems[pSel].run(); } };
addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); openPalette(); return; }
  if (e.key === "Escape") { closePalette(); closeDrawer(); return; }
  if (/input|select|textarea/i.test(document.activeElement.tagName)) return;
  const m = { 1: "overview", 2: "patients", 3: "ask", 4: "map", 5: "lab", 6: "audit" }[e.key]; if (m) go(m);
});

/* ---------- boot ---------- */
(async function boot() {
  try {
    const h = await api("/health"); $("#dot").classList.add("ok"); $("#status-text").textContent = `${h.patients} patients · ${h.chunks} chunks`; $("#src-pill").textContent = h.source;
    const list = await api("/patients"); plistData = list; STRATS = await api("/strategies");
    $("#strategy").innerHTML = STRATS.map(s => `<option value="${esc(s.id)}">${esc(s.name)}</option>`).join(""); showHint();
    $("#scope").innerHTML = `<option value="">Entire cohort</option>` + list.map(p => `<option value="${esc(p.patient_id)}">${esc(p.name)} · ${esc(p.patient_id)}</option>`).join("");
    await loadOverview(); toast("Aurelia ready – press Ctrl K to search");
    const hash = location.hash.slice(1); if (TITLES[hash]) go(hash);
  } catch (e) { $("#status-text").textContent = "offline"; toast("Could not reach the API: " + e.message); }
})();
