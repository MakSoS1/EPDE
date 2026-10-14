/* Browser-first, no server-side data processing. Native PySINDy code lives in science.py. */
"use strict";

const CASES = {
  decay: "du/dt = −0.7u",
  logistic: "du/dt = 1.2u − 0.6u²",
  oscillator: "du/dt = v; dv/dt = −u − 0.2v",
  lotka: "du/dt = 1.5u − uv; dv/dt = uv − 0.7v",
  vdp: "du/dt = v; dv/dt = (1 − u²)v − u",
};
const METHODS = {
  pysindy: "PySINDy · official 2.1.0",
  evo_demo: "EvoSparse · educational, NOT EPDE",
};

const store = {
  db: null, memory: [], worker: null, awaiting: null,
  cancelled: false, records: [], selectedId: null,
  source: "builtin", derivative: "finite", degree: 2,
};

function el(id) { return document.getElementById(id); }
function make(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = String(text);
  return item;
}
function cleanValue(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}
function displayNumber(number) {
  return Number.isFinite(Number(number)) ? Number(number).toFixed(3) : "N/A";
}
function setStatus(text, error) {
  el("status").textContent = text;
  el("status").style.color = error ? "#a53c3c" : "";
}
function setBusy(value) {
  el("run").disabled = value;
  el("cancel").disabled = !value;
  el("progress").classList.toggle("hidden", !value);
}
function showPage(page) {
  for (const nav of document.querySelectorAll("[data-page]")) {
    const active = nav.dataset.page === page;
    nav.classList.toggle("active", active);
    if (active) nav.setAttribute("aria-current", "page");
    else nav.removeAttribute("aria-current");
  }
  for (const section of document.querySelectorAll("[data-view]")) {
    section.classList.toggle("active", section.dataset.view === page);
  }
  if (page === "results") renderResults();
  if (page === "compare") renderCompare();
  history.replaceState(null, "", "#" + page);
}
function switchChoice(attribute, chosen) {
  document.querySelectorAll("[" + attribute + "]").forEach(button => {
    const match = button.dataset[attribute.replace("data-", "")] === String(chosen);
    button.classList.toggle("selected", match);
    button.setAttribute("aria-pressed", String(match));
  });
}
function configureUI() {
  // The scientific WebAssembly engine remains identical in both layouts.
  // Embedded mode keeps PySINDy and its local history visible inside Streamlit,
  // without duplicating EPDE controls or confusing the educational EvoSparse
  // demonstration with PDEComp's native PySINDy adapter.
  if (new URLSearchParams(location.search).get("embed") === "1") {
    document.body.classList.add("embedded");
    el("method-evo").checked = false;
  }
  for (const button of document.querySelectorAll("[data-page]")) {
    button.addEventListener("click", () => showPage(button.dataset.page));
  }
  for (const button of document.querySelectorAll("[data-source]")) {
    button.addEventListener("click", () => {
      store.source = button.dataset.source;
      switchChoice("data-source", store.source);
      el("builtin-options").classList.toggle("hidden", store.source !== "builtin");
      el("csv-options").classList.toggle("hidden", store.source !== "csv");
      el("known-law").classList.toggle("hidden", store.source === "csv");
    });
  }
  for (const button of document.querySelectorAll("[data-deriv]")) {
    button.addEventListener("click", () => {
      store.derivative = button.dataset.deriv;
      switchChoice("data-deriv", store.derivative);
    });
  }
  for (const button of document.querySelectorAll("[data-degree]")) {
    button.addEventListener("click", () => {
      store.degree = Number(button.dataset.degree);
      switchChoice("data-degree", store.degree);
    });
  }
  el("dataset").addEventListener("change", () => {
    el("known-law").textContent = "Reference: " + CASES[el("dataset").value];
  });
  el("csv").addEventListener("change", () => {
    const file = el("csv").files?.[0];
    el("file-status").textContent = file ? file.name + " · " +
      Math.round(file.size / 1024) + " KiB" : "No file chosen";
  });
  el("run").addEventListener("click", run);
  el("cancel").addEventListener("click", cancelRun);
  el("clear").addEventListener("click", clearHistory);
  el("export").addEventListener("click", downloadResults);
  el("load-diagram").addEventListener("click", loadDiagrams);
  const route = location.hash.replace("#", "");
  showPage(["lab", "results", "compare", "learn"].includes(route) ? route : "lab");
}

async function openStorage() {
  return new Promise(resolve => {
    if (!("indexedDB" in window)) { resolve(null); return; }
    const req = indexedDB.open("equation-browser-lab-v1", 1);
    req.onupgradeneeded = () => {
      if (!req.result.objectStoreNames.contains("records")) {
        req.result.createObjectStore("records", {keyPath: "id"});
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
    req.onblocked = () => resolve(null);
  });
}
async function loadRecords() {
  store.db = await openStorage();
  if (!store.db) {
    setStatus("Local browser storage is unavailable. Results last only until this tab closes.", true);
    return;
  }
  store.records = await new Promise(resolve => {
    const tx = store.db.transaction("records", "readonly");
    const req = tx.objectStore("records").getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => resolve([]);
  });
  store.records.sort((a, b) => b.created - a.created);
  updateCount();
}
function persist(result) {
  if (!store.db) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const req = store.db.transaction("records", "readwrite").objectStore("records").put(result);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}
function clearStorage() {
  if (!store.db) return Promise.resolve();
  return new Promise(resolve => {
    const tx = store.db.transaction("records", "readwrite");
    tx.objectStore("records").clear();
    tx.oncomplete = resolve;
    tx.onerror = resolve;
  });
}
function updateCount() {
  el("count").textContent = String(store.records.length);
}
function digestHex(buffer) {
  return Array.from(new Uint8Array(buffer)).map(b => b.toString(16).padStart(2, "0")).join("");
}
async function buildJobBase() {
  const dataset = store.source === "builtin" ? el("dataset").value : "csv";
  const file = store.source === "csv" ? el("csv").files?.[0] : null;
  if (dataset === "csv" && !file) throw new Error("Choose your CSV before running");
  if (file && file.size > 1000000) throw new Error("CSV limit is 1 MB");
  const csv = file ? await file.text() : undefined;
  const noise = cleanValue(el("noise").value);
  const seed = cleanValue(el("seed").value);
  const degree = store.degree;
  const threshold = cleanValue(el("threshold").value);
  const population = cleanValue(el("population").value);
  const epochs = cleanValue(el("epochs").value);
  if (!(noise >= 0 && noise <= 20 && seed >= 0 && seed <= 100000 && Number.isInteger(seed))) {
    throw new Error("Noise must be 0–20%, and seed an integer 0–100,000");
  }
  if (!(threshold > 0 && threshold <= 2)) throw new Error("STLSQ threshold must be between 0 and 2");
  if (!(population >= 4 && population <= 80 && epochs >= 1 && epochs <= 40)) {
    throw new Error("Population or epoch limit exceeded");
  }
  const input = {dataset, csv, noise, seed, degree,
                 threshold, population, epochs, derivatives: store.derivative, samples:401};
  // Method is intentionally omitted: comparisons require identical input conditions.
  const sourceKey = dataset === "csv" ? csv : dataset;
  const buffer = new TextEncoder().encode(JSON.stringify({
    sourceKey, noise, seed, degree, derivatives: store.derivative,
  }));
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return {input, inputKey: digestHex(digest).slice(0, 32)};
}

function nativeWorker() {
  if (store.worker) return store.worker;
  const worker = new Worker("./science_worker.js");
  worker.onmessage = message => {
    const data = message.data;
    if (data.type === "status") {
      setStatus(data.text, false);
      return;
    }
    if (data.type === "capabilities") {
      store.capabilities = data.report;
      try { sessionStorage.setItem("epde_browser_capabilities", JSON.stringify(data.report)); }
      catch (_) {}
      return;
    }
    const pending = store.awaiting;
    if (!pending || data.id !== pending.id) return;
    store.awaiting = null;
    if (data.type === "result") pending.resolve(data.record);
    else pending.reject(new Error(data.message || "Worker failed"));
  };
  worker.onerror = event => {
    const pending = store.awaiting;
    store.awaiting = null;
    if (pending) pending.reject(new Error(event.message || "Browser worker failed"));
    setStatus("Worker error: " + (event.message || "unknown"), true);
  };
  store.worker = worker;
  return worker;
}
function callWorker(job) {
  return new Promise((resolve, reject) => {
    const id = crypto.randomUUID();
    store.awaiting = {id, resolve, reject};
    nativeWorker().postMessage({type:"run", id, job});
  });
}
function cancelRun() {
  store.cancelled = true;
  if (store.worker) store.worker.terminate();
  store.worker = null;
  if (store.awaiting) {
    store.awaiting.reject(new Error("Stopped by user"));
    store.awaiting = null;
  }
  setStatus("Calculation stopped. The next run will reload WebAssembly.", true);
}
async function run() {
  if (el("run").disabled) return;
  const methods = [];
  if (el("method-sindy").checked) methods.push("pysindy");
  if (el("method-evo").checked) methods.push("evo_demo");
  if (!methods.length) {
    setStatus("Select at least one method", true);
    return;
  }
  setBusy(true);
  store.cancelled = false;
  try {
    const {input, inputKey} = await buildJobBase();
    for (let i = 0; i < methods.length; i++) {
      if (store.cancelled) throw new Error("Stopped");
      const method = methods[i];
      setStatus("Initializing " + METHODS[method] + " (job " + (i+1) + "/" + methods.length + ")…");
      const record = await callWorker({...input, method});
      if (store.cancelled) throw new Error("Stopped");
      if (record.status !== "ok" || record.method !== method) {
        throw new Error("The scientific worker returned an invalid result");
      }
      record.id = crypto.randomUUID();
      record.created = Date.now();
      record.input_key = inputKey;
      // Do not store uploaded CSV contents, just a digest and derived charts.
      record.source = input.dataset;
      store.records.unshift(record);
      try { await persist(record); }
      catch (err) { setStatus("Computed, but browser storage failed: " + err.message, true); }
      updateCount();
      store.selectedId = record.id;
      renderResults();
    }
    if (!store.cancelled) {
      setStatus("Finished. Results are saved locally; open Results or Compare.", false);
      showPage("results");
    }
  } catch (err) {
    setStatus("Could not finish: " + String(err.message || err), true);
  } finally {
    setBusy(false);
  }
}
function itemButton(title, subtitle, onClick, selected) {
  const button = make("button", "item" + (selected ? " selected" : ""));
  button.type = "button";
  button.append(make("b", "", title), make("small", "", subtitle));
  button.addEventListener("click", onClick);
  return button;
}
function renderResults() {
  const list = el("result-list");
  list.replaceChildren();
  if (!store.records.length) list.append(make("p", "muted", "No runs yet. Start an experiment in Lab."));
  for (const record of store.records) {
    const title = (record.title || record.dataset) + " · " +
      (record.method === "pysindy" ? "PySINDy" : "EvoSparse");
    const note = new Date(record.created).toLocaleString() +
      " · validation R² " + displayNumber(record.validation?.r2);
    list.append(itemButton(title, note, () => {
      store.selectedId = record.id;
      renderResults();
    }, record.id === store.selectedId));
  }
  const target = el("result-detail");
  target.replaceChildren();
  const record = store.records.find(item => item.id === store.selectedId) || store.records[0];
  if (!record) {
    target.append(make("div", "detail-empty", "Select a completed experiment."));
    return;
  }
  store.selectedId = record.id;
  target.append(make("h2", "", record.title + " — " +
    (record.method === "pysindy" ? "PySINDy" : "EvoSparse teaching demo")));
  target.append(make("p", "small muted", record.details?.implementation || record.method));
  const scores = make("div", "metric-grid");
  [
    ["Validation R²", displayNumber(record.validation?.r2)],
    ["Validation RMSE", displayNumber(record.validation?.rmse)],
    ["Active terms", record.active_terms],
  ].forEach(([label, value]) => {
    const card = make("div", "metric");
    card.append(make("small", "", label), make("strong", "", value));
    scores.append(card);
  });
  target.append(scores);
  target.append(make("h3", "", "Discovered equations"));
  for (const eq of record.equations || []) target.append(make("div", "equation", eq));
  if (record.truth?.length) {
    target.append(make("h3", "", "Reference equations (not used to fit)"));
    for (const eq of record.truth) target.append(make("div", "equation", eq));
  }
  target.append(make("h3", "", "Derivative reconstruction"));
  const plot = make("div", "canvas-wrap");
  const canvas = make("canvas");
  canvas.width = 850; canvas.height = 285;
  canvas.setAttribute("aria-label", "Reference derivative versus discovered derivative on recorded data");
  plot.append(canvas);
  target.append(plot);
  drawChart(canvas, record);
  target.append(make("p", "small muted", record.scope));
  const controls = make("div", "choice");
  const down = make("button", "secondary", "Download result JSON");
  down.addEventListener("click", () => saveFile(
    JSON.stringify(record, null, 2), "equation-" + record.id + ".json"));
  controls.append(down);
  target.append(controls);
}
function drawChart(canvas, record) {
  const ctx = canvas.getContext("2d");
  if (!ctx || !record.data?.t?.length) return;
  const x = record.data.t, trueY = record.data.target_derivative.map(row=>row[0]);
  const predicted = record.data.model_derivative.map(row=>row[0]);
  const width = canvas.width, height = canvas.height;
  const left=58, top=25, right=16, bottom=40;
  const tmin=Math.min(...x), tmax=Math.max(...x);
  const rawY=[...trueY,...predicted].filter(Number.isFinite);
  if (!rawY.length) return;
  let ymin=Math.min(...rawY), ymax=Math.max(...rawY);
  const span=Math.max(1e-10,ymax-ymin);
  ymin-=span*.08; ymax+=span*.08;
  ctx.clearRect(0,0,width,height);
  ctx.fillStyle="#ffffff";ctx.fillRect(0,0,width,height);
  ctx.lineWidth=1;ctx.strokeStyle="#cfd8e5";ctx.beginPath();
  ctx.moveTo(left,top);ctx.lineTo(left,height-bottom);
  ctx.lineTo(width-right,height-bottom);ctx.stroke();
  ctx.fillStyle="#687b91";ctx.font="13px system-ui";
  ctx.fillText(ymax.toPrecision(3),4,top+7);
  ctx.fillText(ymin.toPrecision(3),4,height-bottom);
  ctx.fillText(tmin.toFixed(2),left,height-10);
  ctx.fillText(tmax.toFixed(2),width-right-58,height-10);
  const mapX=t=>(t-tmin)/(tmax-tmin||1)*(width-left-right)+left;
  const mapY=v=>top+(ymax-v)/(ymax-ymin)*(height-top-bottom);
  const line=(array,color)=>{
    ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=2;
    array.forEach((value,i)=>{if(i===0)ctx.moveTo(mapX(x[i]),mapY(value));
      else ctx.lineTo(mapX(x[i]),mapY(value));});ctx.stroke();
  };
  line(trueY,"#1c5184");line(predicted,"#d1833c");
  ctx.fillStyle="#1c5184";ctx.fillRect(width/2-115,9,14,3);
  ctx.fillStyle="#33465c";ctx.fillText("Observed derivative",width/2-96,15);
  ctx.fillStyle="#d1833c";ctx.fillRect(width/2+70,9,14,3);
  ctx.fillStyle="#33465c";ctx.fillText("Recovered",width/2+89,15);
}
function renderCompare() {
  const groups = new Map();
  for (const record of store.records) {
    if (!groups.has(record.input_key)) groups.set(record.input_key, []);
    const rows = groups.get(record.input_key);
    if (!rows.some(x => x.method === record.method)) rows.push(record);
  }
  const list = el("compare-list"), detail = el("compare-detail");
  list.replaceChildren();detail.replaceChildren();
  const matched = [...groups.entries()].filter(([,rows]) => rows.length >= 2);
  if (!matched.length) {
    list.append(make("p","muted","Run PySINDy and EvoSparse on the same equation, noise, seed, degree and differentiation to compare. Different objective scales are not treated as interchangeable."));
    return;
  }
  for (const [key, rows] of matched) {
    list.append(itemButton(
      rows[0].title + " · noise " + rows[0].noise + "%",
      rows.map(r=>r.method).join(" vs "),
      () => showComparison(key, rows),
      false
    ));
  }
  showComparison(matched[0][0], matched[0][1]);
}
function showComparison(key, rows) {
  const target = el("compare-detail");
  target.replaceChildren();
  target.append(make("h3", "", "Same numerical input · independent fitting objectives"));
  const cards = make("div", "compare-grid");
  for (const record of rows) {
    const card = make("div");
    card.append(make("h3", "", record.method === "pysindy" ? "Official PySINDy" : "Educational EvoSparse"));
    card.append(make("p","small muted","Validation R²: " + displayNumber(record.validation?.r2) +
      " · " + record.active_terms + " active terms"));
    for (const eq of record.equations || []) card.append(make("div","equation",eq));
    cards.append(card);
  }
  target.append(cards);
  target.append(make("p","small muted","This is a same-data comparison, not a published EPDE benchmark or a common Pareto ranking."));
}
async function clearHistory() {
  if (!confirm("Clear all results stored in this browser for this site?")) return;
  await clearStorage();
  store.records=[];store.selectedId=null;
  updateCount();renderResults();renderCompare();
}
function saveFile(content, filename) {
  const url = URL.createObjectURL(new Blob([content], {type:"application/json"}));
  const link=document.createElement("a");
  link.href=url;link.download=filename;link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function downloadResults() {
  saveFile(JSON.stringify({
    schema:"browser-results-export/1.0",
    created:new Date().toISOString(),
    records:store.records,
  },null,2),"equation-browser-results.json");
}
function loadDiagrams() {
  const target = el("diagram");
  target.replaceChildren(make("p","muted","Loading the original LikeC4 WebComponent…"));
  const script = document.createElement("script");
  script.src = "./likec4-views.js";
  script.onload = () => {
    const view = document.createElement("likec4-view");
    view.setAttribute("view-id","index");
    view.setAttribute("browser","true");
    view.style.width="100%";
    view.style.height="500px";
    target.replaceChildren(view);
  };
  let triedRawFallback = false;
  script.onerror = () => {
    // A source-file preview has no build step. Load the original, versioned
    // LikeC4 asset directly from the same GitHub repository through the
    // correct-MIME development proxy; GitHub Pages has it locally.
    if (!triedRawFallback) {
      triedRawFallback = true;
      script.src = "https://raw.githack.com/MakSoS1/EPDE/2a338d356cea510972e6057619e3365264baddb3/projects/pic/app/static/likec4-views.js";
    } else {
      target.replaceChildren(make("p","muted",
        "LikeC4 could not load. Open the original architecture in the native demo."));
    }
  };
  document.head.append(script);
  el("load-diagram").disabled=true;
}
document.addEventListener("DOMContentLoaded", async () => {
  configureUI();
  await loadRecords();
  renderResults();renderCompare();
});
