import { loadModel, predictReview } from "./inference.js";

const $ = id => document.getElementById(id);
let resultsData = null;
let examples = [];

const percent = (value, digits = 2) => `${(Number(value) * 100).toFixed(digits)}%`;
const number = value => Number(value).toLocaleString("en-US");
const escapeHTML = value => String(value).replace(/[&<>"']/g, ch => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[ch]));

async function loadResults() {
  const response = await fetch("./data/results.json");
  if (!response.ok) throw new Error(`Could not load project results (${response.status}).`);
  resultsData = await response.json();
  renderStats();
  renderComparisons();
  renderFinalMetrics();
  renderConfusionMatrix();
  renderPayloadSize();
  renderCharts();
}

function renderStats() {
  const d = resultsData.dataset;
  const test = resultsData.held_out_test.metrics;
  const payload = resultsData.browser_payload;
  document.querySelectorAll("[data-stat]").forEach(el => {
    const key = el.dataset.stat;
    const values = {
      original_rows: number(d.original_rows),
      cleaned_rows: number(d.cleaned_rows),
      train_rows: number(d.train_rows),
      test_macro_f1: percent(test["Macro F1-score"]),
      gzip_mb: `${Number(payload.gzip_mb_estimate).toFixed(2)} MB`,
      or_rows: number(d.or_rows),
      cg_rows: number(d.cg_rows)
    };
    if (values[key] !== undefined) el.textContent = values[key];
  });
  const total = d.or_rows + d.cg_rows;
  if (total) {
    $("orBar").style.width = `${100 * d.or_rows / total}%`;
    $("cgBar").style.width = `${100 * d.cg_rows / total}%`;
  }
}

function renderComparisons() {
  const rows = resultsData.tuned_model_comparison || [];
  const host = $("tunedComparison");
  if (!rows.length) {
    host.innerHTML = '<tr><td colspan="4">Tuning comparison data not available.</td></tr>';
    return;
  }
  const order = ["Logistic Regression L1/L2/Elastic Net", "LinearSVC", "Random Forest", "XGBoost"];
  rows.sort((a, b) => {
    const ai = order.indexOf(a.name), bi = order.indexOf(b.name);
    return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
  });
  host.innerHTML = rows.map(row => `
    <tr class="${row.name.startsWith("Logistic") ? "selected-row" : ""}">
      <td>${escapeHTML(row.name)}${row.name.startsWith("Logistic") ? ' <span class="selected-tag">DEPLOYED</span>' : ""}</td>
      <td>${percent(row.cv_macro_f1)}</td>
      <td>${percent(row.training_macro_f1)}</td>
      <td>${(row.generalization_gap * 100).toFixed(2)} pp</td>
    </tr>`).join("");

  const vocabHost = $("vocabComparison");
  const vocabRows = resultsData.linearsvc_vocabulary_tradeoff || [];
  vocabHost.innerHTML = vocabRows.length ? vocabRows.map(row => `
    <tr class="${row.feature_limit === 10000 ? "selected-row" : ""}">
      <td>${number(row.feature_limit)}</td>
      <td>${percent(row.cv_macro_f1, 2)}</td>
      <td>${(row.generalization_gap * 100).toFixed(2)} pp</td>
      <td>${row.json_size_mb.toFixed(3)} MB</td>
    </tr>`).join("") : '<tr><td colspan="4">Vocabulary experiment data not available.</td></tr>';
}

function renderFinalMetrics() {
  const metrics = resultsData.held_out_test.metrics;
  document.querySelectorAll("[data-metric]").forEach(el => {
    const value = metrics[el.dataset.metric];
    if (value !== undefined) el.textContent = percent(value);
  });
  $("testCaveat").textContent = resultsData.held_out_test.note || resultsData.interpretation.test_set_caveat;
}

function renderConfusionMatrix() {
  const cm = resultsData.held_out_test.confusion_matrix;
  $("confusionMatrix").innerHTML = `
    <tr><th>Actual OR</th><td class="matrix-correct">${number(cm[0][0])}</td><td class="matrix-error">${number(cm[0][1])}</td></tr>
    <tr><th>Actual CG</th><td class="matrix-error">${number(cm[1][0])}</td><td class="matrix-correct">${number(cm[1][1])}</td></tr>`;
}

function renderPayloadSize() {
  const payload = resultsData.browser_payload;
  document.querySelectorAll("[data-size]").forEach(el => {
    const value = payload[el.dataset.size];
    el.textContent = value == null ? "—" : `${Number(value).toFixed(3)} MB`;
  });
}

function renderPrediction(result, elapsedMs) {
  const isCG = result.label === "CG";
  const top = result.topContributions;
  const max = Math.max(1e-10, ...top.map(item => Math.abs(item.contribution)));
  $("predictionResult").innerHTML = `
    <div class="prediction-header"><div><small>PREDICTED DATASET LABEL</small><h3>${isCG ? "Computer-generated" : "Original"} <span>(${result.label})</span></h3></div><span class="class-pill ${isCG ? "cg" : "or"}">${isCG ? "CG label" : "OR label"}</span></div>
    <div class="probability-block"><div class="probability-line"><span>Model-assigned P(CG)</span><b>${percent(result.probabilityCG)}</b></div><div class="prob-track"><i style="width:${(result.probabilityCG * 100).toFixed(3)}%"></i></div><div class="probability-line secondary-line"><span>P(OR)</span><b>${percent(result.probabilityOR)}</b></div></div>
    <div class="result-meta"><span>Matched features: <b>${number(result.knownFeatures)}</b></span><span>Inference: <b>${elapsedMs.toFixed(3)} ms</b></span></div>
    <div class="contrib-heading">STRONGEST FEATURE CONTRIBUTIONS</div>
    ${top.length ? top.map(item => {
      const positive = item.contribution >= 0;
      const width = Math.max(2, Math.abs(item.contribution) / max * 100);
      return `<div class="contribution-row"><span title="${escapeHTML(item.term)}">${escapeHTML(item.term)}</span><div class="contribution-track"><i class="${positive ? "positive" : "negative"}" style="width:${width}%"></i></div><b>${item.contribution > 0 ? "+" : ""}${item.contribution.toFixed(4)}</b></div>`;
    }).join("") : '<p class="small-note">No review terms matched the learned vocabulary; the output is based on the model intercept.</p>'}
    <p class="small-note">Positive contributions move the log-odds toward CG; negative contributions move it toward OR. These are associations learned by the classifier, not causal explanations.</p>`;
}


function renderCharts() {
  const comparison = resultsData.tuned_model_comparison || [];
  const cvHost = $("cvChart");
  if (cvHost && comparison.length) {
    const rows = [...comparison].sort((a, b) => b.cv_macro_f1 - a.cv_macro_f1);
    const width = 640, left = 190, right = 78, rowH = 39, top = 14;
    const plotW = width - left - right;
    const min = 0.80, max = 0.96;
    const height = top + rows.length * rowH + 18;
    const bars = rows.map((row, i) => {
      const y = top + i * rowH;
      const value = Math.max(0, Math.min(1, row.cv_macro_f1));
      const w = Math.max(1, (value - min) / (max - min) * plotW);
      const selected = row.name.startsWith("Logistic");
      const label = row.name.replace(" L1/L2/Elastic Net", "");
      return `<text x="${left - 12}" y="${y + 20}" text-anchor="end" font-size="13" fill="#42564a">${escapeHTML(label)}</text><rect x="${left}" y="${y + 5}" width="${plotW}" height="20" rx="5" fill="#edf2ed"/><rect x="${left}" y="${y + 5}" width="${w}" height="20" rx="5" fill="${selected ? "#28785c" : "#91b5a0"}"/><text x="${left + w + 8}" y="${y + 20}" font-size="12" fill="#30463a">${percent(value, 2)}</text>`;
    }).join("");
    cvHost.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Cross-validation Macro F1 comparison">${bars}<text x="${left}" y="${height - 1}" font-size="10" fill="#77857c">Scale: 80% to 96% Macro F1</text></svg>`;
  }

  const testHost = $("testChart");
  if (testHost) {
    const total = resultsData.held_out_test.correct + resultsData.held_out_test.errors;
    const correct = resultsData.held_out_test.correct;
    const errors = resultsData.held_out_test.errors;
    const w = 620, x = 24, y = 45, barW = 570, barH = 34;
    const goodW = barW * correct / total;
    testHost.innerHTML = `<svg viewBox="0 0 ${w} 170" role="img" aria-label="${correct} correct and ${errors} incorrect predictions out of ${total}">
      <text x="${x}" y="25" font-size="13" fill="#40564a">${number(total)} test reviews</text>
      <rect x="${x}" y="${y}" width="${barW}" height="${barH}" rx="7" fill="#f2e6d5"/>
      <path d="M ${x+7} ${y} H ${x+goodW} V ${y+barH} H ${x+7} Q ${x} ${y+barH} ${x} ${y+barH-7} V ${y+7} Q ${x} ${y} ${x+7} ${y} Z" fill="#28785c"/>
      <text x="${x}" y="108" font-size="14" font-weight="700" fill="#28785c">${number(correct)} correct · ${percent(correct/total)}</text>
      <text x="${x}" y="135" font-size="14" font-weight="700" fill="#a56a28">${number(errors)} incorrect · ${percent(errors/total)}</text>
      <text x="${x}" y="158" font-size="11" fill="#77857c">Correct = green · Incorrect = amber</text>
    </svg>`;
  }

  const vocabHost = $("vocabChart");
  const vocab = resultsData.linearsvc_vocabulary_tradeoff || [];
  if (vocabHost && vocab.length) {
    const width = 700, left = 72, right = 92, top = 18, rowH = 37;
    const plotW = width - left - right, height = top + vocab.length * rowH + 18;
    const min = 0.93, max = 0.95;
    const bars = vocab.map((row, i) => {
      const y = top + i * rowH;
      const w = Math.max(2, (row.cv_macro_f1 - min) / (max - min) * plotW);
      const selected = row.feature_limit === 10000;
      return `<text x="${left - 10}" y="${y + 19}" text-anchor="end" font-size="12" fill="#42564a">${number(row.feature_limit)}</text><rect x="${left}" y="${y + 4}" width="${plotW}" height="20" rx="5" fill="#edf2ed"/><rect x="${left}" y="${y + 4}" width="${Math.max(1, Math.min(plotW, w))}" height="20" rx="5" fill="${selected ? "#28785c" : "#91b5a0"}"/><text x="${left + plotW + 10}" y="${y + 13}" font-size="11" fill="#30463a">${percent(row.cv_macro_f1, 2)}</text><text x="${left + plotW + 10}" y="${y + 25}" font-size="9" fill="#77857c">${row.json_size_mb.toFixed(2)} MB</text>`;
    }).join("");
    vocabHost.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="LinearSVC vocabulary size experiment; labels show CV Macro F1 and JSON size">${bars}<text x="${left}" y="${height - 1}" font-size="10" fill="#77857c">Rows: feature limit · bar/value: CV Macro F1 · right: JSON size</text></svg>`;
  }
}

async function ensureExamples() {
  if (examples.length) return examples;
  const response = await fetch("./data/browser_test_cases.json");
  if (!response.ok) throw new Error(`Could not load sample reviews (${response.status}).`);
  examples = await response.json();
  return examples;
}

async function loadExampleReviews() {
  const host = $("exampleButtons");
  try {
    const rows = await ensureExamples();
    host.replaceChildren();
    rows.forEach((example, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example-button";
      const title = document.createElement("b");
      title.textContent = `${example.label} · Example ${index + 1}`;
      const preview = document.createElement("span");
      preview.textContent = example.review;
      button.append(title, preview);
      button.addEventListener("click", () => {
        $("reviewInput").value = example.review;
        analyseReview();
      });
      host.appendChild(button);
    });
  } catch (error) {
    host.textContent = "Examples could not be loaded. You can still enter a review manually.";
    console.error(error);
  }
}

async function analyseReview() {
  const text = $("reviewInput").value.trim();
  if (!text) {
    $("modelStatus").textContent = "Enter a review before analysing.";
    $("reviewInput").focus();
    return;
  }
  const button = $("analyseButton");
  button.disabled = true;
  $("modelStatus").textContent = "Loading the model and analysing…";
  try {
    const loadStart = performance.now();
    await loadModel();
    const loadElapsed = performance.now() - loadStart;
    const start = performance.now();
    const result = predictReview(text);
    const elapsed = performance.now() - start;
    renderPrediction(result, elapsed);
    $("modelStatus").textContent = `Analysis complete in this browser. Model load/check: ${loadElapsed.toFixed(1)} ms.`;
  } catch (error) {
    console.error(error);
    $("modelStatus").textContent = `Could not analyse the review: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

function setupNavigation() {
  const button = $("mobileMenu");
  const nav = $("mainNav");
  button.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    button.setAttribute("aria-expanded", String(open));
  });
  nav.querySelectorAll("a").forEach(link => link.addEventListener("click", () => {
    nav.classList.remove("open");
    button.setAttribute("aria-expanded", "false");
  }));
}

function setupDemo() {
  $("analyseButton").addEventListener("click", analyseReview);
  $("clearButton").addEventListener("click", () => {
    $("reviewInput").value = "";
    $("predictionResult").innerHTML = '<div class="empty-output"><span>✳</span><b>Your result will appear here</b><p>Enter review text to see the predicted dataset label and model-assigned probabilities.</p></div>';
    $("modelStatus").textContent = "Enter another review to analyse.";
    $("reviewInput").focus();
  });
  loadExampleReviews();
}

document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupDemo();
  try {
    await loadResults();
  } catch (error) {
    console.error(error);
    $("modelStatus").textContent = `Project result data could not be loaded: ${error.message}`;
  }
});
