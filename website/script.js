import { loadModel, predictReview } from "./inference.js";

const $ = (id) => document.getElementById(id);
const palette = { green: "#247d60", green2: "#68a987", amber: "#d49a43", muted: "#aebbb4", grid: "#e9efeb" };
let resultsData = null;
let modelChart = null;

async function loadResults() {
  const response = await fetch("./data/results.json");
  if (!response.ok) throw new Error("Could not load data/results.json");
  resultsData = await response.json();
  populateStats();
  buildCharts();
}

function pct(value, digits = 2) {
  return `${(value * 100).toFixed(digits)}%`;
}

function populateStats() {
  const d = resultsData.dataset;
  document.querySelectorAll("[data-stat]").forEach(el => {
    const key = el.dataset.stat;
    if (key === "accuracy") el.textContent = pct(resultsData.held_out_test.accuracy);
    else if (d[key] !== undefined) el.textContent = Number(d[key]).toLocaleString("en-US");
  });
}

function baseOptions() {
  return {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { backgroundColor: "#172b27", padding: 11, titleFont: { family: "DM Sans" }, bodyFont: { family: "DM Sans" } }
    },
    scales: {
      x: { grid: { color: palette.grid }, ticks: { color: "#718079", font: { family: "DM Sans", size: 10 } } },
      y: { grid: { display: false }, ticks: { color: "#52645b", font: { family: "DM Sans", size: 10 } } }
    }
  };
}

function buildCharts() {
  if (!window.Chart || !resultsData) return;
  const d = resultsData.dataset;
  const classCanvas = $("classChart");
  if (classCanvas) new Chart(classCanvas, {
    type: "doughnut",
    data: { labels: ["Original (OR)", "Computer-generated (CG)"], datasets: [{ data: [d.or, d.cg], backgroundColor: [palette.green, palette.amber], borderWidth: 0, hoverOffset: 5 }] },
    options: { responsive: true, maintainAspectRatio: false, cutout: "72%", plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `${c.label}: ${c.raw.toLocaleString()}` } } } }
  });

  const metricSelect = $("metricSelect");
  const modelRows = resultsData.model_comparison_cv;
  const modelCanvas = $("modelChart");
  const metricNames = { accuracy: "Accuracy", macro_f1: "Macro F1", balanced_accuracy: "Balanced accuracy", roc_auc: "ROC-AUC" };
  function renderModelChart() {
    if (!modelCanvas) return;
    const key = metricSelect?.value || "accuracy";
    if (modelChart) modelChart.destroy();
    modelChart = new Chart(modelCanvas, {
      type: "bar",
      data: {
        labels: modelRows.map(r => r.model.replace("Logistic Regression ", "Logistic Reg. ")),
        datasets: [{ label: metricNames[key], data: modelRows.map(r => r[key]), backgroundColor: modelRows.map((_, i) => i === 3 ? palette.green : ["#c3cec7", "#a8b9af", "#83a994"][i]), borderRadius: 7, barThickness: 24 }]
      },
      options: { ...baseOptions(), indexAxis: "y", plugins: { ...baseOptions().plugins, tooltip: { callbacks: { label: c => `${metricNames[key]}: ${pct(c.raw, 2)}` } } }, scales: { x: { min: 0, max: 1, grid: { color: palette.grid }, ticks: { callback: v => `${Math.round(v * 100)}%` } }, y: { grid: { display: false }, ticks: { font: { size: 10 } } } } }
    });
    $("metricCaption").textContent = `${metricNames[key]} · mean across five stratified folds. Higher is better.`;
  }
  renderModelChart();
  metricSelect?.addEventListener("change", renderModelChart);

  const featureCanvas = $("featureChart");
  if (featureCanvas) {
    const rows = resultsData.feature_experiments;
    new Chart(featureCanvas, {
      type: "bar",
      data: { labels: rows.map(r => r.configuration), datasets: [{ data: rows.map(r => r.accuracy), backgroundColor: rows.map((_, i) => i === 4 ? palette.green : "#a9bcb0"), borderRadius: 6, barThickness: 19 }] },
      options: { ...baseOptions(), indexAxis: "y", scales: { x: { min: .93, max: .95, grid: { color: palette.grid }, ticks: { callback: v => `${(v * 100).toFixed(1)}%` } }, y: { grid: { display: false }, ticks: { font: { size: 10 } } } }, plugins: { ...baseOptions().plugins, tooltip: { callbacks: { label: c => `Accuracy: ${pct(c.raw, 3)}` } } } }
    });
  }

  const ratingCanvas = $("ratingChart");
  if (ratingCanvas) {
    const rows = resultsData.oof_error_analysis.rating_error_rates;
    new Chart(ratingCanvas, {
      type: "bar",
      data: { labels: rows.map(r => `${r.rating} star${r.rating > 1 ? "s" : ""}`), datasets: [{ data: rows.map(r => r.error_rate), backgroundColor: palette.amber, borderRadius: 6, barThickness: 27 }] },
      options: { ...baseOptions(), scales: { y: { beginAtZero: true, max: .07, ticks: { callback: v => `${(v * 100).toFixed(0)}%` }, grid: { color: palette.grid } }, x: { grid: { display: false } } }, plugins: { ...baseOptions().plugins, tooltip: { callbacks: { label: c => `Error rate: ${pct(c.raw, 2)}` } } } }
    });
  }

  const categoryCanvas = $("categoryChart");
  if (categoryCanvas) {
    const rows = resultsData.oof_error_analysis.category_error_rates;
    new Chart(categoryCanvas, {
      type: "bar",
      data: { labels: rows.map(r => r.category), datasets: [{ data: rows.map(r => r.error_rate), backgroundColor: rows.map((_, i) => i < 2 ? palette.amber : palette.green2), borderRadius: 5, barThickness: 15 }] },
      options: { ...baseOptions(), indexAxis: "y", scales: { x: { beginAtZero: true, max: .075, ticks: { callback: v => `${(v * 100).toFixed(0)}%` }, grid: { color: palette.grid } }, y: { grid: { display: false }, ticks: { font: { size: 9 } } } }, plugins: { ...baseOptions().plugins, tooltip: { callbacks: { label: c => `Error rate: ${pct(c.raw, 2)} · n=${rows[c.dataIndex].total.toLocaleString()}` } } } }
    });
  }
}

function renderPrediction(result) {
  const container = $("predictionResult");
  const isCG = result.label === "CG";
  const contributions = result.topContributions;
  const max = Math.max(0.000001, ...contributions.map(x => Math.abs(x.contribution)));
  container.innerHTML = `
    <div class="prediction-top">
      <div><div class="prediction-eyebrow">PREDICTED DATASET LABEL</div><div class="prediction-class">${isCG ? "Computer-generated" : "Original"} <span>(${result.label})</span></div></div>
      <span class="prediction-badge">${isCG ? "Positive score" : "Negative score"}</span>
    </div>
    <div class="prediction-score"><b>${result.decisionScore.toFixed(6)}</b><span>LinearSVC decision score · not a probability</span></div>
    <div class="contrib-title">STRONGEST FEATURE CONTRIBUTIONS</div>
    ${contributions.length ? contributions.map(item => {
      const positive = item.contribution > 0;
      const width = Math.max(2, Math.abs(item.contribution) / max * 100);
      return `<div class="contrib-row"><span class="contrib-term" title="${escapeHTML(item.term)}">${escapeHTML(item.term)}</span><div class="contrib-track"><div class="contrib-fill ${positive ? "" : "negative"}" style="width:${width}%"></div></div><span class="contrib-value">${item.contribution > 0 ? "+" : ""}${item.contribution.toFixed(4)}</span></div>`;
    }).join("") : `<p class="demo-status">No review terms matched the model vocabulary. The prediction is therefore based on the model intercept.</p>`}
    <p class="demo-status">Matched ${result.knownFeatures.toLocaleString()} vocabulary features. Positive contributions move the score toward CG; negative contributions move it toward OR.</p>`;
}

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[ch]));
}

async function analyseReview() {
  const text = $("reviewInput")?.value.trim();
  if (!text) { $("modelStatus").textContent = "Enter a review before analysing."; $("reviewInput")?.focus(); return; }
  const button = $("analyseButton");
  button.disabled = true;
  $("modelStatus").textContent = "Loading the trained model and analysing…";
  try {
    await loadModel();
    const result = predictReview(text);
    renderPrediction(result);
    $("modelStatus").textContent = "Analysis complete. Your review was processed in this browser.";
  } catch (err) {
    console.error(err);
    $("modelStatus").textContent = `Could not analyse this review: ${err.message}`;
  } finally { button.disabled = false; }
}

async function loadExampleReviews() {
  const host = $("exampleButtons");
  if (!host) return;
  try {
    const response = await fetch("./data/browser_test_cases.json");
    if (!response.ok) throw new Error(`Could not load examples (${response.status})`);
    const examples = await response.json();
    host.replaceChildren();
    examples.forEach((example, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example-button";
      const label = document.createElement("span");
      label.textContent = `${example.label || example.actual_class || example.predicted_class || "DATASET EXAMPLE"} · ${example.id || `EXAMPLE ${index + 1}`}`;
      const preview = document.createElement("p");
      preview.textContent = example.review || example.text;
      button.append(label, preview);
      button.addEventListener("click", async () => {
        $("reviewInput").value = example.review || example.text;
        await analyseReview();
      });
      host.appendChild(button);
    });
  } catch (error) {
    console.error(error);
    host.textContent = "Examples could not be loaded. You can still enter your own review.";
  }
}

function setupPlayground() {
  loadExampleReviews();
  $("analyseButton")?.addEventListener("click", analyseReview);
  $("clearButton")?.addEventListener("click", () => {
    $("reviewInput").value = "";
    $("predictionResult").innerHTML = `<div class="result-placeholder"><span>✳</span><b>Your prediction will appear here</b><p>Enter review text to see the predicted dataset label, decision score and strongest feature contributions.</p></div>`;
    $("modelStatus").textContent = "Enter another review to analyse.";
    $("reviewInput").focus();
  });
}

function setupMobileNav() {
  const button = $("mobileMenu"), nav = $("mainNav");
  button?.addEventListener("click", () => {
    const open = nav.classList.toggle("open");
    button.setAttribute("aria-expanded", String(open));
  });
  nav?.querySelectorAll("a").forEach(a => a.addEventListener("click", () => {
    nav.classList.remove("open");
    button?.setAttribute("aria-expanded", "false");
  }));
}

document.addEventListener("DOMContentLoaded", async () => {
  setupMobileNav();
  setupPlayground();
  try { await loadResults(); }
  catch (err) { console.error(err); }
});
