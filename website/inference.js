let browserModel = null;

export async function loadModel() {
  if (browserModel) return browserModel;
  const response = await fetch("./data/browser_model.json");
  if (!response.ok) throw new Error(`Could not load model (${response.status}).`);
  browserModel = await response.json();
  return browserModel;
}

function tokenize(text) {
  return text.toLowerCase().match(/[\p{L}\p{N}_]+/gu)
    ?.filter(token => Array.from(token).length >= 2) || [];
}

function getNgrams(tokens) {
  const terms = [...tokens];
  for (let i = 0; i < tokens.length - 1; i++) {
    terms.push(tokens[i] + " " + tokens[i + 1]);
  }
  return terms;
}

export function predictReview(text) {
  if (!browserModel) throw new Error("Model not loaded. Call loadModel() first.");
  if (typeof text !== "string" || !text.trim()) throw new Error("Enter non-empty review text.");

  const { vocabulary, idf, coefficients, intercept } = browserModel;
  const counts = new Map();
  for (const term of getNgrams(tokenize(text))) {
    const index = vocabulary[term];
    if (index !== undefined) counts.set(index, (counts.get(index) || 0) + 1);
  }

  const features = [];
  for (const [index, count] of counts) {
    const tf = 1 + Math.log(count);
    features.push({ index, value: tf * idf[index], term: null });
  }

  let normSquared = 0;
  for (const f of features) normSquared += f.value * f.value;
  const norm = Math.sqrt(normSquared);
  if (norm > 0) for (const f of features) f.value /= norm;

  let score = intercept;
  for (const f of features) score += f.value * coefficients[f.index];

  const termsByIndex = new Map();
  for (const [term, index] of Object.entries(vocabulary)) {
    if (counts.has(index)) termsByIndex.set(index, term);
  }

  const contributions = features.map(f => ({
    term: termsByIndex.get(f.index) || `feature ${f.index}`,
    contribution: f.value * coefficients[f.index]
  })).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  return {
    prediction: score > 0 ? 1 : 0,
    label: score > 0 ? "CG" : "OR",
    decisionScore: score,
    knownFeatures: features.length,
    topContributions: contributions.slice(0, 8)
  };
}
