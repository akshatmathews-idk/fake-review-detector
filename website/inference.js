let browserModel = null;
let modelLoadPromise = null;

export async function loadModel() {
  if (browserModel) return browserModel;
  if (!modelLoadPromise) {
    modelLoadPromise = fetch("./data/browser_model.json")
      .then(response => {
        if (!response.ok) throw new Error(`Could not load model (${response.status}).`);
        return response.json();
      })
      .then(model => {
        if (model.model_family !== "logistic_regression") {
          throw new Error("The deployed payload is not the selected Logistic Regression model.");
        }
        if (Object.keys(model.vocabulary || {}).length === 0 || model.idf.length !== model.coefficients.length) {
          throw new Error("The exported model payload is incomplete.");
        }
        browserModel = model;
        return browserModel;
      })
      .catch(error => {
        modelLoadPromise = null;
        throw error;
      });
  }
  return modelLoadPromise;
}

function tokenize(text) {
  // Mirrors TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", lowercase=True).
  return text.toLowerCase().match(/[\p{L}\p{N}_]+/gu)
    ?.filter(token => Array.from(token).length >= 2) || [];
}

function getNgrams(tokens) {
  const terms = [...tokens];
  for (let i = 0; i < tokens.length - 1; i++) {
    terms.push(`${tokens[i]} ${tokens[i + 1]}`);
  }
  return terms;
}

function sigmoid(value) {
  // Numerically stable logistic function.
  if (value >= 0) {
    const z = Math.exp(-value);
    return 1 / (1 + z);
  }
  const z = Math.exp(value);
  return z / (1 + z);
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
    // sublinear_tf=True -> 1 + log(count); then multiply by IDF.
    const tf = 1 + Math.log(count);
    features.push({ index, value: tf * idf[index] });
  }

  // norm="l2" in TfidfVectorizer.
  let normSquared = 0;
  for (const feature of features) normSquared += feature.value * feature.value;
  const norm = Math.sqrt(normSquared);
  if (norm > 0) for (const feature of features) feature.value /= norm;

  let decisionScore = intercept;
  const contributions = [];
  const termsByIndex = new Map();
  for (const [term, index] of Object.entries(vocabulary)) {
    if (counts.has(index)) termsByIndex.set(index, term);
  }
  for (const feature of features) {
    const contribution = feature.value * coefficients[feature.index];
    decisionScore += contribution;
    contributions.push({
      term: termsByIndex.get(feature.index) || `feature ${feature.index}`,
      contribution
    });
  }
  contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  const probabilityCG = sigmoid(decisionScore);
  const prediction = probabilityCG >= 0.5 ? 1 : 0;
  return {
    prediction,
    label: prediction === 1 ? "CG" : "OR",
    decisionScore,
    probabilityCG,
    probabilityOR: 1 - probabilityCG,
    knownFeatures: features.length,
    topContributions: contributions.slice(0, 8)
  };
}
