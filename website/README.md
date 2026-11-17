# Fake Review Detector — static website

This folder is the deployable GitHub Pages site. It uses `data/browser_model.json` for in-browser TF–IDF + Logistic Regression inference and `data/results.json` for metrics and dynamically rendered SVG charts. No Python backend or third-party charting library is required.

## Run locally

From the project root:

```bash
python -m http.server 8000 --directory website
```

Open `http://localhost:8000`. Do not open `index.html` directly as a `file://` URL because the browser blocks the `fetch()` calls used to load model and results JSON.

## Deployment model

The deployed model is the selected Logistic Regression pipeline. Experimental LinearSVC candidates are not downloaded by the website. If the model is retrained, regenerate `data/browser_model.json`, run the browser parity test cases, and confirm the export schema still matches `inference.js`.

## Charts and sources

The charts are generated from `data/results.json` using native SVG in `script.js`. The website itself uses HTML, CSS and JavaScript; it does not load Plotly, Chart.js or another chart package. Dataset, research-paper and Python-library citations, plus the AI-use disclosure and editable team contribution fields, are included in the Credits section of `index.html`.
