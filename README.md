# Review Signals — Computer-Generated vs Original Consumer Review Classification

An ES335 machine-learning project investigating whether review text contains useful signals for distinguishing the `CG` (computer-generated) and `OR` (original) labels in a public consumer-review dataset.

> **Scope warning:** This project reproduces labels in one dataset. It is not a universal AI detector and cannot establish authorship, deception, fraud, authenticity, or review truthfulness.

## Final model

The selected deployment model is **TF-IDF (unigrams + bigrams) + Logistic Regression (L2)**.

| Item | Result |
|---|---:|
| Cross-validation Macro F1 | **93.9769%** |
| CV training Macro F1 | 97.9406% |
| CV train–validation gap | 3.9637 percentage points |
| Existing held-out test accuracy | **94.4816%** |
| Existing held-out test Macro F1 | **94.4810%** |
| Existing held-out test ROC-AUC | **98.7064%** |
| Browser model JSON | about 1.438 MB uncompressed; about 0.492 MB gzip estimate |

The best tuned LinearSVC candidate reached approximately 94.38% CV Macro F1, a little higher than Logistic Regression, but had a larger train–validation gap (about 5.27 percentage points). Logistic Regression was retained for deployment because the observed CV gap was smaller and it natively provides class probabilities. A compact LinearSVC (10k features) is retained as a size-focused alternative, not the deployed model.

**Test-set caveat:** the existing test split was inspected during earlier development, including the later model comparison. These figures are the project's held-out split results, not an untouched external validation. External data from a different source would be needed for a stronger generalisation claim.

## Dataset

Source: [Fake Reviews Dataset — mexwell on Kaggle](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset)

Original dataset: 40,432 rows. After trimming surrounding whitespace and removing 22 duplicate review-text occurrences: 40,410 unique texts. The labels are nearly balanced:

- `OR` → 0
- `CG` → 1

The stratified split contains 32,328 training examples and 8,082 test examples (`test_size=0.20`, `random_state=42`). The classifier uses review text only. Product category and rating are retained for EDA and error analysis, not as model inputs.

The raw CSV is included in the supplied project bundle for convenience. It is ignored by Git because it is source data from Kaggle; when cloning the repository, download the dataset separately and place `fake reviews dataset.csv` inside `data/raw/`.

## Repository structure

```text
fake_review_detector_project/
├── .github/workflows/deploy-pages.yml
├── data/
│   ├── raw/                     # Kaggle source CSV (local; Git-ignored)
│   └── processed/               # train.csv, test.csv, metadata (regenerated)
├── models/                      # fitted joblib models (generated; Git-ignored)
├── notebooks/                   # numbered analysis and experiment notebooks
├── results/                     # CV, tuning, evaluation and comparison outputs
├── src/
│   ├── prepare_data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── export_browser_model.py
│   ├── generate_browser_test_cases.py
│   └── export_website.py
├── website/                     # static website + browser model JSON
├── requirements.txt
└── README.md
```

## Setup (Fedora / Linux)

Python 3.12 is recommended. From the project root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name review-signals --display-name "Python 3.12 (Review Signals)"
```

Select **Python 3.12 (Review Signals)** as the Jupyter kernel in VS Code.

## Reproduce the pipeline with Python scripts

Run these commands from the project root, in order:

```bash
python src/prepare_data.py
python src/train.py
python src/evaluate.py
python src/export_browser_model.py
python src/generate_browser_test_cases.py
python src/export_website.py
```

The scripts write the fitted model under `models/final/`, evaluation outputs under `results/final_logistic_regression/`, and the browser model/results under `website/data/`.

For a single CLI prediction:

```bash
python src/predict.py "The product works exactly as expected."
```

For interactive predictions:

```bash
python src/predict.py --interactive
```

## Notebook workflow

Run numbered notebooks `01` through `10` in order. See [`notebooks/README.md`](notebooks/README.md) for the purpose of every notebook. The model-improvement notebook (Notebook 06) is the longest and requires XGBoost. All notebooks are delivered without saved cell outputs so results are generated from the files in this project rather than stale cached outputs.

Start Jupyter from the notebook directory if using a terminal:

```bash
cd notebooks
jupyter lab
```

## Run the website locally

From the project root:

```bash
python -m http.server 8000 --directory website
```

Open `http://localhost:8000`. Use a local HTTP server; opening `index.html` directly as a `file://` URL prevents the browser from fetching the model JSON.

The demo runs entirely in the browser. It downloads the Logistic Regression JSON model, tokenises review text, constructs TF-IDF features and applies the sigmoid to the model's log-odds score. The displayed probabilities are model outputs for the dataset classes, not calibrated proof of authorship.

## Deploy to GitHub Pages

1. Push the project to GitHub.
2. In repository **Settings → Pages**, select **GitHub Actions** as the build/deployment source.
3. Push a change to `website/` on `main` or `website-development`, or manually run the `Deploy static website to GitHub Pages` workflow.
4. The workflow publishes the contents of `website/`.

Update the repository links in `website/index.html` if you use a different GitHub repository.

## Key methodological notes

- TF-IDF is inside each scikit-learn pipeline, so vocabulary and IDF are learned independently in each CV training fold.
- Macro F1 is the primary selection metric because it weights both classes equally.
- The generalisation gap is a diagnostic (`mean training-fold Macro F1 − mean validation-fold Macro F1`), not a formal proof of overfitting.
- Logistic Regression probabilities are not guaranteed to be calibrated.
- Browser JSON size is not the same as total page size or guaranteed transfer size; actual loading depends on compression, network, caching and parsing.
- Website inference should be checked against Python predictions after any model re-training/export.
