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


## Why we made these choices

This section records the reasoning behind the main methodological and engineering decisions. The explanations describe the choices made in this project; they should not be read as universal rules.

### 1. Dataset and prediction input

**We used the mexwell Fake Reviews Dataset on Kaggle** because it provides labelled consumer-review text with two classes (`OR` and `CG`), making it suitable for a supervised binary-classification experiment.

**We used review text as the only model input.** The dataset also contains product category and star rating, but these fields were reserved for exploratory analysis and error analysis. This keeps the prediction task focused on linguistic patterns and avoids letting metadata such as rating dominate the classifier. It also means the deployed demo needs only a review's text.

**We removed repeated review texts before splitting.** There were 22 duplicate text occurrences. Removing duplicates reduces the risk that identical reviews appear in both training and test partitions and inflate evaluation scores. Text was normalised for surrounding whitespace before duplicate removal.

**We used a stratified 80/20 train/test split with `random_state=42`.** Stratification preserves the near-equal OR/CG class proportions in both partitions. Fixing the random seed makes the split reproducible. This produced 32,328 training rows and 8,082 test rows.

### 2. Text representation: TF–IDF with unigrams and bigrams

**We chose TF–IDF rather than raw word counts.** TF–IDF scales a term according to its frequency in a review and how common it is across the corpus. This reduces the influence of terms that appear in nearly every review and gives more weight to terms that are relatively distinctive.

**We included unigrams and bigrams (`ngram_range=(1, 2)`).** Unigrams capture individual words; bigrams capture short phrases and local word combinations. This allows the classifier to use signals such as a phrase rather than relying only on isolated words. We retained the default token pattern for words of at least two characters.

**We did not remove English stop words in the final configuration.** The experiments without stop-word removal performed better than the earlier English-stop-word configuration. Common words and short function-word patterns can still carry stylistic information in this task, so removing them was not justified by our validation results.

**We used `min_df=10`.** Terms appearing in fewer than ten training documents were excluded. This reduces very rare features that may be noise or one-off spelling/product references, while retaining recurring vocabulary.

**We used sublinear term frequency scaling.** `sublinear_tf=True` replaces raw term frequency with a logarithmically scaled value. This limits the influence of a word merely being repeated many times in one review.

**We kept TF–IDF inside the scikit-learn pipeline during cross-validation.** This is important: the vocabulary and IDF weights are learned separately within each fold's training portion, rather than being calculated once using all data before cross-validation. This reduces information leakage.

### 3. Evaluation metric and validation strategy

**We used Macro F1 as the primary model-selection metric.** The two classes are close to balanced, but Macro F1 still makes the evaluation explicit: it calculates F1 for each class and gives both classes equal weight. Accuracy alone can hide uneven performance between classes.

**We used stratified five-fold cross-validation on the training split.** Each fold maintains approximately the same class proportions. Five folds provide multiple validation estimates while keeping the computational cost manageable for the set of models and parameter combinations explored.

**We reported the train–validation gap as a diagnostic, not as proof of overfitting.** The gap is the mean training-fold Macro F1 minus mean validation-fold Macro F1. A large gap can indicate that a model fits its training folds more strongly than it generalises, but the gap must be interpreted alongside validation performance and variability.

**We retained a DummyClassifier baseline.** A baseline gives a minimum reference point. A more complex classifier is useful only if it improves substantially over a simple strategy.

### 4. Why compare linear and tree-based models?

**We compared Logistic Regression and LinearSVC** because linear classifiers are strong, efficient baselines for high-dimensional sparse text features. They can work well with TF–IDF without requiring dense feature matrices.

**We also evaluated Random Forest and XGBoost**, including experiments with additional handcrafted style features. This tested whether non-linear tree ensembles and simple review-style statistics could add useful information. In our experiments, these approaches did not outperform the strongest linear TF–IDF models. The combined text-plus-style features helped the tree models relative to their text-only versions, but did not change the final model choice.

### 5. Why Logistic Regression is the deployed model

The tuned LinearSVC candidate achieved approximately **94.38% cross-validation Macro F1**, compared with **93.98%** for Logistic Regression. LinearSVC therefore had the slightly higher CV score.

We nevertheless selected **Logistic Regression with L2 regularisation** for deployment for two practical reasons:

1. Its observed train–validation gap was smaller: approximately **3.96 percentage points**, compared with approximately **5.27 percentage points** for the best tuned LinearSVC candidate.
2. Logistic Regression naturally provides class-probability outputs, which are convenient for the website's result display. LinearSVC produces decision margins rather than probabilities; probability estimates would require an additional calibration step.

The choice is a trade-off, not a claim that Logistic Regression is universally better. The CV score difference is small, and LinearSVC remains a strong benchmark.

The selected Logistic Regression configuration uses TF–IDF unigrams and bigrams, no stop-word removal, `min_df=10`, sublinear term frequency, up to 100,000 features, and L2-regularised Logistic Regression with `C=3` and the `liblinear` solver.

### 6. Why we tested LinearSVC vocabulary sizes

We separately tested LinearSVC with feature limits of 5,000, 10,000, 25,000 and 50,000 to understand the trade-off between predictive performance and browser payload size.

The 10,000-feature version produced a much smaller JSON payload (about 0.513 MB) while remaining close to the best LinearSVC CV score. Increasing the feature limit from 25,000 to 50,000 did not improve the measured CV Macro F1, and the 50,000 limit did not produce 50,000 actual vocabulary terms.

This experiment was useful for deployment planning, but it did not replace Logistic Regression as the selected website model.

### 7. Why browser-based inference?

**We deployed a static HTML/CSS/JavaScript website using an exported JSON representation of the trained Logistic Regression model.** GitHub Pages hosts static files and does not run the project's Python training or prediction scripts as a backend. Exporting the vocabulary, IDF values, coefficients and intercept allows the browser to reproduce the TF–IDF transformation and Logistic Regression score locally.

This keeps hosting simple and avoids requiring a separate server. It also means that after retraining, the browser model and displayed result files must be exported again and committed to the website.

**The displayed probabilities are model outputs, not proof of authorship.** Logistic Regression probabilities are not guaranteed to be calibrated, and the classifier has only been evaluated on this dataset's label scheme.

### 8. Limitations that affect how we interpret the choices

- The test split was inspected during development, including later model comparisons. Its results should therefore be described as the project's existing held-out split evaluation, not a completely untouched final validation.
- The dataset's OR/CG labels may not perfectly represent real-world authorship.
- Performance on other platforms, product categories, languages or time periods is unknown without external evaluation.
- TF–IDF captures word and phrase patterns, not full meaning, intent, sarcasm or factual correctness.
- The model is a dataset-specific text classifier, not a universal AI-writing or fraud detector.


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

## Key methodological notes

- TF-IDF is inside each scikit-learn pipeline, so vocabulary and IDF are learned independently in each CV training fold.
- Macro F1 is the primary selection metric because it weights both classes equally.
- The generalisation gap is a diagnostic (`mean training-fold Macro F1 − mean validation-fold Macro F1`), not a formal proof of overfitting.
- Logistic Regression probabilities are not guaranteed to be calibrated.
- Browser JSON size is not the same as total page size or guaranteed transfer size; actual loading depends on compression, network, caching and parsing.
- Website inference should be checked against Python predictions after any model re-training/export.
