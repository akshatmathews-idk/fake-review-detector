# Notebook sequence

Open Jupyter from the `notebooks/` folder (or open the notebook in VS Code; each notebook contains a project-root setup cell). Run the numbered notebooks in order.

| Notebook | Purpose |
|---|---|
| `01_data_understanding` | Dataset structure, labels, missing values, duplicates, categories, ratings and review length. |
| `02_data_preprocessing` | Trim and deduplicate texts, encode labels, stratified 80/20 split, leakage checks. |
| `03_baseline_model` | Dummy baseline and initial Logistic Regression / LinearSVC runs. |
| `04_model_comparison` | Analyse the initial CV comparison. |
| `05_feature_experiments` | Controlled LinearSVC TF-IDF feature experiments. |
| `06_model_improvement_and_tuning` | Stop-word, regularisation, LinearSVC, tree-model and style-feature experiments. This is the longest notebook. |
| `07_error_analysis_final_logistic_regression` | Out-of-fold error analysis for the final Logistic Regression model. |
| `08_final_evaluation_logistic_regression` | Refit and evaluate the final Logistic Regression model on the existing test split. |
| `09_linearsvc_vocabulary_size_experiment` | Compare compact LinearSVC vocabulary limits and browser payload sizes. |
| `10_final_model_comparison` | Direct comparison of Logistic Regression and the 10k LinearSVC alternative, including size and latency. |

**Model-selection distinction:** Notebook 06 records the highest-CV-Macro-F1 tuned candidate (LinearSVC). The project chooses Logistic Regression for deployment due to the smaller observed train–validation gap and native probability output. Notebook 07, Notebook 08, `src/`, and the website all use this final Logistic Regression configuration.

Notebooks are delivered without saved cell outputs to avoid stale results. The CSV/JSON files under `results/` preserve the recorded experiment summaries supplied with the project. Re-run cells to regenerate outputs.
