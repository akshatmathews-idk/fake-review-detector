# Source scripts

Run all scripts from the project root with `python src/<script>.py`.

- `prepare_data.py`: whitespace cleaning, duplicate removal, OR/CG target mapping, stratified split.
- `train.py`: fits the selected TF-IDF + Logistic Regression pipeline. Also exposes `build_pipeline()` and the recorded CV metrics for notebook reuse.
- `evaluate.py`: evaluates the fitted model on the existing test split and writes metrics/predictions.
- `predict.py`: CLI prediction.
- `export_browser_model.py`: exports TF-IDF vocabulary/IDF and Logistic Regression coefficients for JavaScript inference.
- `generate_browser_test_cases.py`: selects six correctly classified dataset examples for the demo.
- `export_website.py`: writes the website's `results.json` from saved project outputs.

The browser demo is a separate JavaScript implementation of the fitted pipeline. The website JSON payload is not a Python model file; the `.joblib` artifact is used by the Python scripts.
