"""Build the submission notebook from small, reviewable cells."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3"}

cells = [
    nbf.v4.new_markdown_cell(
        "# A3: Predicting Car Price III\n\n"
        "**Student:** st127314  \n"
        "**Problem:** convert cleaned selling prices into four classes, implement multinomial "
        "logistic regression and classification metrics from scratch, add optional L2 regularisation, "
        "track experiments in MLflow, and deploy the selected model."
    ),
    nbf.v4.new_markdown_cell(
        "## 1. Reproducible setup\n\n"
        "The shared modules below are used by this notebook, the unit tests, and the Dash app. "
        "Keeping one implementation avoids a notebook-only class that cannot be unpickled in deployment."
    ),
    nbf.v4.new_code_cell(
        "from pathlib import Path\nimport json, sys\nimport joblib\nimport matplotlib.pyplot as plt\n"
        "import numpy as np\nimport pandas as pd\nfrom sklearn.metrics import classification_report\n\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "sys.path.insert(0, str(ROOT / 'app' / 'code'))\n"
        "sys.path.insert(0, str(ROOT / 'training'))\n"
        "from data_prep import load_clean_data, bucket_prices, price_bin_edges\n"
        "from logistic_regression import LogisticRegression, classification_report_from_scratch\n"
        "from train import EXPERIMENT_NAME, MODEL_NAME, TRACKING_URI\n"
        "print(EXPERIMENT_NAME, MODEL_NAME, TRACKING_URI)"
    ),
    nbf.v4.new_markdown_cell(
        "## 2. A1/A2 preprocessing and four price classes\n\n"
        "I reuse the A1/A2 rules: remove duplicate listings, test-drive cars, incompatible CNG/LPG "
        "mileage rows and an impossible odometer value; parse numeric units; extract brand; and group "
        "rare brands. Numeric median imputation includes the A1/A2 missing-value indicators; "
        "categorical values use most-frequent imputation and one-hot encoding. "
        "The three quartile boundaries are learned from training prices only. The outer "
        "test split therefore cannot influence either the bins or model selection."
    ),
    nbf.v4.new_code_cell(
        "from train import load_split\n"
        "X_train, X_test, y_train, y_test, edges = load_split(ROOT / 'datasets' / 'Cars.csv')\n"
        "pd.DataFrame({'class': range(4), 'train_support': np.bincount(y_train, minlength=4), "
        "'test_support': np.bincount(y_test, minlength=4), "
        "'lower': edges[:-1], 'upper': edges[1:]})"
    ),
    nbf.v4.new_markdown_cell(
        "## 3. Metrics from scratch\n\n"
        "For each class, TP, FP and FN are counted in a one-vs-rest view. Macro averaging gives every "
        "class equal weight. Weighted averaging uses `support / total_support` as the class weight. "
        "The `/ 4` printed after the weighted sum in the brief would make the result four times too "
        "small and would not match scikit-learn, so it is not applied."
    ),
    nbf.v4.new_code_cell(
        "y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3])\n"
        "y_pred = np.array([0, 1, 1, 1, 2, 3, 3, 0, 3])\n"
        "ours = classification_report_from_scratch(y_true, y_pred, labels=[0,1,2,3])\n"
        "theirs = classification_report(y_true, y_pred, labels=[0,1,2,3], output_dict=True, zero_division=0)\n"
        "comparison = pd.DataFrame({\n"
        "    'scratch': [ours['accuracy'], ours['macro avg']['f1-score'], ours['weighted avg']['f1-score']],\n"
        "    'sklearn': [theirs['accuracy'], theirs['macro avg']['f1-score'], theirs['weighted avg']['f1-score']],\n"
        "}, index=['accuracy', 'macro F1', 'weighted F1'])\n"
        "comparison['absolute difference'] = (comparison.scratch - comparison.sklearn).abs()\ncomparison"
    ),
    nbf.v4.new_markdown_cell(
        "**Support** is the number of true observations belonging to a class in the evaluated data. "
        "It is a count, not a performance score. It supplies the weights in weighted averages."
    ),
    nbf.v4.new_markdown_cell(
        "## 4. Multinomial and ridge logistic regression\n\n"
        "The estimator applies a numerically stable softmax to four linear scores and minimises mean "
        "cross-entropy with mini-batch gradient descent. `l2_lambda=0` disables regularisation. A positive "
        "value adds `lambda * sum(W**2)` and gradient `2 * lambda * W`; the intercept is not penalised. "
        "The complete commented implementation is in `app/code/logistic_regression.py`."
    ),
    nbf.v4.new_code_cell(
        "import inspect\nprint(inspect.getsource(LogisticRegression))"
    ),
    nbf.v4.new_markdown_cell(
        "## 5. Experiment\n\n"
        "Eight combinations compare two learning rates and four L2 strengths. Candidates are trained "
        "on an inner fit split and selected by validation macro F1; only the winner is refitted on all "
        "outer-training rows and evaluated once on the test set. This avoids selecting on the test set."
    ),
    nbf.v4.new_code_cell(
        "results = json.loads((ROOT / 'figures' / 'experiment_results.json').read_text())\n"
        "runs = pd.DataFrame(results['all_runs']).sort_values('validation_macro_f1', ascending=False)\nruns"
    ),
    nbf.v4.new_code_cell(
        "ax = runs.pivot(index='l2_lambda', columns='learning_rate', "
        "values='validation_macro_f1').sort_index().plot(kind='bar', figsize=(8,4), ylim=(0.68,0.75))\n"
        "ax.set(title='Validation macro F1 by L2 strength', ylabel='Macro F1', xlabel='L2 lambda')\n"
        "plt.tight_layout()\nplt.savefig(ROOT / 'figures' / 'a3_experiment_comparison.png', dpi=160)\nplt.show()"
    ),
    nbf.v4.new_code_cell(
        "model = joblib.load(ROOT / 'app' / 'models' / 'car_price_classifier.joblib')\n"
        "pd.DataFrame(model.test_report_).T"
    ),
    nbf.v4.new_markdown_cell(
        "The selected unregularised model (`lambda=0`, learning rate `0.05`) achieved **0.742 validation "
        "macro F1** and, after a full training refit, **0.738 test accuracy** and **0.732 test macro F1**. "
        "Small L2 values tied or nearly tied the winner; stronger L2 slightly underfit. The scratch and "
        "scikit-learn report values match numerically."
    ),
    nbf.v4.new_markdown_cell(
        "## 6. MLflow model registry and deployment\n\n"
        "MLflow may use a configured remote server or a local server. The app tries the "
        "remote registry first and falls back to its local registry when it is unavailable. "
        "Training runs are logged to the selected URI; local and remote histories are separate. "
        "`training/train.py` defaults to tracking URI `http://127.0.0.1:5000`, experiment "
        "`st127314-a3`, logs parameters/metrics without logging the dataset, saves the final pipeline, "
        "and can register `st127314-a3-model` at Staging. Run:\n\n"
        "```bash\npython training/train.py --log-mlflow --register\n```\n\n"
        "The repository also contains the Dash application, Docker image, two model-interface unit tests, "
        "and a GitHub Actions workflow. Every push runs tests; a passing push builds and publishes the "
        "Docker image. Docker Hub credentials are stored as GitHub secrets, never in the repository."
    ),
    nbf.v4.new_markdown_cell(
        "### MLflow UI screenshots\n\n"
        "The runs above are visible in the MLflow tracking UI, and the best refit is registered\n"
        "in the Model Registry at the Staging stage.\n\n"
        "**Experiment runs (`st127314-a3`):**\n\n"
        "![MLflow experiment runs UI](../figures/mlflow_experiment_ui.png)\n\n"
        "**Best model run (`best-model-refit`) with logged test metrics:**\n\n"
        "![MLflow best run UI](../figures/mlflow_best_run_ui.png)\n\n"
        "**Registered model (`st127314-a3-model`, Version 1, Stage: Staging):**\n\n"
        "![MLflow model registry UI](../figures/mlflow_model_registry_ui.png)"
    ),
]

nb["cells"] = cells
nbf.write(nb, ROOT / "notebooks" / "03_car_price_classification.ipynb")
