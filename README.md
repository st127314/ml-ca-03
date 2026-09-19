# A3: Predicting Car Price III

Student `st127314`. This repository continues the A1/A2 used-car project, but treats
selling price as a four-class classification problem and implements multinomial logistic
regression from scratch.

## Deliverables

- `notebooks/03_car_price_classification.ipynb` - complete analysis, metric comparison,
  ridge experiment, measured results, and MLflow/deployment notes.
- `app/` - the fitted model, responsive Dash interface, Dockerfile, and VM compose file.
- `app/code/logistic_regression.py` - softmax regression, optional L2 penalty, and every
  requested classification metric using NumPy.
- `tests/` - unit tests, including the two explicitly required model input/output tests.
- `.github/workflows/ci-cd.yml` - test first, then build and publish only after tests pass.

## What is implemented

The data cleaning is intentionally the same as A1/A2. After an outer 80/20 split, training
price quartiles define these classes:

| Class | Selling-price band |
| --- | ---: |
| 0 | up to 250,000 |
| 1 | 250,000 to 409,999 |
| 2 | 409,999 to 645,000 |
| 3 | above 645,000 |

Numeric values are median-imputed and standardised. Categorical values are most-frequent
imputed and one-hot encoded. The classifier uses stable softmax, cross-entropy and
mini-batch gradient descent. Set `l2_lambda=0` for ordinary logistic regression or a
positive value for ridge logistic regression. The intercept is not penalised.

Accuracy, per-class precision/recall/F1, macro averages and weighted averages are written
from scratch. Weighted metrics use each class's support divided by total support. The
extra `/4` shown in the brief's weighted-precision example is not used because the class
proportions already sum to one; applying it would disagree with scikit-learn.

**Support** means the number of true samples in each class in the evaluated dataset.

## Experiment results

Eight configurations compare learning rates `0.03` and `0.05` with L2 strengths `0`,
`1e-5`, `1e-4`, and `1e-3`. Selection uses an inner validation split and macro F1, so the
outer test data is touched only once after the winner is refitted on all training rows.

| Result | Value |
| --- | ---: |
| Best validation macro F1 | 0.729 |
| Test accuracy | 0.740 |
| Test macro F1 | 0.734 |
| Test weighted F1 | 0.738 |
| Selected learning rate | 0.05 |
| Selected L2 lambda | 0 (ordinary logistic regression) |

Small L2 penalties tied or nearly tied the best validation result, while `1e-3` reduced
macro F1. `figures/experiment_results.json` contains every measured configuration. The
notebook verifies that the scratch metrics match `sklearn.metrics.classification_report`.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python app/code/app.py
```

Open `http://127.0.0.1:8050`. To rebuild the included fitted artifact without writing to
MLflow:

```bash
python training/train.py
```

## MLflow and model registry

The training script uses the exact assignment names:

- tracking URI: `http://mlflow.ml.brain.cs.ait.ac.th/`
- experiment: `st127314-a3`
- registered model: `st127314-a3-model`
- requested stage: `Staging`

The dataset is never logged. To intentionally write the experiment runs to the course
server, save the final model, register the best refit, and move it to Staging:

```bash
python training/train.py --log-mlflow --register
```

The remote command is explicit so importing modules, running tests, and opening the
notebook cannot accidentally create server runs.

## CI/CD and deployment

Add GitHub repository secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`. On a push to
`main` or `master`, GitHub Actions runs pytest. Only after it passes does the workflow
build and push both `latest` and commit-SHA Docker tags. The supplied
`app/docker-compose.yaml` matches the course VM's Traefik pattern (host
`web-st127314-a3.ml.brain.cs.ait.ac.th`) and pulls the `latest` image. Replace the Docker
Hub namespace or set `DOCKERHUB_USER` on the VM if needed.

The course VM is not reachable from GitHub's hosted runners, so the final `deploy` job
runs on a **self-hosted runner installed on the VM itself** — the runner only makes
outbound requests to GitHub to fetch jobs, so no inbound access to the VM is required.
One-time setup on the VM:

1. In the GitHub repo, go to **Settings → Actions → Runners → New self-hosted runner**
   and follow the download/config commands shown for Linux x64.
2. When prompted for labels, add `st127314-a3` (matches `runs-on: [self-hosted,
   st127314-a3]` in the workflow) so this runner only picks up this repo's deploy jobs.
3. Install it as a background service so it survives reboots/logout:
   ```bash
   sudo ./svc.sh install
   sudo ./svc.sh start
   ```
4. Confirm `docker` and the external `web` Traefik network already exist on the VM
   (`docker network ls | grep web`), since `docker-compose.yaml` depends on both.

Once the runner is online, every push to `main`/`master` that passes tests will build,
push, and then automatically run `docker compose pull && docker compose up -d` on the VM.
