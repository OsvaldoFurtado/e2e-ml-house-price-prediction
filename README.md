# ML Project Template

Starter template for your Machine Learning/AI projects.

Features:

- [`dvc`](https://dvc.org/) for data versioning and pipeline management (reproducibility)
- [`FastAPI`](https://fastapi.tiangolo.com/) for serving the model
- [`conda`](https://docs.conda.io/projects/conda/en/stable/index.html) package and environment manager
- [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting
- [`pytest`](https://docs.pytest.org/en/stable/) for testing
- [`loguru`](https://loguru.readthedocs.io/en/stable/) for logging
- [`Docker`](https://www.docker.com/) for containerization


## Install

Make sure you have [`conda` installed](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).

Clone the repository:

```bash
git clone git@github.com:OsvaldoFurtado/machine-learning-project-template.git .
cd machine-learning-project-template
```

Create Python 3.12.8 Virtual Environment using conda:

```bash
conda create --name ml_template python=3.12.8
```

Activate a virtual environment:

```bash
conda activate ml_template
```

Install dependencies:

```bash
pip-sync requirements-dev.txt
```

Install package in editable mode:

```bash
pip install -e .
```

Install pre-commit hooks:

```bash
pre-commit install
```

## Reproduce

The project contains three different stages defined in `dvc.yaml`.

- Create a dataset from the raw data:

```bash
dvc repro build-dataset
```

- Train a model using the dataset:

```bash
dvc repro train-model
```

- Evaluate the model using the test dataset:

```bash
dvc repro evaluate
```

## API server

Start the FastAPI server:

```bash
python app.py
```

Test the API:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "lets launch now"}'
```

## Tests

```bash
pytest
```

## Docker

The template includes a `Dockerfile` to build a Docker image:

```bash
docker build -t prophet:latest .
```

Run the Docker container:

```bash
docker run -d -p 8000:8000 --name prophet prophet:latest
```