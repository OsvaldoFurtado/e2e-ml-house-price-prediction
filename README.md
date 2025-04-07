# Airbnb NYC Price Prediction

Deep Neural Network model with TensorFlow/Keras to predict Airbnb prices in NYC and deploy it as REST API using FastAPI.

Features:

- [`dvc`](https://dvc.org/) for data versioning and pipeline management (reproducibility)
- [`FastAPI`](https://fastapi.tiangolo.com/) for serving the model
- [`TensorFlow`](https://www.tensorflow.org/) for deep learning
- [`sklearn`](https://scikit-learn.org/) for data preprocessing
- [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting
- [`pytest`](https://docs.pytest.org/en/stable/) for testing
- [`loguru`](https://loguru.readthedocs.io/en/stable/) for logging
- [`Docker`](https://www.docker.com/) for containerization

## Install

Make sure you have [`conda` installed](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).

Clone the repository:

```bash
git clone git@github.com:YourUsername/airbnb-price-prediction.git
cd airbnb-price-prediction
```

Create Python 3.12.8 Virtual Environment using conda:

```bash
conda create --name airbnb_prediction python=3.12.8
```

Activate a virtual environment:

```bash
conda activate airbnb_prediction
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

## Dataset

This project uses the NYC Airbnb dataset. Ensure the dataset is available at:
```
artefacts/raw_dataset/airbnb_nyc.csv
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
     -d '{
        "latitude": 40.7456,
        "longitude": -73.9852,
        "room_type": "Entire home/apt",
        "neighbourhood_group": "Manhattan",
        "minimum_nights": 3,
        "number_of_reviews": 20,
        "calculated_host_listings_count": 2,
        "availability_365": 200
     }'
```

## Tests

```bash
pytest
```

## Docker

The template includes a `Dockerfile` to build a Docker image:

```bash
docker build -t airbnb-predictor:latest .
```

Run the Docker container:

```bash
docker run -d -p 8000:8000 --name airbnb-predictor airbnb-predictor:latest
```

## Model Architecture

The model is a deep neural network with the following architecture:
- Input layer based on feature dimensions
- Dense layer with 64 units and ReLU activation
- Dropout layer with 0.3 rate
- Dense layer with 32 units and ReLU activation
- Dropout layer with 0.5 rate
- Output layer with 1 unit (price prediction)

## Features Used

- `latitude` and `longitude`: Location coordinates
- `room_type`: Type of the room
- `neighbourhood_group`: NYC borough
- `minimum_nights`: Minimum stay requirement
- `number_of_reviews`: Number of reviews received
- `calculated_host_listings_count`: Number of listings the host has
- `availability_365`: Number of days available in a year