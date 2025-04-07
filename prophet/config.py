import os
import random
import sys
from pathlib import Path

import numpy as np
from loguru import logger


class Config:
    SEED = 42

    class Path:
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        ARTEFACTS_DIR = APP_HOME / "artefacts"
        DATA_DIR = ARTEFACTS_DIR / "data"
        RAW_DATA_DIR = ARTEFACTS_DIR / "raw_dataset"
        MODELS_DIR = ARTEFACTS_DIR / "models"
        REPORTS_DIR = ARTEFACTS_DIR / "reports"
        TRANSFORMERS_DIR = ARTEFACTS_DIR / "transformers"

    class Dataset:
        TEST_SIZE = 0.2
        TRAIN_FILE = "train.parquet"
        TEST_FILE = "test.parquet"
        PROCESSED_FILE = "processed.parquet"
        RAW_FILE = "airbnb_nyc.csv"

    class Model:
        FILE_NAME = "price_prediction_model.h5"
        TRANSFORMER_FILE = "data_transformer.joblib"
        BATCH_SIZE = 32
        LEARNING_RATE = 0.0001
        EPOCHS = 100
        PATIENCE = 10

    class Evaluation:
        REPORT_FILE = "evaluation.json"


def seed_everything(seed: int = Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        logger.warning("TensorFlow not found, skipping TF seed setting")


def configure_logging():
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "colorize": True,
                "format": "<green>{time:YYYY-MM-DD - HH:mm:ss}</green> | <level>{level}</level> | {message}",
            },
        ]
    }
    logger.configure(**config)