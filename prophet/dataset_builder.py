from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from prophet.config import Config


def build_dataset(save_dir: Path) -> pd.DataFrame:
    """Process the Airbnb NYC dataset and prepare it for model training."""
    logger.debug(f"Building dataset at {save_dir}")

    # Create necessary directories
    Config.Path.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Read raw data
    raw_file_path = Config.Path.RAW_DATA_DIR / Config.Dataset.RAW_FILE
    logger.debug(f"Reading raw data from {raw_file_path}")
    df = pd.read_csv(raw_file_path)

    # Data cleaning and preprocessing
    logger.debug("Cleaning and preprocessing data")
    df = df.drop(
        ["id", "name", "host_id", "host_name", "reviews_per_month", "last_review", "neighbourhood"],
        axis=1,
    )

    # Handle missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"Found {missing.sum()} missing values")
        df = df.dropna()  # Drop rows with missing values for simplicity

    # Save processed data
    processed_file = save_dir / Config.Dataset.PROCESSED_FILE
    df.to_parquet(processed_file)
    logger.debug(f"Saved processed data to {processed_file}")

    # Split into train and test sets
    X = df.drop("price", axis=1)
    y = np.log1p(df.price.values)  # Log transform for price prediction

    # Create a combined dataframe for storage
    X["price_log"] = y

    # Random split (no need for sklearn here as we just store the data)
    train_idx = np.random.choice(
        np.arange(len(X)), size=int(len(X) * (1 - Config.Dataset.TEST_SIZE)), replace=False
    )
    test_idx = np.array(list(set(range(len(X))) - set(train_idx)))

    # Save train and test datasets
    df_train = X.iloc[train_idx]
    df_test = X.iloc[test_idx]

    train_file = save_dir / Config.Dataset.TRAIN_FILE
    test_file = save_dir / Config.Dataset.TEST_FILE

    df_train.to_parquet(train_file)
    df_test.to_parquet(test_file)

    logger.debug(f"Saved {len(df_train)} training samples to {train_file}")
    logger.debug(f"Saved {len(df_test)} test samples to {test_file}")

    return df
