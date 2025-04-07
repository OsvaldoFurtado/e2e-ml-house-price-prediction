from pathlib import Path

import pandas as pd
from loguru import logger

from prophet.config import Config
from prophet.price_predictor import PricePredictor


def train(data_dir: Path, save_dir: Path):
    logger.debug("Model training started...")
    
    # Create directories if they don't exist
    Config.Path.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    Config.Path.TRANSFORMERS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load training and testing data
    train_df = pd.read_parquet(data_dir / Config.Dataset.TRAIN_FILE)
    test_df = pd.read_parquet(data_dir / Config.Dataset.TEST_FILE)
    
    logger.debug(f"Training data shape: {train_df.shape}")
    logger.debug(f"Test data shape: {test_df.shape}")
    
    # Create and train the model
    model = PricePredictor()
    history = model.fit(train_df, test_df)
    
    # Log training results
    final_epoch = len(history.history['mse'])
    train_mse = history.history['mse'][-1]
    val_mse = history.history['val_mse'][-1]
    logger.debug(f"Training completed in {final_epoch} epochs")
    logger.debug(f"Final training MSE: {train_mse:.4f}")
    logger.debug(f"Final validation MSE: {val_mse:.4f}")
    
    # Save model and transformer
    model_path = save_dir / Config.Model.FILE_NAME
    transformer_path = Config.Path.TRANSFORMERS_DIR / Config.Model.TRANSFORMER_FILE
    model.save(model_path, transformer_path)
    
    logger.debug(f"Model saved to {model_path}")
    logger.debug(f"Transformer saved to {transformer_path}")