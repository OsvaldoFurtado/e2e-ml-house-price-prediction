import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prophet.config import Config
from prophet.price_predictor import PricePredictor


def evaluate(test_data_file: Path, model_file: Path, report_path: Path) -> dict:
    """Evaluate the model on test data and save metrics to a report file."""
    # Load test data
    df = pd.read_parquet(test_data_file)
    logger.debug(f"Evaluating model with {len(df)} samples")
    
    # Create reports directory if it doesn't exist
    Config.Path.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    predictor = PricePredictor.load(model_file)
    
    # Prepare test data
    X_test = df.drop('price_log', axis=1)
    y_test = df['price_log'].values
    
    # Make predictions
    log_y_pred = predictor.model.predict(predictor.transformer.transform(X_test))
    
    # Calculate metrics on log-transformed data
    mse = mean_squared_error(y_test, log_y_pred)
    mae = mean_absolute_error(y_test, log_y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, log_y_pred)
    
    # Convert predictions back to original scale for additional context
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(log_y_pred)
    
    # Calculate metrics on original scale
    mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    rmse_orig = np.sqrt(mse_orig)
    
    # Create metrics dictionary
    metrics = {
        "log_scale": {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        },
        "original_scale": {
            "mse": float(mse_orig),
            "mae": float(mae_orig),
            "rmse": float(rmse_orig)
        }
    }
    
    # Log metrics
    logger.debug(f"Log-scale metrics: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    logger.debug(f"Original-scale metrics: MSE={mse_orig:.2f}, MAE={mae_orig:.2f}, RMSE={rmse_orig:.2f}")
    
    # Save metrics to report file
    report_path.write_text(json.dumps(metrics, indent=2))
    logger.debug(f"Evaluation report saved to {report_path}")
    
    return metrics