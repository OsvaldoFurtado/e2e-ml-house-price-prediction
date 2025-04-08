from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow import keras

from prophet.config import Config


class PricePredictor:
    def __init__(self, model=None, transformer=None):
        self.model = model
        self.transformer = transformer

    def _create_transformer(self):
        """Create a column transformer for feature preprocessing."""
        return make_column_transformer(
            (
                MinMaxScaler(),
                [
                    "latitude",
                    "longitude",
                    "minimum_nights",
                    "number_of_reviews",
                    "calculated_host_listings_count",
                    "availability_365",
                ],
            ),
            (OneHotEncoder(handle_unknown="ignore"), ["neighbourhood_group", "room_type"]),
        )

    def _create_model(self, input_shape):
        """Create a neural network model for price prediction."""
        model = keras.Sequential()
        model.add(keras.layers.Dense(units=64, activation="relu", input_shape=[input_shape]))
        model.add(keras.layers.Dropout(rate=0.3))
        model.add(keras.layers.Dense(units=32, activation="relu"))
        model.add(keras.layers.Dropout(rate=0.5))
        model.add(keras.layers.Dense(1))

        model.compile(
            optimizer=keras.optimizers.Adam(Config.Model.LEARNING_RATE), loss="mse", metrics=["mse"]
        )
        return model

    def fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None):
        """Train the price prediction model."""
        # Separate features and target
        X_train = train_data.drop("price_log", axis=1)
        y_train = train_data["price_log"].values

        # Create and fit the transformer
        self.transformer = self._create_transformer()
        self.transformer.fit(X_train)

        # Transform the features
        X_train_transformed = self.transformer.transform(X_train)

        # Create and compile the model
        self.model = self._create_model(X_train_transformed.shape[1])

        # Prepare validation data if test_data is provided
        validation_data = None
        if test_data is not None:
            X_val = test_data.drop("price_log", axis=1)
            y_val = test_data["price_log"].values
            X_val_transformed = self.transformer.transform(X_val)
            validation_data = (X_val_transformed, y_val)

        # Define early stopping callback
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_mse" if validation_data else "mse",
            mode="min",
            patience=Config.Model.PATIENCE,
        )

        # Train the model
        history = self.model.fit(
            x=X_train_transformed,
            y=y_train,
            shuffle=True,
            epochs=Config.Model.EPOCHS,
            validation_data=validation_data,
            validation_split=0.2 if validation_data is None else 0.0,
            batch_size=Config.Model.BATCH_SIZE,
            callbacks=[early_stop],
        )

        return history

    def predict(self, data):
        """
        Predict price from input data.

        Args:
            data: Can be a dict, DataFrame row, or DataFrame

        Returns:
            Predicted price (not log-transformed)
        """
        # Convert input to DataFrame if it's a dict
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # Ensure input is in DataFrame format
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a dict or pandas DataFrame")

        # Transform features
        X_transformed = self.transformer.transform(data)

        # Make prediction (log-transformed)
        log_price_pred = self.model.predict(X_transformed)

        # Convert from log space back to original price
        price_pred = np.expm1(log_price_pred)

        return price_pred.flatten()[0] if len(price_pred) == 1 else price_pred.flatten()

    def save(self, model_path: Path, transformer_path: Path = None):
        """Save model and transformer to files."""
        # Ensure directory exists
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the model
        self.model.save(model_path)
        logger.debug(f"Model saved to {model_path}")

        # Save the transformer
        if transformer_path is None:
            transformer_path = model_path.parent / Config.Model.TRANSFORMER_FILE

        Config.Path.TRANSFORMERS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.transformer, transformer_path)
        logger.debug(f"Transformer saved to {transformer_path}")

    @staticmethod
    def load(model_path: Path, transformer_path: Path = None) -> "PricePredictor":
        """Load model and transformer from files."""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load the model
        model = keras.models.load_model(model_path, compile=False)

        # Load the transformer
        if transformer_path is None:
            transformer_path = (
                model_path.parent.parent / "transformers" / Config.Model.TRANSFORMER_FILE
            )

        if not transformer_path.exists():
            raise FileNotFoundError(f"Transformer file not found: {transformer_path}")

        transformer = joblib.load(transformer_path)

        return PricePredictor(model, transformer)
