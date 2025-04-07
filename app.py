from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prophet.config import Config
from prophet.data import AirbnbFeatures, PricePrediction
from prophet.price_predictor import PricePredictor


app = FastAPI(
    title="Airbnb Price Predictor API",
    description="Predict Airbnb listing prices in NYC",
    version="0.1.0",
)

# Load the model at startup
model = PricePredictor.load(
    Config.Path.MODELS_DIR / Config.Model.FILE_NAME,
    Config.Path.TRANSFORMERS_DIR / Config.Model.TRANSFORMER_FILE
)


@app.post(
    "/predict",
    response_model=PricePrediction,
    summary="Predict Airbnb listing price",
)
async def predict(features: AirbnbFeatures):
    try:
        # Convert Pydantic model to dict
        features_dict = features.dict()
        
        # Make prediction
        predicted_price = model.predict(features_dict)
        
        # Create response with prediction and category
        price_range = PricePrediction.categorize_price(predicted_price)
        
        return PricePrediction(
            price=float(predicted_price),
            price_range=price_range,
            features=features
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)