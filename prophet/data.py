from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class AirbnbFeatures(BaseModel):
    latitude: float 
    longitude: float
    room_type: str = Field(description="Type of room (e.g., 'Entire home/apt', 'Private room', 'Shared room')")
    neighbourhood_group: str = Field(description="NYC borough (e.g., 'Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island')")
    minimum_nights: int = Field(description="Minimum nights required to book")
    number_of_reviews: int = Field(description="Number of reviews for the listing")
    calculated_host_listings_count: int = Field(description="Number of listings the host has")
    availability_365: int = Field(description="Number of days available in a year")
    
    class Config:
        schema_extra = {
            "example": {
                "latitude": 40.7456,
                "longitude": -73.9852,
                "room_type": "Entire home/apt",
                "neighbourhood_group": "Manhattan",
                "minimum_nights": 3,
                "number_of_reviews": 20,
                "calculated_host_listings_count": 2,
                "availability_365": 200
            }
        }


class PriceRange(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PricePrediction(BaseModel):
    price: float = Field(description="Predicted price in USD")
    price_range: PriceRange = Field(description="Price category")
    features: AirbnbFeatures = Field(description="Input features used for prediction")
    
    @classmethod
    def categorize_price(cls, price: float) -> PriceRange:
        """Categorize price into a price range."""
        if price < 100:
            return PriceRange.LOW
        elif price < 200:
            return PriceRange.MEDIUM
        elif price < 350:
            return PriceRange.HIGH
        else:
            return PriceRange.VERY_HIGH