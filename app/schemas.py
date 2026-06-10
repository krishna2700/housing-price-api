from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class HouseFeatures(BaseModel):
    square_footage: float = Field(..., gt=0, description="Total square footage of the house", example=1850)
    bedrooms: int = Field(..., ge=1, description="Number of bedrooms", example=3)
    bathrooms: float = Field(..., gt=0, description="Number of bathrooms (0.5 increments allowed)", example=2.0)
    year_built: int = Field(..., ge=1800, le=2100, description="Year the house was built", example=1998)
    lot_size: float = Field(..., gt=0, description="Lot size in square feet", example=7500)
    distance_to_city_center: float = Field(..., ge=0, description="Distance to city center in miles", example=5.6)
    school_rating: float = Field(..., ge=0, le=10, description="Local school rating (0–10)", example=8.2)

    model_config = {
        "json_schema_extra": {
            "example": {
                "square_footage": 1850,
                "bedrooms": 3,
                "bathrooms": 2.0,
                "year_built": 1998,
                "lot_size": 7500,
                "distance_to_city_center": 5.6,
                "school_rating": 8.2,
            }
        }
    }


class BatchPredictionRequest(BaseModel):
    houses: List[HouseFeatures] = Field(..., min_length=1, description="List of houses to predict prices for")

    model_config = {
        "json_schema_extra": {
            "example": {
                "houses": [
                    {
                        "square_footage": 1850,
                        "bedrooms": 3,
                        "bathrooms": 2.0,
                        "year_built": 1998,
                        "lot_size": 7500,
                        "distance_to_city_center": 5.6,
                        "school_rating": 8.2,
                    },
                    {
                        "square_footage": 2400,
                        "bedrooms": 4,
                        "bathrooms": 3.0,
                        "year_built": 2010,
                        "lot_size": 10500,
                        "distance_to_city_center": 8.2,
                        "school_rating": 9.0,
                    },
                ]
            }
        }
    }


class PredictionResult(BaseModel):
    predicted_price: float = Field(..., description="Predicted house price in USD")
    input_features: HouseFeatures


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    count: int = Field(..., description="Number of predictions returned")


class ModelCoefficients(BaseModel):
    feature: str
    coefficient: float
    description: str


class ModelMetrics(BaseModel):
    r2_score: float = Field(..., description="R² score on test split (higher is better, max 1.0)")
    mae: float = Field(..., description="Mean Absolute Error in USD")
    rmse: float = Field(..., description="Root Mean Squared Error in USD")
    training_samples: int = Field(..., description="Number of samples used for training")
    test_samples: int = Field(..., description="Number of samples used for evaluation")


class ModelInfoResponse(BaseModel):
    model_type: str
    intercept: float
    coefficients: List[ModelCoefficients]
    metrics: ModelMetrics
    feature_names: List[str]
    trained_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    version: str = "1.0.0"
