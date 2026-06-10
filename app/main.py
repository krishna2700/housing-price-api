from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.model import model_manager, FEATURE_NAMES
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HouseFeatures,
    HealthResponse,
    ModelInfoResponse,
    ModelCoefficients,
    ModelMetrics,
    PredictionResult,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model on startup."""
    loaded = model_manager.load()
    if not loaded:
        raise RuntimeError(
            "Model file not found. Please run 'python -m app.train' first."
        )
    print("Model loaded successfully.")
    yield


app = FastAPI(
    title="Housing Price Prediction API",
    description=(
        "A regression-based API that predicts housing prices from property features. "
        "Supports single and batch predictions, exposes model coefficients and performance metrics."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"],
)
def health_check():
    """Returns the API health status and whether the model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=model_manager.is_loaded,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0",
    )


# ---------------------------------------------------------------------------
# Predict endpoint  (single + batch)
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=BatchPredictionResponse,
    summary="Predict Housing Price(s)",
    tags=["Prediction"],
)
def predict(request: BatchPredictionRequest):
    """
    Accepts one or more houses and returns predicted prices.

    - **Single prediction**: pass a list with one house.
    - **Batch prediction**: pass a list with multiple houses.

    All 7 features are required for each house.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    feature_vectors = [
        [
            house.square_footage,
            house.bedrooms,
            house.bathrooms,
            house.year_built,
            house.lot_size,
            house.distance_to_city_center,
            house.school_rating,
        ]
        for house in request.houses
    ]

    try:
        prices = model_manager.predict(feature_vectors)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")

    results = [
        PredictionResult(predicted_price=round(price, 2), input_features=house)
        for house, price in zip(request.houses, prices)
    ]

    return BatchPredictionResponse(predictions=results, count=len(results))


# ---------------------------------------------------------------------------
# Model-info endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Coefficients & Performance Metrics",
    tags=["Model"],
)
def model_info():
    """
    Returns:
    - **Model type** and training timestamp
    - **Coefficients** for each feature (with descriptions)
    - **Intercept**
    - **Performance metrics**: R², MAE, RMSE on the held-out test split
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    info = model_manager.get_info()

    coefficients = [
        ModelCoefficients(
            feature=c["feature"],
            coefficient=c["coefficient"],
            description=c["description"],
        )
        for c in info["coefficients"]
    ]

    metrics = ModelMetrics(
        r2_score=round(info["metrics"]["r2_score"], 6),
        mae=round(info["metrics"]["mae"], 2),
        rmse=round(info["metrics"]["rmse"], 2),
        training_samples=info["metrics"]["training_samples"],
        test_samples=info["metrics"]["test_samples"],
    )

    return ModelInfoResponse(
        model_type=info["model_type"],
        intercept=round(info["intercept"], 4),
        coefficients=coefficients,
        metrics=metrics,
        feature_names=info["feature_names"],
        trained_at=info["trained_at"],
    )
