# 🏠 Housing Price Prediction API

A production-ready REST API that predicts housing prices using a **Linear Regression** model trained on property features. Built with **FastAPI**, **Scikit-learn**, and containerised with **Docker**.

---

## Features

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Health check — API status & model state |
| `/predict` | `POST` | Single **or** batch price prediction |
| `/model-info` | `GET` | Model coefficients, intercept & performance metrics |
| `/docs` | `GET` | Interactive Swagger UI |
| `/redoc` | `GET` | ReDoc API documentation |

---

## Tech Stack

- **Python 3.12**
- **FastAPI** — API framework
- **Scikit-learn** — Linear Regression + StandardScaler
- **Pydantic v2** — Request/response validation
- **Joblib** — Model persistence
- **Docker** — Containerisation

---

## Quick Start

### Option 1 — Docker (Recommended)

```bash
# Build the image (trains the model automatically)
docker build -t housing-price-api .

# Run the container
docker run -p 8000:8000 housing-price-api
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

### Option 2 — Local Development

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python -m app.train

# 4. Start the API
uvicorn app.main:app --reload --port 8000
```

---

## API Usage

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "timestamp": "2026-06-10T10:00:00+00:00",
  "version": "1.0.0"
}
```

---

### `POST /predict` — Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "houses": [{
      "square_footage": 1850,
      "bedrooms": 3,
      "bathrooms": 2.0,
      "year_built": 1998,
      "lot_size": 7500,
      "distance_to_city_center": 5.6,
      "school_rating": 8.2
    }]
  }'
```

```json
{
  "predictions": [
    {
      "predicted_price": 265432.18,
      "input_features": { ... }
    }
  ],
  "count": 1
}
```

---

### `POST /predict` — Batch Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "houses": [
      {
        "square_footage": 1850, "bedrooms": 3, "bathrooms": 2.0,
        "year_built": 1998, "lot_size": 7500,
        "distance_to_city_center": 5.6, "school_rating": 8.2
      },
      {
        "square_footage": 2400, "bedrooms": 4, "bathrooms": 3.0,
        "year_built": 2010, "lot_size": 10500,
        "distance_to_city_center": 8.2, "school_rating": 9.0
      }
    ]
  }'
```

---

### `GET /model-info`

```bash
curl http://localhost:8000/model-info
```

```json
{
  "model_type": "Linear Regression",
  "intercept": -4521234.56,
  "coefficients": [
    { "feature": "square_footage", "coefficient": 89.23, "description": "..." },
    ...
  ],
  "metrics": {
    "r2_score": 0.9987,
    "mae": 4521.34,
    "rmse": 5812.67,
    "training_samples": 40,
    "test_samples": 10
  },
  "feature_names": ["square_footage", "bedrooms", ...],
  "trained_at": "2026-06-10T10:00:00Z"
}
```

---

## Input Features

| Feature | Type | Description |
|---|---|---|
| `square_footage` | float | Total square footage |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | float | Number of bathrooms (0.5 increments) |
| `year_built` | int | Year the house was built |
| `lot_size` | float | Lot size in square feet |
| `distance_to_city_center` | float | Distance to city center (miles) |
| `school_rating` | float | Local school rating (0–10) |

---

## Project Structure

```
housing-price-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app & endpoints
│   ├── model.py         # ModelManager (train, load, predict, info)
│   ├── schemas.py       # Pydantic request/response models
│   └── train.py         # Standalone training script
├── data/
│   ├── House Price Dataset.csv
│   └── Test Data For Prediction.csv
├── models/
│   └── housing_model.pkl   # Generated at build/train time
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```
