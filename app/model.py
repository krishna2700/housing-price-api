import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import List, Dict, Any

FEATURE_NAMES = [
    "square_footage",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size",
    "distance_to_city_center",
    "school_rating",
]

FEATURE_DESCRIPTIONS = {
    "square_footage": "Total square footage of the house",
    "bedrooms": "Number of bedrooms",
    "bathrooms": "Number of bathrooms",
    "year_built": "Year the house was built",
    "lot_size": "Lot size in square feet",
    "distance_to_city_center": "Distance to city center in miles",
    "school_rating": "Local school rating (0–10)",
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "housing_model.pkl")


class ModelManager:
    def __init__(self):
        self._model: LinearRegression | None = None
        self._scaler: StandardScaler | None = None
        self._metrics: Dict[str, Any] = {}
        self._trained_at: str = ""
        self._training_samples: int = 0
        self._test_samples: int = 0

    def train(self, data_path: str) -> None:
        df = pd.read_csv(data_path)
        X = df[FEATURE_NAMES].values
        y = df["price"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        self._model = LinearRegression()
        self._model.fit(X_train_scaled, y_train)

        y_pred = self._model.predict(X_test_scaled)
        self._metrics = {
            "r2_score": float(r2_score(y_test, y_pred)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }
        self._training_samples = len(X_train)
        self._test_samples = len(X_test)
        self._trained_at = datetime.utcnow().isoformat() + "Z"

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "scaler": self._scaler,
                "metrics": self._metrics,
                "trained_at": self._trained_at,
                "training_samples": self._training_samples,
                "test_samples": self._test_samples,
            },
            MODEL_PATH,
        )
        print(f"Model trained and saved to {MODEL_PATH}")
        print(f"  R²:   {self._metrics['r2_score']:.4f}")
        print(f"  MAE:  ${self._metrics['mae']:,.2f}")
        print(f"  RMSE: ${self._metrics['rmse']:,.2f}")

    def load(self) -> bool:
        if not os.path.exists(MODEL_PATH):
            return False
        payload = joblib.load(MODEL_PATH)
        self._model = payload["model"]
        self._scaler = payload["scaler"]
        self._metrics = payload["metrics"]
        self._trained_at = payload["trained_at"]
        self._training_samples = payload["training_samples"]
        self._test_samples = payload["test_samples"]
        return True

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._scaler is not None

    def predict(self, features: List[List[float]]) -> List[float]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() or train() first.")
        X = np.array(features)
        X_scaled = self._scaler.transform(X)
        predictions = self._model.predict(X_scaled)
        return [float(p) for p in predictions]

    def get_info(self) -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")
        coefficients = [
            {
                "feature": name,
                "coefficient": float(coef),
                "description": FEATURE_DESCRIPTIONS[name],
            }
            for name, coef in zip(FEATURE_NAMES, self._model.coef_)
        ]
        return {
            "model_type": "Linear Regression",
            "intercept": float(self._model.intercept_),
            "coefficients": coefficients,
            "metrics": {
                **self._metrics,
                "training_samples": self._training_samples,
                "test_samples": self._test_samples,
            },
            "feature_names": FEATURE_NAMES,
            "trained_at": self._trained_at,
        }


# Singleton instance shared across the app
model_manager = ModelManager()
