"""
Standalone training script.
Run this once to train the model and persist it to models/housing_model.pkl.

Usage:
    python -m app.train
"""
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model import model_manager

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "House Price Dataset.csv"
)


def main():
    print("Training Housing Price Prediction Model...")
    model_manager.train(DATA_PATH)
    print("Training complete.")


if __name__ == "__main__":
    main()
