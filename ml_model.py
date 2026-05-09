"""Traditional machine-learning model for AG News classification."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from data_loader import LABEL_MAP
from text_preprocessing import clean_text


def build_ml_pipeline(max_features: int = 10000) -> Pipeline:
    """Create a TF-IDF + Logistic Regression text classifier."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=max_features)),
            ("classifier", LogisticRegression(max_iter=300, n_jobs=-1)),
        ]
    )


def train_ml_model(
    train_df: pd.DataFrame,
    text_column: str = "clean_text",
    label_column: str = "label",
    max_features: int = 10000,
) -> Pipeline:
    """Train and return the ML classification pipeline."""
    model = build_ml_pipeline(max_features=max_features)
    model.fit(train_df[text_column], train_df[label_column])
    return model


def evaluate_ml_model(
    model: Pipeline,
    test_df: pd.DataFrame,
    text_column: str = "clean_text",
    label_column: str = "label",
) -> dict:
    """Evaluate the classifier and return accuracy plus a text report."""
    predictions = model.predict(test_df[text_column])
    return {
        "accuracy": accuracy_score(test_df[label_column], predictions),
        "classification_report": classification_report(test_df[label_column], predictions),
    }


def predict_category(model: Pipeline, text: str) -> tuple[int, str]:
    """Predict the numeric label and readable category for one news item."""
    cleaned = clean_text(text)
    label = int(model.predict([cleaned])[0])
    return label, LABEL_MAP[label]


def save_model(model: Pipeline, path: str = "ml_model.joblib") -> None:
    """Persist the trained scikit-learn pipeline."""
    joblib.dump(model, path)


def load_model(path: str = "ml_model.joblib") -> Pipeline:
    """Load a saved scikit-learn pipeline."""
    return joblib.load(path)

