from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.config import config
from app.db import fetch_annotated_tweets


MODEL_PATH = Path(config.model_dir) / "sentiment_model.joblib"


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_model() -> dict:
    rows = fetch_annotated_tweets()
    if len(rows) < 4:
        raise ValueError("At least 4 annotated tweets are required in MySQL to train the model.")

    texts = [row.text for row in rows]
    positive_labels = [row.positive for row in rows]
    negative_labels = [row.negative for row in rows]

    positive_model = _build_pipeline()
    negative_model = _build_pipeline()

    stratify = positive_labels if len(set(positive_labels)) > 1 else None
    split = train_test_split(
        texts,
        positive_labels,
        negative_labels,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )
    x_train, x_valid, y_pos_train, y_pos_valid, y_neg_train, y_neg_valid = split

    positive_model.fit(x_train, y_pos_train)
    negative_model.fit(x_train, y_neg_train)

    metrics = {
        "positive": _evaluate(positive_model, x_valid, y_pos_valid),
        "negative": _evaluate(negative_model, x_valid, y_neg_valid),
        "training_size": len(x_train),
        "validation_size": len(x_valid),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "positive_model": positive_model,
            "negative_model": negative_model,
            "metrics": metrics,
        },
        MODEL_PATH,
    )
    return metrics


def _evaluate(model: Pipeline, texts: list[str], labels: list[int]) -> dict:
    predictions = model.predict(texts)
    return {
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        ),
    }


def load_model() -> dict:
    if not MODEL_PATH.exists():
        train_model()
    return joblib.load(MODEL_PATH)


def predict_scores(tweets: list[str]) -> dict[str, float]:
    bundle = load_model()
    positive_model = bundle["positive_model"]
    negative_model = bundle["negative_model"]

    positive_scores = positive_model.predict_proba(tweets)[:, 1]
    negative_scores = negative_model.predict_proba(tweets)[:, 1]

    results = {}
    for tweet, positive_score, negative_score in zip(tweets, positive_scores, negative_scores):
        score = float(positive_score - negative_score)
        results[tweet] = round(max(-1.0, min(1.0, score)), 4)
    return results
