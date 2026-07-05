from __future__ import annotations

from flask import Flask, jsonify, request

from app.model import predict_scores, train_model


app = Flask(__name__)


def _extract_tweets(payload) -> tuple[list[str] | None, str | None]:
    if isinstance(payload, list):
        tweets = payload
    elif isinstance(payload, dict):
        tweets = payload.get("tweets")
    else:
        return None, "Request body must be a JSON list or an object with a 'tweets' list."

    if not isinstance(tweets, list):
        return None, "'tweets' must be a list."
    if not tweets:
        return None, "'tweets' cannot be empty."
    if not all(isinstance(tweet, str) for tweet in tweets):
        return None, "Every tweet must be a string."

    cleaned = [tweet.strip() for tweet in tweets]
    if any(not tweet for tweet in cleaned):
        return None, "Tweets cannot be empty strings."
    return cleaned, None


@app.post("/analyze")
@app.post("/sentiment")
def analyze_sentiments():
    payload = request.get_json(silent=True)
    tweets, error = _extract_tweets(payload)
    if error:
        return jsonify({"error": error}), 400

    return jsonify(predict_scores(tweets))


@app.post("/retrain")
def retrain():
    try:
        metrics = train_model()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify(metrics)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})
