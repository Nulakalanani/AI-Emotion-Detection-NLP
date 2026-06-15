"""
predict.py
----------
Inference module for the Emotion Detection project.

Loads the serialised Logistic Regression model and TF-IDF vectorizer
from the `models/` directory and exposes a `predict_emotion()` function
for real-time text classification.

Usage
-----
    from src.predict import predict_emotion

    print(predict_emotion("I got selected for an internship!"))
    # → joy
"""

import os
import pickle
from pathlib import Path

from src.preprocess import preprocess_text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent.parent  # project root
_MODELS_DIR = _BASE_DIR / "models"
_MODEL_PATH = _MODELS_DIR / "emotion_model.pkl"
_TFIDF_PATH = _MODELS_DIR / "tfidf_vectorizer.pkl"

# ---------------------------------------------------------------------------
# Emotion label map (matches training label encoding)
# ---------------------------------------------------------------------------
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}


def _load_artifacts():
    """Load model and vectorizer from disk. Raises FileNotFoundError if missing."""
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at '{_MODEL_PATH}'.\n"
            "Please run the notebook (emotion_detection.ipynb) first to train and save the model."
        )
    if not _TFIDF_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer file not found at '{_TFIDF_PATH}'.\n"
            "Please run the notebook (emotion_detection.ipynb) first to train and save the vectorizer."
        )

    with open(_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(_TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)

    return model, tfidf


# Lazy-load at first call to avoid overhead on import
_model = None
_tfidf = None


def predict_emotion(text: str) -> str:
    """
    Predict the emotion expressed in the input text.

    Parameters
    ----------
    text : str
        Raw user input text.

    Returns
    -------
    str
        Predicted emotion label: one of
        'sadness', 'joy', 'love', 'anger', 'fear', 'surprise'.

    Examples
    --------
    >>> predict_emotion("I got selected for an internship!")
    'joy'
    >>> predict_emotion("I lost my wallet")
    'sadness'
    """
    global _model, _tfidf
    if _model is None or _tfidf is None:
        _model, _tfidf = _load_artifacts()

    cleaned = preprocess_text(text)
    vector = _tfidf.transform([cleaned])
    prediction = _model.predict(vector)[0]
    return EMOTION_LABELS[int(prediction)]


def predict_with_confidence(text: str) -> dict:
    """
    Return emotion prediction along with probability scores for all classes.

    Parameters
    ----------
    text : str
        Raw user input text.

    Returns
    -------
    dict
        {
            'predicted_emotion': str,
            'confidence': float,          # probability of top prediction
            'all_scores': dict[str, float]
        }
    """
    global _model, _tfidf
    if _model is None or _tfidf is None:
        _model, _tfidf = _load_artifacts()

    cleaned = preprocess_text(text)
    vector = _tfidf.transform([cleaned])
    probabilities = _model.predict_proba(vector)[0]
    predicted_idx = probabilities.argmax()

    return {
        "predicted_emotion": EMOTION_LABELS[int(predicted_idx)],
        "confidence": round(float(probabilities[predicted_idx]), 4),
        "all_scores": {
            EMOTION_LABELS[i]: round(float(p), 4)
            for i, p in enumerate(probabilities)
        },
    }


# ---------------------------------------------------------------------------
# Quick CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_sentences = [
        "I got selected for an internship! I'm so thrilled!",
        "I lost my wallet and I feel terrible.",
        "I am absolutely terrified about the exam results.",
        "I love spending time with my family.",
        "I cannot believe he said that — I'm furious!",
        "What just happened? That was completely unexpected!",
    ]

    print("=" * 60)
    print("  Emotion Detection — Inference Demo")
    print("=" * 60)

    for sentence in test_sentences:
        result = predict_with_confidence(sentence)
        print(f"\nInput      : {sentence}")
        print(f"Prediction : {result['predicted_emotion'].upper()} "
              f"(confidence: {result['confidence']*100:.1f}%)")
        scores = "  |  ".join(
            f"{k}: {v:.2f}" for k, v in result["all_scores"].items()
        )
        print(f"All scores : {scores}")

    print("\n" + "=" * 60)
