"""
models/ml/scam_classifier.py — ScamClassifier: TF-IDF + Logistic Regression.

Architecture overview:
  ┌────────────────────────────────────────────────────────┐
  │  raw text                                              │
  │     ↓  scam_preprocessor.preprocess()                 │
  │  cleaned + tokenised text                              │
  │     ↓  TfidfVectorizer (unigrams + bigrams)            │
  │  sparse feature matrix                                 │
  │     ↓  LogisticRegression (class_weight='balanced')    │
  │  P(scam) probability                                   │
  │     ↓  threshold comparison                            │
  │  label ("scam" / "safe") + confidence %               │
  └────────────────────────────────────────────────────────┘

Design decisions:
  - TF-IDF sublinear_tf=True  → log-normalises term frequency so a word
    repeated 100 times doesn't drown out other features.
  - ngram_range=(1, 2)         → bigrams capture phrases like "verify account",
    "share otp", "click link", which are strong scam signals.
  - class_weight='balanced'    → automatically up-weights the minority scam
    class so the model doesn't just learn "everything is safe".
  - C=3.0                      → moderate regularisation; tighter than the
    default (1.0) to allow the model to be more decisive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from app.utils.scam_preprocessor import preprocess

# Where the trained pipeline is persisted on disk
MODEL_PATH: Path = Path(__file__).parent / "scam_classifier.joblib"

# Default confidence threshold — anything above this is labelled "scam"
DEFAULT_THRESHOLD: float = 0.55


def _build_pipeline() -> Pipeline:
    """
    Construct a fresh, untrained sklearn Pipeline.

    Kept as a module-level function so both ScamClassifier.train() and the
    training script can call it without instantiating the class.
    """
    tfidf = TfidfVectorizer(
        # Unigrams + bigrams: "share otp", "click link", "verify account"
        ngram_range=(1, 2),
        # Caps vocabulary size; 40k covers all meaningful n-grams in SMS data
        max_features=40_000,
        # log(1 + tf) — prevents high-frequency words dominating the vector
        sublinear_tf=True,
        # Normalise accented chars (common in unicode-trick scam messages)
        strip_accents="unicode",
        # Only consider tokens of 2+ characters
        token_pattern=r"\w{2,}",
        # Ignore terms that appear in fewer than 2 documents (noise)
        min_df=2,
        # Ignore terms that appear in more than 95% of documents (stop-words)
        max_df=0.95,
    )

    clf = LogisticRegression(
        # C=3.0 → slightly less regularisation than default; lets the model
        # be more aggressive about strong scam-keyword signals
        C=3.0,
        max_iter=1000,
        solver="lbfgs",
        # Automatically handles class imbalance (fewer scam than safe msgs)
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


class ScamClassifier:
    """
    Thin wrapper around the sklearn pipeline.

    Lifecycle:
      1. Training:    ScamClassifier().train(texts, labels).save()
      2. Inference:   ScamClassifier.from_disk().predict(text)
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self._pipeline: Optional[Pipeline] = None
        self._threshold = threshold

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_disk(
        cls,
        path: Path = MODEL_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> "ScamClassifier":
        """Load a previously trained classifier from disk."""
        instance = cls(threshold=threshold)
        instance.load(path)
        return instance

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, texts: list[str], labels: list[int]) -> "ScamClassifier":
        """
        Fit the TF-IDF + LR pipeline on pre-processed text.

        Parameters
        ----------
        texts  : list of cleaned strings (run through scam_preprocessor first)
        labels : list of ints — 0 = safe, 1 = scam
        """
        self._pipeline = _build_pipeline()
        self._pipeline.fit(texts, labels)
        return self     # enables chaining: classifier.train(X, y).save()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, raw_text: str) -> dict:
        """
        Classify a single raw message.

        Returns
        -------
        dict with keys:
          label      — "scam" or "safe"
          confidence — P(scam) expressed as a percentage (0–100)
        """
        if self._pipeline is None:
            raise RuntimeError("Model is not trained or loaded. Call train() or from_disk() first.")

        # Preprocessing mirrors what the training script applied to training data
        cleaned = preprocess(raw_text)

        # predict_proba returns [[P(safe), P(scam)]] — take the scam column
        proba = self._pipeline.predict_proba([cleaned])[0]
        scam_prob = float(proba[1])

        label = "scam" if scam_prob >= self._threshold else "safe"
        confidence = round(scam_prob * 100, 2)

        return {"label": label, "confidence": confidence}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = MODEL_PATH) -> None:
        """Persist the trained pipeline to disk with joblib."""
        if self._pipeline is None:
            raise RuntimeError("Nothing to save — model has not been trained.")
        joblib.dump(self._pipeline, path)

    def load(self, path: Path = MODEL_PATH) -> None:
        """Load a pipeline from disk. Raises FileNotFoundError if absent."""
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model found at {path}. "
                "Run scripts/train_model.py to train and save one."
            )
        self._pipeline = joblib.load(path)

    # ------------------------------------------------------------------
    # Introspection (useful for debugging and tests)
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError("Threshold must be in (0, 1)")
        self._threshold = value
