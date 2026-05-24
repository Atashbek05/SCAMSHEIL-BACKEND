"""
services/detection_service.py — Scam detection business logic.

Single integration point between the FastAPI routes and the ML classifier.

Pipeline (POST /api/v1/analyze):
  raw message
    → preprocess()             token replacement + cleaning (once)
    → predict_preprocessed()   TF-IDF → LogisticRegression → P(scam)
    → _find_keywords()         curated keyword scan on the raw message
    → AnalyzeResponse          label, risk%, scam_probability, keywords, analyzed_text

The model is loaded once at import time and reused across all requests.
If the .joblib file is absent the service degrades gracefully (label="unknown")
so the API stays up while the model is being trained.
"""

from __future__ import annotations

import re
from typing import List

from loguru import logger

from app.core.config import settings
from app.models.ml.scam_classifier import ScamClassifier, MODEL_PATH
from app.models.schemas.detection import DetectionRequest, DetectionResult, ScamType
from app.utils.scam_preprocessor import preprocess
from app.services import openai_analyzer


# ---------------------------------------------------------------------------
# Scam keyword vocabulary
#
# Scanned against the raw (un-preprocessed) message text so keywords survive
# even if tokenisation collapses them.  Ordered longest-first so multi-word
# phrases ("share otp") match before their sub-phrases ("otp").
# ---------------------------------------------------------------------------

_SCAM_KEYWORDS: List[str] = [
    # OTP / credential theft
    "one time password",
    "authentication code",
    "verification code",
    "share otp",
    "send otp",
    "enter otp",
    "verify otp",
    "share code",
    "send code",
    "personal details",
    "otp",
    "cvv",
    "pin",
    # Urgency / pressure tactics
    "expires soon",
    "last chance",
    "final notice",
    "limited time",
    "act now",
    "urgent",
    "immediately",
    "deadline",
    "expire",
    "suspended",
    "blocked",
    "restricted",
    "deactivated",
    # Account / banking scams
    "account suspended",
    "account blocked",
    "verify your account",
    "verify account",
    "kyc update",
    "unauthorized transaction",
    "transaction failed",
    "net banking",
    "online banking",
    "kyc",
    # Prize / lottery
    "lucky winner",
    "lucky draw",
    "claim your",
    "claim now",
    "free money",
    "free gift",
    "congratulations",
    "lottery",
    "winner",
    "prize",
    "reward",
    "won",
    # Government impersonation
    "government notice",
    "income tax",
    "it department",
    "tds refund",
    "gst refund",
    "pan card",
    "aadhar",
    "uidai",
    "epfo",
    # Link / download pressure
    "download now",
    "install app",
    "update app",
    "click here",
    "click link",
    "click below",
    "verify now",
    "confirm now",
    "activate now",
    # Personal information requests
    "provide your",
    "submit your",
    "share your",
    "send your",
    "credentials",
    "password",
    # Investment / crypto scams
    "guaranteed profit",
    "guaranteed return",
    "double your money",
    "trading platform",
    "invest now",
    "bitcoin",
    "crypto",
    # Refund scams
    "money back",
    "compensation",
    "cashback",
    "refund",
]


def _find_keywords(text: str) -> List[str]:
    """
    Return entries from _SCAM_KEYWORDS that appear in *text* (case-insensitive).

    Uses word-boundary anchors so short keywords like "pin" don't fire inside
    words like "opinion".  Each keyword is reported at most once.
    """
    t = text.lower()
    found: List[str] = []
    for kw in _SCAM_KEYWORDS:  # already sorted longest-first in the list above
        pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
        if re.search(pattern, t):
            found.append(kw)
    return found


class DetectionService:
    """
    Orchestrates the full scam-detection pipeline.

    Loaded once at module import; reused for every request.
    Degrades gracefully when the model file is missing.
    """

    def __init__(self) -> None:
        self._model: ScamClassifier | None = None
        self._load_model()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the trained classifier from disk, using the configured threshold."""
        try:
            self._model = ScamClassifier.from_disk(
                MODEL_PATH,
                # Threshold from config (default 0.75): probability above which
                # the message is labelled "scam".  Higher = fewer false positives.
                threshold=settings.MODEL_CONFIDENCE_THRESHOLD,
            )
            logger.info(
                "ScamClassifier loaded from {} (threshold={:.2f})",
                MODEL_PATH,
                settings.MODEL_CONFIDENCE_THRESHOLD,
            )
        except FileNotFoundError:
            logger.warning(
                "No trained model found at {}. "
                "Run `python scripts/train_model.py` to train one. "
                "API will return label='unknown' until then.",
                MODEL_PATH,
            )
            self._model = None

    # ------------------------------------------------------------------
    # Public API — POST /api/v1/analyze  (Android-facing)
    # ------------------------------------------------------------------

    def analyze(self, message: str) -> "AnalyzeResponse":  # type: ignore[name-defined]
        """
        Full pipeline for a single raw message.

        1. Try ChatGPT (openai_analyzer) — use its result if available.
        2. On any OpenAI failure, fall back to TF-IDF → LogisticRegression.
        3. Keyword scan always runs against the raw message for transparency.
        4. Response includes `source` field: "chatgpt" or "local_ml".
        """
        from app.models.schemas.analyze import AnalyzeResponse

        keywords = _find_keywords(message)
        analyzed_text = preprocess(message)

        # --- Try ChatGPT first ---
        gpt_result = openai_analyzer.analyze(message)
        if gpt_result is not None:
            probability = float(gpt_result["scam_probability"])
            # Merge keywords: prefer ChatGPT's list, supplement with local scan
            gpt_keywords = gpt_result.get("suspicious_keywords") or []
            merged_keywords = gpt_keywords if gpt_keywords else keywords
            return AnalyzeResponse(
                label=gpt_result["label"],
                risk=round(probability * 100, 2),
                scam_probability=probability,
                suspicious_keywords=merged_keywords,
                analyzed_text=analyzed_text,
                source="chatgpt",
            )

        # --- Fallback: local ML model ---
        logger.info("ChatGPT unavailable — using local ML model")

        if self._model is None:
            return AnalyzeResponse(
                label="unknown",
                risk=0.0,
                scam_probability=0.0,
                suspicious_keywords=[],
                analyzed_text=analyzed_text,
                source="local_ml",
            )

        raw = self._model.predict_preprocessed(analyzed_text)
        risk = raw["confidence"]
        scam_probability = risk / 100.0

        return AnalyzeResponse(
            label=raw["label"],
            risk=risk,
            scam_probability=scam_probability,
            suspicious_keywords=keywords,
            analyzed_text=analyzed_text,
            source="local_ml",
        )

    # ------------------------------------------------------------------
    # Public API — legacy /analyse endpoint (DetectionRequest schema)
    # ------------------------------------------------------------------

    def analyse(self, request: DetectionRequest) -> DetectionResult:
        """
        Classify a single message and return a DetectionResult.
        Uses the same preprocessing-once pattern as analyze() to avoid
        running the tokeniser twice.
        """
        if self._model is None:
            return self._model_unavailable_response()

        analyzed_text = preprocess(request.text)
        raw = self._model.predict_preprocessed(analyzed_text)
        return self._postprocess(raw, request.text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _postprocess(self, raw: dict, original_text: str) -> DetectionResult:
        """Map the classifier's raw dict to the DetectionResult Pydantic schema."""
        is_scam = raw["label"] == "scam"
        confidence = raw["confidence"] / 100.0  # convert % → [0, 1]

        scam_type = (
            self._infer_scam_type(original_text) if is_scam else ScamType.UNKNOWN
        )

        return DetectionResult(
            is_scam=is_scam,
            confidence=confidence,
            scam_type=scam_type,
        )

    def _infer_scam_type(self, text: str) -> ScamType:
        """
        Lightweight keyword heuristic to categorise the kind of scam.
        Replaced by a multi-class model in a future sprint.
        """
        t = text.lower()

        if any(k in t for k in ("otp", "pin", "share code", "send code", "cvv")):
            return ScamType.PHISHING

        if any(k in t for k in ("call", "phone", "speak", "press 1", "agent")):
            return ScamType.VISHING

        if any(k in t for k in ("won", "prize", "lottery", "reward", "lucky")):
            return ScamType.LOTTERY

        if any(k in t for k in ("bank", "account", "credit", "debit", "kyc", "rbi")):
            return ScamType.PHISHING

        if any(k in t for k in ("invest", "profit", "return", "trading", "crypto")):
            return ScamType.INVESTMENT

        if any(k in t for k in ("sms", "text", "mobile")):
            return ScamType.SMISHING

        return ScamType.PHISHING  # credential phishing is the most common catch-all

    def _model_unavailable_response(self) -> DetectionResult:
        return DetectionResult(
            is_scam=False,
            confidence=0.0,
            scam_type=ScamType.UNKNOWN,
            explanation="Model not trained. Run scripts/train_model.py first.",
        )
