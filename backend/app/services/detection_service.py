"""
services/detection_service.py — Scam detection business logic.

This is the single integration point between the FastAPI routes and the
ML classifier. Routes call analyse() or analyze(); everything below stays private.

Pipeline:
  DetectionRequest.text / AnalyzeRequest.message
    → _preprocess()          (scam_preprocessor.preprocess)
    → _run_model()           (ScamClassifier.predict)
    → _postprocess()         (map raw dict → response schema)
    → DetectionResult / AnalyzeResponse
"""

from __future__ import annotations

import re
from typing import List

from loguru import logger

from app.models.ml.scam_classifier import ScamClassifier, MODEL_PATH
from app.models.schemas.detection import DetectionRequest, DetectionResult, ScamType
from app.utils.scam_preprocessor import preprocess


# ---------------------------------------------------------------------------
# Scam keyword vocabulary — checked against raw (un-preprocessed) message text.
# Ordered roughly by signal strength; all lowercase for case-insensitive matching.
# ---------------------------------------------------------------------------

_SCAM_KEYWORDS: List[str] = [
    # OTP / credential theft
    "otp",
    "one time password",
    "share otp",
    "send otp",
    "enter otp",
    "verify otp",
    "share code",
    "send code",
    "authentication code",
    "verification code",
    "cvv",
    "pin",
    # Urgency / pressure tactics
    "urgent",
    "immediately",
    "expire",
    "expires soon",
    "last chance",
    "final notice",
    "act now",
    "limited time",
    "deadline",
    "suspended",
    "blocked",
    "restricted",
    "deactivated",
    # Account / banking scams
    "kyc",
    "kyc update",
    "account suspended",
    "account blocked",
    "verify account",
    "verify your account",
    "net banking",
    "online banking",
    "transaction failed",
    "unauthorized transaction",
    # Prize / lottery
    "won",
    "winner",
    "lottery",
    "prize",
    "reward",
    "congratulations",
    "lucky draw",
    "lucky winner",
    "claim now",
    "claim your",
    "free gift",
    "free money",
    # Government impersonation
    "income tax",
    "it department",
    "epfo",
    "uidai",
    "aadhar",
    "pan card",
    "tds refund",
    "gst refund",
    "government notice",
    # Link / download pressure
    "click here",
    "click link",
    "click below",
    "download now",
    "install app",
    "update app",
    "verify now",
    "confirm now",
    "activate now",
    # Personal information requests
    "share your",
    "send your",
    "provide your",
    "submit your",
    "password",
    "credentials",
    "personal details",
    # Investment / crypto scams
    "invest now",
    "guaranteed profit",
    "guaranteed return",
    "double your money",
    "crypto",
    "bitcoin",
    "trading platform",
    # Refund scams
    "refund",
    "cashback",
    "money back",
    "compensation",
]

# Pre-compiled pattern for fast word-boundary matching across all keywords.
# Longer phrases are checked first via sorted order (longest first wins).
_SORTED_KEYWORDS = sorted(_SCAM_KEYWORDS, key=len, reverse=True)


def _find_keywords(text: str) -> List[str]:
    """
    Return which entries from _SCAM_KEYWORDS appear in *text* (case-insensitive).

    Matches on word boundaries so "pin" won't fire inside "opinion".
    Each keyword is reported at most once even if repeated in the text.
    """
    t = text.lower()
    found: List[str] = []
    for kw in _SORTED_KEYWORDS:
        # Use word-boundary anchors; wrap multi-word phrases in \\b only at edges
        pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
        if re.search(pattern, t):
            found.append(kw)
    return found


class DetectionService:
    """
    Orchestrates the full scam-detection pipeline.

    The classifier is loaded once on __init__ and reused across all requests.
    If the model file is absent (not yet trained), the service degrades
    gracefully and returns an explicit "model unavailable" response rather
    than crashing the entire API.
    """

    def __init__(self) -> None:
        self._model: ScamClassifier | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Try to load the trained classifier from disk at startup."""
        try:
            self._model = ScamClassifier.from_disk(MODEL_PATH)
            logger.info(f"ScamClassifier loaded from {MODEL_PATH}")
        except FileNotFoundError:
            logger.warning(
                "No trained model found at {path}. "
                "Run `python scripts/train_model.py` to train one. "
                "API will return placeholder results until then.",
                path=MODEL_PATH,
            )
            self._model = None

    # ------------------------------------------------------------------
    # Public API — /analyze endpoint (Android-facing)
    # ------------------------------------------------------------------

    def analyze(self, message: str) -> "AnalyzeResponse":  # type: ignore[name-defined]
        """
        Classify *message* and return scam_probability, label, suspicious_keywords.

        This is the method called by the POST /analyze route handler.
        Import is deferred to avoid a top-level circular import.
        """
        from app.models.schemas.analyze import AnalyzeResponse

        keywords = _find_keywords(message)

        if self._model is None:
            # Model not trained yet — return a safe-looking placeholder with no keywords
            return AnalyzeResponse(
                scam_probability=0.0,
                label="unknown",
                suspicious_keywords=[],
            )

        cleaned = self._preprocess(message)
        raw = self._run_model(cleaned)

        scam_probability = raw["confidence"] / 100.0  # convert % → [0, 1]
        label = raw["label"]

        return AnalyzeResponse(
            scam_probability=scam_probability,
            label=label,
            suspicious_keywords=keywords,
        )

    # ------------------------------------------------------------------
    # Public API — legacy /analyse endpoint (existing schema)
    # ------------------------------------------------------------------

    def analyse(self, request: DetectionRequest) -> DetectionResult:
        """
        Classify a single message and return a structured detection result.
        Called directly by the route handler.
        """
        if self._model is None:
            return self._model_unavailable_response()

        cleaned = self._preprocess(request.text)
        raw = self._run_model(cleaned)
        return self._postprocess(raw, request.text)

    # ------------------------------------------------------------------
    # Private pipeline steps
    # ------------------------------------------------------------------

    def _preprocess(self, text: str) -> str:
        """
        Apply scam-aware preprocessing: token replacement + cleaning.
        Mirrors exactly what was done to training data so the feature space matches.
        """
        return preprocess(text)

    def _run_model(self, cleaned_text: str) -> dict:
        """
        Call the classifier. Returns {"label": "scam"|"safe", "confidence": float}.
        The classifier already expects pre-processed text.
        """
        return self._model.predict(cleaned_text)  # type: ignore[union-attr]

    def _postprocess(self, raw: dict, original_text: str) -> DetectionResult:
        """
        Map the classifier's raw output to the Pydantic response schema.
        Scam type is inferred from keyword heuristics on the original text
        until a multi-class model is implemented.
        """
        is_scam = raw["label"] == "scam"
        confidence = raw["confidence"] / 100.0  # convert % → [0, 1] for schema

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
        Lightweight keyword heuristic to label the *kind* of scam.
        Replaced by a multi-class model in a future sprint.
        """
        t = text.lower()

        if any(k in t for k in ("otp", "pin", "share code", "send code")):
            return ScamType.PHISHING  # OTP-theft treated as credential phishing

        if any(k in t for k in ("sms", "text", "message", "mobile")):
            return ScamType.SMISHING

        if any(k in t for k in ("call", "phone", "speak", "press 1", "agent")):
            return ScamType.VISHING

        if any(k in t for k in ("won", "prize", "lottery", "reward", "lucky")):
            return ScamType.LOTTERY

        if any(k in t for k in ("bank", "account", "credit", "debit", "kyc", "rbi")):
            return ScamType.PHISHING

        if any(k in t for k in ("invest", "profit", "return", "trading", "crypto")):
            return ScamType.INVESTMENT

        return ScamType.PHISHING  # default — most scam messages are credential phishing

    # ------------------------------------------------------------------
    # Fallback when model is not trained yet
    # ------------------------------------------------------------------

    def _model_unavailable_response(self) -> DetectionResult:
        return DetectionResult(
            is_scam=False,
            confidence=0.0,
            scam_type=ScamType.UNKNOWN,
            explanation="Model not trained. Run scripts/train_model.py first.",
        )
