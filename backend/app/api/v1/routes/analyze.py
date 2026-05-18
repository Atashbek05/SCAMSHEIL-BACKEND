"""
api/v1/routes/analyze.py — POST /api/v1/analyze

Primary scam-detection endpoint consumed by Android clients and any HTTP caller.

── Example curl request ──────────────────────────────────────────────────────

  # Check a suspicious message
  curl -s -X POST https://<your-host>/api/v1/analyze \
       -H "Content-Type: application/json" \
       -d '{"message": "URGENT: Share OTP 482716 to unlock your SBI account."}' \
       | python -m json.tool

  # Safe message
  curl -s -X POST https://<your-host>/api/v1/analyze \
       -H "Content-Type: application/json" \
       -d '{"message": "Hey, are we still meeting for lunch tomorrow?"}' \
       | python -m json.tool

  # Local dev
  curl -s -X POST http://localhost:8000/api/v1/analyze \
       -H "Content-Type: application/json" \
       -d '{"message": "Congratulations! You won Rs 50,000. Claim: http://prize.tk"}'

── Example responses ─────────────────────────────────────────────────────────

  Scam detected:
  {
    "label": "scam",
    "risk": 97.4,
    "scam_probability": 0.974,
    "suspicious_keywords": ["urgent", "share otp", "otp", "suspended"],
    "analyzed_text": "urgent share otptoken to unlock your sbi account"
  }

  Safe message:
  {
    "label": "safe",
    "risk": 3.1,
    "scam_probability": 0.031,
    "suspicious_keywords": [],
    "analyzed_text": "hey are we still meeting for lunch tomorrow"
  }

── Android integration notes ─────────────────────────────────────────────────

  • Android's OkHttp/Retrofit does NOT send an Origin header for native requests
    so CORS restrictions do not apply to native Android clients.
  • Use `risk` as the primary integer shown to the user (0–100 scale).
  • Show `suspicious_keywords` as chips/tags to explain the decision.
  • Store nothing — this endpoint is intentionally stateless.
"""

from fastapi import APIRouter, HTTPException, Request, status
from loguru import logger

from app.models.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.detection_service import DetectionService

router = APIRouter()

# ---------------------------------------------------------------------------
# Service singleton — the .joblib model is loaded once at import time and
# reused across all requests.  Loading per-request would add ~200 ms latency.
# ---------------------------------------------------------------------------
_service = DetectionService()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyse a text message for scam content",
    description=(
        "Classifies a raw text message and returns a structured risk assessment.\n\n"
        "**Response fields**\n"
        "- `label` — `\"scam\"` | `\"safe\"` | `\"unknown\"` (unknown = model not trained)\n"
        "- `risk` — 0–100 float; the primary signal for mobile UI\n"
        "- `scam_probability` — same as `risk / 100`, for analytics consumers\n"
        "- `suspicious_keywords` — human-readable signals found in the message\n"
        "- `analyzed_text` — preprocessed version the model actually classified\n\n"
        "The endpoint is **stateless** — no message content is stored or logged in full."
    ),
    responses={
        200: {
            "description": "Analysis complete",
            "content": {
                "application/json": {
                    "example": {
                        "label": "scam",
                        "risk": 97.4,
                        "scam_probability": 0.974,
                        "suspicious_keywords": ["urgent", "share otp", "otp"],
                        "analyzed_text": "urgent share otptoken to unlock your sbi account",
                    }
                }
            },
        },
        422: {"description": "Validation error — message missing or exceeds 10 000 chars"},
        500: {"description": "Internal server error during ML inference"},
    },
    tags=["Detection"],
)
def analyze(request: AnalyzeRequest, http_request: Request) -> AnalyzeResponse:
    """
    Classify a single text message for scam likelihood.

    **Scam types detected**
    - OTP / credential theft
    - Fake banking alerts and account-suspension threats
    - Phishing links (URL patterns replaced with URLTOKEN for model input)
    - Prize, lottery, and government-impersonation scams
    - Social engineering using urgency and fear
    - Fake job offers and investment fraud

    **How the risk score works**

    The backend runs a TF-IDF → Logistic Regression pipeline trained on the
    UCI SMS Spam Collection plus domain-specific South Asian scam examples.
    The pipeline outputs P(scam) which is scaled to 0–100 and returned as
    `risk`.  Messages with `risk` above the server's configured threshold
    (default 75) receive `label = "scam"`.

    **Privacy**

    Only the first 60 characters of the message are written to the server log
    (for debugging).  The full text is never persisted.
    """
    # Log truncated preview only — avoid writing sensitive message content to disk
    logger.info(
        "POST /analyze | client={} length={} preview={!r}",
        http_request.client.host if http_request.client else "unknown",
        len(request.message),
        request.message[:60],
    )

    try:
        result = _service.analyze(request.message)

    except ValueError as exc:
        # Raised by downstream validation (e.g. empty cleaned text after preprocessing)
        logger.warning("Validation error in /analyze: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    except Exception as exc:
        # Catch-all: ML inference failures must not leak stack traces to clients
        logger.exception("Unexpected error in /analyze: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while analysing the message.",
        )

    logger.info(
        "Analysis complete | label={} risk={:.1f} keywords={}",
        result.label,
        result.risk,
        result.suspicious_keywords,
    )

    return result
