"""
api/v1/routes/analyze.py — POST /analyze endpoint.

Primary entry point for Android clients and any HTTP consumer that wants
to check whether a message is a scam.

URL: POST /api/v1/analyze
"""

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from app.models.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.detection_service import DetectionService

router = APIRouter()

# The ML model is loaded once here when the module is first imported.
# Keeping the service at module level avoids reloading the .joblib on every request.
_service = DetectionService()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Check a message for scam content",
    description=(
        "Accepts a raw text message and returns:\n"
        "- **scam_probability** — float in [0, 1]; higher means more likely a scam\n"
        "- **label** — `\"scam\"` or `\"safe\"`\n"
        "- **suspicious_keywords** — high-signal words/phrases found in the message"
    ),
    tags=["Detection"],
)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Classify a single message for scam likelihood.

    **Request body**
    ```json
    { "message": "URGENT: Share OTP 482716 to unlock your bank account." }
    ```

    **Response**
    ```json
    {
      "scam_probability": 0.97,
      "label": "scam",
      "suspicious_keywords": ["urgent", "share otp", "otp", "bank account"]
    }
    ```

    The endpoint is intentionally stateless — no message text is stored.
    """
    logger.info(
        "POST /analyze | message_length={} preview={!r}",
        len(request.message),
        # Log only first 60 chars to avoid leaking sensitive content to log files
        request.message[:60],
    )

    try:
        result = _service.analyze(request.message)
    except ValueError as exc:
        # Raised when input fails a downstream validation check
        logger.warning("Validation error in analyze: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        # Catch-all so an unexpected ML failure doesn't leak a 500 stack trace
        logger.exception("Unexpected error in analyze: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while analysing the message.",
        )

    logger.info(
        "Analysis complete | label={} probability={:.3f} keywords={}",
        result.label,
        result.scam_probability,
        result.suspicious_keywords,
    )

    return result
