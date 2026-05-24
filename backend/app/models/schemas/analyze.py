"""
models/schemas/analyze.py — Request and response shapes for POST /api/v1/analyze.

Kept separate from detection.py so each endpoint owns its own contract.
Android clients send AnalyzeRequest; the API replies with AnalyzeResponse.

Response field guide
--------------------
label            : "scam" | "safe" | "unknown" (unknown = model not trained yet)
risk             : 0–100 float — the headline number shown to the user
scam_probability : 0.0–1.0 float — same value, kept for ML consumers and dashboards
analyzed_text    : what the model actually saw (tokens replace URLs, OTPs, amounts, …)
suspicious_keywords : human-readable signals extracted from the raw message
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Payload the Android client (or any HTTP consumer) sends for a scam check."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description=(
            "Raw text to analyse — SMS body, notification, chat message, email snippet, etc. "
            "The API accepts any plain-text string; pre-processing is handled server-side."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": (
                        "URGENT: Your SBI account is suspended. "
                        "Share OTP 482716 with our agent to restore access."
                    )
                }
            ]
        }
    }


class AnalyzeResponse(BaseModel):
    """What the API returns after running the scam classifier."""

    label: str = Field(
        ...,
        description=(
            'Classification result: "scam", "safe", or "unknown". '
            '"unknown" is only returned before the model has been trained.'
        ),
    )
    risk: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description=(
            "Scam risk score as a percentage (0–100). "
            "Values above the server-configured threshold (default 75) produce label='scam'. "
            "Use this as the primary signal in Android UI."
        ),
    )
    scam_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Same as risk / 100 — kept for ML consumers and analytics dashboards "
            "that prefer a probability in [0, 1]."
        ),
    )
    suspicious_keywords: List[str] = Field(
        default_factory=list,
        description=(
            "High-signal scam words and phrases found in the original message. "
            "Useful for explaining the decision to the end user."
        ),
    )
    analyzed_text: str = Field(
        ...,
        description=(
            "The preprocessed version of the input the model actually classified. "
            "URLs are replaced with URLTOKEN, OTPs with OTPTOKEN, phone numbers with PHONETOKEN, "
            "monetary amounts with AMOUNTTOKEN. Useful for debugging and transparency."
        ),
    )
    source: str = Field(
        default="local_ml",
        description=(
            'Which backend produced the result: "chatgpt" (OpenAI GPT-4o-mini) '
            'or "local_ml" (TF-IDF + Logistic Regression fallback).'
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "label": "scam",
                    "risk": 97.4,
                    "scam_probability": 0.974,
                    "suspicious_keywords": ["urgent", "share otp", "otp", "suspended"],
                    "analyzed_text": (
                        "urgent your sbi account is suspended "
                        "share otptoken with our agent to restore access"
                    ),
                    "source": "chatgpt",
                }
            ]
        }
    }
