"""
models/schemas/analyze.py — Request and response shapes for POST /analyze.

Kept separate from detection.py so each endpoint owns its own contract.
Android clients send AnalyzeRequest; the API replies with AnalyzeResponse.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    """Payload the Android client (or any HTTP client) sends for a scam check."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Raw text to analyse (SMS body, notification text, chat message, etc.)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "URGENT: Your bank account is suspended. Share OTP 482716 to restore access."
                }
            ]
        }
    }


class AnalyzeResponse(BaseModel):
    """What the API returns after running the scam classifier."""

    scam_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability that the message is a scam, in [0, 1]. "
                    "1.0 = certain scam, 0.0 = certainly safe.",
    )
    label: str = Field(
        ...,
        description='Classification result — "scam" or "safe".',
    )
    suspicious_keywords: List[str] = Field(
        default_factory=list,
        description="High-signal scam words/phrases found in the message.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "scam_probability": 0.94,
                    "label": "scam",
                    "suspicious_keywords": ["otp", "urgent", "suspended", "share"],
                }
            ]
        }
    }
