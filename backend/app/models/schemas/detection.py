"""
models/schemas/detection.py — Pydantic request/response schemas for scam detection.

These are the data shapes the API accepts and returns.
Actual AI model implementation goes in services/detection_service.py.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ScamType(str, Enum):
    PHISHING = "phishing"
    SMISHING = "smishing"       # SMS-based phishing
    VISHING = "vishing"         # voice/phone phishing
    ADVANCE_FEE = "advance_fee"
    ROMANCE = "romance"
    INVESTMENT = "investment"
    LOTTERY = "lottery"
    UNKNOWN = "unknown"


class DetectionRequest(BaseModel):
    """Payload the client sends for a scam-check."""
    text: str = Field(..., min_length=1, max_length=10_000, description="Text to analyse")
    source: Optional[str] = Field(None, description="Origin channel (email, sms, call, …)")


class DetectionResult(BaseModel):
    """What the API returns after running the scam classifier."""
    is_scam: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    scam_type: ScamType
    explanation: Optional[str] = None   # human-readable reason (future feature)
