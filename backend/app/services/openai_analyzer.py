"""
services/openai_analyzer.py — GPT-4o-mini scam analysis via ChatGPT API.

Sends the raw message to OpenAI and parses a structured JSON response.
Returns None on any error (timeout, auth failure, malformed JSON) so the
caller can fall back to the local ML model.
"""

from __future__ import annotations

import json
import os
from typing import TypedDict

from loguru import logger


class OpenAIResult(TypedDict):
    scam_probability: float
    label: str
    suspicious_keywords: list[str]
    reason: str


_SYSTEM_PROMPT = "You are a scam detection expert. Analyze the message and respond ONLY with JSON."

_USER_PROMPT_TEMPLATE = (
    "Analyze this message for scam indicators. Message: {text}\n"
    "Return JSON: {{\n"
    '  "scam_probability": float 0.0-1.0,\n'
    '  "label": "scam" or "safe",\n'
    '  "suspicious_keywords": [list of suspicious words found],\n'
    '  "reason": "brief explanation"\n'
    "}}"
)


def _get_client():
    """Build an OpenAI client using the CHATGPT env var as the API key."""
    import openai  # lazy import so missing package only fails at call time

    api_key = os.environ.get("CHATGPT")
    if not api_key:
        raise ValueError("CHATGPT environment variable is not set")
    return openai.OpenAI(api_key=api_key)


def analyze(text: str) -> OpenAIResult | None:
    """
    Send *text* to gpt-4o-mini and return structured scam analysis.

    Returns None if the API is unavailable, times out, or returns
    unparseable output — callers should treat None as "no signal".
    """
    try:
        client = _get_client()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(text=text)},
            ],
            temperature=0,
            timeout=10,
        )

        raw_content = response.choices[0].message.content or ""

        # Strip markdown code fences if the model wraps the JSON
        content = raw_content.strip()
        if content.startswith("```"):
            content = content.split("```", 2)[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.rstrip("`").strip()

        data = json.loads(content)

        return OpenAIResult(
            scam_probability=float(data["scam_probability"]),
            label=str(data["label"]),
            suspicious_keywords=list(data.get("suspicious_keywords", [])),
            reason=str(data.get("reason", "")),
        )

    except Exception as exc:
        logger.warning("OpenAI analyzer failed: {}", exc)
        return None
