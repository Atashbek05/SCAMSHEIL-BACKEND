"""
utils/validators.py — Input validation helpers beyond Pydantic field constraints.

Centralise any non-trivial validation logic here so route handlers stay thin.
"""

import re


def is_valid_phone_number(value: str) -> bool:
    """Basic E.164 format check — tighten with a library (phonenumbers) later."""
    return bool(re.match(r"^\+?[1-9]\d{6,14}$", value.strip()))


def is_suspicious_url(url: str) -> bool:
    """
    Lightweight heuristic for obviously suspicious URLs.
    Replace with a proper URL reputation service in production.
    """
    suspicious_patterns = [
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # raw IP address
        r"bit\.ly|tinyurl|t\.co",                  # URL shorteners
        r"[a-z0-9]{20,}\.",                         # very long random subdomains
    ]
    return any(re.search(p, url, re.IGNORECASE) for p in suspicious_patterns)
