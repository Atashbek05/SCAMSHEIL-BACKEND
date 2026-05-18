"""
utils/scam_preprocessor.py — Scam-aware token replacement pipeline.

Runs BEFORE generic text cleaning so high-signal patterns are preserved
as named tokens (URLTOKEN, OTPTOKEN, …) rather than stripped away.
TF-IDF then learns these tokens as strong scam features.

Call order:
  raw text → scam_preprocessor.preprocess() → text_cleaner.clean() → model
"""

import re
from app.utils.text_cleaner import clean


# ---------------------------------------------------------------------------
# Pattern → replacement token mapping
# ---------------------------------------------------------------------------

# Matches http/https URLs and bare www. domains
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Phone numbers — covers South Asian (5+5), global (3+4+4), and compact formats.
# Structure: optional country code, then 8–10 local digits with optional separators.
# Examples matched: +91 98765 43210 | 9876543210 | +1 555-123-4567
_PHONE_RE = re.compile(
    r"(?<!\w)"                          # not preceded by a word character
    r"(?:\+?\d{1,3}[\s\-.])"           # country code (+91 , +1-) — separator required
    r"?\d{4,5}[\s\-.]?\d{4,5}"         # 8–10 local digits, optional mid-separator
    r"(?!\d)"                           # not followed by more digits
)

# Monetary amounts: Rs., INR, $, ₹ followed by digits with optional commas
_AMOUNT_RE = re.compile(
    r"(rs\.?|inr|usd|\$|₹)\s*[\d,]+(\.\d{1,2})?",
    re.IGNORECASE,
)

# 4–8 digit codes used as OTPs / PINs
# Negative lookbehind/ahead prevents matching parts of longer numbers
_OTP_RE = re.compile(r"(?<!\d)\d{4,8}(?!\d)")

# Card / account numbers: 12–19 consecutive digits (with optional spaces)
_ACCOUNT_RE = re.compile(r"(?<!\d)(\d[\s]?){12,19}(?!\d)")

# ---------------------------------------------------------------------------
# Individual replacement functions (pure — no side effects)
# ---------------------------------------------------------------------------

def replace_urls(text: str) -> str:
    """Replace all URLs with URLTOKEN — scam messages heavily use short/fake URLs."""
    return _URL_RE.sub(" URLTOKEN ", text)


def replace_phones(text: str) -> str:
    """Replace phone numbers with PHONETOKEN."""
    return _PHONE_RE.sub(" PHONETOKEN ", text)


def replace_amounts(text: str) -> str:
    """Replace monetary amounts with AMOUNTTOKEN (prize/refund scams)."""
    return _AMOUNT_RE.sub(" AMOUNTTOKEN ", text)


def replace_accounts(text: str) -> str:
    """Replace card/account number patterns with ACCOUNTTOKEN."""
    return _ACCOUNT_RE.sub(" ACCOUNTTOKEN ", text)


def replace_otps(text: str) -> str:
    """
    Replace 4–8 digit sequences with OTPTOKEN.
    Run AFTER replace_accounts so long digit strings are already tokenised.
    Short digit codes are the most abused pattern in OTP-theft scams.
    """
    return _OTP_RE.sub(" OTPTOKEN ", text)


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def preprocess(text: str) -> str:
    """
    Full scam-aware preprocessing pipeline.

    Step 1 – replace high-signal patterns with named tokens.
    Step 2 – apply generic cleaning (lowercase, remove punctuation, etc.)

    This is the function DetectionService and the training script both call.
    """
    text = replace_urls(text)
    text = replace_phones(text)
    text = replace_amounts(text)
    text = replace_accounts(text)
    text = replace_otps(text)
    return clean(text)          # text_cleaner.clean() handles the rest
