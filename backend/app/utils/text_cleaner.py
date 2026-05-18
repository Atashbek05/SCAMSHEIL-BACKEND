"""
utils/text_cleaner.py — Generic, reusable text normalisation helpers.

These functions operate on plain text with no knowledge of scam patterns.
For scam-specific token replacement (URLs → URLTOKEN, etc.) see scam_preprocessor.py.

Call order when used together:
  scam_preprocessor.preprocess(text)   ← calls clean() internally at the end
"""

import re
import unicodedata


def remove_urls(text: str) -> str:
    """Remove raw HTTP/HTTPS URLs and bare www. links."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_special_chars(text: str) -> str:
    """
    Keep only alphanumeric characters and whitespace.
    Uppercase TOKEN placeholders (URLTOKEN, OTPTOKEN, …) survive because
    they are alphanumeric — this is intentional.
    """
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


def to_ascii(text: str) -> str:
    """
    Strip accents and non-ASCII characters.
    Handles common scam tricks like using lookalike unicode letters
    (e.g. Ꮯ instead of C, ṡ instead of s) to bypass keyword filters.
    """
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def clean(text: str) -> str:
    """
    Full cleaning pipeline used as the final step in scam_preprocessor.preprocess().
    Lowercase + ASCII + remove specials + normalise whitespace.
    """
    text = to_ascii(text)
    text = remove_special_chars(text)
    text = normalise_whitespace(text)
    return text.lower().strip()


def clean_for_display(text: str) -> str:
    """
    Lighter version of clean() that keeps punctuation — safe for showing
    processed text back to end users or in log output.
    """
    text = to_ascii(text)
    text = normalise_whitespace(text)
    return text.strip()
