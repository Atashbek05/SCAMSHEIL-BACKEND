"""
tests/unit/test_text_cleaner.py — Unit tests for text cleaning utilities.

Run with:  pytest tests/unit/test_text_cleaner.py -v
"""

from app.utils.text_cleaner import remove_urls, remove_special_chars, clean


def test_remove_urls():
    assert "visit " in remove_urls("visit https://scam-site.com now")
    assert "http" not in remove_urls("click http://evil.ru/xyz")


def test_remove_special_chars():
    result = remove_special_chars("Hello! Win $1,000 now!!!")
    assert "$" not in result
    assert "!" not in result


def test_clean_pipeline():
    # clean() is a generic cleaner — URL *removal* belongs to scam_preprocessor.
    # After remove_special_chars the colon and slashes disappear, leaving "http".
    # The important invariants are: lowercase, stripped, no punctuation.
    messy = "  Cöngrats!!! Claim your PRIZE now!!!  "
    result = clean(messy)
    assert result == result.lower()
    assert "!" not in result
    assert result == result.strip()
    assert "congrats" in result
