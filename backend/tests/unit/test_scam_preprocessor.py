"""
tests/unit/test_scam_preprocessor.py — Tests for scam token replacement.

Run with:  pytest tests/unit/test_scam_preprocessor.py -v
"""

from app.utils.scam_preprocessor import (
    replace_urls,
    replace_phones,
    replace_amounts,
    replace_otps,
    preprocess,
)


def test_replace_urls_http():
    result = replace_urls("Click here: https://evil-bank.tk/steal")
    assert "URLTOKEN" in result
    assert "evil-bank" not in result


def test_replace_urls_www():
    result = replace_urls("Visit www.scam-site.com now")
    assert "URLTOKEN" in result


def test_replace_phones():
    result = replace_phones("Call us at +91 98765 43210 immediately")
    assert "PHONETOKEN" in result
    assert "98765" not in result


def test_replace_amounts_rupee():
    result = replace_amounts("You won Rs. 50,000 today!")
    assert "AMOUNTTOKEN" in result
    assert "50,000" not in result


def test_replace_amounts_inr():
    result = replace_amounts("Transfer INR 10000 to claim prize")
    assert "AMOUNTTOKEN" in result


def test_replace_otps_six_digit():
    result = replace_otps("Your OTP is 847362 valid for 10 minutes")
    assert "OTPTOKEN" in result
    assert "847362" not in result


def test_replace_otps_does_not_match_short():
    # 3-digit numbers should not become OTPTOKEN
    result = replace_otps("Call 100 for emergency")
    assert "OTPTOKEN" not in result


def test_preprocess_otp_theft_scam():
    msg = "Share OTP 628374 with our agent to unlock your account at https://bank-secure.tk"
    result = preprocess(msg)
    # Tokens should be present in lowercase
    assert "urltoken" in result
    assert "otptoken" in result
    assert "628374" not in result
    assert "https" not in result


def test_preprocess_lowercases():
    result = preprocess("URGENT: Verify your ACCOUNT now!")
    assert result == result.lower()


def test_preprocess_removes_special_chars():
    result = preprocess("Win $$$!!! Click => HERE!!!")
    assert "$" not in result
    assert "!" not in result
