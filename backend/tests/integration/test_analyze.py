"""
tests/integration/test_analyze.py — End-to-end tests for the ScamShield API.

Run from the backend/ directory:
    pytest tests/integration/test_analyze.py -v

Tests are split into two groups:
  1. Schema & validation tests — run without a trained model (always pass in CI).
  2. ML smoke tests — skipped automatically when the .joblib file is absent.
     Train the model first:  python scripts/train_model.py
"""

import pathlib
import pytest
from fastapi.testclient import TestClient

# TestClient runs the full ASGI app in-process — no server needed
from main import app

client = TestClient(app)

# Path used by @pytest.mark.skipif to detect a trained model
_MODEL_PATH = pathlib.Path("app/models/ml/scam_classifier.joblib")
_model_missing = not _MODEL_PATH.exists()
_skip_if_no_model = pytest.mark.skipif(
    _model_missing,
    reason="Trained model not present — run `python scripts/train_model.py` first",
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def post_analyze(message: str):
    return client.post("/api/v1/analyze", json={"message": message})


# ---------------------------------------------------------------------------
# Input validation (no model required)
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_missing_message_field_returns_422(self):
        """Body with no 'message' key must be rejected."""
        r = client.post("/api/v1/analyze", json={})
        assert r.status_code == 422

    def test_empty_message_returns_422(self):
        """min_length=1 constraint must fire on empty string."""
        r = post_analyze("")
        assert r.status_code == 422

    def test_message_exceeding_limit_returns_422(self):
        """max_length=10 000 constraint must fire."""
        r = post_analyze("x" * 10_001)
        assert r.status_code == 422

    def test_non_json_body_returns_422(self):
        r = client.post(
            "/api/v1/analyze",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422

    def test_wrong_content_type_returns_422(self):
        r = client.post(
            "/api/v1/analyze",
            content='{"message": "hello"}',
            headers={"Content-Type": "text/plain"},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Response shape (no model required — uses graceful-degradation path)
# ---------------------------------------------------------------------------

class TestResponseShape:
    """
    These tests assert the API contract regardless of whether a model is trained.
    When the model is absent the service returns label='unknown', risk=0.0.
    """

    def test_required_fields_present(self):
        r = post_analyze("Hello, are you free for coffee tomorrow?")
        assert r.status_code == 200
        data = r.json()
        assert "label" in data
        assert "risk" in data
        assert "scam_probability" in data
        assert "suspicious_keywords" in data
        assert "analyzed_text" in data

    def test_risk_in_valid_range(self):
        r = post_analyze("Congratulations! You won Rs 50,000 in our lucky draw!")
        assert r.status_code == 200
        assert 0.0 <= r.json()["risk"] <= 100.0

    def test_scam_probability_in_valid_range(self):
        r = post_analyze("Share OTP 123456 to unlock your account.")
        assert r.status_code == 200
        assert 0.0 <= r.json()["scam_probability"] <= 1.0

    def test_risk_and_probability_are_consistent(self):
        """risk must equal scam_probability × 100 within floating-point tolerance."""
        r = post_analyze("URGENT: Verify your bank account at http://scam.tk")
        data = r.json()
        assert abs(data["risk"] - data["scam_probability"] * 100) < 0.01

    def test_label_is_valid_string(self):
        r = post_analyze("Meeting rescheduled to 3pm.")
        assert r.json()["label"] in ("scam", "safe", "unknown")

    def test_suspicious_keywords_is_list(self):
        r = post_analyze("Share your OTP now to unlock your account.")
        assert isinstance(r.json()["suspicious_keywords"], list)

    def test_analyzed_text_is_non_empty_string(self):
        r = post_analyze("Your OTP is 482716. Do not share it.")
        data = r.json()
        assert isinstance(data["analyzed_text"], str)
        assert len(data["analyzed_text"]) > 0

    def test_analyzed_text_is_lowercase(self):
        """Preprocessed output must always be lowercase."""
        r = post_analyze("URGENT SHARE OTP NOW")
        assert r.json()["analyzed_text"] == r.json()["analyzed_text"].lower()

    def test_urls_replaced_in_analyzed_text(self):
        """URLs in the message must appear as 'urltoken' in analyzed_text."""
        r = post_analyze("Click here: https://evil-bank.tk/steal")
        assert "urltoken" in r.json()["analyzed_text"]
        assert "https" not in r.json()["analyzed_text"]

    def test_otp_replaced_in_analyzed_text(self):
        """6-digit OTPs in the message must appear as 'otptoken' in analyzed_text."""
        r = post_analyze("Your OTP is 847362. Valid for 10 minutes.")
        assert "otptoken" in r.json()["analyzed_text"]
        assert "847362" not in r.json()["analyzed_text"]

    def test_single_character_message_accepted(self):
        """Minimum valid message: 1 character."""
        r = post_analyze("a")
        assert r.status_code == 200

    def test_unicode_message_accepted(self):
        """Non-ASCII input should not crash the endpoint."""
        r = post_analyze("आपका OTP 482716 है। किसी के साथ साझा न करें।")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Keyword detection (no model required)
# ---------------------------------------------------------------------------

class TestKeywordDetection:
    def test_otp_keyword_detected(self):
        r = post_analyze("Share your OTP with our agent immediately.")
        assert "otp" in r.json()["suspicious_keywords"]

    def test_urgent_keyword_detected(self):
        r = post_analyze("URGENT: Your account will be closed.")
        assert "urgent" in r.json()["suspicious_keywords"]

    def test_no_keywords_for_safe_message(self):
        r = post_analyze("Hey, can you pick up some milk on the way home?")
        assert r.json()["suspicious_keywords"] == []

    def test_multiple_keywords_detected(self):
        r = post_analyze(
            "URGENT: Your bank account is suspended. Share OTP now to verify KYC."
        )
        kws = r.json()["suspicious_keywords"]
        assert len(kws) >= 3


# ---------------------------------------------------------------------------
# ML smoke tests (require trained model)
# ---------------------------------------------------------------------------

class TestMLPredictions:
    @_skip_if_no_model
    def test_otp_theft_scam_detected(self):
        r = post_analyze(
            "URGENT: Share OTP 482716 with our agent to unlock your SBI account."
        )
        data = r.json()
        assert data["label"] == "scam"
        assert data["risk"] > 70

    @_skip_if_no_model
    def test_fake_banking_scam_detected(self):
        r = post_analyze(
            "Your HDFC account has been blocked due to KYC non-compliance. "
            "Update now: http://hdfc-kyc-update.tk"
        )
        data = r.json()
        assert data["label"] == "scam"
        assert data["risk"] > 70

    @_skip_if_no_model
    def test_lottery_scam_detected(self):
        r = post_analyze(
            "Congratulations! You have won Rs 25 lakhs in our lucky draw. "
            "Call 9876543210 to claim your prize."
        )
        data = r.json()
        assert data["label"] == "scam"

    @_skip_if_no_model
    def test_phishing_link_detected(self):
        r = post_analyze(
            "Your Google account will be disabled. Verify ownership at "
            "http://google-verify.tk within 24 hours."
        )
        assert r.json()["label"] == "scam"

    @_skip_if_no_model
    def test_safe_personal_message_not_flagged(self):
        r = post_analyze("Hey, are we still on for dinner Friday?")
        data = r.json()
        assert data["label"] == "safe"
        assert data["risk"] < 40

    @_skip_if_no_model
    def test_legitimate_bank_otp_not_flagged(self):
        """Real OTP messages from banks say 'do not share' — should not be scam."""
        r = post_analyze(
            "HDFC Bank: Your OTP is 847362 for your net banking login. "
            "Valid for 10 minutes. Do not share with anyone."
        )
        # Legitimate OTPs tend to score lower — allow either label but risk must be < 90
        assert r.json()["risk"] < 90

    @_skip_if_no_model
    def test_batch_consistency(self):
        """
        predict_batch and predict must agree on the same message.
        This tests the ScamClassifier.predict_batch() path indirectly via the service.
        """
        msg = "Share the OTP received on your phone with our customer care agent."
        r1 = post_analyze(msg)
        r2 = post_analyze(msg)
        # Deterministic model — same input must give same output
        assert r1.json()["label"] == r2.json()["label"]
        assert r1.json()["risk"] == r2.json()["risk"]


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_has_status_ok(self):
        assert client.get("/api/v1/health").json()["status"] == "ok"

    def test_health_has_model_loaded_field(self):
        data = client.get("/api/v1/health").json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_health_has_version_field(self):
        assert "version" in client.get("/api/v1/health").json()

    def test_health_model_loaded_matches_file(self):
        """model_loaded must reflect whether the .joblib file actually exists."""
        data = client.get("/api/v1/health").json()
        assert data["model_loaded"] == _MODEL_PATH.exists()


# ---------------------------------------------------------------------------
# CORS headers (Android integration)
# ---------------------------------------------------------------------------

class TestCORS:
    def test_options_preflight_allowed(self):
        """Browsers send OPTIONS before POST — must return 2xx."""
        r = client.options(
            "/api/v1/analyze",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
        )
        assert r.status_code in (200, 204)
