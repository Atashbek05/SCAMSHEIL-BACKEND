"""
scripts/train_model.py — Train and persist the ScamShield scam classifier.

Usage (run from backend/ directory):
    python scripts/train_model.py

What this script does:
  1. Downloads the SMS Spam Collection dataset from UCI if not cached locally
  2. Augments it with domain-specific scam/safe examples (OTP theft,
     fake banking, phishing, lottery, government, e-commerce scams)
  3. Pre-processes every message through scam_preprocessor.preprocess()
  4. Splits data into 80% train / 20% test (stratified by label)
  5. Trains the TF-IDF + Logistic Regression pipeline
  6. Prints a full evaluation report (accuracy, precision, recall, F1,
     confusion matrix, and top scam-signal features)
  7. Saves the trained pipeline to app/models/ml/scam_classifier.joblib
"""

import sys
import io
import zipfile
import urllib.request
from pathlib import Path

# Allow running as a script from the backend/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from loguru import logger

from app.models.ml.scam_classifier import ScamClassifier, MODEL_PATH
from app.utils.scam_preprocessor import preprocess

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BACKEND_DIR / "datasets" / "raw"
SMS_SPAM_FILE = RAW_DIR / "SMSSpamCollection"

# UCI archive — the canonical source for this dataset
SMS_SPAM_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

# ---------------------------------------------------------------------------
# Domain-specific augmentation data
#
# Each tuple is (message_text, label) where label 0=safe, 1=scam.
# These examples teach the model patterns not well-represented in the
# SMS Spam Collection: OTP theft, fake banking, government impersonation,
# and e-commerce scams common in South Asian markets.
# ---------------------------------------------------------------------------

AUGMENTATION_DATA: list[tuple[str, int]] = [
    # ── OTP theft scams ──────────────────────────────────────────────────────
    # Key signal: asking the recipient to SHARE or SEND the OTP back
    ("Please share the OTP you received to complete your KYC verification.", 1),
    ("Call us and provide the OTP sent to your number to unlock your account.", 1),
    ("Send us the 6-digit OTP on your phone to claim your cashback reward.", 1),
    ("Our agent will call you. Please share the OTP to verify your identity.", 1),
    ("Your account will be closed. Share OTP received to stop this action.", 1),
    ("Enter the OTP shared by customer to complete the transaction on your device.", 1),
    ("To reverse the failed transaction share OTP OTPTOKEN with our support.", 1),
    # Real bank OTPs never ask you to share — these are safe
    ("Your OTP for SBI net banking login is OTPTOKEN. Do not share with anyone.", 0),
    ("HDFC Bank: OTP OTPTOKEN is valid for 10 minutes for your transaction.", 0),
    ("Use OTP OTPTOKEN to complete your UPI payment. Never share this code.", 0),
    ("ICICI: Your OTP is OTPTOKEN. Bank officials will never ask for this.", 0),

    # ── Fake banking / account suspension scams ───────────────────────────────
    ("Your SBI account has been suspended due to KYC non-compliance. Update now: URLTOKEN", 1),
    ("HDFC Alert: Unusual activity detected. Click to secure your account: URLTOKEN", 1),
    ("Dear customer your net banking access is blocked. Verify at URLTOKEN immediately.", 1),
    ("Your credit card will be deactivated in 24 hours. Re-verify KYC at URLTOKEN", 1),
    ("RBI Notice: Your bank account is under review. Submit documents at URLTOKEN", 1),
    ("PNB: Your account limit has been exhausted. To increase click here: URLTOKEN", 1),
    ("Axis Bank: Suspicious login detected from another device. Secure now: URLTOKEN", 1),
    ("Your debit card ACCOUNTTOKEN has been blocked. Call PHONETOKEN to unblock.", 1),
    # Real bank alerts — no links, no urgency to click
    ("SBI: A transaction of AMOUNTTOKEN was made on your account ending OTPTOKEN.", 0),
    ("HDFC Bank: Your credit card bill of AMOUNTTOKEN is due on 15th. Pay now.", 0),
    ("ICICI: Your EMI of AMOUNTTOKEN has been debited from your account.", 0),

    # ── Phishing / prize / lottery scams ─────────────────────────────────────
    ("Congratulations! You have won AMOUNTTOKEN in our lucky draw. Claim: URLTOKEN", 1),
    ("You are selected for a Google reward of AMOUNTTOKEN. Claim now: URLTOKEN", 1),
    ("Amazon customer survey winner! Get your AMOUNTTOKEN gift card: URLTOKEN", 1),
    ("KBC lottery winner! You have won Rs 25 lakhs. Call PHONETOKEN to claim.", 1),
    ("TRAI: Your number has been selected for a AMOUNTTOKEN cash prize.", 1),
    ("WhatsApp lucky spin winner! You won an iPhone. Claim here: URLTOKEN", 1),
    ("Your email has been selected for a AMOUNTTOKEN international lottery prize.", 1),

    # ── Government impersonation scams ────────────────────────────────────────
    ("Income Tax Department: Refund of AMOUNTTOKEN pending. Update bank at URLTOKEN", 1),
    ("EPFO: Your PF account has an unclaimed amount. Withdraw now: URLTOKEN", 1),
    ("PM relief fund: You are eligible for AMOUNTTOKEN. Apply at URLTOKEN today.", 1),
    ("IT Dept: You have an outstanding tax refund of AMOUNTTOKEN. Verify PAN: URLTOKEN", 1),
    ("UIDAI Aadhaar: Your Aadhaar has been deactivated. Reactivate at URLTOKEN", 1),
    ("Ministry of Finance: Covid relief grant of AMOUNTTOKEN approved for you.", 1),

    # ── Fake e-commerce / delivery scams ──────────────────────────────────────
    ("Your package could not be delivered. Pay customs fee AMOUNTTOKEN: URLTOKEN", 1),
    ("Flipkart: Your order is stuck. Pay AMOUNTTOKEN shipping charges to release it.", 1),
    ("Amazon: Your account is locked due to suspicious activity. Verify: URLTOKEN", 1),
    ("Your COD order of AMOUNTTOKEN is out for delivery. Confirm address: URLTOKEN", 1),
    ("DTDC courier: Your parcel is held at customs. Pay AMOUNTTOKEN to release.", 1),
    # Real delivery notifications
    ("Your Flipkart order has been shipped and will arrive by tomorrow.", 0),
    ("Amazon: Your order for has been delivered. Rate your experience.", 0),
    ("Zomato: Your order is out for delivery. Track: URLTOKEN", 0),

    # ── Vishing / phone scam setups ───────────────────────────────────────────
    ("Call PHONETOKEN immediately to prevent your account from being closed.", 1),
    ("This is an automated message. Press 1 to speak to a bank fraud specialist.", 1),
    ("Your computer has been hacked. Call PHONETOKEN for immediate assistance.", 1),
    ("CBI Officer calling. Your Aadhaar is linked to illegal activity. Call PHONETOKEN", 1),

    # ── Safe / normal messages ────────────────────────────────────────────────
    ("Hey are you free for lunch tomorrow?", 0),
    ("The meeting is rescheduled to 3pm. Please update your calendar.", 0),
    ("Your electricity bill of AMOUNTTOKEN is due. Pay at URLTOKEN", 0),
    ("Reminder: Your appointment is confirmed for tomorrow at 10 AM.", 0),
    ("Thanks for your payment. Your receipt number is OTPTOKEN.", 0),
    ("Your mobile recharge of AMOUNTTOKEN was successful. New balance AMOUNTTOKEN.", 0),
    ("OLA: Your ride has been booked. Driver PHONETOKEN is on the way.", 0),
    ("Aadhaar update successful. Your request reference number is OTPTOKEN.", 0),
    ("Your PF withdrawal of AMOUNTTOKEN has been processed. It will credit in 3 days.", 0),
    ("Class cancelled today. See you on Thursday.", 0),
    ("Can you pick up milk on the way home?", 0),
    ("Your subscription has been renewed for AMOUNTTOKEN. Thank you.", 0),
]


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_sms_spam_dataset() -> None:
    """
    Download the SMS Spam Collection from UCI and save to datasets/raw/.
    Skips download if the file already exists.
    """
    if SMS_SPAM_FILE.exists():
        logger.info(f"Dataset already cached at {SMS_SPAM_FILE}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading SMS Spam Collection from {SMS_SPAM_URL} …")

    try:
        with urllib.request.urlopen(SMS_SPAM_URL, timeout=30) as response:
            zip_bytes = response.read()
    except Exception as exc:
        logger.error(f"Download failed: {exc}")
        logger.error(
            "Manual download instructions:\n"
            f"  1. Go to {SMS_SPAM_URL}\n"
            f"  2. Extract the file named 'SMSSpamCollection'\n"
            f"  3. Place it at {SMS_SPAM_FILE}"
        )
        sys.exit(1)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        # The archive contains 'SMSSpamCollection' (no extension)
        zf.extract("SMSSpamCollection", RAW_DIR)

    logger.success(f"Dataset saved to {SMS_SPAM_FILE}")


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset() -> tuple[list[str], list[int]]:
    """
    Load the SMS Spam Collection and merge augmentation data.

    Returns
    -------
    texts  : preprocessed message strings
    labels : 0 = safe, 1 = scam
    """
    logger.info("Loading SMS Spam Collection …")
    df = pd.read_csv(
        SMS_SPAM_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
        on_bad_lines="skip",
    )

    # Map UCI labels to binary integers
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].astype(int)

    base_texts = df["message"].tolist()
    base_labels = df["label"].tolist()

    logger.info(
        f"Base dataset: {len(base_texts):,} messages "
        f"({sum(base_labels):,} scam / {len(base_labels) - sum(base_labels):,} safe)"
    )

    # Merge augmentation
    aug_texts  = [pair[0] for pair in AUGMENTATION_DATA]
    aug_labels = [pair[1] for pair in AUGMENTATION_DATA]

    all_texts  = base_texts  + aug_texts
    all_labels = base_labels + aug_labels

    logger.info(
        f"After augmentation: {len(all_texts):,} messages "
        f"({sum(all_labels):,} scam / {len(all_labels) - sum(all_labels):,} safe)"
    )

    # Apply full scam-aware preprocessing to every message
    logger.info("Preprocessing messages …")
    processed = [preprocess(t) for t in all_texts]

    return processed, all_labels


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def print_evaluation(y_true: list, y_pred: list, y_proba: np.ndarray) -> None:
    """Print accuracy, classification report, confusion matrix, and AUC."""
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    cm  = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print("  MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["safe", "scam"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"  {'':10s}  safe   scam")
    print(f"  actual safe  {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  actual scam  {cm[1][0]:5d}  {cm[1][1]:5d}")
    print("=" * 60)


def print_top_features(classifier: ScamClassifier, n: int = 20) -> None:
    """Display the n features with the highest scam-class log-odds weight."""
    pipeline = classifier._pipeline
    tfidf    = pipeline.named_steps["tfidf"]
    lr       = pipeline.named_steps["clf"]

    feature_names = tfidf.get_feature_names_out()
    # Class 1 (scam) log-odds coefficients
    scam_coefs = lr.coef_[0]

    top_indices = np.argsort(scam_coefs)[-n:][::-1]
    print(f"\nTop {n} scam-signal features (highest log-odds weight):")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:<30s} {scam_coefs[idx]:.4f}")
    print()


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_and_save() -> None:
    download_sms_spam_dataset()

    texts, labels = load_dataset()

    # Stratified split keeps scam/safe ratio equal in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    logger.info(
        f"Train: {len(X_train):,} | Test: {len(X_test):,}"
    )

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("Training TF-IDF + Logistic Regression pipeline …")
    classifier = ScamClassifier()
    classifier.train(X_train, y_train)
    logger.success("Training complete.")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_pred  = classifier._pipeline.predict(X_test)
    y_proba = classifier._pipeline.predict_proba(X_test)[:, 1]

    print_evaluation(y_test, y_pred, y_proba)
    print_top_features(classifier)

    # ── Save ─────────────────────────────────────────────────────────────────
    classifier.save(MODEL_PATH)
    logger.success(f"Model saved to {MODEL_PATH}")

    # Quick smoke-test on three hand-crafted examples
    logger.info("Smoke-test predictions:")
    test_cases = [
        "Share the OTP received on your phone with our agent immediately.",
        "Your SBI account is suspended. Verify KYC at http://sbi-update.tk",
        "Hey, are we still meeting for coffee tomorrow?",
    ]
    for msg in test_cases:
        result = classifier.predict(msg)
        logger.info(f"  [{result['label'].upper():4s} {result['confidence']:5.1f}%] {msg[:70]}")


if __name__ == "__main__":
    train_and_save()
