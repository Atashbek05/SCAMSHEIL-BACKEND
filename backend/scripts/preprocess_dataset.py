"""
scripts/preprocess_dataset.py — One-off script to clean a raw dataset.

Run from the backend/ directory:
    python scripts/preprocess_dataset.py --input sms_spam.csv --output sms_spam_clean.csv

Output is saved to datasets/processed/.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

# Allow imports from the app package when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.text_cleaner import clean
from app.services.dataset_service import DatasetService


def main():
    parser = argparse.ArgumentParser(description="Preprocess a raw scam dataset")
    parser.add_argument("--input",  required=True, help="Filename inside datasets/raw/")
    parser.add_argument("--output", required=True, help="Filename to write in datasets/processed/")
    parser.add_argument("--text-col", default="text", help="Column containing the message text")
    parser.add_argument("--label-col", default="label", help="Column containing the label (scam/ham)")
    args = parser.parse_args()

    svc = DatasetService()
    df = svc.load_raw(args.input)

    print(f"Loaded {len(df):,} rows from {args.input}")

    df[args.text_col] = df[args.text_col].astype(str).apply(clean)
    df = df.dropna(subset=[args.text_col, args.label_col])
    df = df.drop_duplicates(subset=[args.text_col])

    svc.save_processed(df, args.output)
    print(f"Saved {len(df):,} clean rows to datasets/processed/{args.output}")


if __name__ == "__main__":
    main()
