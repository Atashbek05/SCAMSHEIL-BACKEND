"""
services/dataset_service.py — Dataset loading and management utilities.

Responsible for reading raw/processed datasets from the datasets/ directory
and exposing them to training scripts and exploratory notebooks.
"""

import pandas as pd
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets"
RAW_DIR = DATASETS_DIR / "raw"
PROCESSED_DIR = DATASETS_DIR / "processed"


class DatasetService:
    """Provides access to scam/ham training datasets."""

    def load_raw(self, filename: str) -> pd.DataFrame:
        """Load a CSV from datasets/raw/."""
        path = RAW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Raw dataset not found: {path}")
        return pd.read_csv(path)

    def load_processed(self, filename: str) -> pd.DataFrame:
        """Load a CSV from datasets/processed/."""
        path = PROCESSED_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {path}")
        return pd.read_csv(path)

    def save_processed(self, df: pd.DataFrame, filename: str) -> None:
        """Persist a cleaned/transformed DataFrame to datasets/processed/."""
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DIR / filename, index=False)
