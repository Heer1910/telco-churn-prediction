"""
Download the Telco Customer Churn dataset.

Source: IBM Watson Analytics sample dataset (hosted on GitHub).
"""

import os
import urllib.request
from pathlib import Path


DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

DEST_DIR = Path(__file__).parent / "data" / "raw"
DEST_FILE = DEST_DIR / "telco_churn.csv"


def download():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    if DEST_FILE.exists():
        print(f"✅ Dataset already exists at {DEST_FILE}")
        return

    print(f"⬇  Downloading Telco Customer Churn dataset ...")
    urllib.request.urlretrieve(DATASET_URL, DEST_FILE)
    size_kb = os.path.getsize(DEST_FILE) / 1024
    print(f"✅ Saved to {DEST_FILE} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    download()
