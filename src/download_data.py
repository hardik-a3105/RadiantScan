"""
download_data.py

One-time script to download and unzip the chest X-ray pneumonia dataset
from Kaggle into data/raw/.

Setup (one-time, per machine):
    1. Create a Kaggle account, go to Account -> Create New API Token
    2. This downloads kaggle.json -- place it at ~/.kaggle/kaggle.json
       (on Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json)
    3. chmod 600 ~/.kaggle/kaggle.json   (Linux/Mac)

Usage:
    python src/download_data.py
"""

import subprocess
import zipfile
from pathlib import Path

DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"
RAW_DIR = Path("data/raw")


def download_and_extract():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "chest-xray-pneumonia.zip"

    print(f"Downloading {DATASET_SLUG} via Kaggle API...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(RAW_DIR)],
        check=True,
    )

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    print(f"Done. Dataset extracted to {RAW_DIR}/")
    print("Expected structure: data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/")


if __name__ == "__main__":
    download_and_extract()
