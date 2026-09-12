"""
download_data.py

One-time script to download and unzip the chest X-ray pneumonia dataset
from Kaggle into data/raw/.

"""

import subprocess
import zipfile
from pathlib import Path

DATASET = "paultimothymooney/chest-xray-pneumonia"
DATA_DIR = Path("data/raw")

DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading dataset...")

subprocess.run([
    "kaggle",
    "datasets",
    "download",
    "-d", DATASET,
    "-p", str(DATA_DIR)
], check=True)

print("Extracting dataset...")

zip_file = DATA_DIR / "chest-xray-pneumonia.zip"

with zipfile.ZipFile(zip_file, "r") as zip:
    zip.extractall(DATA_DIR)

print("Dataset downloaded and extracted successfully!")