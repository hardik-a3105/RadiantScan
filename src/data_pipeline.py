"""
data_pipeline.py

Handles dataset loading, stratified train/val/test splitting, augmentation,
and DataLoader construction for the RadiantScan chest X-ray classifier.

Expected raw data layout (Kaggle "chest-xray-pneumonia" format):

    data/raw/chest_xray/
        train/
            NORMAL/*.jpeg
            PNEUMONIA/*.jpeg
        test/
            NORMAL/*.jpeg
            PNEUMONIA/*.jpeg
        val/
            NORMAL/*.jpeg
            PNEUMONIA/*.jpeg

NOTE: The Kaggle val/ split is tiny (~16 images) and not reliable on its own.
This pipeline pools train+val together and re-splits with stratification
so validation numbers are actually meaningful.
"""

import os
import random
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


SEED = 42
IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        import torch
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_manifest(raw_dir: str) -> pd.DataFrame:
    """
    Walk the raw dataset directory and build a manifest DataFrame with
    columns: filepath, label, label_idx.

    Pools together the original train/ and val/ folders from the Kaggle
    dataset (keeping test/ separate) so we can do a proper stratified split.
    """
    raw_dir = Path(raw_dir)
    rows = []

    for split_folder in ["train", "val"]:
        for class_name in CLASS_NAMES:
            class_dir = raw_dir / split_folder / class_name
            if not class_dir.exists():
                continue
            for ext in ("*.jpeg", "*.jpg", "*.png"):
                for img_path in class_dir.glob(ext):
                    rows.append({
                        "filepath": str(img_path),
                        "label": class_name,
                        "label_idx": CLASS_NAMES.index(class_name),
                        "source_split": split_folder,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(
            f"No images found under {raw_dir}. "
            "Check that the dataset was downloaded and unzipped correctly."
        )
    return df


def build_test_manifest(raw_dir: str) -> pd.DataFrame:
    """Build manifest for the held-out Kaggle test/ folder. Never touched
    until final evaluation."""
    raw_dir = Path(raw_dir)
    rows = []
    for class_name in CLASS_NAMES:
        class_dir = raw_dir / "test" / class_name
        if not class_dir.exists():
            continue
        for ext in ("*.jpeg", "*.jpg", "*.png"):
            for img_path in class_dir.glob(ext):
                rows.append({
                    "filepath": str(img_path),
                    "label": class_name,
                    "label_idx": CLASS_NAMES.index(class_name),
                })
    return pd.DataFrame(rows)


def stratified_split(df: pd.DataFrame, val_size: float = 0.15, seed: int = SEED):
    """Stratified train/val split preserving class ratio in both sets."""
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["label_idx"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def class_distribution(df: pd.DataFrame) -> pd.Series:
    """Quick sanity-check helper: prints/returns class counts and ratio."""
    counts = df["label"].value_counts()
    return counts


class ChestXrayDataset(Dataset):
    """PyTorch Dataset wrapping a manifest DataFrame of (filepath, label_idx)."""

    def __init__(self, manifest_df: pd.DataFrame, transform=None):
        self.df = manifest_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = int(row["label_idx"])
        if self.transform:
            image = self.transform(image)
        return image, label


# ImageNet normalization stats -- required since we use ImageNet-pretrained backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """No augmentation -- used for val/test so metrics are reproducible."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_dataloaders(
    raw_dir: str,
    batch_size: int = 32,
    val_size: float = 0.15,
    num_workers: int = 2,
    seed: int = SEED,
) -> DataLoaders:
    """End-to-end: manifest -> split -> datasets -> dataloaders."""
    set_seed(seed)

    full_df = build_manifest(raw_dir)
    train_df, val_df = stratified_split(full_df, val_size=val_size, seed=seed)
    test_df = build_test_manifest(raw_dir)

    train_ds = ChestXrayDataset(train_df, transform=get_train_transforms())
    val_ds = ChestXrayDataset(val_df, transform=get_eval_transforms())
    test_ds = ChestXrayDataset(test_df, transform=get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return DataLoaders(train=train_loader, val=val_loader, test=test_loader)


if __name__ == "__main__":
    # Quick manual sanity check when run directly:
    #   python src/data_pipeline.py
    RAW_DIR = "data/raw/chest_xray"
    df = build_manifest(RAW_DIR)
    print("Full pooled train+val manifest:")
    print(class_distribution(df))

    train_df, val_df = stratified_split(df)
    print("\nTrain split:")
    print(class_distribution(train_df))
    print("\nVal split:")
    print(class_distribution(val_df))

    test_df = build_test_manifest(RAW_DIR)
    print("\nTest split (held out, Kaggle's original test/):")
    print(class_distribution(test_df))
