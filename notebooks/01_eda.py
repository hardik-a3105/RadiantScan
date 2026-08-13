"""
01_eda.py

Phase 1 exploratory data analysis for RadiantScan.
Run this after downloading the dataset to sanity-check class balance,
image sizes, and spot-check samples before building the training pipeline.

Convert to a Jupyter notebook (jupytext) or just run as a script:
    python notebooks/01_eda.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from PIL import Image

from src.data_pipeline import build_manifest, build_test_manifest, class_distribution

RAW_DIR = "data/raw/chest_xray"


def plot_class_distribution(df, title, save_path=None):
    counts = class_distribution(df)
    ax = counts.plot(kind="bar", color=["#4C72B0", "#DD8452"])
    ax.set_title(title)
    ax.set_ylabel("Number of images")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def sample_grid(df, n=6, title="Sample images"):
    fig, axes = plt.subplots(2, n // 2, figsize=(12, 5))
    sample_df = df.sample(n=n, random_state=42)
    for ax, (_, row) in zip(axes.flatten(), sample_df.iterrows()):
        img = Image.open(row["filepath"])
        ax.imshow(img, cmap="gray")
        ax.set_title(row["label"])
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def image_size_stats(df, n_sample=200):
    sizes = []
    for _, row in df.sample(n=min(n_sample, len(df)), random_state=42).iterrows():
        with Image.open(row["filepath"]) as img:
            sizes.append(img.size)
    widths = [w for w, h in sizes]
    heights = [h for w, h in sizes]
    print(f"Width  -> min: {min(widths)}, max: {max(widths)}, avg: {sum(widths)/len(widths):.0f}")
    print(f"Height -> min: {min(heights)}, max: {max(heights)}, avg: {sum(heights)/len(heights):.0f}")


if __name__ == "__main__":
    print("Building manifest from raw data...")
    df = build_manifest(RAW_DIR)
    test_df = build_test_manifest(RAW_DIR)

    print(f"\nTotal pooled (train+val) images: {len(df)}")
    print(f"Total held-out test images: {len(test_df)}")

    print("\nClass distribution (train+val pool):")
    print(class_distribution(df))
    imbalance_ratio = class_distribution(df).max() / class_distribution(df).min()
    print(f"Imbalance ratio: {imbalance_ratio:.2f}x")

    print("\nImage size statistics (sampled):")
    image_size_stats(df)

    plot_class_distribution(df, "Class Distribution: Train+Val Pool")
    sample_grid(df, n=6, title="Sample Chest X-Rays")
