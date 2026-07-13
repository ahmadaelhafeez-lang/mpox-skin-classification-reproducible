#!/usr/bin/env python3
"""Prepare a cleaned four-class mpox skin image dataset split.

This script indexes image files, removes unreadable and duplicate files using
content hashing, and creates a stratified train/validation/test split.
"""

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

CLASSES = ["Monkeypox", "Chickenpox", "Measles", "Normal"]
CLASS_TO_LABEL = {c: i for i, c in enumerate(CLASSES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def find_class_folders(data_root: Path):
    mapping = {}
    for cls in CLASSES:
        direct = data_root / cls
        if direct.exists():
            mapping[cls] = direct
    if set(mapping) != set(CLASSES):
        raise RuntimeError(f"Could not find all class folders under {data_root}. Found: {mapping}")
    return mapping


def build_clean_index(data_root: Path) -> pd.DataFrame:
    folders = find_class_folders(data_root)
    rows = []
    seen_hashes = set()
    for cls in CLASSES:
        for path in sorted(folders[cls].rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if not validate_image(path):
                continue
            md5 = file_md5(path)
            if md5 in seen_hashes:
                continue
            seen_hashes.add(md5)
            rows.append({"path": str(path), "filename": path.name, "class": cls, "label": CLASS_TO_LABEL[cls], "md5": md5})
    return pd.DataFrame(rows)


def stratified_split(df: pd.DataFrame, seed: int):
    train_df, temp_df = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to the folder containing the four class folders.")
    parser.add_argument("--output_dir", default="outputs", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.output_dir) / "csv"
    out.mkdir(parents=True, exist_ok=True)

    df = build_clean_index(data_root)
    train_df, val_df, test_df = stratified_split(df, args.seed)

    df.to_csv(out / "cleaned_image_index_v2.csv", index=False)
    train_df.to_csv(out / "train_split_v2.csv", index=False)
    val_df.to_csv(out / "validation_split_v2.csv", index=False)
    test_df.to_csv(out / "test_split_v2.csv", index=False)

    class_distribution = df.groupby("class").size().reindex(CLASSES).reset_index(name="count")
    class_distribution.to_csv(out / "class_distribution_v2.csv", index=False)

    split_distribution = pd.DataFrame({
        "split": ["train", "validation", "test"],
        "n": [len(train_df), len(val_df), len(test_df)]
    })
    split_distribution.to_csv(out / "split_distribution_v2.csv", index=False)

    weights = compute_class_weight("balanced", classes=np.arange(len(CLASSES)), y=train_df["label"].values)
    pd.DataFrame({"class": CLASSES, "label": range(len(CLASSES)), "class_weight": weights}).to_csv(out / "class_weights_v2.csv", index=False)

    print("Cleaned images:", len(df))
    print("Split sizes:", len(train_df), len(val_df), len(test_df))
    print(class_distribution)


if __name__ == "__main__":
    main()
