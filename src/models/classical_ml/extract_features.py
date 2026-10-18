"""
Extract handcrafted features from DeepWeeds images for the classical ML baseline.

Phase 3/4 of project_execution_algorithm.md — draws on the Cluster 2 papers in
docs/literature_assignment_plan.md (Islam et al. reached 94-96% with RF/SVM on
colour + texture + shape features; Tufail et al. used a similar feature set).

Features extracted per image:
  - Colour indices: ExG (Excess Green), ExGR (Excess Green minus Excess Red)
  - Texture: GLCM (grey-level co-occurrence matrix) — contrast, dissimilarity,
    homogeneity, energy, correlation, ASM, averaged across 4 angles
  - Shape: HOG (Histogram of Oriented Gradients), summarized to mean/std/max
    since raw HOG vectors are too high-dimensional for a clean feature table

Reads the FROZEN split (data/deepweeds_splits.csv — do not regenerate this)
and writes data/deepweeds_features.csv: one row per image, all features as
columns, plus species label and split. train_ml.py reads from this file.

Note: dimensionality reduction (PCA, per configs/classical_ml.yaml) is
deliberately NOT done here — it must be fit on the train split only and
applied to val/test, which only makes sense at training time, not here.

Run: python -m src.models.classical_ml.extract_features
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, hog
from tqdm import tqdm

SPLITS_CSV = Path("data/deepweeds_splits.csv")
OUTPUT_CSV = Path("data/deepweeds_features.csv")
IMG_SIZE = 128  # resize for consistent, tractable feature extraction


def extract_color_indices(img_rgb: np.ndarray) -> dict:
    """ExG and ExGR colour indices on normalized RGB channels."""
    img = img_rgb.astype(np.float32)
    total = img.sum(axis=2) + 1e-6  # avoid divide-by-zero on pure black pixels
    r = img[:, :, 0] / total
    g = img[:, :, 1] / total
    b = img[:, :, 2] / total

    exg = 2 * g - r - b
    exr = 1.4 * r - g
    exgr = exg - exr

    return {
        "exg_mean": float(np.mean(exg)),
        "exg_std": float(np.std(exg)),
        "exgr_mean": float(np.mean(exgr)),
        "exgr_std": float(np.std(exgr)),
    }


def extract_glcm_texture(gray_img: np.ndarray) -> dict:
    """GLCM texture features at 4 angles, averaged for rotation-invariance."""
    gray_uint8 = (gray_img * 255).astype(np.uint8)
    glcm = graycomatrix(
        gray_uint8,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )
    features = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        values = graycoprops(glcm, prop)[0]
        features[f"glcm_{prop}_mean"] = float(np.mean(values))
    return features


def extract_hog_shape(gray_img: np.ndarray) -> dict:
    """HOG descriptor, summarized to mean/std/max for a tabular feature set."""
    hog_features = hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )
    return {
        "hog_mean": float(np.mean(hog_features)),
        "hog_std": float(np.std(hog_features)),
        "hog_max": float(np.max(hog_features)),
    }


def extract_features_for_image(filepath: str) -> dict | None:
    img_bgr = cv2.imread(filepath)
    if img_bgr is None:
        return None
    img_bgr = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = rgb2gray(img_rgb)

    features = {}
    features.update(extract_color_indices(img_rgb))
    features.update(extract_glcm_texture(gray))
    features.update(extract_hog_shape(gray))
    return features


def main() -> None:
    if not SPLITS_CSV.exists():
        print(f"ERROR: {SPLITS_CSV} not found. Run `python -m src.data.make_splits` first.")
        return

    if OUTPUT_CSV.exists():
        print(f"{OUTPUT_CSV} already exists. Delete it manually first if you genuinely "
              f"want to regenerate it (e.g. after changing IMG_SIZE or the feature set).")
        return

    splits_df = pd.read_csv(SPLITS_CSV)
    print(f"Extracting features from {len(splits_df)} images (this will take a while — "
          f"GLCM and HOG aren't free)...")

    rows = []
    failed = []
    for _, row in tqdm(splits_df.iterrows(), total=len(splits_df)):
        features = extract_features_for_image(row["filepath"])
        if features is None:
            failed.append(row["filepath"])
            continue
        features["filepath"] = row["filepath"]
        features["species"] = row["species"]
        features["split"] = row["split"]
        rows.append(features)

    features_df = pd.DataFrame(rows)
    features_df.to_csv(OUTPUT_CSV, index=False)

    feature_cols = [c for c in features_df.columns if c not in ("filepath", "species", "split")]
    print(f"\nSaved {len(features_df)} feature rows to {OUTPUT_CSV}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    if failed:
        print(f"\nWARNING: {len(failed)} images failed to load/process:")
        for f in failed[:10]:
            print(f"  {f}")


if __name__ == "__main__":
    main()
