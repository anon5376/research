"""Plate-matched DMSO centroids as a compact basal-state encoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tahoe.metadata import default_cache_dir
from tahoe.plates import CELL_EVAL_REPO, normalize_plate

DMSO_HVG = "zeroshot/cell_line_centroids_dmso_hvg.parquet"


def download_dmso_centroids(cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / DMSO_HVG
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=CELL_EVAL_REPO,
            filename=DMSO_HVG,
            repo_type="dataset",
            local_dir=str(cache_dir),
        )
    )


def load_dmso_centroids(cache_dir: Path | None = None) -> pd.DataFrame:
    path = download_dmso_centroids(cache_dir)
    raw = pd.read_parquet(path)
    frame = raw.rename(columns={"cell_line_id": "cell_id"})
    frame["cell_id"] = frame["cell_id"].astype(str)
    frame["plate"] = frame["plate"].map(normalize_plate)
    return frame


def pca_fit(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, components) with components shape (k, n_genes)."""
    if matrix.shape[0] < 2:
        raise ValueError("PCA needs at least two rows")
    k = min(k, matrix.shape[0] - 1, matrix.shape[1])
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return mean, vt[:k]


def pca_transform(matrix: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (matrix - mean) @ components.T


def attach_basal_pcs(
    frame: pd.DataFrame,
    centroids: pd.DataFrame,
    train_mask: pd.Series,
    *,
    k: int = 8,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """Fit DMSO HVG PCA on training (cell, plate) pairs; attach PCs to every row."""
    gene_cols = [c for c in centroids.columns if c not in {"cell_id", "plate", "cell_line_id"}]
    keys = frame.loc[train_mask, ["cell_id", "plate"]].drop_duplicates()
    train_basal = centroids.merge(keys, on=["cell_id", "plate"], how="inner")
    if len(train_basal) < 2:
        raise ValueError("not enough training DMSO centroids to fit PCA")
    mean, components = pca_fit(train_basal[gene_cols].to_numpy(dtype=np.float64), k)
    pcs = pca_transform(centroids[gene_cols].to_numpy(dtype=np.float64), mean, components)
    names = [f"basal_pc{i}" for i in range(components.shape[0])]
    encoded = pd.DataFrame(pcs, columns=names)
    encoded["cell_id"] = centroids["cell_id"].to_numpy()
    encoded["plate"] = centroids["plate"].to_numpy()
    merged = frame.merge(encoded, on=["cell_id", "plate"], how="left")
    merged[names] = merged[names].fillna(0.0)
    return merged, names, mean, components
