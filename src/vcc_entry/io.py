from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

VALIDATION_CONTEXTS = ("A", "B", "C")
FINAL_CONTEXTS = ("D", "E", "F")
EXPECTED_GENE_COUNT = 18_533
CELLS_PER_PERT = 400


def load_gene_names(path: Path) -> list[str]:
    """Load the official headerless gene list, preserving order."""
    names = pd.read_csv(path, header=None, names=["gene"], dtype=str)["gene"].tolist()
    if not names:
        raise ValueError(f"empty gene list: {path}")
    return names


def load_perturbations(path: Path) -> list[str]:
    """Load the official perturbation list (gene symbols, not construct ids)."""
    frame = pd.read_csv(path, header=None, dtype=str)
    names = frame.iloc[:, 0].tolist()
    if not names:
        raise ValueError(f"empty perturbation list: {path}")
    if any(name.endswith("-1") or name.lower() == "non-targeting" for name in names):
        raise ValueError(
            "perturbation labels look like construct ids or controls; "
            "the panel uses gene symbols such as ADNP"
        )
    return names


def load_control(path: Path) -> ad.AnnData:
    adata = ad.read_h5ad(path)
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"empty control AnnData: {path}")
    return adata


def control_path(data_dir: Path, context: str) -> Path:
    path = data_dir / f"context_{context}.h5ad"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path.name}; unpack the controls bundle into {data_dir}"
        )
    return path
