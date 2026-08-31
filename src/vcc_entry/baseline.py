"""Context-mean baseline: paste the average unperturbed cell onto every perturbation.

On the 2026 scoring scale this is the 0-point: no better than predicting the
cell-context mean for every knockdown. It is still a useful first submission
because it exercises the full download → predict → `vcc prep` path against the
real gene list, perturbation list, and context labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from vcc_entry.io import (
    CELLS_PER_PERT,
    VALIDATION_CONTEXTS,
    control_path,
    load_control,
    load_gene_names,
    load_perturbations,
)


def _as_csr(matrix) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return sparse.csr_matrix(matrix)


def align_genes(adata: ad.AnnData, gene_names: list[str]) -> ad.AnnData:
    """Reorder and restrict variables to the official gene list."""
    var_names = adata.var_names.astype(str)
    missing = [name for name in gene_names if name not in var_names]
    if missing:
        preview = ", ".join(missing[:8])
        raise ValueError(
            f"{len(missing)} official genes missing from control (e.g. {preview})"
        )
    return adata[:, gene_names].copy()


def mean_counts(adata: ad.AnnData) -> np.ndarray:
    """Mean expression as non-negative integer counts."""
    mean = np.asarray(_as_csr(adata.X).mean(axis=0)).ravel()
    return np.clip(np.rint(mean), 0, None).astype(np.float32)


def tile_mean(mean: np.ndarray, n_cells: int) -> sparse.csr_matrix:
    """Repeat one sparse count vector `n_cells` times without densifying."""
    nonzero = np.flatnonzero(mean)
    n_nz = int(nonzero.size)
    n_genes = int(mean.size)
    if n_cells == 0:
        return sparse.csr_matrix((0, n_genes), dtype=np.float32)
    if n_nz == 0:
        return sparse.csr_matrix((n_cells, n_genes), dtype=np.float32)
    data = np.tile(mean[nonzero], n_cells)
    indices = np.tile(nonzero.astype(np.int32), n_cells)
    indptr = np.arange(0, (n_cells + 1) * n_nz, n_nz, dtype=np.int64)
    return sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_genes))


def build_context_mean_prediction(
    data_dir: Path,
    gene_names: list[str],
    perturbations: list[str],
    contexts: tuple[str, ...] = VALIDATION_CONTEXTS,
    cells_per_pert: int = CELLS_PER_PERT,
) -> ad.AnnData:
    blocks: list[ad.AnnData] = []
    n_perts = len(perturbations)
    n_cells_per_context = n_perts * cells_per_pert

    for context in contexts:
        control = align_genes(load_control(control_path(data_dir, context)), gene_names)
        mean = mean_counts(control)
        if mean.size != len(gene_names):
            raise ValueError(
                f"context {context}: expected {len(gene_names)} genes, got {mean.size}"
            )
        x = tile_mean(mean, n_cells_per_context)
        obs = pd.DataFrame(
            {
                "target_gene": np.repeat(perturbations, cells_per_pert),
                "context": context,
            }
        )
        blocks.append(ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=gene_names)))

    prediction = ad.concat(blocks, axis=0, join="inner", index_unique="-")
    prediction.obs_names_make_unique()
    return prediction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the official 0-point context-mean baseline prediction."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory with unpacked controls (default: data/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/baseline_mean.h5ad"),
        help="Where to write the prediction AnnData",
    )
    parser.add_argument(
        "--contexts",
        default="A,B,C",
        help="Comma-separated context labels (A,B,C for validation; D,E,F for final)",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir
    gene_names = load_gene_names(data_dir / "gene_names.csv")
    perturbations = load_perturbations(data_dir / "pert_counts.csv")
    contexts = tuple(part.strip() for part in args.contexts.split(",") if part.strip())
    if not contexts:
        raise SystemExit("no contexts given")

    prediction = build_context_mean_prediction(
        data_dir, gene_names, perturbations, contexts=contexts
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_h5ad(args.output, compression="gzip")
    print(
        f"wrote {args.output} "
        f"({prediction.n_obs} cells × {prediction.n_vars} genes, "
        f"{prediction.X.nnz} stored entries)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
