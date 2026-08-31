from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from vcc_entry.baseline import (
    align_genes,
    build_context_mean_prediction,
    mean_counts,
    tile_mean,
)
from vcc_entry.io import load_gene_names, load_perturbations


def test_tile_mean_repeats_sparse_row() -> None:
    mean = np.array([3.0, 0.0, 5.0], dtype=np.float32)
    matrix = tile_mean(mean, 4)
    assert matrix.shape == (4, 3)
    assert sparse.issparse(matrix)
    dense = matrix.toarray()
    assert np.array_equal(dense, np.tile(mean, (4, 1)))


def test_mean_counts_rounds_and_clips() -> None:
    x = sparse.csr_matrix(np.array([[1.4, 0.0], [1.6, 0.0]], dtype=np.float32))
    adata = ad.AnnData(X=x)
    mean = mean_counts(adata)
    assert mean.dtype == np.float32
    assert np.array_equal(mean, np.array([2.0, 0.0], dtype=np.float32))


def test_align_genes_reorders(tmp_path: Path) -> None:
    adata = ad.AnnData(
        X=np.array([[1, 2, 3]], dtype=np.float32),
        var=pd.DataFrame(index=["C", "A", "B"]),
    )
    aligned = align_genes(adata, ["A", "B", "C"])
    assert list(aligned.var_names) == ["A", "B", "C"]
    assert np.array_equal(aligned.X, np.array([[2, 3, 1]], dtype=np.float32))


def test_load_lists(tmp_path: Path) -> None:
    genes = tmp_path / "gene_names.csv"
    perts = tmp_path / "pert_counts.csv"
    genes.write_text("GAPDH\nACTB\n")
    perts.write_text("TP53\nMYC\n")
    assert load_gene_names(genes) == ["GAPDH", "ACTB"]
    assert load_perturbations(perts) == ["TP53", "MYC"]


def test_build_context_mean_prediction(tmp_path: Path) -> None:
    genes = ["G1", "G2", "G3"]
    perts = ["TP53", "MYC"]
    (tmp_path / "gene_names.csv").write_text("\n".join(genes) + "\n")
    (tmp_path / "pert_counts.csv").write_text("\n".join(perts) + "\n")

    for context, scale in (("A", 1.0), ("B", 10.0)):
        x = np.array([[scale, 0.0, 2 * scale], [scale, 0.0, 2 * scale]], dtype=np.float32)
        adata = ad.AnnData(X=x, var=pd.DataFrame(index=genes))
        adata.write_h5ad(tmp_path / f"context_{context}.h5ad")

    prediction = build_context_mean_prediction(
        tmp_path, genes, perts, contexts=("A", "B"), cells_per_pert=3
    )
    assert prediction.n_obs == 2 * 2 * 3
    assert list(prediction.var_names) == genes
    assert set(prediction.obs["context"]) == {"A", "B"}
    assert list(prediction.obs["target_gene"].unique()) == perts
    assert (prediction.obs.groupby(["context", "target_gene"]).size() == 3).all()
    assert "non-targeting" not in set(prediction.obs["target_gene"])
    assert sparse.issparse(prediction.X)
    a_block = prediction[prediction.obs["context"] == "A"].X.toarray()
    assert np.allclose(a_block, np.tile([1.0, 0.0, 2.0], (6, 1)))
