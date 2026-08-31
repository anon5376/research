"""Basal transcriptome + drug (+ drivers) → expression delta."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tahoe.basal import pca_transform
from tahoe.features import FEATURE_NAMES, design_matrix
from tahoe.model import GenotypeRidge


def _as_basal_vector(
    basal: pd.Series | dict[str, float] | np.ndarray,
    pca_genes: list[str],
    pca_mean: np.ndarray,
) -> np.ndarray:
    if isinstance(basal, np.ndarray):
        vec = np.asarray(basal, dtype=np.float64).reshape(-1)
        if vec.shape[0] == len(pca_genes):
            return vec
        raise ValueError(f"basal length {vec.shape[0]} != {len(pca_genes)} PCA genes")
    series = pd.Series(basal)
    out = pca_mean.copy()
    for i, gene in enumerate(pca_genes):
        if gene in series.index and pd.notna(series[gene]):
            out[i] = float(series[gene])
    return out


@dataclass
class PerturbationModel:
    """f(DMSO HVG, drug, KRAS/BRAF/TP53) → treated−DMSO on the same genes."""

    genes: list[str]
    feature_names: list[str]
    weights: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_genes: list[str]
    lam: float = 1.0

    def encode_basal(self, basal: pd.Series | dict[str, float] | np.ndarray) -> np.ndarray:
        if isinstance(basal, np.ndarray) and np.asarray(basal).reshape(-1).shape[0] == self.pca_components.shape[0]:
            return np.asarray(basal, dtype=np.float64).reshape(-1)
        vector = _as_basal_vector(basal, self.pca_genes, self.pca_mean)
        return pca_transform(vector.reshape(1, -1), self.pca_mean, self.pca_components)[0]

    def predict(
        self,
        *,
        basal: pd.Series | dict[str, float] | np.ndarray,
        drug: str,
        concentration: float,
        drivers: dict[str, object],
    ) -> pd.Series:
        pcs = self.encode_basal(basal)
        row = pd.DataFrame([{**drivers, "drug": drug, "concentration": concentration}])
        geno, names = design_matrix(row)
        if names != list(FEATURE_NAMES):
            raise ValueError("genotype feature layout changed")
        extra = [name for name in self.feature_names if name not in FEATURE_NAMES]
        if extra:
            pc_map = {f"basal_pc{i}": pcs[i] for i in range(len(pcs))}
            addon = np.array([[pc_map.get(name, 0.0) for name in extra]], dtype=np.float64)
            x = np.hstack([geno, addon])
        else:
            x = geno
        if x.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {x.shape[1]}")
        model = GenotypeRidge(lam=self.lam)
        model.weights = self.weights
        return pd.Series(model.predict(x)[0], index=self.genes)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            weights=self.weights,
            lam=np.asarray([self.lam]),
            genes=np.asarray(self.genes),
            feature_names=np.asarray(self.feature_names),
            pca_mean=self.pca_mean,
            pca_components=self.pca_components,
            pca_genes=np.asarray(self.pca_genes),
        )

    @classmethod
    def load(cls, path: Path) -> PerturbationModel:
        data = np.load(path, allow_pickle=False)
        return cls(
            genes=[str(g) for g in data["genes"]],
            feature_names=[str(n) for n in data["feature_names"]],
            weights=data["weights"],
            pca_mean=data["pca_mean"],
            pca_components=data["pca_components"],
            pca_genes=[str(g) for g in data["pca_genes"]],
            lam=float(data["lam"][0]),
        )
