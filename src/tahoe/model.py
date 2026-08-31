"""Ridge models for perturbation deltas. No cell-line identity features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _add_bias(x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((x.shape[0], 1))])


def _ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Solve (X'X + λI) W = X'Y. Y is (n, g). Last column of X is bias."""
    n_features = x.shape[1]
    xtx = x.T @ x
    reg = lam * np.eye(n_features)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xtx + reg, x.T @ y)


@dataclass
class ZeroBaseline:
    n_genes: int = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> ZeroBaseline:
        self.n_genes = y.shape[1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.zeros((x.shape[0], self.n_genes), dtype=np.float64)


@dataclass
class DrugMeanBaseline:
    """Mean training delta per drug class (RAF / RAS / MEK). Ignores genotype."""

    class_index: tuple[int, ...] = (6, 7, 8)  # is_raf, is_ras, is_mek in FEATURE_NAMES
    means: np.ndarray | None = None
    global_mean: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> DrugMeanBaseline:
        self.global_mean = y.mean(axis=0)
        n_classes = len(self.class_index)
        self.means = np.tile(self.global_mean, (n_classes, 1))
        for i, col in enumerate(self.class_index):
            mask = x[:, col] > 0.5
            if mask.any():
                self.means[i] = y[mask].mean(axis=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.means is not None and self.global_mean is not None
        out = np.tile(self.global_mean, (x.shape[0], 1))
        for i, col in enumerate(self.class_index):
            mask = x[:, col] > 0.5
            out[mask] = self.means[i]
        return out


@dataclass
class GenotypeRidge:
    lam: float = 1.0
    weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> GenotypeRidge:
        self.weights = _ridge_fit(_add_bias(x), y, self.lam)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.weights is not None
        return _add_bias(x) @ self.weights


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def identity_onehot(cell_ids: pd.Series, train_ids: list[str]) -> np.ndarray:
    """One-hot of training lines. Unseen (held-out) lines are the zero vector."""
    index = {cid: i for i, cid in enumerate(train_ids)}
    out = np.zeros((len(cell_ids), len(train_ids)), dtype=np.float64)
    for row, cid in enumerate(cell_ids.tolist()):
        col = index.get(str(cid))
        if col is not None:
            out[row, col] = 1.0
    return out


def pearson_per_row(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores: list[float] = []
    for i in range(y_true.shape[0]):
        a, b = y_true[i], y_pred[i]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        scores.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(scores)) if scores else float("nan")
