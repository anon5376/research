from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tahoe.basal import pca_fit, pca_transform
from tahoe.bundle import PerturbationModel
from tahoe.features import FEATURE_NAMES, design_matrix, drug_class_flags
from tahoe.model import identity_onehot
from tahoe.slice import canonicalize_drug
from tahoe.split import line_holdout
from tahoe.train import evaluate, load_ridge, predict_delta, save_ridge, synthetic_slice


def test_canonicalize_trametinib_solvate() -> None:
    assert canonicalize_drug("Trametinib (DMSO_TF solvate)") == "Trametinib"
    assert drug_class_flags("Trametinib (DMSO_TF solvate)")["is_mek"] == 1


def test_design_matrix_has_braf_raf_interaction() -> None:
    frame = pd.DataFrame(
        [
            {
                "drug": "Dabrafenib",
                "concentration": 0.5,
                "has_kras": False,
                "has_braf": True,
                "has_tp53": True,
                "kras_g12": False,
                "braf_v600e": True,
                "tp53_lof": True,
            }
        ]
    )
    x, names = design_matrix(frame)
    assert names == list(FEATURE_NAMES)
    assert x.shape == (1, len(FEATURE_NAMES))
    assert x[0, names.index("is_raf")] == 1.0
    assert x[0, names.index("braf_v600e__is_raf")] == 1.0
    assert x[0, names.index("kras_g12__is_raf")] == 0.0


def test_line_holdout_keeps_each_class_in_train() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": [f"L{i}" for i in range(12)],
            "ras_raf_class": ["BRAF_V600E"] * 4 + ["KRAS_GOF"] * 4 + ["RAS_RAF_WT"] * 4,
        }
    )
    train, test = line_holdout(frame, seed=0, per_class={"BRAF_V600E": 2, "KRAS_GOF": 2, "RAS_RAF_WT": 2})
    assert int(test.sum()) > 0
    train_classes = set(frame.loc[train, "ras_raf_class"])
    assert train_classes == {"BRAF_V600E", "KRAS_GOF", "RAS_RAF_WT"}


def test_synthetic_ridge_beats_drug_mean_and_roundtrips(tmp_path: Path) -> None:
    frame, genes = synthetic_slice(seed=0)
    result = evaluate(frame, genes, lam=1.0, seed=0)
    ridge = result["ridge"]
    models = result["models"]
    assert models["genotype_ridge"]["mse"] < models["drug_mean"]["mse"]
    assert models["genotype_ridge"]["mse"] < models["zero"]["mse"]
    assert models["genotype_ridge"]["mse"] < models["identity_ridge"]["mse"]
    dab = result["dabrafenib_mapk_all"]
    assert dab["BRAF_V600E"] < dab["KRAS_GOF"]
    assert dab["KRAS_GOF"] < dab["RAS_RAF_WT"] + 0.2

    path = tmp_path / "ridge.npz"
    save_ridge(path, ridge, genes, FEATURE_NAMES)
    loaded, loaded_genes, loaded_names = load_ridge(path)
    assert loaded_genes == genes
    assert loaded_names == list(FEATURE_NAMES)
    drivers = {
        "has_kras": False,
        "has_braf": True,
        "has_tp53": True,
        "kras_g12": False,
        "braf_v600e": True,
        "tp53_lof": True,
    }
    pred = predict_delta(loaded, loaded_genes, drivers=drivers, drug="Dabrafenib", concentration=0.5)
    assert pred.mean() < -1.0
    kras_pred = predict_delta(
        loaded,
        loaded_genes,
        drivers={
            "has_kras": True,
            "has_braf": False,
            "has_tp53": False,
            "kras_g12": True,
            "braf_v600e": False,
            "tp53_lof": False,
        },
        drug="Dabrafenib",
        concentration=0.5,
    )
    assert pred.mean() < kras_pred.mean()
    weights = result["interaction_weights"]
    assert weights["braf_v600e__is_raf"] < weights["kras_g12__is_raf"]
    np.testing.assert_allclose(loaded.predict(design_matrix(frame)[0]), ridge.predict(design_matrix(frame)[0]))


def test_perturbation_model_uses_basal_and_drivers(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    pca_genes = ["G1", "G2", "G3", "G4"]
    mean = np.zeros(4)
    components = np.eye(4)[:2]
    genes = ["DUSP4", "EGR1"]
    feature_names = list(FEATURE_NAMES) + ["basal_pc0", "basal_pc1"]
    # Hand-built weights: last two features (PCs) plus braf×raf and bias.
    weights = np.zeros((len(feature_names) + 1, 2))
    weights[FEATURE_NAMES.index("braf_v600e__is_raf"), :] = -1.5
    weights[len(FEATURE_NAMES), :] = 0.4  # basal_pc0
    weights[-1, :] = -0.2
    model = PerturbationModel(
        genes=genes,
        feature_names=feature_names,
        weights=weights,
        pca_mean=mean,
        pca_components=components,
        pca_genes=pca_genes,
        lam=1.0,
    )
    path = tmp_path / "perturbation.npz"
    model.save(path)
    loaded = PerturbationModel.load(path)
    drivers_braf = {
        "has_kras": False,
        "has_braf": True,
        "has_tp53": True,
        "kras_g12": False,
        "braf_v600e": True,
        "tp53_lof": True,
    }
    high = pd.Series({"G1": 2.0, "G2": 0.0, "G3": 0.0, "G4": 0.0})
    low = pd.Series({"G1": 0.0, "G2": 0.0, "G3": 0.0, "G4": 0.0})
    pred_high = loaded.predict(basal=high, drug="Dabrafenib", concentration=0.5, drivers=drivers_braf)
    pred_low = loaded.predict(basal=low, drug="Dabrafenib", concentration=0.5, drivers=drivers_braf)
    pred_kras = loaded.predict(
        basal=low,
        drug="Dabrafenib",
        concentration=0.5,
        drivers={
            "has_kras": True,
            "has_braf": False,
            "has_tp53": False,
            "kras_g12": True,
            "braf_v600e": False,
            "tp53_lof": False,
        },
    )
    assert pred_high.mean() > pred_low.mean()
    assert pred_low.mean() < pred_kras.mean()
    _ = rng


def test_identity_onehot_zeros_unseen_lines() -> None:
    ids = pd.Series(["A", "B", "C"])
    encoded = identity_onehot(ids, ["A", "B"])
    assert encoded.shape == (3, 2)
    assert encoded[0, 0] == 1.0
    assert encoded[2].sum() == 0.0


def test_pca_roundtrip_recovers_rank1() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(12, 1))
    matrix = z @ np.array([[1.0, 2.0, 3.0]]) + 0.4
    mean, components = pca_fit(matrix, k=1)
    reconstructed = pca_transform(matrix, mean, components) @ components + mean
    assert np.mean((matrix - reconstructed) ** 2) < 1e-10
