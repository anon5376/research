"""Fit genotype × drug-class ridge vs baselines on a cell-line holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tahoe.features import DRUG_CLASS_COLS, FEATURE_NAMES, design_matrix
from tahoe.metadata import default_cache_dir
from tahoe.model import (
    DrugMeanBaseline,
    GenotypeRidge,
    ZeroBaseline,
    identity_onehot,
    mse,
    pearson_per_row,
)
from tahoe.slice import MAPK_OUTPUT_GENES
from tahoe.split import line_holdout


def synthetic_slice(seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    """BRAF × RAF and KRAS × RAS suppress MAPK output; WT barely moves."""
    rng = np.random.default_rng(seed)
    classes = (
        [("BRAF_V600E", True, False, True)] * 6
        + [("KRAS_GOF", False, True, False)] * 8
        + [("RAS_RAF_WT", False, False, False)] * 6
    )
    genes = ["DUSP4", "EGR1", "SPRY1"]
    drugs = (
        ("Dabrafenib", "raf", 0.5),
        ("Vemurafenib", "raf", 0.5),
        ("Adagrasib", "ras", 0.5),
        ("RMC-6236", "ras", 0.5),
        ("Trametinib", "mek", 0.5),
        ("Binimetinib", "mek", 0.5),
    )
    rows: list[dict[str, object]] = []
    for i, (klass, braf, kras, tp53) in enumerate(classes):
        cell_id = f"CVCL_{i:04d}"
        for drug, dclass, conc in drugs:
            effect = 0.0
            if dclass == "raf" and braf:
                effect = -1.8
            elif dclass == "raf" and kras:
                effect = -0.2
            elif dclass == "ras" and kras:
                effect = -1.4
            elif dclass == "mek" and (braf or kras):
                effect = -1.1
            elif dclass == "mek":
                effect = -0.3
            noise = rng.normal(0.0, 0.12, size=len(genes))
            record: dict[str, object] = {
                "cell_id": cell_id,
                "drug": drug,
                "concentration": conc,
                "ras_raf_class": klass,
                "has_kras": kras,
                "has_braf": braf,
                "has_tp53": tp53,
                "kras_g12": kras,
                "braf_v600e": braf,
                "tp53_lof": tp53,
            }
            for gene, delta in zip(genes, noise, strict=True):
                record[gene] = float(effect + delta)
            record["mapk_score"] = float(np.mean([record[g] for g in genes]))
            rows.append(record)
    return pd.DataFrame(rows), genes


def save_ridge(path: Path, model: GenotypeRidge, genes: list[str], feature_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert model.weights is not None
    np.savez(
        path,
        weights=model.weights,
        lam=np.asarray([model.lam]),
        genes=np.asarray(genes),
        feature_names=np.asarray(feature_names),
    )


def load_ridge(path: Path) -> tuple[GenotypeRidge, list[str], list[str]]:
    data = np.load(path, allow_pickle=False)
    model = GenotypeRidge(lam=float(data["lam"][0]))
    model.weights = data["weights"]
    return model, [str(g) for g in data["genes"]], [str(n) for n in data["feature_names"]]


def predict_delta(
    model: GenotypeRidge,
    genes: list[str],
    *,
    drivers: dict[str, object],
    drug: str,
    concentration: float,
) -> pd.Series:
    row = pd.DataFrame([{**drivers, "drug": drug, "concentration": concentration}])
    x, _ = design_matrix(row)
    return pd.Series(model.predict(x)[0], index=genes)


def _dabrafenib_class_means(frame: pd.DataFrame, mask: pd.Series) -> dict[str, float]:
    block = frame.loc[mask & (frame["drug"] == "Dabrafenib")]
    if block.empty or "mapk_score" not in block.columns:
        return {}
    return {str(k): float(v) for k, v in block.groupby("ras_raf_class")["mapk_score"].mean().items()}


def _score(y_true: np.ndarray, y_pred: np.ndarray, mapk_idx: list[int]) -> dict[str, float]:
    mapped = {
        "mse": mse(y_true, y_pred),
        "pearson": pearson_per_row(y_true, y_pred),
    }
    if mapk_idx:
        mapped["mapk_mse"] = float(
            np.mean((y_true[:, mapk_idx].mean(axis=1) - y_pred[:, mapk_idx].mean(axis=1)) ** 2)
        )
    else:
        mapped["mapk_mse"] = float(np.mean((y_true.mean(axis=1) - y_pred.mean(axis=1)) ** 2))
    return mapped


def _mapk_index(genes: list[str]) -> list[int]:
    return [i for i, gene in enumerate(genes) if gene in MAPK_OUTPUT_GENES]


def interaction_weights(ridge: GenotypeRidge, genes: list[str], feature_names: list[str]) -> dict[str, float]:
    """Mean weight across MAPK-output genes (or all genes if none overlap)."""
    assert ridge.weights is not None
    idx = _mapk_index(genes)
    columns = ridge.weights[:, idx] if idx else ridge.weights
    mean_w = columns.mean(axis=1)
    names = list(feature_names) + ["bias"]
    return {name: float(value) for name, value in zip(names, mean_w, strict=True)}


def evaluate(
    frame: pd.DataFrame,
    genes: list[str],
    *,
    lam: float = 1.0,
    seed: int = 0,
    centroids: pd.DataFrame | None = None,
    basal_k: int = 8,
) -> dict:
    train, test = line_holdout(frame, seed=seed)
    train_np = train.to_numpy()
    test_np = test.to_numpy()
    basal_cols: list[str] = []
    if centroids is not None:
        from tahoe.basal import attach_basal_pcs

        frame, basal_cols, _, _ = attach_basal_pcs(frame, centroids, train, k=basal_k)

    x_geno, names = design_matrix(frame)
    drug_idx = [names.index(col) for col in DRUG_CLASS_COLS] + [names.index("log10_um")]
    x_drug = x_geno[:, drug_idx]
    train_ids = sorted(frame.loc[train, "cell_id"].unique().tolist())
    x_id = np.hstack([x_drug, identity_onehot(frame["cell_id"], train_ids)])
    y = frame[genes].to_numpy(dtype=np.float64)
    y_train, y_test = y[train_np], y[test_np]
    mapk_idx = _mapk_index(genes)

    models: dict[str, GenotypeRidge | ZeroBaseline | DrugMeanBaseline] = {
        "zero": ZeroBaseline().fit(x_drug[train_np], y_train),
        "drug_mean": DrugMeanBaseline(class_index=tuple(range(len(DRUG_CLASS_COLS)))).fit(
            x_drug[train_np], y_train
        ),
        "genotype_ridge": GenotypeRidge(lam=lam).fit(x_geno[train_np], y_train),
        "identity_ridge": GenotypeRidge(lam=lam).fit(x_id[train_np], y_train),
    }
    predict_x = {
        "zero": x_drug,
        "drug_mean": x_drug,
        "genotype_ridge": x_geno,
        "identity_ridge": x_id,
    }
    geno_basal_names = names
    if basal_cols:
        basal_mat = frame[basal_cols].to_numpy(dtype=np.float64)
        x_basal_drug = np.hstack([x_drug, basal_mat])
        x_geno_basal = np.hstack([x_geno, basal_mat])
        models["basal_drug_ridge"] = GenotypeRidge(lam=lam).fit(x_basal_drug[train_np], y_train)
        models["genotype_basal_ridge"] = GenotypeRidge(lam=lam).fit(x_geno_basal[train_np], y_train)
        predict_x["basal_drug_ridge"] = x_basal_drug
        predict_x["genotype_basal_ridge"] = x_geno_basal
        geno_basal_names = names + basal_cols

    metrics: dict[str, object] = {
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "n_train_lines": int(frame.loc[train, "cell_id"].nunique()),
        "n_test_lines": int(frame.loc[test, "cell_id"].nunique()),
        "holdout_ids": sorted(frame.loc[test, "cell_id"].unique().tolist()),
        "genes": genes,
        "feature_names": names,
        "basal_cols": basal_cols,
        "models": {},
        "dabrafenib_mapk_test": _dabrafenib_class_means(frame, test),
        "dabrafenib_mapk_all": _dabrafenib_class_means(frame, pd.Series(True, index=frame.index)),
        "train_mse": {},
    }
    for label, model in models.items():
        pred_test = model.predict(predict_x[label][test_np])
        metrics["models"][label] = _score(y_test, pred_test, mapk_idx)  # type: ignore[index]
        pred_train = model.predict(predict_x[label][train_np])
        metrics["train_mse"][label] = mse(y_train, pred_train)  # type: ignore[index]

    ridge = models["genotype_ridge"]
    assert isinstance(ridge, GenotypeRidge)
    metrics["ridge"] = ridge
    metrics["interaction_weights"] = interaction_weights(ridge, genes, names)
    if "genotype_basal_ridge" in models:
        both = models["genotype_basal_ridge"]
        assert isinstance(both, GenotypeRidge)
        metrics["ridge_basal"] = both
        metrics["genotype_basal_feature_names"] = geno_basal_names
    return metrics


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return None
        return number
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--synthetic", action="store_true", help="Fit the biology toy slice (no download).")
    parser.add_argument(
        "--plates",
        default="1",
        help="Comma-separated plate numbers, or 'all'. Ignored with --synthetic.",
    )
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-um",
        type=float,
        default=0.0,
        help="Drop treatments below this micromolar dose (plate 1 is 0.05).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Weight file (default: data/cache/genotype_ridge.npz, or _synthetic.npz).",
    )
    parser.add_argument("--no-basal", action="store_true", help="Do not attach DMSO HVG principal components.")
    parser.add_argument(
        "--genes",
        choices=("mapk", "hvg"),
        default="mapk",
        help="Predict the MAPK-output panel or the full 2k HVG delta.",
    )
    args = parser.parse_args(argv)

    centroids = None
    if args.synthetic:
        frame, genes = synthetic_slice(seed=args.seed)
        source = "synthetic"
    else:
        from tahoe.plates import N_PLATES, load_ras_raf_slice

        plate_spec = args.plates.strip().lower()
        plates = list(range(1, N_PLATES + 1)) if plate_spec == "all" else [int(p) for p in plate_spec.split(",")]
        frame, genes = load_ras_raf_slice(
            cache_dir=args.cache_dir, plates=plates, gene_mode=args.genes
        )
        source = f"cell_eval plates {plates} genes={args.genes}"
        if args.min_um > 0:
            frame = frame.loc[frame["concentration"] >= args.min_um].reset_index(drop=True)
            source += f", conc>={args.min_um} uM"
            if frame.empty:
                print("no rows after --min-um filter")
                return 1
        if not args.no_basal:
            from tahoe.basal import load_dmso_centroids

            centroids = load_dmso_centroids(args.cache_dir)
            source += ", basal DMSO PCA"

    result = evaluate(frame, genes, lam=args.lam, seed=args.seed, centroids=centroids)
    ridge = result.pop("ridge")
    ridge_basal = result.pop("ridge_basal", None)
    default_name = "genotype_ridge_synthetic.npz" if args.synthetic else "genotype_ridge.npz"
    out_model = args.out or (args.cache_dir / default_name)
    save_ridge(out_model, ridge, genes, FEATURE_NAMES)
    if ridge_basal is not None:
        extra_names = result.get("genotype_basal_feature_names") or FEATURE_NAMES
        save_ridge(out_model.with_name(out_model.stem + "_basal.npz"), ridge_basal, genes, list(extra_names))
    metrics_path = out_model.with_suffix(".json")
    payload = {"source": source, **_jsonable({k: v for k, v in result.items()})}
    metrics_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"source: {source}")
    print(f"rows {result['n_train']} train / {result['n_test']} test; genes {len(genes)}")
    print(f"holdout lines: {result['holdout_ids']}")
    print("Dabrafenib MAPK mean (all):", result["dabrafenib_mapk_all"])
    print("Dabrafenib MAPK mean (test):", result["dabrafenib_mapk_test"])
    print("interaction weights (MAPK mean):", result.get("interaction_weights"))
    print("train MSE:", result.get("train_mse"))
    for name, scores in result["models"].items():  # type: ignore[union-attr]
        print(
            f"{name:22s}  mse={scores['mse']:.5f}  mapk_mse={scores['mapk_mse']:.5f}  pearson={scores['pearson']:.3f}"
        )
    print(f"wrote {out_model}")
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
