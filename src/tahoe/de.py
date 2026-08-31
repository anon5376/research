"""Aggregate MAPK-output log2 fold-change by cell-line RAS/RAF class."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pandas as pd

from tahoe.genotype import summarize_drivers
from tahoe.slice import MAPK_OUTPUT_GENES, is_ras_raf_drug, parse_treatment


def _drug_name(row: pd.Series) -> str | None:
    if "drug" in row and pd.notna(row["drug"]):
        return str(row["drug"])
    if "target" in row and pd.notna(row["target"]):
        name, _ = parse_treatment(row["target"])
        return name
    if "treatment" in row and pd.notna(row["treatment"]):
        name, _ = parse_treatment(row["treatment"])
        return name
    return None


def _gene_name(row: pd.Series) -> str | None:
    for key in ("gene_name", "feature", "gene"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return None


def _lfc(row: pd.Series) -> float:
    for key in ("log2FoldChange", "fold_change"):
        if key in row and pd.notna(row[key]):
            value = float(row[key])
            if value == float("-inf") or value == float("inf"):
                return 0.0
            return value
    return 0.0


def _cell_id(row: pd.Series) -> str | None:
    for key in ("Cell_ID_Cellosaur", "cell_line", "cell_id"):
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return None


def _concentration(row: pd.Series) -> float | None:
    if "concentration" in row and pd.notna(row["concentration"]):
        return float(row["concentration"])
    treatment = row["target"] if "target" in row else row.get("treatment")
    _, conc = parse_treatment(treatment)
    return conc


def iter_signature_rows(
    records: Iterable[dict],
    *,
    drugs: Iterable[str] | None = None,
    genes: Iterable[str] = MAPK_OUTPUT_GENES,
    concentration: float | None = 0.5,
) -> Iterator[dict[str, object]]:
    """Yield long-form DE rows that sit in the RAS/RAF slice."""
    gene_set = set(genes)
    drug_set = {name.casefold() for name in drugs} if drugs is not None else None
    for record in records:
        row = pd.Series(record)
        gene = _gene_name(row)
        if gene not in gene_set:
            continue
        name = _drug_name(row)
        if drug_set is not None:
            if not name or name.casefold() not in drug_set:
                continue
        elif not is_ras_raf_drug(name):
            continue
        if concentration is not None:
            conc = _concentration(row)
            if conc is None or abs(conc - concentration) > 1e-9:
                continue
        cell = _cell_id(row)
        if not cell:
            continue
        yield {
            "cell_id": cell,
            "drug": name,
            "gene": gene,
            "log2fc": _lfc(row),
            "concentration": _concentration(row),
        }


def signature_by_genotype(
    de_rows: pd.DataFrame,
    drivers: pd.DataFrame,
) -> pd.DataFrame:
    """Mean MAPK-output log2FC per (cell line, drug), joined to RAS/RAF class."""
    if de_rows.empty:
        return de_rows
    per_line = (
        de_rows.groupby(["cell_id", "drug"], as_index=False)["log2fc"]
        .mean()
        .rename(columns={"log2fc": "mapk_log2fc"})
    )
    labels = drivers[["cell_id", "cell_name", "organ", "ras_raf_class", "braf_v600e", "kras_g12", "tp53_lof"]]
    return per_line.merge(labels, on="cell_id", how="left")


def class_means(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    return (
        scored.groupby(["drug", "ras_raf_class"], as_index=False)
        .agg(n_lines=("cell_id", "nunique"), mean_mapk_log2fc=("mapk_log2fc", "mean"))
        .sort_values(["drug", "ras_raf_class"])
        .reset_index(drop=True)
    )


def drivers_from_raw(raw_cell_lines: pd.DataFrame) -> pd.DataFrame:
    return summarize_drivers(raw_cell_lines)
