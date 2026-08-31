"""Driver-mutation labels for Tahoe cell lines.

Tahoe ships one row per (cell line, driver). We collapse that into flags the
model can condition on: KRAS / BRAF / TP53 status, and the RAS/RAF class used
in the Tahoe paper's Dabrafenib analysis.
"""

from __future__ import annotations

import pandas as pd

DRIVER_GENES = ("KRAS", "BRAF", "TP53")

_BRAF_V600E = ("V600E", "p.V600E")
_KRAS_G12 = ("G12", "p.G12")


def _effect(row: pd.Series) -> str:
    value = row.get("Driver_ProtEffect_or_CdnaEffect")
    return "" if pd.isna(value) else str(value)


def _mech(row: pd.Series) -> str:
    value = row.get("Driver_Mech_InferDM")
    return "" if pd.isna(value) else str(value)


def summarize_drivers(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per cell line with KRAS / BRAF / TP53 flags."""
    required = {"Cell_ID_Cellosaur", "Driver_Gene_Symbol"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"cell-line metadata missing columns: {sorted(missing)}")

    records: list[dict[str, object]] = []
    grouped = raw.groupby("Cell_ID_Cellosaur", dropna=False, sort=False)
    for cell_id, block in grouped:
        if pd.isna(cell_id):
            continue
        genes = set(block["Driver_Gene_Symbol"].dropna().astype(str))
        kras_rows = block[block["Driver_Gene_Symbol"] == "KRAS"]
        braf_rows = block[block["Driver_Gene_Symbol"] == "BRAF"]
        tp53_rows = block[block["Driver_Gene_Symbol"] == "TP53"]

        kras_g12 = any(any(tag in _effect(row) for tag in _KRAS_G12) for _, row in kras_rows.iterrows())
        braf_v600e = any(
            any(tag in _effect(row) for tag in _BRAF_V600E) for _, row in braf_rows.iterrows()
        )
        tp53_lof = any(_mech(row) == "LoF" for _, row in tp53_rows.iterrows()) or ("TP53" in genes)

        first = block.iloc[0]
        records.append(
            {
                "cell_id": str(cell_id),
                "cell_name": first.get("cell_name"),
                "organ": first.get("Organ"),
                "depmap_id": first.get("Cell_ID_DepMap"),
                "has_kras": "KRAS" in genes,
                "has_braf": "BRAF" in genes,
                "has_tp53": "TP53" in genes,
                "kras_g12": kras_g12,
                "braf_v600e": braf_v600e,
                "tp53_lof": bool(tp53_lof),
                "driver_genes": ",".join(sorted(genes)),
            }
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame["ras_raf_class"] = frame.apply(ras_raf_class, axis=1)
    return frame.reset_index(drop=True)


def ras_raf_class(row: pd.Series) -> str:
    """Paper-style strata: BRAF-V600E, KRAS GoF, dual-WT, or mixed/other."""
    braf_v600e = bool(row.get("braf_v600e"))
    has_braf = bool(row.get("has_braf"))
    kras = bool(row.get("has_kras") or row.get("kras_g12"))
    if braf_v600e and not kras:
        return "BRAF_V600E"
    if kras and not has_braf:
        return "KRAS_GOF"
    if not kras and not has_braf:
        return "RAS_RAF_WT"
    return "other"
