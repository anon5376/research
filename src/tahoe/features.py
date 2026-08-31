"""Feature construction: drivers × drug class, never cell-line identity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tahoe.slice import (
    MEK_INHIBITORS,
    RAF_INHIBITORS,
    RAS_INHIBITORS,
    canonicalize_drug,
    parse_treatment,
)

GENOTYPE_COLS = (
    "has_kras",
    "has_braf",
    "has_tp53",
    "kras_g12",
    "braf_v600e",
    "tp53_lof",
)

DRUG_CLASS_COLS = ("is_raf", "is_ras", "is_mek")

INTERACTION_PAIRS = (
    ("braf_v600e", "is_raf"),
    ("kras_g12", "is_ras"),
    ("kras_g12", "is_raf"),
    ("has_kras", "is_mek"),
    ("braf_v600e", "is_mek"),
    ("tp53_lof", "is_raf"),
)

FEATURE_NAMES = (
    list(GENOTYPE_COLS)
    + list(DRUG_CLASS_COLS)
    + ["log10_um"]
    + [f"{g}__{d}" for g, d in INTERACTION_PAIRS]
)


def drug_class_flags(name: str | None) -> dict[str, int]:
    canon = canonicalize_drug(name) or name or ""
    key = str(canon).casefold()
    return {
        "is_raf": int(any(d.casefold() == key for d in RAF_INHIBITORS)),
        "is_ras": int(any(d.casefold() == key for d in RAS_INHIBITORS)),
        "is_mek": int(any(d.casefold() == key for d in MEK_INHIBITORS)),
    }


def _value(row: pd.Series, key: str):
    if key not in row.index:
        return None
    value = row[key]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and not value.strip():
        return None
    return value


def design_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Rows = (cell line, treatment). Columns = FEATURE_NAMES."""
    rows: list[list[float]] = []
    for _, row in frame.iterrows():
        flags = {col: float(bool(_value(row, col) or False)) for col in GENOTYPE_COLS}
        name = _value(row, "drug")
        if name is None:
            name, _ = parse_treatment(_value(row, "treatment") or _value(row, "target"))
        classes = drug_class_flags(None if name is None else str(name))
        conc = _value(row, "concentration")
        if conc is None:
            _, parsed = parse_treatment(_value(row, "treatment") or _value(row, "target"))
            conc = parsed
        try:
            conc_f = float(conc) if conc is not None else None
        except (TypeError, ValueError):
            conc_f = None
        log_dose = float(np.log10(conc_f)) if conc_f not in (None, 0) else -2.0
        vector = [flags[c] for c in GENOTYPE_COLS]
        vector += [float(classes[c]) for c in DRUG_CLASS_COLS]
        vector.append(log_dose)
        vector += [flags[g] * classes[d] for g, d in INTERACTION_PAIRS]
        rows.append(vector)
    return np.asarray(rows, dtype=np.float64), list(FEATURE_NAMES)
