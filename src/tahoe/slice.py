"""Fixed first slice: RAS/RAF drugs and a compact MAPK-output signature.

The Tahoe paper scores a ~600-gene MSigDB RAS/RAF set. We start with a small
literature MAPK-output panel so a streamed DE subset is interpretable and
testable without downloading the full atlas.
"""

from __future__ import annotations

import ast
import re

# Exact Tahoe drug_metadata names for genotype-conditioned MAPK response.
RAS_RAF_DRUGS = (
    "Dabrafenib",
    "Vemurafenib",
    "Encorafenib",
    "Adagrasib",
    "RMC-6236",
    "BI-3406",
    "Trametinib",
    "Cobimetinib",
    "Binimetinib",
    "TAK-733",
)

RAF_INHIBITORS = ("Dabrafenib", "Vemurafenib", "Encorafenib")
RAS_INHIBITORS = ("Adagrasib", "RMC-6236", "BI-3406")
MEK_INHIBITORS = ("Trametinib", "Cobimetinib", "Binimetinib", "TAK-733")

# Immediate-early / feedback genes induced by MAPK signaling.
# Tahoe cell_eval is a 2k HVG panel; only a subset is present (DUSP4, EGR1,
# SPRY1, IER3, …). Intersection happens at load time.
MAPK_OUTPUT_GENES = (
    "DUSP4",
    "DUSP5",
    "DUSP6",
    "SPRY1",
    "SPRY2",
    "SPRY4",
    "ETV1",
    "ETV4",
    "ETV5",
    "FOSL1",
    "EGR1",
    "EGR4",
    "CCND1",
    "CCND2",
    "MYC",
    "PHLDA1",
    "IER3",
    "TRIB2",
)

_TREATMENT_RE = re.compile(r"\['?\"?([^'\"\]]+)'?\"?\s*,\s*([0-9.]+)")


def canonicalize_drug(name: str | None) -> str | None:
    """Map Tahoe labels like `Trametinib (DMSO_TF solvate)` onto RAS_RAF_DRUGS."""
    if not name:
        return None
    lowered = name.casefold().strip()
    for drug in RAS_RAF_DRUGS:
        key = drug.casefold()
        if lowered == key or lowered.startswith(f"{key} ") or lowered.startswith(f"{key}("):
            return drug
    return None


def parse_treatment(raw: object) -> tuple[str | None, float | None]:
    """Parse Tahoe treatment labels such as `[('Dabrafenib', 0.5, 'uM')]`."""
    if raw is None:
        return None, None
    text = str(raw).strip()
    if not text:
        return None, None
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and parsed:
            first = parsed[0]
            if isinstance(first, (list, tuple)) and first:
                name = str(first[0])
                conc = float(first[1]) if len(first) > 1 else None
                return name, conc
    except (SyntaxError, ValueError, TypeError):
        pass
    match = _TREATMENT_RE.search(text)
    if match:
        return match.group(1), float(match.group(2))
    return text, None


def is_ras_raf_drug(name: str | None) -> bool:
    return canonicalize_drug(name) is not None
