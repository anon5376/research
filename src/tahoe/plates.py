"""Download and slice tahoe-de-rhaister cell_eval plates (~70 MB each)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tahoe.genotype import summarize_drivers
from tahoe.metadata import default_cache_dir, load_table
from tahoe.slice import MAPK_OUTPUT_GENES, canonicalize_drug, is_ras_raf_drug, parse_treatment

CELL_EVAL_REPO = "tahoebio/tahoe-de-rhaister"
N_PLATES = 14
META_COLS = ("cell_line", "treatment", "plate", "target")


def normalize_plate(value: object) -> int | None:
    if value is None:
        return None
    text = str(value)
    if text.startswith("plate_plate"):
        return int(text.removeprefix("plate_plate"))
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def plate_filename(plate: int) -> str:
    if plate < 1 or plate > N_PLATES:
        raise ValueError(f"plate must be 1–{N_PLATES}, got {plate}")
    return f"cell_eval/plate_plate{plate}.parquet"


def download_plate(plate: int, cache_dir: Path | None = None) -> Path:
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = plate_filename(plate)
    local = cache_dir / filename
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=CELL_EVAL_REPO,
        filename=filename,
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    return Path(downloaded)


def mapk_columns(columns: list[str]) -> list[str]:
    present = {str(c) for c in columns}
    return [gene for gene in MAPK_OUTPUT_GENES if gene in present]


def gene_columns(raw: pd.DataFrame, mode: str) -> list[str]:
    mode = mode.lower()
    if mode == "mapk":
        genes = mapk_columns(list(raw.columns))
        if not genes:
            raise ValueError(f"none of {MAPK_OUTPUT_GENES} in plate columns")
        return genes
    if mode != "hvg":
        raise ValueError(f"gene mode must be mapk or hvg, got {mode}")
    skip = set(META_COLS)
    return [str(c) for c in raw.columns if str(c) not in skip]


def load_ras_raf_slice(
    *,
    cache_dir: Path | None = None,
    plates: list[int] | None = None,
    drivers: pd.DataFrame | None = None,
    gene_mode: str = "mapk",
) -> tuple[pd.DataFrame, list[str]]:
    """Wide deltas for RAS/RAF drugs, joined to driver flags."""
    cache_dir = cache_dir or default_cache_dir()
    plates = plates or [1]
    if drivers is None:
        drivers = summarize_drivers(load_table("cell_line_metadata", cache_dir=cache_dir))

    blocks: list[pd.DataFrame] = []
    genes_used: list[str] | None = None
    for plate in plates:
        path = download_plate(plate, cache_dir=cache_dir)
        raw = pd.read_parquet(path)
        genes = gene_columns(raw, gene_mode)
        if genes_used is None:
            genes_used = genes
        else:
            genes = [g for g in genes_used if g in raw.columns]
        keep = [c for c in META_COLS if c in raw.columns] + genes
        block = raw.loc[:, keep].copy()
        if "plate" not in block.columns:
            block["plate"] = plate
        else:
            block["plate"] = block["plate"].map(normalize_plate)
        blocks.append(block)

    assert genes_used is not None
    frame = pd.concat(blocks, ignore_index=True)
    treatment_col = "treatment" if "treatment" in frame.columns else "target"
    parsed = frame[treatment_col].map(parse_treatment)
    frame["drug_raw"] = [pair[0] for pair in parsed]
    frame["concentration"] = [pair[1] for pair in parsed]
    frame["drug"] = frame["drug_raw"].map(canonicalize_drug)
    frame = frame.loc[frame["drug_raw"].map(is_ras_raf_drug)].copy()
    frame["cell_id"] = frame["cell_line"].astype(str)
    if "plate" in frame.columns:
        frame["plate"] = frame["plate"].map(normalize_plate)
    labels = drivers.drop_duplicates(subset=["cell_id"])
    frame = frame.merge(labels, on="cell_id", how="inner")
    mapk_present = [g for g in MAPK_OUTPUT_GENES if g in frame.columns]
    frame["mapk_score"] = frame[mapk_present].mean(axis=1) if mapk_present else np.nan
    return frame.reset_index(drop=True), genes_used
