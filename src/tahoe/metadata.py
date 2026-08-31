"""Load Tahoe-100M metadata tables (small) and cache them locally."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HF_DATASET = "tahoebio/Tahoe-100M"


def default_cache_dir() -> Path:
    return Path("data") / "cache"


def load_table(name: str, cache_dir: Path | None = None) -> pd.DataFrame:
    """Load a named Hugging Face config (`cell_line_metadata`, `drug_metadata`, …)."""
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"{name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    from datasets import load_dataset

    frame = load_dataset(HF_DATASET, name, split="train").to_pandas()
    frame.to_csv(csv_path, index=False)
    return frame
