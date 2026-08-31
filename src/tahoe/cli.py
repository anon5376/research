"""Fetch Tahoe metadata and score Dabrafenib MAPK output by RAS/RAF genotype."""

from __future__ import annotations

import argparse
from pathlib import Path

from tahoe.de import class_means, iter_signature_rows, signature_by_genotype
from tahoe.genotype import summarize_drivers
from tahoe.metadata import default_cache_dir, load_table
from tahoe.slice import MAPK_OUTPUT_GENES


def _stream_de(max_rows: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(
        "tahoebio/Tahoe-100M",
        "pseudobulk_differential_expression",
        split="train",
        streaming=True,
    )
    kept: list[dict] = []
    scanned = 0
    for record in ds:
        scanned += 1
        kept.extend(
            iter_signature_rows(
                [record],
                drugs=("Dabrafenib",),
                genes=MAPK_OUTPUT_GENES,
                concentration=0.5,
            )
        )
        if max_rows is not None and scanned >= max_rows:
            break
        if scanned % 100_000 == 0:
            print(f"scanned {scanned:,} DE rows, kept {len(kept):,}", flush=True)
    print(f"done: scanned {scanned:,}, kept {len(kept):,}", flush=True)
    return kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument(
        "--max-de-rows",
        type=int,
        default=None,
        help="Stop after scanning this many DE rows (smoke test). Default: full scan.",
    )
    parser.add_argument(
        "--skip-de",
        action="store_true",
        help="Only refresh cell-line / drug metadata.",
    )
    args = parser.parse_args(argv)

    cache = args.cache_dir
    cell_lines = load_table("cell_line_metadata", cache_dir=cache)
    drugs = load_table("drug_metadata", cache_dir=cache)
    drivers = summarize_drivers(cell_lines)
    drivers_path = cache / "cell_line_drivers.csv"
    drivers.to_csv(drivers_path, index=False)
    print(f"wrote {drivers_path} ({len(drivers)} lines)")
    print(drivers["ras_raf_class"].value_counts().to_string())
    print(f"drug metadata: {len(drugs)} compounds")

    if args.skip_de:
        return 0

    import pandas as pd

    rows = _stream_de(args.max_de_rows)
    de = pd.DataFrame(rows)
    if de.empty:
        print("no Dabrafenib / MAPK-output DE rows in the scanned window")
        return 0
    scored = signature_by_genotype(de, drivers)
    summary = class_means(scored)
    out = cache / "dabrafenib_mapk_by_genotype.csv"
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
