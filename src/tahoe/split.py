"""Hold out entire cell lines, stratified by RAS/RAF class."""

from __future__ import annotations

import pandas as pd


def line_holdout(
    frame: pd.DataFrame,
    *,
    seed: int = 0,
    per_class: dict[str, int] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return boolean masks (train, test) over `frame` rows.

    Never hold out a whole class: at least one line of each present class stays
    in training so genotype × drug interactions remain identifiable.
    """
    per_class = per_class or {"BRAF_V600E": 2, "KRAS_GOF": 4, "RAS_RAF_WT": 3, "other": 1}
    hold: set[str] = set()
    lines = frame[["cell_id", "ras_raf_class"]].drop_duplicates()
    for klass, n_hold in per_class.items():
        pool = lines.loc[lines["ras_raf_class"] == klass, "cell_id"].tolist()
        if len(pool) < 2:
            continue
        shuffled = pd.Series(pool).sample(frac=1.0, random_state=seed).tolist()
        cap = max(1, len(shuffled) // 3)
        hold.update(shuffled[: min(n_hold, cap)])
    test = frame["cell_id"].isin(hold)
    train = ~test
    if int(train.sum()) == 0 or int(test.sum()) == 0:
        raise ValueError("line holdout produced an empty split")
    return train, test
