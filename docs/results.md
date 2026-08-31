# Genotype-aware chemical perturbation response on Tahoe-100M

Tahoe-100M is a CC0 mosaic drug screen (~50 cancer lines × ~379 compounds).
This note is a first public model of **when a drug-induced expression shift
tracks driver mutations** rather than the cell-line name.

Chemical inhibition is not CRISPRi. The claim is narrower: RAF/RAS/MEK
inhibitors move a MAPK-output panel more in BRAF-V600E lines than in KRAS-mutant
ones, and a linear model that sees only those flags transfers to held-out lines.

## What you need

```bash
uv sync --extra dev
uv run pytest
uv run tahoe-train --synthetic
uv run tahoe-train --plates all --min-um 0.5
```

The first real run downloads ~1 GB of `tahoebio/tahoe-de-rhaister` `cell_eval`
plates plus a 10 MB DMSO centroid table (gitignored). Fitted MAPK weights that
match this note live in `models/genotype_ridge.npz`.

Predict for one line:

```python
from tahoe.train import load_ridge, predict_delta

model, genes, _ = load_ridge("models/genotype_ridge.npz")
delta = predict_delta(
    model,
    genes,
    drivers={
        "has_kras": False,
        "has_braf": True,
        "has_tp53": True,
        "kras_g12": False,
        "braf_v600e": True,
        "tp53_lof": True,
    },
    drug="Dabrafenib",
    concentration=5.0,
)
```

Features are driver flags × drug class (RAF / RAS / MEK) and log-dose. There is
no cell-line identity column.

## Slice

| Axis | What |
| --- | --- |
| Lines | 50 Cellosaurus IDs in the mosaic. **7 BRAF-V600E**, 22 KRAS GoF, 19 RAS/RAF WT, 2 other (dual BRAF+KRAS). |
| BRAF-V600E | COLO 205, HT-29, RKO, C32, LOX-IMVI, RPMI-7951, SW1417 |
| Drugs | Dabrafenib, Vemurafenib, Encorafenib, Adagrasib, RMC-6236, BI-3406, Trametinib, Cobimetinib, Binimetinib, TAK-733 |
| Dose | ≥ 0.5 µM (0.05 µM is a weak separate plate) |
| Response | plate-matched `mean(treated) − mean(DMSO)` on the 2k HVG panel |
| MAPK genes present | DUSP4, SPRY1, EGR1, EGR4, CCND2, IER3, TRIB2 (DUSP6/SPRY2/ETV4 are not in the HVG file) |

Split: **10 entire cell lines** held out, stratified by RAS/RAF class (RKO and
SW1417 for BRAF; A-427, NCI-H1573, NCI-H1792, SHP-77 for KRAS; SK-MEL-2, A498,
HepG2/C3A for WT; LS 180 as other). Every test drug is still seen in training.

## Biology check

Dabrafenib MAPK-output mean (linear delta, 7 genes):

| Dose | BRAF-V600E | KRAS GoF | RAS/RAF WT |
| --- | --- | --- | --- |
| 0.05 µM | −0.059 | −0.013 | −0.016 |
| 0.5 µM | −0.046 | −0.028 | −0.020 |
| 5 µM | **−0.082** | −0.022 | −0.026 |

Direction matches the Tahoe paper: Dabrafenib hits BRAF-V600E harder than KRAS.

The fitted interaction `braf_v600e × RAF` is negative on the MAPK mean
(−0.032); `kras_g12 × RAF` is positive (+0.012). RAF inhibitors are allowed to
do less in KRAS lines. `tp53_lof × RAF` is also negative (−0.018) — TP53 is not
a clean negative control on this panel.

## Held-out lines (MAPK panel)

Same 10-line split, concentration ≥ 0.5 µM, 1028 train / 258 test rows.

| Model | Test MSE | Train MSE | What it is allowed to see |
| --- | --- | --- | --- |
| Zero | 0.01301 | 0.01603 | nothing |
| Drug-class mean | 0.00443 | 0.00732 | RAF/RAS/MEK average on train lines |
| **Genotype × drug ridge** | **0.00381** | 0.00635 | KRAS/BRAF/TP53 flags × drug class |
| Line-identity × drug ridge | 0.00425 | **0.00429** | one-hot cell-line name (zeros on test lines) |
| Basal DMSO PCA × drug | 0.00453 | 0.00638 | 8 PCs of plate-matched DMSO HVG |
| Genotype + basal × drug | 0.00409 | 0.00595 | both |

Identity is the best fit **on training lines** and worse than genotype on
lines it has never seen. That is the genetics check: memorizing names does not
transfer; driver flags do.

DMSO principal components (the basal transcriptome the objective asked for)
help on the training set and do **not** beat the drug mean on held-out MAPK
response. For this slice, the BRAF/KRAS labels carry the Dabrafenib interaction
more cleanly than eight DMSO PCs. The genotype+basal model is saved as
`data/cache/genotype_ridge_basal.npz` when you train with plates cached; it is
not the default public weights.

## When it is not genotype

On the **full 2k HVG delta** (same split), genotype loses on overall MSE
(0.00190) to a drug mean (0.00181) and to basal DMSO PCs (0.00172). The MAPK
subset of that 2k run still prefers genotype (MAPK MSE 0.00080 vs drug mean
0.00087). Driver dependence shows up in the RAS/RAF output genes, not as a
uniform transcriptome shift.

Failure cases already in the split: SK-MEL-2 is NRAS-mutant and labelled
RAS/RAF WT (we only flag KRAS/BRAF/TP53). LS 180 is dual BRAF+KRAS and sits in
`other`. Both are expected misses for a KRAS-vs-BRAF-V600E story.

## Not this

Full 100M cells, Tahoe-x1, STATE, SMILES encoders, and any face-scanned
challenge. Next gain on this slice is DUSP6/SPRY2/ETV4 from long-form pdex, not
a deeper net.
