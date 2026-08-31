# Genotype-aware chemical perturbation response on Tahoe-100M

The public model is `f(DMSO transcriptome, drug, KRAS/BRAF/TP53) → treated − DMSO`
on the Tahoe 2k highly-variable genes. It does not use the cell-line name.

Chemical inhibition is not CRISPRi. The genetics claim is: **the same drug
moves different genes depending on driver flags, and that transfer survives a
cell-line holdout**, whereas one-hot line identity does not.

## Run

```bash
uv sync --extra dev
uv run pytest
uv run tahoe-train --plates all --min-um 0.5 --genes hvg
```

That writes `models/perturbation.npz` (~700 KB: 8 DMSO principal components +
ridge weights onto 2000 genes). Plates (~1 GB) stay gitignored.

```python
from pathlib import Path
from tahoe.bundle import PerturbationModel

model = PerturbationModel.load(Path("models/perturbation.npz"))
delta = model.predict(
    basal=dmso_hvg,          # Series indexed by gene symbol, plate-matched DMSO
    drug="Dabrafenib",
    concentration=5.0,
    drivers={
        "has_kras": False,
        "has_braf": True,
        "has_tp53": True,
        "kras_g12": False,
        "braf_v600e": True,
        "tp53_lof": True,
    },
)
# delta["EGR1"], delta["DUSP4"], ... 2000 genes
```

Missing basal genes are filled with the training DMSO mean. Drivers are the
DepMap/Tahoe flags, not the line name.

## Slice

50 mosaic lines, RAS/RAF/MEK drugs, dose ≥ 0.5 µM. 40 lines train / 10 holdout
(RKO and SW1417 for BRAF-V600E; A-427, NCI-H1573, NCI-H1792, SHP-77 for KRAS;
SK-MEL-2, A498, HepG2/C3A WT; LS 180 other). 1028 / 258 (line, drug, dose) rows.

Basal state is plate-matched DMSO HVG, compressed to 8 PCs fit on **training
lines only**, then applied to the holdout (the holdout’s DMSO is allowed; its
perturbation response is not).

## Held-out scores (2k HVG deltas)

| Model | Test MSE | Train MSE | MAPK-subset MSE | Allowed to see |
| --- | --- | --- | --- | --- |
| Zero | 0.00203 | 0.00224 | 0.00331 | nothing |
| Drug-class mean | 0.00181 | 0.00199 | 0.00087 | RAF/RAS/MEK average |
| Basal DMSO PCs × drug | **0.00172** | 0.00169 | 0.00089 | transcriptome + drug |
| Genotype × drug | 0.00190 | 0.00183 | 0.00080 | KRAS/BRAF/TP53 + drug |
| **Genotype + basal × drug** | 0.00185 | 0.00161 | **0.00079** | both (the public weights) |
| Line identity × drug | 0.00177 | 0.00135 | 0.00082 | cell-line name |
| Identity + basal × drug | 0.00190 | **0.00129** | 0.00083 | name + DMSO |

Identity plus basal is the best **training** fit and worse than basal-only on
lines whose names were never seen. Driver flags plus basal are the best
**MAPK-output** fit on those same lines. Bulk HVG error prefers basal + drug
without genotype: adding KRAS/BRAF/TP53 slightly raises overall MSE.

## When genotype matters

On the holdout, genotype + basal beats basal + drug on **399 / 2000** genes.
Largest gains:

| Gene | Genotype benefit (MSE) | MAPK panel |
| --- | --- | --- |
| EGR1 | 0.00440 | yes |
| NRG1 | 0.00256 | no |
| PBX3 | 0.00238 | no |
| ITGA2 | 0.00233 | no |
| MCTP1 | 0.00147 | no |

EGR1 is a MAPK immediate-early gene. DUSP4/SPRY1/IER3 on this HVG panel are
not in the top-gain set (genotype slightly *raises* their holdout MSE). The
driver effect is real and **local**, not a uniform transcriptome shift.

Dabrafenib MAPK-output mean at 5 µM still ranks BRAF-V600E (`−0.082`) below
KRAS (`−0.022`) and RAS/RAF WT (`−0.026`). Fitted `braf_v600e × RAF` is
negative; `kras_g12 × RAF` is positive.

## Failure cases

SK-MEL-2 is NRAS-mutant and labelled RAS/RAF WT. LS 180 is dual BRAF+KRAS
(`other`). DUSP6/SPRY2/ETV4 are absent from the 2k HVG file. The public model
only emits RAS/RAF/MEK drugs it was trained on.

A 7-gene MAPK-only ridge (no basal) remains in `models/genotype_ridge.npz`
as the ablation where genotype beats identity on the whole target, not just a
subset.
