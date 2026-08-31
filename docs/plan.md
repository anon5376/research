# Research plan: genotype-aware perturbation response on Tahoe-100M

## Objective

Given a cancer cell line’s basal (DMSO) state and a small molecule, predict how gene expression shifts. Show that the shift tracks **driver mutations** (KRAS, BRAF-V600E, TP53), not the cell-line name.

Dataset: [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M) (CC0). No contest KYC.

## Why this is the right first problem

The 2026 Virtual Cell Challenge is the same scientific question with CRISPRi and a face scan. Tahoe is chemical perturbation across ~50–100 cancer lines with published driver annotations. The Tahoe paper already reports that Dabrafenib’s RAS/RAF signature is stronger in BRAF-V600E lines than in KRAS-mutant ones. That is a falsifiable biology check before any fancy model.

Chemical inhibition ≠ CRISPRi knockdown. Treat that as a feature of the write-up, not a bug.

## Data we actually use (do not download 429 GB)

| Piece | Where | Size | Role |
| --- | --- | --- | --- |
| Cell-line drivers | `tahoebio/Tahoe-100M` `cell_line_metadata` | tiny | KRAS / BRAF / TP53 flags |
| Drug table | same, `drug_metadata` | tiny | RAF / RAS / MEK labels, SMILES |
| Pseudobulk deltas | `tahoebio/tahoe-de-rhaister` `cell_eval` | ~1 GB, 14 plates | `mean(treated) − mean(DMSO)` on ~2k genes |
| DMSO centroids | same, `zeroshot/cell_line_centroids_dmso_hvg.parquet` | ~10 MB | basal state for a later encoder |

Plate-matched DMSO is already baked into `cell_eval`. Long-format pdex (~40 GB, 4B rows) is the fallback if a gene is missing from the 2k panel.

## Slice

**Drugs (from Tahoe metadata, exact names):**
RAF: Dabrafenib, Encorafenib, Vemurafenib.
RAS: Adagrasib, RMC-6236, BI-3406.
MEK: Trametinib, Cobimetinib, Binimetinib, TAK-733.

**Lines:** collapse `cell_line_metadata` to one row per Cellosaurus ID with `ras_raf_class ∈ {BRAF_V600E, KRAS_GOF, RAS_RAF_WT, other}`.

**Genes:** MAPK-output panel intersected with the cell_eval 2k HVG set. On plate 1 that intersection is `DUSP4, EGR1, SPRY1, IER3, EGR4, CCND2, TRIB2` — DUSP6/SPRY2/ETV4 are not in the HVG file. Expand to the full 2k only after this panel works. Each plate is one dose (plate 1 is 0.05 µM); use `--plates all` for 0.5 and 5.0 µM.

## Model (draft in `src/tahoe/model.py`)

1. **Zero baseline** — predict no change (DMSO). Official “you learned nothing.”
2. **Drug-mean baseline** — ignore genotype; predict the mean delta of that drug on *training* lines. This is the thing we must beat on held-out lines.
3. **Genotype × drug-class ridge** — features:
   - binary drivers: `kras_g12`, `braf_v600e`, `tp53_lof`, `has_kras`, `has_braf`, `has_tp53`
   - drug class: RAF / RAS / MEK
   - interactions: `braf_v600e × RAF`, `kras_g12 × RAS`, `kras_g12 × RAF` (expected weaker Dabrafenib effect), `tp53_lof × *` as a control (should be weaker / unstructured)
   - optional: log-dose
   Multi-output ridge onto the MAPK (then HVG) delta vector.

No cell-line identity feature. If the model only works when you one-hot the line name, it has not learned genetics.

Later, if this ridge is real: replace drug class with a short SMILES encoder and basal DMSO HVG with a small MLP. Do not start there.

## Splits

Hold out **entire cell lines**, stratified by `ras_raf_class` (at least one BRAF-V600E, several KRAS, several WT). Every test drug is still seen in training, on other genotypes. That is the claim: transfer across genetic background.

Do not hold out a random 10% of wells. That leaks the same line’s other drugs and overstates performance.

## Success

- Reproducible: `uv run tahoe-train` from this repo with cached plates.
- On held-out lines, genotype ridge beats drug-mean on MAPK-output MSE and on Dabrafenib BRAF-vs-KRAS direction (BRAF-V600E more negative MAPK score).
- Write-up names failure cases (NRAS-only melanoma, dual BRAF+KRAS “other” class).
- Full 100M cells, Tahoe-x1, STATE, and CRISPRi stay out of scope until that check is true.

## Order of work

1. Metadata + driver table (done).
2. Pull `cell_eval` plates, score MAPK output by RAS/RAF class (Dabrafenib sanity).
3. Fit ridge vs baselines on a line-holdout split.
4. Expand gene set; add SMILES / basal encoder only if (3) is positive.
5. Public note in this repo with figures.

### Status (overnight)

All 14 `cell_eval` plates are cached locally (~1 GB, gitignored). The 50-line mosaic has **7 BRAF-V600E**, 22 KRAS GoF, 19 RAS/RAF WT, 2 other.

Dabrafenib MAPK-output mean (7 HVG genes, linear delta) at **5 µM**: BRAF-V600E `−0.082` vs KRAS `−0.022` vs WT `−0.026`. Direction matches the paper.

On a 10-line holdout, concentration ≥ 0.5 µM:

| Model | MSE | MAPK MSE | Pearson |
| --- | --- | --- | --- |
| Zero | 0.01301 | 0.00331 | — |
| Drug-class mean | 0.00443 | 0.00087 | 0.838 |
| Genotype × drug ridge | **0.00381** | **0.00080** | 0.824 |

`uv run tahoe-train --plates all --min-um 0.5` writes local cache weights. Public copy: `models/genotype_ridge.npz`. Write-up: `docs/results.md`.

The public predictor is `PerturbationModel` in `models/perturbation.npz`: DMSO HVG (8 PCs) + drug + KRAS/BRAF/TP53 → 2000-gene delta. `uv run tahoe-train --plates all --min-um 0.5 --genes hvg`. Write-up: `docs/results.md`.

On a 10-line holdout, **identity + basal** is the best training fit (MSE 0.00129) and worse than basal-only on unseen lines (0.00190 vs 0.00172). **Genotype + basal** is the best MAPK-output fit (0.00079) and helps 399/2000 genes (EGR1 first). Bulk HVG prefers basal + drug without genotype.
