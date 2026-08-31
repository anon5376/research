# Virtual Cell Challenge

Personal entry for the [2026 Arc Institute Virtual Cell Challenge](https://virtualcellchallenge.org/). The task is to predict how gene expression changes after CRISPRi knockdown in cell lines the model has never seen perturbed.

This repository is a working lab notebook: challenge notes, a format-correct **context-mean baseline**, and the path from control cells to a scored submission. The Challenge is an annual competition (grand prize $100,000; $50,000 / $25,000 for second and third) sponsored by NVIDIA, 10x Genomics, and Ultima Genomics.

## The 2026 task

There is **no Challenge-specific training set**. For each held-out cell line you receive:

1. Unperturbed (non-targeting control) expression profiles — the basal state of that context
2. The gene symbols to knock down

You must predict the post-perturbation count matrices that would be measured after CRISPRi + 10x Flex profiling. Ground truth is withheld.

Validation uses three unlabeled contexts `A`, `B`, `C`. The final phase switches to three different cell lines labeled `D`, `E`, `F`. Do not carry validation labels into a final submission.

| Date | Milestone |
| --- | --- |
| 20 Aug 2026 | Validation data live, leaderboard open |
| 22 Oct 2026 | Final test set released |
| 5 Nov 2026, 23:59 UTC | Final submissions due |
| Mid–late Nov 2026 | Winners announced |

Official write-up: [The 2026 Virtual Cell Challenge](https://arcinstitute.org/news/virtual-cell-challenge-2026).

## Scoring

Six metrics, each scaled so that **0** is the cell-context mean (this repo's first baseline) and **1** is a split-half replicate of the real experiment. Overall is their unweighted mean.

- Perturbation discrimination (`pds`)
- Expression accuracy (`mse`)
- DE log-FC accuracy (`nmae`)
- DE direction fidelity (`fid`)
- DE direction reach (`reach`)
- DE significance overlap (`jac`)

A score above 1 is possible on five of the six metrics (better than experimental replicate noise). Expression accuracy is capped at 1. Negative scores mean worse than the mean baseline.

## Submission shape (validation)

One `.vcc` file covering all three contexts:

- 300 perturbations × 400 cells × 3 contexts = **360,000 cells**
- **18,533** genes in official order
- Raw non-negative integer counts, stored sparse
- `obs["target_gene"]` = gene symbols (`ADNP`, not `ADNP-1`)
- `obs["context"]` = `A` / `B` / `C`, copied from the control files, never reassigned
- No non-targeting control cells in the prediction

`vcc prep` checks this locally before upload. Full rules: [VCC CLI wiki](https://vcc-cli-wiki.virtualcellchallenge.org/).

## Setup

Python 3.11+, [uv](https://docs.astral.sh/uv/), and a Challenge account.

```bash
uv sync --extra dev
uv tool install vcc-cli
```

Register at [virtualcellchallenge.org](https://virtualcellchallenge.org/), then mint an API key under Account → Credentials. Keys display once; a new key revokes the old one.

```bash
cp .env.example .env
# put the key in VCC_TOKEN, or:
echo "$VCC_TOKEN" | vcc login --token-stdin
vcc whoami
```

Download and unpack controls (hundreds of MB; not committed):

```bash
vcc datasets download controls -d data
unzip -o -j data/vcc_2026_controls.zip \
  "gene_names.csv" "pert_counts.csv" \
  "context_A.h5ad" "context_B.h5ad" "context_C.h5ad" \
  -d data
```

## Context-mean baseline

Paste each context's average unperturbed cell onto every perturbation. This is the official 0-point. It is still worth submitting once: it proves the pipeline and gives a reference score.

```bash
uv run vcc-baseline --data-dir data --output data/baseline_mean.h5ad
vcc prep data/baseline_mean.h5ad \
  -g data/gene_names.csv \
  --perts data/pert_counts.csv \
  -o data/baseline_mean.vcc
vcc submit data/baseline_mean.vcc -m "context-mean baseline" --wait
```

In `vcc prep`, `--perts` is the perturbation list. `-p` is a **column name** (`target_gene` by default). Do not copy `-p pert_counts.csv` from `vcc sample`.

## Layout

```
src/vcc_entry/     baseline + AnnData I/O
tests/             format checks on synthetic controls
data/              local controls and predictions (gitignored)
```

```bash
uv run pytest
```

## Approach (working)

2025 asked for interpolation inside one measured cell type (H1 hESC) given some training perturbations. 2026 asks for **zero-shot transfer across cell types**. Public Perturb-seq and the [Arc Virtual Cell Atlas](https://arcinstitute.org/tools/virtual-cell-atlas) are in bounds; so is Arc's [STATE](https://github.com/ArcInstitute/state) model (non-commercial).

Order of work:

1. Register, download controls, land the mean baseline on the leaderboard
2. Inspect basal states of A/B/C (which genes are on, library size, likely lineage) without trying to deanonymize them from the names — the labels are opaque on purpose
3. Train or adapt a perturbation model on public data so it can take (control cells, target gene) → predicted counts in an unseen context
4. Keep predictions in raw counts, sparse, 400 cells per perturbation, labels glued to the control file they came from

Useful reading:

- [STATE](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2) — Arc's virtual cell model (SE embeddings + ST transitions)
- [cell-eval](https://github.com/ArcInstitute/cell-eval) — the metric suite behind scoring
- [2025 wrap-up](https://arcinstitute.org/news/virtual-cell-challenge-2025-wrap-up) — Altos Labs won the generalist prize with a flow-matching model
- [Toward a Turing test for the virtual cell](https://www.cell.com/cell/fulltext/S0092-8674(25)00674-4) (*Cell*, 2025)

## License

MIT. Challenge data and third-party models keep their own terms (STATE is non-commercial).
