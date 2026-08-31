# Research

Public notes, code, and experiments. Mostly computational biology: gene expression, perturbations, and related biotech.

This used to be a Virtual Cell Challenge 2026 signup. That contest requires a Persona face scan, so it is not an active target. Leftover format-sketch code under `src/vcc_entry/` is not a submission.

**Current result:** a genotype × drug ridge on [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M) (CC0). Given KRAS / BRAF / TP53 flags and a RAS/RAF/MEK inhibitor, it predicts a MAPK-output expression shift. On held-out cell lines it beats a drug-class mean, and it beats a model that memorizes cell-line names. Dabrafenib still hits BRAF-V600E lines harder than KRAS lines.

Write-up: [`docs/results.md`](docs/results.md). Plan: [`docs/plan.md`](docs/plan.md). Weights: [`models/genotype_ridge.npz`](models/genotype_ridge.npz).

```bash
uv sync --extra dev
uv run pytest
uv run tahoe-train --synthetic
uv run tahoe-train --plates all --min-um 0.5
```

The plate download (~1 GB) stays local and uncommitted. Copy `.env.example` to `.env` for tokens; never push secrets.
