# Research

Public notes, code, and experiments. Mostly computational biology: gene expression, perturbations, and related biotech.

This used to be a Virtual Cell Challenge 2026 signup. That contest requires a Persona face scan, so it is not an active target. Leftover format-sketch code under `src/vcc_entry/` is not a submission.

**Current result:** an open model of chemical perturbation on [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M) (CC0). Input is a cell line’s DMSO transcriptome, a RAS/RAF/MEK drug, and KRAS/BRAF/TP53 flags — not the line name. Output is the 2k-gene treated−DMSO vector. On held-out lines, memorizing names overfits; driver flags help a subset of genes (EGR1 first), not the whole transcriptome.

Write-up: [`docs/results.md`](docs/results.md). Weights: [`models/perturbation.npz`](models/perturbation.npz).

```bash
uv sync --extra dev
uv run pytest
uv run tahoe-train --plates all --min-um 0.5 --genes hvg
```

```python
from pathlib import Path
from tahoe.bundle import PerturbationModel

model = PerturbationModel.load(Path("models/perturbation.npz"))
delta = model.predict(basal=dmso_hvg, drug="Dabrafenib", concentration=5.0, drivers=flags)
```

Plate downloads (~1 GB) stay local. Copy `.env.example` to `.env` for tokens; never push secrets.
