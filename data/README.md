# Local challenge data

Download the 2026 controls bundle here. Nothing in this directory is committed except this file.

```bash
uv tool install vcc-cli
vcc datasets download controls -d data
unzip -o -j data/vcc_2026_controls.zip \
  "gene_names.csv" "pert_counts.csv" \
  "context_A.h5ad" "context_B.h5ad" "context_C.h5ad" \
  -d data
```

Expected files after unpacking:

- `context_A.h5ad`, `context_B.h5ad`, `context_C.h5ad` — 18,400 non-targeting control cells each
- `gene_names.csv` — 18,533 gene symbols, headerless, fixed order
- `pert_counts.csv` — 300 CRISPRi targets as gene symbols (one column)

Keep API tokens out of this folder. Use `.env` or `vcc login --token-stdin`.
