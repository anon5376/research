# Research

Public notes, code, and experiments. Mostly computational biology: gene expression, perturbations, and related biotech.

This used to be a Virtual Cell Challenge 2026 signup. That contest requires a Persona face scan, so it is not an active target. The leftover baseline code under `src/` is a format sketch, not a submission.

Current thread: [Tahoe-100M](https://huggingface.co/datasets/tahoebio/Tahoe-100M) — ~100M single-cell profiles, 50 cancer lines × small-molecule treatments, CC0.

```bash
uv sync --extra dev
uv run pytest
```

Data stays local and uncommitted. Copy `.env.example` to `.env` for tokens; never push secrets.
