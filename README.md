# Red River Drought ML (CNN–LSTM + SHAP)

Reproducible code & minimal dataset scaffold for the manuscript.

## What’s inside
- End-to-end pipeline: data → train → evaluate → interpret (SHAP)
- Minimal sample data folder: `data/sample/` (put tiny files here if needed; use Git LFS for larger ones)
- Config-driven experiments (`configs/baseline.yaml`)

## Quickstart (Conda)
```bash
conda env create -f environment.yml
conda activate drought-ml
```

### Run all (placeholder scripts)
```bash
bash scripts/run_all.sh
```

## Data availability
- Large-scale raw data are NOT included. Use `scripts/download_data.py` to fetch public sources (edit the script to point to ERA5/GLDAS, etc.).
- Minimal sample dataset: place a few tiny CSV/NetCDF slices under `data/sample/` for reproducibility.

## Citation
See `CITATION.cff` (fill in after acceptance or Zenodo DOI).

## License
Code: MIT (see `LICENSE`). Data follow original providers’ licenses.
