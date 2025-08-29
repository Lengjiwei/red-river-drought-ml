# XGBoost-Based Monthly SRI Prediction with SHAP Explainability

This repository provides a complete pipeline to reproduce the results of our machine learning-based analysis for predicting Standardized Runoff Index (SRI) using XGBoost and interpreting the model with SHAP.

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
- Large-scale raw data are NOT included. Use `scripts/download_data.py` to fetch public sources (edit the script to point to ERA5-Land/MSWEP, etc.).
- Minimal sample dataset: place a few tiny CSV/NetCDF slices under `data/sample/` for reproducibility.

## Citation
See `CITATION.cff` (fill in after acceptance or Zenodo DOI).

## License
Code: MIT (see `LICENSE`). Data follow original providers’ licenses.
