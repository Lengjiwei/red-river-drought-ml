#!/usr/bin/env bash
set -e
python scripts/download_data.py --dest data/sample || true
python src/train.py || true
python src/evaluate.py || true
python src/interpret.py || true
