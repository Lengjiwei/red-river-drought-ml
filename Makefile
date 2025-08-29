.PHONY: all setup train evaluate interpret

setup:
	python -m pip install -r requirements.txt

train:
	python src/train.py --config configs/baseline.yaml

evaluate:
	python src/evaluate.py --config configs/baseline.yaml

interpret:
	python src/interpret.py --config configs/baseline.yaml

all: train evaluate interpret
