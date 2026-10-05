# Convenience targets; everything runs from the repository root with the framework on the path.
PY ?= python
export PYTHONPATH := src

.PHONY: install test lint format verify smoke figures index

install:          ## create the environment (uv)
	uv venv --python 3.12 && uv sync --frozen

test:             ## reference, config, runtime and end-to-end tests
	$(PY) -m pytest -q tests

lint:             ## the CI lint
	ruff check src scripts tests

format:           ## ruff format (single quotes, line length 120)
	ruff format src scripts tests

verify:           ## independent checks on Clothing (graphs, architecture, SASRec branch)
	$(PY) scripts/verify_graphs.py Clothing
	$(PY) scripts/indep_check_architecture.py --config configs/train/clothing64/05_full_l00.json --data data/Clothing
	$(PY) scripts/indep_check_sasrec.py --config configs/train/clothing64/sasrec_inbatch_l00.json --mclsr-config configs/train/clothing64/01_orig.json

smoke:            ## the whole pipeline on synthetic data, ~5 s on a CPU
	$(PY) -m pytest -q tests/test_end_to_end.py

figures:          ## RESULTS.md figures and the interval plot
	$(PY) scripts/plot_study_figures.py && $(PY) scripts/plot_paired_bootstrap.py

index:            ## RUN_INDEX.md
	$(PY) scripts/build_run_index.py
