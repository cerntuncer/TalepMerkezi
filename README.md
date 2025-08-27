# BERT Turkish Training Project

This project provides a professional, reproducible pipeline to fine-tune Turkish BERT models with strong overfitting controls (early stopping, label smoothing, class weights, stratified splits, dedup, leakage guard).

## Day-by-Day Plan

Day 1: Data and scaffold
- Prepare raw data under `data/raw/` (CSV with `text,label`).
- Run data prep: deduplicate, stratified train/val/test split, optional group-based leakage control.
- Verify class balance and split metrics.

Day 2: Baseline training
- Run baseline training with early stopping and label smoothing.
- Track metrics (accuracy, macro-F1) and confusion matrix.
- Save best checkpoint and tokenizer to `artifacts/`.

Day 3: Tuning and robustness
- Small hyperparameter sweep (lr, weight_decay, dropout, max_length).
- Optional: class weighting vs. sampler; calibrate probabilities.
- Export final model, report test metrics.

## Quickstart

1) Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

2) Put your raw dataset as CSV under `data/raw/` with columns: `text,label`.

3) Run data prep
```bash
python src/data_prep.py --input data/raw/dataset.csv \
  --text_col text --label_col label \
  --val_size 0.1 --test_size 0.1 \
  --group_col "" --seed 42
```

4) Train
```bash
python src/train.py --config src/config.yaml
```

Outputs are saved under `outputs/` and the best model under `artifacts/`.

## Data format

CSV with columns:
- `text`: input string
- `label`: class id or class name (strings allowed; will be mapped)

## Notes

- Early stopping and `load_best_model_at_end` are enabled.
- Label smoothing and optional class weights reduce overfitting.
- For long texts, increase `max_length` or chunk upstream.
