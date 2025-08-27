# ai-service Training Guide

Training is integrated under `ai-service/train/` and models are saved under `ai-service/models/bert`.

## Day-by-Day Plan

Day 1: Data and split
- Place raw CSV at `ai-service/data/raw/dataset.csv` with columns `text,label`.
- Run data prep to deduplicate and stratify into train/val/test.

Day 2: Baseline training
- Train BERT (dbmdz/bert-base-turkish-cased) with early stopping + label smoothing.

Day 3: Tuning
- Small HPO sweep for lr, wd, dropout, max_length; optional class weights.

Day 4: Finalize
- Final train, test report, export to `ai-service/models/bert`.

## Commands

Install deps (service):
```bash
pip install -r ai-service/requirements.txt
```

Split data:
```bash
python ai-service/train/data_prep.py --input ai-service/data/raw/dataset.csv \
  --text_col text --label_col label --val_size 0.1 --test_size 0.1 --seed 42
```

Train:
```bash
python ai-service/train/train_seqcls.py \
  --model_name_or_path dbmdz/bert-base-turkish-cased \
  --train_file ai-service/data/processed/train.csv \
  --validation_file ai-service/data/processed/val.csv \
  --output_dir ai-service/models/bert \
  --num_train_epochs 5 \
  --per_device_batch_size 16 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --max_length 128 \
  --label_smoothing 0.1 \
  --early_stopping_patience 2 \
  --use_class_weights
```

Service uses `SEQ_CLS_MODEL_DIR` (default `ai-service/models/bert`).
