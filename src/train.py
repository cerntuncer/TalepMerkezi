import os
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
import evaluate


@dataclass
class Config:
    model_name: str
    output_dir: str
    artifacts_dir: str
    seed: int
    data: Dict[str, Any]
    training: Dict[str, Any]


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return Config(**cfg)


def compute_class_weights(train_csv: str, label_col: str) -> Optional[torch.Tensor]:
    df = pd.read_csv(train_csv)
    labels = df[label_col].values
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) <= 1:
        return None
    # inverse frequency weights
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.sum() * len(classes)
    weights_t = torch.tensor(weights, dtype=torch.float)
    print("Class weights:", {int(c): float(w) for c, w in zip(classes, weights_t)})
    return weights_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.artifacts_dir, exist_ok=True)

    model_name = cfg.model_name

    # Load datasets from CSV
    data_files = {
        "train": cfg.data["train_file"],
        "validation": cfg.data["val_file"],
        "test": cfg.data["test_file"],
    }
    raw_ds = load_dataset("csv", data_files=data_files)

    text_col = cfg.data["text_column"]
    label_col = cfg.data["label_column"]

    # Map string labels if any
    if raw_ds["train"].features[label_col].dtype == "string":
        classes = sorted(list(set(raw_ds["train"][label_col])))
        class_label = ClassLabel(names=classes)
        def map_labels(example):
            example[label_col] = class_label.str2int(example[label_col])
            return example
        raw_ds = raw_ds.map(map_labels)

    num_labels = len(set(raw_ds["train"][label_col]))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    max_length = int(cfg.training.get("max_length", 128))
    def tokenize_fn(example):
        return tokenizer(example[text_col], truncation=True, max_length=max_length)

    tokenized = raw_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model config with custom dropout if provided
    model_config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    dropout_prob = float(cfg.training.get("dropout_prob", 0.1))
    if hasattr(model_config, "hidden_dropout_prob"):
        model_config.hidden_dropout_prob = dropout_prob
    if hasattr(model_config, "attention_probs_dropout_prob"):
        model_config.attention_probs_dropout_prob = dropout_prob

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=model_config,
    )

    # Loss with label smoothing and optional class weights
    label_smoothing = float(cfg.training.get("label_smoothing_factor", 0.0))
    use_class_weights = bool(cfg.training.get("use_class_weights", False))
    class_weights = None
    if use_class_weights:
        cw = compute_class_weights(cfg.data["train_file"], label_col)
        if cw is not None:
            class_weights = cw.to(model.device)

    def custom_loss_fn(outputs, labels):
        logits = outputs.logits
        if label_smoothing > 0:
            # CrossEntropy with label smoothing
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=label_smoothing
            )
        else:
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        return loss_fn(logits, labels)

    # Metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    # Training arguments
    tcfg = cfg.training
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        evaluation_strategy=tcfg.get("eval_strategy", "epoch"),
        save_strategy=tcfg.get("save_strategy", "epoch"),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 32)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(tcfg.get("learning_rate", 2e-5)),
        num_train_epochs=float(tcfg.get("num_train_epochs", 3)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.1)),
        logging_steps=int(tcfg.get("logging_steps", 50)),
        load_best_model_at_end=True,
        metric_for_best_model=tcfg.get("metric_for_best_model", "f1_macro"),
        greater_is_better=bool(tcfg.get("greater_is_better", True)),
        fp16=bool(tcfg.get("fp16", True)),
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        report_to=["tensorboard"],
    )

    # Custom Trainer to inject custom loss
    class SmoothTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            loss = custom_loss_fn(outputs, labels)
            return (loss, outputs) if return_outputs else loss

    callbacks = []
    patience = int(cfg.training.get("early_stopping_patience", 2))
    if patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = SmoothTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    val_metrics = trainer.evaluate()
    print("Validation:", val_metrics)

    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])
    print("Test:", test_metrics)

    trainer.save_model(cfg.artifacts_dir)
    tokenizer.save_pretrained(cfg.artifacts_dir)

    # Save metrics
    pd.Series(val_metrics).to_json(os.path.join(cfg.output_dir, "val_metrics.json"))
    pd.Series(test_metrics).to_json(os.path.join(cfg.output_dir, "test_metrics.json"))


if __name__ == "__main__":
    main()

