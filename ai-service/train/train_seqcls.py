import argparse
import os
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sequence classification")
    parser.add_argument("--model_name_or_path", type=str, default="dbmdz/bert-base-turkish-cased")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data (CSV/JSON)")
    parser.add_argument("--validation_file", type=str, required=False, help="Path to validation data (CSV/JSON)")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--output_dir", type=str, default="/workspace/ai-service/models/seqcls")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(train_file: str, validation_file: str | None) -> Dict[str, object]:
    data_files = {"train": train_file}
    if validation_file:
        data_files["validation"] = validation_file
    ext = os.path.splitext(train_file)[1].lstrip(".")
    dataset = load_dataset(ext, data_files=data_files)
    return dataset


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_data(args.train_file, args.validation_file)

    # Create label mapping
    if "train" not in dataset:
        raise RuntimeError("Dataset must contain a train split")
    train_labels = list(set(dataset["train"][args.label_column]))
    # Keep label order deterministic
    train_labels = sorted(train_labels)
    label2id = {label: idx for idx, label in enumerate(train_labels)}
    id2label = {v: k for k, v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    def preprocess(examples: Dict[str, List[str]]):
        texts = examples[args.text_column]
        enc = tokenizer(texts, truncation=True, max_length=args.max_length)
        enc["labels"] = [label2id[label] for label in examples[args.label_column]]
        return enc

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).astype(np.float32).mean().item()
        return {"accuracy": accuracy}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        evaluation_strategy="epoch" if "validation" in tokenized else "no",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_strategy="epoch",
        logging_steps=50,
        seed=args.seed,
        load_best_model_at_end=("validation" in tokenized),
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if "validation" in tokenized else None,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()