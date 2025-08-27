import argparse
import os
from typing import Dict, List

import numpy as np
from datasets import load_dataset
import evaluate
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
    parser.add_argument("--output_dir", type=str, default="/workspace/ai-service/models/bert")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--use_class_weights", action="store_true")
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

    # Create label mapping from all available splits to avoid unseen-label errors
    if "train" not in dataset:
        raise RuntimeError("Dataset must contain a train split")
    all_labels = set(dataset["train"][args.label_column])
    if "validation" in dataset:
        all_labels.update(dataset["validation"][args.label_column])
    if "test" in dataset:
        all_labels.update(dataset["test"][args.label_column])
    labels_sorted = sorted(list(all_labels))
    label2id = {label: idx for idx, label in enumerate(labels_sorted)}
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

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    # Enable mixed precision only when CUDA or bfloat16 is available
    import torch
    use_fp16 = torch.cuda.is_available()
    use_bf16 = (not use_fp16) and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

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
        metric_for_best_model="f1_macro" if "validation" in tokenized else None,
        greater_is_better=True,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_accumulation_steps=1 if len(tokenized["train"]) >= 16 else 2,
    )

    # Optional class weights and label smoothing
    class_weights = None
    if args.use_class_weights:
        # compute from training set
        from collections import Counter
        labels_list = tokenized["train"]["labels"]
        label_counts = Counter(labels_list)
        num_classes = len(label_counts)
        total = sum(label_counts.values())
        weights = []
        for i in range(num_classes):
            cnt = label_counts.get(i, 1)
            weights.append(total / (num_classes * cnt))
        import torch
        class_weights = torch.tensor(weights, dtype=torch.float)

    def custom_loss(outputs, labels):
        import torch
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing if args.label_smoothing > 0 else 0.0)
        return loss_fn(outputs.logits, labels)

    class SmoothTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
            loss = custom_loss(outputs, labels)
            return (loss, outputs) if return_outputs else loss

    from transformers import EarlyStoppingCallback

    trainer = SmoothTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if "validation" in tokenized else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)] if "validation" in tokenized else None,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()