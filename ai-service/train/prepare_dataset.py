import argparse
import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(df: pd.DataFrame, label_col: str, seed: int, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Try stratified split when feasible; fall back gracefully for tiny/imbalanced data
    try:
        stratify_labels = df[label_col] if df[label_col].nunique() > 1 else None
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_labels,
        )
    except ValueError:
        # Fallback: non-stratified split
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )
    # Ensure validation is non-empty
    if len(val_df) == 0:
        val_df = train_df.sample(n=min(1, len(train_df)), random_state=seed)
        train_df = train_df.drop(val_df.index)
    return train_df, val_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset: stratified train/validation split")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="/workspace/ai-service/data/processed")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    # Basic cleaning: drop NA, strip whitespace
    df = df[[args.text_column, args.label_column]].dropna()
    df[args.text_column] = df[args.text_column].astype(str).str.strip()
    df[args.label_column] = df[args.label_column].astype(str).str.strip()
    df = df[df[args.text_column].str.len() > 0]
    # Drop duplicates
    df = df.drop_duplicates(subset=[args.text_column, args.label_column])

    # For extremely tiny datasets, increase validation share slightly to ensure at least 1 example
    effective_test_size = args.test_size
    if len(df) < 10:
        effective_test_size = min(0.33, max(0.2, args.test_size))

    train_df, val_df = stratified_split(df, args.label_column, args.seed, effective_test_size)

    train_path = os.path.join(args.out_dir, "train.csv")
    val_path = os.path.join(args.out_dir, "validation.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    print(f"Saved train to {train_path} ({len(train_df)} rows), validation to {val_path} ({len(val_df)} rows)")


if __name__ == "__main__":
    main()

