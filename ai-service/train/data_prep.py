import argparse
import os
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def deduplicate(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)
    print(f"Removed {before - len(df)} duplicates")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--text_col", default="text")
    p.add_argument("--label_col", default="label")
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs("/workspace/ai-service/data/processed", exist_ok=True)
    df = pd.read_csv(args.input)

    assert args.text_col in df.columns and args.label_col in df.columns, "Missing required columns"

    df = deduplicate(df, args.text_col, args.label_col)

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=df[args.label_col], random_state=args.seed
    )
    rel_val = args.val_size / (1.0 - args.test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=rel_val, stratify=train_df[args.label_col], random_state=args.seed
    )

    out_dir = "/workspace/ai-service/data/processed"
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)
    print(f"Saved splits to {out_dir}")


if __name__ == "__main__":
    main()
