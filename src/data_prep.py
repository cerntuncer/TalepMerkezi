import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold


def deduplicate_dataframe(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)
    after = len(df)
    print(f"Deduplicated: {before-after} duplicates removed (from {before} to {after}).")
    return df


def stratified_split(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    val_size: float,
    test_size: float,
    seed: int,
    group_col: Optional[str] = None,
):
    if group_col and group_col in df.columns and df[group_col].notna().any():
        print(f"Using group-aware stratified split on '{group_col}'.")
        # First split train+val vs test using groups
        gkf = StratifiedGroupKFold(n_splits=int(1 / test_size), shuffle=True, random_state=seed)
        # Take first split as test
        X = df[text_col].values
        y = df[label_col].values
        groups = df[group_col].values
        trainval_idx, test_idx = next(gkf.split(X, y, groups))
        df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        # Now split train vs val within trainval
        gkf_val = StratifiedGroupKFold(n_splits=int(1 / val_size), shuffle=True, random_state=seed)
        X_tv = df_trainval[text_col].values
        y_tv = df_trainval[label_col].values
        groups_tv = df_trainval[group_col].values
        train_idx, val_idx = next(gkf_val.split(X_tv, y_tv, groups_tv))
        df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
        df_val = df_trainval.iloc[val_idx].reset_index(drop=True)
    else:
        print("Using standard stratified split.")
        df_trainval, df_test = train_test_split(
            df,
            test_size=test_size,
            stratify=df[label_col],
            random_state=seed,
        )
        rel_val = val_size / (1.0 - test_size)
        df_train, df_val = train_test_split(
            df_trainval,
            test_size=rel_val,
            stratify=df_trainval[label_col],
            random_state=seed,
        )

    return df_train, df_val, df_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="Path to raw CSV file")
    parser.add_argument("--text_col", default="text", type=str)
    parser.add_argument("--label_col", default="label", type=str)
    parser.add_argument("--group_col", default="", type=str)
    parser.add_argument("--val_size", default=0.1, type=float)
    parser.add_argument("--test_size", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(args.input)
    assert args.text_col in df.columns and args.label_col in df.columns, "Input CSV must have text and label columns"

    # Clean label dtype: convert to string for mapping then keep mapping file
    if df[args.label_col].dtype.name != "int64" and df[args.label_col].dtype.name != "int32":
        print("Mapping string labels to integers...")
        class_names = sorted(df[args.label_col].astype(str).unique())
        name_to_id = {name: idx for idx, name in enumerate(class_names)}
        df[args.label_col] = df[args.label_col].astype(str).map(name_to_id)
        pd.Series(class_names).to_csv("data/processed/label_names.csv", index=False, header=False)
    else:
        # still save inferred label order
        class_names = sorted(df[args.label_col].unique().tolist())
        pd.Series(class_names).to_csv("data/processed/label_names.csv", index=False, header=False)

    df = deduplicate_dataframe(df, args.text_col, args.label_col)

    group_col = args.group_col if args.group_col else None
    train_df, val_df, test_df = stratified_split(
        df, args.text_col, args.label_col, args.val_size, args.test_size, args.seed, group_col
    )

    print("Class distribution (train/val/test):")
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = part[args.label_col].value_counts(normalize=True).sort_index()
        print(name, counts.to_dict())

    train_df.to_csv("data/processed/train.csv", index=False)
    val_df.to_csv("data/processed/val.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print("Saved processed splits under data/processed/.")


if __name__ == "__main__":
    main()

