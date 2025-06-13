import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
from langdetect import DetectorFactory
from job_fraud_detection.data.general_preprocessing import (
    read_csv, read_user_input, preprocess_dataframe, remove_feature_name_row)
import sys
sys.path.append('.')

DetectorFactory.seed = 0


def split_and_save(df: pd.DataFrame, output_dir: str) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["fraudulent"], test_size=0.3, random_state=42,
        stratify=df["fraudulent"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(
        f"{output_dir}/train_bert.csv", index=False)
    pd.DataFrame({"text": X_val, "label": y_val}).to_csv(
        f"{output_dir}/val_bert.csv", index=False)
    pd.DataFrame({"text": X_test, "label": y_test}).to_csv(
        f"{output_dir}/test_bert.csv", index=False)


def analyze_tokens(df: pd.DataFrame) -> None:
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    token_lengths = df["text"].apply(
        lambda x: len(tokenizer(x, truncation=False)["input_ids"]))
    print(f"Average token length: {token_lengths.mean():.2f}")
    print(f"Max token length: {token_lengths.max()}")
    print(f"Median token length: {token_lengths.median():.2f}")
    print(f"95th percentile: {np.percentile(token_lengths, 95):.2f}")
    print(f"99th percentile: {np.percentile(token_lengths, 99):.2f}")
    df["token_length"] = token_lengths
    max_idx = df["token_length"].idxmax()
    print(f"Longest text has {df.loc[max_idx, 'token_length']} tokens:\n")
    print(df.loc[max_idx, "text"])
    print(f"Label : {df.loc[max_idx, 'fraudulent']}")
    print(df["fraudulent"].value_counts(normalize=True))


def compute_class_weights(y_train: pd.Series) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced",
                                   classes=classes, y=y_train)
    return dict(zip(classes, weights))


def main(data: dict = None, input_dir=None, output_dir=None,
         multimodality: bool = False):
    if data:
        df = read_user_input(data)
    else:
        df = read_csv(input_dir)

    if not multimodality:
        df = preprocess_dataframe(df)
        df = remove_feature_name_row(df)

    # uncomment this to view statistics of the BERT tokenizer:
    # analyze_tokens(df)

    if data is not None:
        return df
    else:
        split_and_save(df, output_dir)


if __name__ == "__main__":
    main(input_dir="data/raw", output_dir="data/processed")
