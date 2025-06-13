import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from langdetect import DetectorFactory
import json
from job_fraud_detection.saver import Saver
from job_fraud_detection.data.general_preprocessing import (
    read_csv, read_user_input, preprocess_dataframe, remove_feature_name_row)
import sys
sys.path.append('.')

DetectorFactory.seed = 0

encoder_saver = Saver()


def edit_salary(df: pd.DataFrame) -> pd.DataFrame:
    split = df["salary_range"].str.split("-", expand=True)
    salary_min = pd.to_numeric(split[0], errors="coerce")
    salary_max = pd.to_numeric(split[1], errors="coerce")

    df["salary_range"] = (salary_min + salary_max) / 2
    df["salary_range"] = df["salary_range"].fillna(-1)

    conditions = [
        df["salary_range"] < 1,
        (df["salary_range"] < 30000) & (df["salary_range"] >= 1),
        (df["salary_range"] < 70000) & (df["salary_range"] >= 30000),
        (df["salary_range"] < 250000) & (df["salary_range"] >= 70000),
        (df["salary_range"] >= 250000)
        ]

    labels = ["missing", "low salary", "medium salary", "high salary",
              "top 1 percent"]

    df["salary_category"] = np.select(conditions, labels)

    return df


def split_and_encode(df: pd.DataFrame) -> tuple:
    # splits the dataset and one-hot encodes categorical columns
    cat_columns = ["employment_type", "required_education",
                   "required_experience", "salary_category",
                   "function", "location"]

    X_train_raw, X_other_raw, y_train, y_other = train_test_split(df.drop(
                    columns="fraudulent"), df["fraudulent"],
                    test_size=0.3, stratify=df["fraudulent"], random_state=42)

    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_other_raw,
                                                            y_other,
                                                            test_size=0.5,
                                                            stratify=y_other,
                                                            random_state=42)

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    one_hot_encoded = enc.fit_transform(X_train_raw[cat_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded,
                              columns=enc.get_feature_names_out(cat_columns),
                              index=X_train_raw.index)
    X_train = pd.concat([X_train_raw.drop(cat_columns, axis=1), one_hot_df],
                        axis=1)

    one_hot_encoded = enc.transform(X_val_raw[cat_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded,
                              columns=enc.get_feature_names_out(cat_columns),
                              index=X_val_raw.index)
    X_val = pd.concat([X_val_raw.drop(cat_columns, axis=1), one_hot_df],
                      axis=1)

    one_hot_encoded = enc.transform(X_test_raw[cat_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded,
                              columns=enc.get_feature_names_out(cat_columns),
                              index=X_test_raw.index)
    X_test = pd.concat([X_test_raw.drop(cat_columns, axis=1), one_hot_df],
                       axis=1)

    return enc, X_train, X_val, X_test, y_train, y_val, y_test


def save(enc: OneHotEncoder, X_train: pd.DataFrame, X_val: pd.DataFrame,
         X_test: pd.DataFrame, y_train: pd.Series, y_val: pd.Series,
         y_test: pd.Series) -> None:
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_val.to_csv("data/processed/X_val.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)

    pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
    pd.DataFrame(y_val).to_csv("data/processed/y_val.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

    encoder_saver.save(model=enc, name='ohe_encoder.pkl')

    with open("data/processed/feature_columns.json", "w") as f:
        json.dump(X_train.columns.tolist(), f)


def main(data=None, input_path=None, output_dir=None):
    if data:
        df = read_user_input(data)
    else:
        df = read_csv(input_path)

    df = edit_salary(df)

    df = preprocess_dataframe(df)

    df["desc_length"] = df["description"].str.len()

    df = remove_feature_name_row(df)

    df = df.replace("", "missing")

    if not data:
        (enc, X_train, X_val, X_test,
         y_train, y_val, y_test) = split_and_encode(df)

        save(enc, X_train, X_val, X_test, y_train, y_val, y_test)

    return df


if __name__ == "__main__":
    main(input_path="data/raw", output_dir="data/processed")
