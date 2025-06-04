import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import unicodedata
import joblib
import re
import json
import os
DetectorFactory.seed = 0

def non_latin_ratio(text):
    if not isinstance(text, str):
        return 1.0

    total = len(text)
    if total == 0:
        return 1.0

    latin_count = sum(
        1 for c in text
        if 'LATIN' in unicodedata.name(c, '') or c.isdigit() or c in " []()-_.:,"
    )
    return 1 - (latin_count / total)

def detect_desc_lang(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def strip_html(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'#URL_[a-f0-9]{64}', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'mso-[\w-]+:[^;"]+;?', '', text)
    text = re.sub(r'st1:[^ \n]+', '', text)

    text = re.sub(r'\b(false|true)\b', '', text)

    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('&nbsp;', '')
    text = text.replace('&amp;', '')
    text = text.replace('\xa0', '')
    text = re.sub(r'#\{[^}]*\}', '', text)
    text = re.sub(r'#(EMAIL|PHONE)_[a-f0-9]{64}#', '', text)
    text = re.sub(r'\.(\S)', r'. \1', text)
    return text.strip()

def clean_and_mark(row):
    return (
        f"[TITLE] {strip_html(row['title'])} "
        f"[DESC] {strip_html(row['description'])[:1000]} "
        f"[REQ] {strip_html(row['requirements'])[:500]} "
        f"[PROFILE] {strip_html(row['company_profile'])[:500]} "
        f"[BENEFITS] {strip_html(row['benefits'])[:500]} "
    ).strip()

def main(input_path, output_dir):
    df = pd.read_csv(f"{input_path}/fake_job_postings.csv")

    df = df.replace(np.nan, '', regex=True)

    df["desc_lang"] = df["description"].apply(detect_desc_lang)

    df = df[df["desc_lang"] == "en"]

    df["text"] = df.apply(clean_and_mark, axis=1)
    df["non_latin_ratio"] = df["text"].apply(non_latin_ratio)
    df = df[df["non_latin_ratio"] <= 0.10]

    split = df["salary_range"].str.split("-", expand=True)
    salary_min = pd.to_numeric(split[0], errors="coerce")
    salary_max = pd.to_numeric(split[1], errors="coerce")

    df["salary_range"] = (salary_min + salary_max) / 2
    df["salary_range"] = df["salary_range"].fillna(-1)

    conditions = [df["salary_range"] < 1,
                (df["salary_range"] < 30000) & (df["salary_range"] >= 1),
                (df["salary_range"] < 70000) & (df["salary_range"] >= 30000),
                (df["salary_range"] < 250000) & (df["salary_range"] >= 70000),
                (df["salary_range"] >= 250000)]

    labels = ["missing", "low salary", "medium salary", "high salary", "top 1 percent"]

    df["salary_category"] = np.select(conditions, labels)

    df["desc_length"] = df["description"].str.len()

    edited_location = df['location'].str.split(',').str[0]
    df['location'] = edited_location.values
    print(df['function'].nunique())

    df = df.drop(columns=["job_id", "title", "description", "department", "company_profile", "requirements", "benefits",
                        "industry", "salary_range", "text", "desc_lang", "non_latin_ratio"])

    # keep employment_type: 5+empty,
    # has_questions: 2,
    # required_education: 8+empty,
    # required_experience: 7+1,
    # has_company_logo: 2,
    # telecommuting: 2,
    # salary_range: 5
    # function: 38
    # desc_length: continous

    df = df.replace("", "missing")

    cat_columns = ["employment_type", "required_education", "required_experience", "salary_category", "function", "location"]

    X_train_raw, X_other_raw, y_train, y_other = train_test_split(df.drop(columns="fraudulent"), df["fraudulent"], test_size=0.3, stratify=df["fraudulent"], random_state=42)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_other_raw, y_other, test_size=0.5, stratify=y_other, random_state=42)


    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    one_hot_encoded = enc.fit_transform(X_train_raw[cat_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=enc.get_feature_names_out(cat_columns), index=X_train_raw.index)
    X_train = pd.concat([X_train_raw.drop(cat_columns, axis=1), one_hot_df], axis=1)

    one_hot_encoded = enc.transform(X_val_raw[cat_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=enc.get_feature_names_out(cat_columns), index=X_val_raw.index)
    X_val = pd.concat([X_val_raw.drop(cat_columns, axis=1), one_hot_df], axis=1)

    one_hot_encoded = enc.transform(X_test_raw[cat_columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=enc.get_feature_names_out(cat_columns), index=X_test_raw.index)
    X_test = pd.concat([X_test_raw.drop(cat_columns, axis=1), one_hot_df], axis=1)

    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_val.npy", X_val)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/y_test.npy", y_test)

    joblib.dump(enc, "models/ohe_encoder.pkl")
    with open("data/processed/feature_columns.json", "w") as f:
        json.dump(X_train.columns.tolist(), f)

if __name__ == "__main__":
    main(input_path="data/raw", output_dir="data/processed")
