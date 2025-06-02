import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
from langdetect import detect, DetectorFactory
import unicodedata
import re
from langdetect.lang_detect_exception import LangDetectException
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

def main(input_path, output_dir):
    df = pd.read_csv(f"{input_path}/fake_job_postings.csv")
    df = df.replace(np.nan, "", regex=True)

    df["location"] = df["location"].str.split(",").str[0]

    def strip_html(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"#URL_[a-f0-9]{64}", "", text)
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

    df["desc_lang"] = df["description"].apply(detect_desc_lang)

    df = df[df["desc_lang"] == "en"]

    def clean_and_mark(row):
        return (
            f"[TITLE] {strip_html(row['title'])} "
            f"[DESC] {strip_html(row['description'])[:1000]} "
            f"[REQ] {strip_html(row['requirements'])[:500]} "
            f"[PROFILE] {strip_html(row['company_profile'])[:500]} "
            f"[BENEFITS] {strip_html(row['benefits'])[:500]} "
        ).strip()

    df["text"] = df.apply(clean_and_mark, axis=1)

    df["non_latin_ratio"] = df["text"].apply(non_latin_ratio)
    df = df[df["non_latin_ratio"] <= 0.10]

    df = df.drop(
        columns=[
            "job_id",
            "title",
            "description",
            "location",
            "department",
            "company_profile",
            "requirements",
            "benefits",
            "employment_type",
            "required_experience",
            "required_education",
            "industry",
            "function",
        ]
    )

    split = df["salary_range"].str.split("-", expand=True)
    salary_min = pd.to_numeric(split[0], errors="coerce")
    salary_max = pd.to_numeric(split[1], errors="coerce")
    df["salary_range"] = (salary_min + salary_max) / 2
    df["salary_range"] = df["salary_range"].fillna(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["fraudulent"],
        test_size=0.3,
        random_state=42,
        stratify=df["fraudulent"],
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    classes_dictionary = dict(zip(classes, weights))
    print(classes_dictionary)

    pd.DataFrame({"text": X_train, "label": y_train}).to_csv(
        f"{output_dir}/train_bert.csv", index=False
    )
    pd.DataFrame({"text": X_val, "label": y_val}).to_csv(f"{output_dir}/val_bert.csv", index=False)
    pd.DataFrame({"text": X_test, "label": y_test}).to_csv(f"{output_dir}/test_bert.csv", index=False)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    token_lengths = df["text"].apply(lambda x: len(tokenizer(x, truncation=False)["input_ids"]))

    print(f"Average token length: {token_lengths.mean():.2f}")
    print(f"Max token length: {token_lengths.max()}")
    print(f"Median token length: {token_lengths.median():.2f}")
    print(f"95th percentile: {np.percentile(token_lengths, 95):.2f}")
    print(f"99th percentile: {np.percentile(token_lengths, 99):.2f}")

    df["token_length"] = df["text"].apply(lambda x: len(tokenizer(x, truncation=False)["input_ids"]))

    max_idx = df["token_length"].idxmax()
    longest_text = df.loc[max_idx, "text"]
    longest_token_count = df.loc[max_idx, "token_length"]
    print(f"Longest text has {longest_token_count} tokens:\n")
    print(longest_text)
    print(f"Label : {df.loc[max_idx, 'fraudulent']}")

    print(df["fraudulent"].value_counts(normalize=True))


if __name__ == "__main__":
    input_path = "data/raw"
    output_dir = "data/processed"
    main(input_path, output_dir)
