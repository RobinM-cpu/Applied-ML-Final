import numpy as np
import pandas as pd
import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import unicodedata


def read_csv(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(f"{input_path}/fake_job_postings.csv")
    df = df.replace(np.nan, "", regex=True)

    return df


def read_user_input(data: dict) -> pd.DataFrame:
    return pd.DataFrame([data])


def detect_desc_lang(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


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
    text = text.replace('&nbsp;', '').replace('&amp;', '').replace('\xa0', '')
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


def preprocess_dataframe(df: pd.DataFrame, log_reg: bool = False
                         ) -> pd.DataFrame:
    # if location contains more information than the country code
    if df["location"].astype(str).str.contains(",", na=False).any():
        edited_location = df["location"].str.split(",").str[0]
        df["location"] = edited_location.values

    # removes non-english listings
    df['desc_lang'] = df['description'].apply(detect_desc_lang)
    df = df[df['desc_lang'] == 'en']

    if log_reg:
        # merge text features
        df["text"] = (
            df["title"]
            + " "
            + df["description"]
            + " "
            + df["location"]
            + " "
            + df["department"]
            + " "
            + df["company_profile"]
            + " "
            + df["requirements"]
            + " "
            + df["benefits"]
            + " "
            + df["employment_type"]
            + " "
            + df["required_experience"]
            + " "
            + df["required_education"]
            + " "
            + df["industry"]
            + " "
            + df["function"]
        )
        
        # decapitalize and replace
        df["text"] = df["text"].str.lower()
        df["text"] = df["text"].str.replace(r"[^\w\s]", " ", regex=True)
    else:   
        df['text'] = df.apply(clean_and_mark, axis=1)

    df['non_latin_ratio'] = df['text'].apply(non_latin_ratio)
    df = df[df['non_latin_ratio'] <= 0.10]
    return df


def remove_feature_name_row(df: pd.DataFrame, log_reg=False) -> pd.DataFrame:
    if log_reg:
        return df.drop(
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
                "non_latin_ratio",
                "desc_lang"
            ]
        )
    else:
        return df.drop(columns=[
                "job_id",
                "title",
                "description",
                "department",
                "company_profile",
                "requirements",
                "benefits",
                "industry",
                "non_latin_ratio",
                "desc_lang"
            ]
        )
