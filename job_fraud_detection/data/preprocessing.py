import pandas as pd
import numpy as np
import nltk
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import unicodedata

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")


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


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    # if location contains more information than the country code
    if df["location"].astype(str).str.contains(",", na=False).any():
        edited_location = df["location"].str.split(",").str[0]
        df["location"] = edited_location.values

    #removes non-english listings
    df["desc_lang"] = df["description"].apply(detect_desc_lang)
    df = df[df["desc_lang"] == "en"].copy()
    
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

    df["non_latin_ratio"] = df["text"].apply(non_latin_ratio)
    df = df[df["non_latin_ratio"] <= 0.10].copy()
    
    return df


def remove_stopword(sentence: str) -> str:
    stop_words = set(stopwords.words("english"))
    words = sentence.split()
    filtered = [word for word in words if word not in stop_words and word.isalnum()]
    return " ".join(filtered)

  
def get_wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  

       
def lemmatizer(sentence: str) -> str:
    wnl = WordNetLemmatizer()
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    filtered = [wnl.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
    return " ".join(filtered)


def remove_feature_name_row(df: pd.DataFrame) -> pd.DataFrame:
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


def edit_salary_feature(df: pd.DataFrame) -> None:
    split = df["salary_range"].str.split("-", expand=True)
    salary_min = pd.to_numeric(split[0], errors="coerce")
    salary_max = pd.to_numeric(split[1], errors="coerce")

    df["salary_range"] = (salary_min + salary_max) / 2
    df["salary_range"] = df["salary_range"].fillna(-1)


def save_preprocessed_csv(df: pd.DataFrame, output_dir: str) -> None:
    df[["text", "fraudulent"]].to_csv(
        f"{output_dir}/preprocessed_text_labels.csv", index=False)


def main(data: dict=None, input_path: str=None, output_dir: str=None) -> None:
    if input_path:
        df = read_csv(input_path)
    else:
        df = read_user_input(data)
    print(df)


    df = preprocessing(df)
    df["text"] = df["text"].apply(remove_stopword)
    df["text"] = df["text"].apply(lemmatizer)
    df = remove_feature_name_row(df)
    print(df)

    if output_dir:
        save_preprocessed_csv(df, output_dir)
    else:
        return df


if __name__ == "__main__":
    main(input_path="data/raw", output_dir="data/processed")

