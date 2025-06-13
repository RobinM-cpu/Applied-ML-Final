import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize

import sys
sys.path.append('.')

from job_fraud_detection.data.general_preprocessing import (
                                   read_csv, read_user_input,
                                   preprocess_dataframe,
                                   remove_feature_name_row)

nltk.download("stopwords")

nltk.download("wordnet")
nltk.download("omw-1.4")

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")


def remove_stopword(sentence: str) -> str:
    stop_words = set(stopwords.words("english"))
    words = sentence.split()
    filtered = [word for word in words if word not in stop_words
                and word.isalnum()]
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
    filtered = [wnl.lemmatize(word, get_wordnet_pos(tag))
                for word, tag in pos_tags]
    return " ".join(filtered)


def save_preprocessed_csv(df: pd.DataFrame, output_dir: str) -> None:
    df[["text", "fraudulent"]].to_csv(
        f"{output_dir}/preprocessed_text_labels.csv", index=False)


def main(data: dict = None, input_path: str = None,
         output_dir: str = None) -> None:
    if data:
        df = read_user_input(data)
    else:
        df = read_csv(input_path)

    df = preprocess_dataframe(df, log_reg=True)
    df["text"] = df["text"].apply(remove_stopword)
    df["text"] = df["text"].apply(lemmatizer)
    df = remove_feature_name_row(df, log_reg=True)

    if data:
        return df
    else:
        save_preprocessed_csv(df, output_dir)


if __name__ == "__main__":
    main(input_path="data/raw", output_dir="data/processed")
