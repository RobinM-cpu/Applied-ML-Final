import pandas as pd
import numpy as np
import nltk

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


def main(input_path, output_dir):
    df = pd.read_csv(f"{input_path}/fake_job_postings.csv")

    print(df.head())
    df = df.replace(np.nan, "", regex=True)

    edited_location = df["location"].str.split(",").str[0]
    df["location"] = edited_location.values

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

    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].str.replace(r"[^\w\s]", " ", regex=True)

    def remove_stopword(sentence):
        words = sentence.split()
        filtered = [word for word in words if word not in stop_words and word.isalnum()]
        return " ".join(filtered)

    stop_words = set(stopwords.words("english"))
    df["text"] = df["text"].apply(remove_stopword)
    wnl = WordNetLemmatizer()

    def get_wordnet_pos(tag):
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

    def lemmatizer(sentence):
        words = word_tokenize(sentence)
        pos_tags = pos_tag(words)
        filtered = [wnl.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
        return " ".join(filtered)

    df["text"] = df["text"].apply(lemmatizer)

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

    df[["text", "fraudulent"]].to_csv(f"{output_dir}/preprocessed_text_labels.csv", index=False)


if __name__ == "__main__":
    main()
