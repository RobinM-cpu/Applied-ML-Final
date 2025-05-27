import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def main():
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "raw",
            "fake_job_postings.csv",
        )
    )
    df = df.replace(np.nan, "", regex=True)

    df["location"] = df["location"].str.split(",").str[0]

    def strip_html(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"#URL_[a-f0-9]{64}", "", text)
        return text.strip()

    def clean_and_mark(row):
        return (
            f"[TITLE] {strip_html(row['title'])[:100]} "
            f"[DESC] {strip_html(row['description'])[:1000]} "
            f"[REQ] {strip_html(row['requirements'])[:500]} "
            f"[PROFILE] {strip_html(row['company_profile'])[:500]} "
            f"[BENEFITS] {strip_html(row['benefits'])[:500]} "
            f"[LOC] {strip_html(row['location'].split(',')[0])} "
            f"[EXP] {strip_html(row['required_experience'])} "
            f"[TYPE] {strip_html(row['employment_type'])} "
            f"[DEPT] {strip_html(row['department'])} "
            f"[EDU] {strip_html(row['required_education'])} "
            f"[INDUSTRY] {strip_html(row['industry'])} "
            f"[FUNC] {strip_html(row['function'])}"
        ).strip()

    df["text"] = df.apply(clean_and_mark, axis=1)

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
        "train_bert.csv", index=False
    )
    pd.DataFrame({"text": X_val, "label": y_val}).to_csv("val_bert.csv", index=False)
    pd.DataFrame({"text": X_test, "label": y_test}).to_csv("test_bert.csv", index=False)


if __name__ == "__main__":
    main()
