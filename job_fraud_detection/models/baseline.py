from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple
import pandas as pd
import os
from scipy.sparse import spmatrix


def data_splitting(df: pd.DataFrame) -> Tuple[pd.DataFrame]:
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

    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_class_weights(y_train: pd.DataFrame) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced",
                                   classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    return class_weights


def vectorize(X_train: pd.DataFrame, X_val: pd.DataFrame,
              X_test: pd.DataFrame) -> Tuple[spmatrix]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    train_tfidf = vectorizer.fit_transform(X_train)
    val_tfidf = vectorizer.transform(X_val)
    test_tfidf = vectorizer.transform(X_test)

    return train_tfidf, val_tfidf, test_tfidf


def fit(model: type[LogisticRegression], train_tfidf: spmatrix,
        y_train: pd.DataFrame) -> None:
    return model.fit(train_tfidf, y_train)


def predict(model: type[LogisticRegression], tfidf_matrix: spmatrix
            ) -> np.ndarray:
    return model.predict(tfidf_matrix)


def validation_set_metrics(model: type[LogisticRegression],
                           y_val: pd.DataFrame,
                           y_val_pred: np.ndarray,
                           val_tfidf: spmatrix) -> None:
    print("Validation Set Performance")
    print(classification_report(y_val, y_val_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"F1: {f1_score(y_val, y_val_pred)}")

    y_val_probs = model.predict_proba(val_tfidf)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_probs)
    print(f"AUC (Validation): {val_auc:.4f}")


def test_set_metrics(model: type[LogisticRegression],
                    y_test: pd.DataFrame,
                    y_test_pred: np.ndarray,
                    test_tfidf: spmatrix) -> None:
    y_test_pred = model.predict(test_tfidf)
    print("\nTest Set Performance")
    print(classification_report(y_test, y_test_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"F1: {f1_score(y_test, y_test_pred)}")

    y_test_probs = model.predict_proba(test_tfidf)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_probs)
    print(f"AUC (Test): {test_auc:.4f}")


def main(df_user_input: pd.DataFrame=None, user_input: bool=False,
         return_metrics: bool=True):
    df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "processed",
            "preprocessed_text_labels.csv",
        )
    )

    X_train, X_val, X_test, y_train, y_val, y_test = data_splitting(df)
    class_weights = calculate_class_weights(y_train)

    model = LogisticRegression(class_weight=class_weights, max_iter=1000)

    train_tfidf, val_tfidf, test_tfidf = vectorize(X_train, X_val, X_test)
    fit(model, train_tfidf, y_train)

    y_val_pred = predict(model, val_tfidf)
    y_test_pred = predict(model, test_tfidf)

    if return_metrics:
        validation_set_metrics(model, y_val, y_val_pred, val_tfidf)
        test_set_metrics(model, y_test, y_test_pred, test_tfidf)

    if user_input:
        df = df_user_input
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        vectorizer.fit_transform(X_train)
        user_input_tfidf = vectorizer.transform(df['text'])
        return predict(model, user_input_tfidf)


if __name__ == "__main__":
    main()
