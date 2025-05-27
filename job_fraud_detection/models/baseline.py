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


def train_baseline_model(df):

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
    class_weights = dict(zip(classes, weights))
    print(class_weights)


    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    train_tfidf = vectorizer.fit_transform(X_train)
    val_tfidf = vectorizer.transform(X_val)
    test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(class_weight=class_weights, max_iter=1000)
    model.fit(train_tfidf, y_train)

    y_val_pred = model.predict(val_tfidf)
    print("Validation Set Performance")
    print(classification_report(y_val, y_val_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_val_pred))
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"F1: {f1_score(y_val, y_val_pred)}")

    y_test_pred = model.predict(test_tfidf)
    print("\nTest Set Performance")
    print(classification_report(y_test, y_test_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"F1: {f1_score(y_test, y_test_pred)}")

    y_val_probs = model.predict_proba(val_tfidf)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_probs)
    print(f"AUC (Validation): {val_auc:.4f}")

    y_test_probs = model.predict_proba(test_tfidf)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_probs)
    print(f"AUC (Test): {test_auc:.4f}")


def main():
    import os
    import pandas as pd

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

    train_baseline_model(df)


if __name__ == "__main__":
    main()

