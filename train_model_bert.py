import pandas as pd
from job_fraud_detection.models.bert import train_bert_model


def main():
    train_df = pd.read_csv("data/processed/train_bert.csv")
    val_df = pd.read_csv("data/processed/val_bert.csv")
    test_df = pd.read_csv("data/processed/test_bert.csv")

    train_bert_model(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
