import pandas as pd
from job_fraud_detection.models.baseline import train_baseline_model


def main():
    df = pd.read_csv("data/processed/preprocessed_text_labels.csv")
    train_baseline_model(df)


if __name__ == "__main__":
    main()
