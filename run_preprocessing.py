from job_fraud_detection.data import preprocessing, preprocess_bert


def main():
    print("Running baseline preprocessing")
    preprocessing.main("data/raw", "data/processed")

    print("Running bert preprocessing")
    preprocess_bert.main("data/raw", "data/processed")

    print("All preprocessing complete. Ready to train models")


if __name__ == "__main__":
    main()
