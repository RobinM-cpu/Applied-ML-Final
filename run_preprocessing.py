from job_fraud_detection.data import preprocessing, preprocess_bert, \
    multimodal_preprocess


def main():
    print("Running baseline preprocessing")
    preprocessing.main(input_path="data/raw", output_dir="data/processed")

    print("Running bert preprocessing")
    preprocess_bert.main(input_dir="data/raw", output_dir="data/processed")

    print("Running multimodality preprocessing")
    multimodal_preprocess.main(input_path="data/raw",
                               output_dir="data/processed")

    print("All preprocessing complete. Ready to train models")


if __name__ == "__main__":
    main()
