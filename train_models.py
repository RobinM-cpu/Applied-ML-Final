from job_fraud_detection.models import baseline, multimodality, rf


if __name__ == "__main__":
    print('Training baseline Logistic Regression model...')
    baseline.main()

    print('Training Random Forest model...')
    rf.main(base_path="data/processed")

    print('Training multimodality implementation of BERT...')
    multimodality.main(bert_path="models/tuned_bert_model",
                       rf_path="models/rf_model.pkl")
