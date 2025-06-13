def test_fusion_pipeline():
    import joblib
    import pandas as pd
    from transformers import AutoTokenizer, \
        TFAutoModelForSequenceClassification
    from datasets import Dataset
    import tensorflow as tf

    tokenizer = AutoTokenizer.from_pretrained("models/tuned_bert_model")
    bert_model = TFAutoModelForSequenceClassification.from_pretrained(
        "models/tuned_bert_model")
    rf_model = joblib.load("models/rf_model.pkl")

    df = pd.read_csv("data/processed/val_bert.csv").sample(10, random_state=42)
    ds = Dataset.from_pandas(df)

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, padding=True,
                         max_length=512)

    tokenized = ds.map(preprocess, batched=True)
    tf_ds = bert_model.prepare_tf_dataset(tokenized, shuffle=False,
                                          batch_size=2)

    logits = bert_model.predict(tf_ds).logits
    probs_bert = tf.nn.softmax(logits, axis=1).numpy()[:, 1]

    X_val = pd.read_csv("data/processed/X_val.csv")[:10]
    X_val = X_val.drop(columns=["text"])
    y_val = pd.read_csv("data/processed/y_val.csv")[:10]
    y_val = y_val.to_numpy()
    probs_rf = rf_model.predict_proba(X_val)[:, 1]

    assert probs_bert.shape == probs_rf.shape
    assert len(y_val) == len(probs_bert)
    print("Fusion pipeline tested successfully!")


if __name__ == "__main__":
    test_fusion_pipeline()
