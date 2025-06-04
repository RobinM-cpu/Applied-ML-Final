import numpy as np
import tensorflow as tf
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TFAutoModelForSequenceClassification,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve
from datasets import Dataset
from transformers import create_optimizer


id2label = {0: "REAL", 1: "FRAUD"}
label2id = {"REAL": 0, "FRAUD": 1}
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)


def print_metrics(true, pred, probs=None):
    print("\nClassification Report:")
    print(classification_report(true, pred))
    print(f"F1 Score: {f1_score(true, pred):.4f}")
    if probs is not None:
        print(f"AUC Score: {roc_auc_score(true, probs):.4f}")


def get_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=labels
    )
    return dict(zip(classes, weights))


def train_bert_model(train_df, val_df, test_df):
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_val = val_ds.map(preprocess_function, batched=True)
    tokenized_test = test_ds.map(preprocess_function, batched=True)

    class_weights = get_class_weights(train_ds["label"])

    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )

    batch_size = 8
    train_tf_ds = model.prepare_tf_dataset(
        tokenized_train, shuffle=True, batch_size=batch_size
    )
    val_tf_ds = model.prepare_tf_dataset(
        tokenized_val, shuffle=False, batch_size=batch_size
    )
    test_tf_ds = model.prepare_tf_dataset(
        tokenized_test, shuffle=False, batch_size=batch_size
    )

    train_examples = len(train_ds)
    steps_per_epoch = train_examples // batch_size
    num_train_steps = steps_per_epoch * 4
    num_warmup_steps = int(0.1 * num_train_steps)

    optimizer, lr_schedule = create_optimizer(
    init_lr=2e-5,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    weight_decay_rate=0.01
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )

    model.fit(
        train_tf_ds,
        validation_data=val_tf_ds,
        epochs=4,
        class_weight=class_weights,
        callbacks=[early_stop],
    )

    print("\nEvaluating on test set:")
    eval_loss, eval_acc = model.evaluate(test_tf_ds)
    print(f"Test Accuracy: {eval_acc:.4f}")

    pred_logits = model.predict(test_tf_ds).logits
    probs = tf.nn.softmax(pred_logits, axis=1).numpy()[:, 1]
    y_pred = (probs >= 0.5).astype(int)
    y_true = test_df["label"].values

    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_thresh = thresholds[np.argmax(f1_scores)]
    print(f"Best Threshold for F1: {best_thresh:.4f}")

    y_pred = (probs >= best_thresh).astype(int)
    print_metrics(y_true, y_pred, probs)

    model.save_pretrained("final_fraud_model")
    tokenizer.save_pretrained("final_fraud_model")


def main():
    import os
    import pandas as pd

    train_df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "processed",
            "train_bert.csv",
        )
    )
    val_df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "processed",
            "val_bert.csv",
        )
    )
    test_df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "processed",
            "test_bert.csv",
        )
    )

    train_bert_model(train_df, val_df, test_df)


if __name__ == "__main__":
    main()
