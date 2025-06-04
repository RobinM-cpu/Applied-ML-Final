import numpy as np
import pandas as pd
import os
import optuna
import tensorflow as tf
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    create_optimizer,
    AutoConfig,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from datasets import Dataset

id2label = {0: "REAL", 1: "FRAUD"}
label2id = {"REAL": 0, "FRAUD": 1}
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def get_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return dict(zip(classes, weights))


def objective(trial):
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

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    learning_rate = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    epochs = trial.suggest_int("epochs", 3, 6)
    max_length = trial.suggest_int("max_length", 64, 512, step=64)
    warmup_proportion = trial.suggest_float("warmup_proportion", 0.05, 0.3)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

    def preprocess_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding=True, max_length=max_length
        )

    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_val = val_ds.map(preprocess_function, batched=True)

    class_weights = get_class_weights(train_df["label"])

    config = AutoConfig.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=dropout_rate,
        attention_probs_dropout_prob=dropout_rate,
    )
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        config=config,
    )

    train_tf_ds = model.prepare_tf_dataset(
        tokenized_train, shuffle=True, batch_size=batch_size
    )
    val_tf_ds = model.prepare_tf_dataset(
        tokenized_val, shuffle=False, batch_size=batch_size
    )

    steps_per_epoch = len(train_df) // batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(warmup_proportion * num_train_steps)

    optimizer, lr_schedule = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        weight_decay_rate=weight_decay,
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
        epochs=epochs,
        class_weight=class_weights,
        callbacks=[early_stop],
    )

    pred_logits = model.predict(val_tf_ds).logits
    probs = tf.nn.softmax(pred_logits, axis=1).numpy()[:, 1]
    y_true = val_df["label"].values

    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_thresh = thresholds[np.argmax(f1_scores)]
    best_f1 = np.max(f1_scores)

    final_preds = (probs >= best_thresh).astype(int)

    auc = roc_auc_score(y_true, probs)
    precision = precision_score(y_true, final_preds)
    recall = recall_score(y_true, final_preds)

    trial.set_user_attr("best_thresh", best_thresh)
    trial.set_user_attr("f1", best_f1)
    trial.set_user_attr("auc", auc)
    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)

    return best_f1


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    top_trials = study.trials_dataframe().sort_values("value", ascending=False).head(10)
    print("\nTop 10 Trials:")
    print(top_trials)

    all_trials = [
        {
            **trial.params,
            "f1": trial.user_attrs.get("f1"),
            "auc": trial.user_attrs.get("auc"),
            "precision": trial.user_attrs.get("precision"),
            "recall": trial.user_attrs.get("recall"),
            "best_thresh": trial.user_attrs.get("best_thresh"),
            "value": trial.value,
        }
        for trial in study.trials
    ]

    all_df = pd.DataFrame(all_trials)
    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "all_trials.csv")
    all_df.to_csv(csv_path, index=False)
    print("Best hyperparameters:", study.best_params)


if __name__ == "__main__":
    main()
