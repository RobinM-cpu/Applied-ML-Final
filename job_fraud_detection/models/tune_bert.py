import os
import optuna
import numpy as np
import pandas as pd
import tensorflow as tf
import random

from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    create_optimizer,
    AutoConfig,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from optuna.trial import Trial
from datasets import Dataset

SEED = 1
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()


id2label = {0: "REAL", 1: "FRAUD"}
label2id = {"REAL": 0, "FRAUD": 1}
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def get_class_weights(labels: pd.Series) -> dict:
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes,
                                   y=labels)
    return dict(zip(classes, weights))


# sets a global variable for calculating the F1 score during optimization
best_f1_so_far = -1.0


def objective(trial: Trial) -> float:
    global best_f1_so_far

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

    # all of the hyperparameters
    max_length = 512
    learning_rate = trial.suggest_float("lr", 2e-5, 3e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    weight_decay = trial.suggest_float("weight_decay", 0.1, 0.18)
    epochs = trial.suggest_int("epochs", 5, 7)
    warmup_proportion = trial.suggest_float("warmup_proportion", 0.08, 0.15)
    dropout_rate = trial.suggest_float("dropout_rate", 0.08, 0.2)

    def preprocess_function(examples: dict) -> dict:
        return tokenizer(
            examples["text"], truncation=True, padding=True,
            max_length=max_length)

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

    if best_f1 > best_f1_so_far:
        print(f"new best f1 = {best_f1:.4f} — saving model")
        model.save_pretrained("models/tuned_bert_model")
        tokenizer.save_pretrained("models/tuned_bert_model")
        best_f1_so_far = best_f1

    return best_f1


def save_trials(study: optuna.Study, trial: Trial) -> None:

    all_trials = [
        {
            **t.params,
            "f1": t.user_attrs.get("f1"),
            "auc": t.user_attrs.get("auc"),
            "precision": t.user_attrs.get("precision"),
            "recall": t.user_attrs.get("recall"),
            "best_thresh": t.user_attrs.get("best_thresh"),
            "value": t.value,
        }
        for t in study.trials
    ]
    all_df = pd.DataFrame(all_trials)

    script_dir = os.path.dirname(__file__)
    csv_path = os.path.join(script_dir, "all_trials5.csv")
    all_df.to_csv(csv_path, index=False)


def main() -> None:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, callbacks=[save_trials])

    top_trials = study.trials_dataframe().sort_values(
        "value", ascending=False).head(10)
    print("\nTop 10 Trials:")
    print(top_trials)

    print("Best hyperparameters:", study.best_params)


if __name__ == "__main__":
    main()
