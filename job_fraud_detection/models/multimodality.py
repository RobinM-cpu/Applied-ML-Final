import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification
)
from datasets import Dataset
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    roc_auc_score,
    f1_score
)
from rf import rf_saver


def main(bert_path, rf_path):

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = TFAutoModelForSequenceClassification.from_pretrained(
        bert_path
    )

    rf_model = rf_saver.load(name='rf_model.pkl')

    def preprocess_for_bert(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512
        )

    #BERT validation
    bert_val_df = pd.read_csv("data/processed/val_bert.csv")
    bert_val_ds = Dataset.from_pandas(bert_val_df)

    tokenized_val = bert_val_ds.map(preprocess_for_bert, batched=True)
    val_tf_ds = bert_model.prepare_tf_dataset(
        tokenized_val,
        shuffle=False,
        batch_size=8
    )

    val_logits = bert_model.predict(val_tf_ds).logits
    val_probs_bert = tf.nn.softmax(val_logits, axis=1).numpy()[:, 1]
    y_val = bert_val_df["label"].values

    precision_b, recall_b, thresholds_b = precision_recall_curve(y_val, val_probs_bert)
    f1_scores_b = 2 * precision_b * recall_b / (precision_b + recall_b + 1e-9)
    best_idx_b = np.argmax(f1_scores_b)
    best_thresh_bert_val = thresholds_b[best_idx_b]
    print(f"VAL: Best BERT threshold = {best_thresh_bert_val:.4f} (F1 = {f1_scores_b[best_idx_b]:.4f})")


    # Random forest validation
    X_val = np.load("data/processed/X_val.npy", allow_pickle=True)
    y_val_rf = np.load("data/processed/y_val.npy", allow_pickle=True)

    #check everything is the same
    if not np.array_equal(y_val, y_val_rf):
        raise ValueError("Validation labels for BERT and RF do not match.")

    val_probs_rf = rf_model.predict_proba(X_val)[:, 1]

    precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_val, val_probs_rf)
    f1_scores_rf = 2 * precision_rf * recall_rf / (precision_rf + recall_rf + 1e-9)
    best_idx_rf = np.argmax(f1_scores_rf)
    best_thresh_rf_val = thresholds_rf[best_idx_rf]
    print(f"VAL: Best RF threshold = {best_thresh_rf_val:.4f} (F1 = {f1_scores_rf[best_idx_rf]:.4f})")

    # Grid search
    alphas = np.arange(0.0, 1.01, 0.1)
    best_alpha = None
    best_thresh_fusion_val = None
    best_f1_fusion_val = -1.0

    for value in alphas:
        fused_probs_val = value * val_probs_bert + (1.0 - value) * val_probs_rf
        precision_f, recall_f, thresholds_f = precision_recall_curve(y_val, fused_probs_val)
        f1_scores_f = 2 * precision_f * recall_f / (precision_f + recall_f + 1e-9)

        idx_best_f = np.argmax(f1_scores_f)
        f1_f_val = f1_scores_f[idx_best_f]
        if f1_f_val > best_f1_fusion_val:
            best_f1_fusion_val = f1_f_val
            best_alpha = value
            best_thresh_fusion_val = thresholds_f[idx_best_f]

    print(
        f"VAL Best fusion alpha = {best_alpha:.1f}, "
        f"fusion threshold = {best_thresh_fusion_val:.4f} "
        f"(F1 = {best_f1_fusion_val:.4f})")

   #TEST SET
    bert_test_df = pd.read_csv("data/processed/test_bert.csv")
    bert_test_ds = Dataset.from_pandas(bert_test_df)

    tokenized_test = bert_test_ds.map(preprocess_for_bert, batched=True)
    test_tf_ds = bert_model.prepare_tf_dataset(
        tokenized_test,
        shuffle=False,
        batch_size=8
    )

    test_logits = bert_model.predict(test_tf_ds).logits
    test_probs_bert = tf.nn.softmax(test_logits, axis=1).numpy()[:, 1]
    y_test = bert_test_df["label"].values

    # RF test
    X_test = np.load("data/processed/X_test.npy")
    y_test_rf = np.load("data/processed/y_test.npy")

    if not np.array_equal(y_test, y_test_rf):
        raise ValueError("Test labels for BERT and RF do not match")

    test_probs_rf = rf_model.predict_proba(X_test)[:, 1]

    # BERT results
    y_pred_bert_test = (test_probs_bert >= best_thresh_bert_val).astype(int)
    print("\n----- BERT results -----")
    print(f"Applied threshold (from VAL): {best_thresh_bert_val:.4f}")
    print(classification_report(y_test, y_pred_bert_test, digits=4))
    print(f"Test AUC (BERT): {roc_auc_score(y_test, test_probs_bert):.4f}")
    print(f"Test F1 (BERT): {f1_score(y_test, y_pred_bert_test):.4f}")

    #RF results
    y_pred_rf_test = (test_probs_rf >= best_thresh_rf_val).astype(int)
    print("\n----- RF results -----")
    print(f"Applied threshold (from VAL): {best_thresh_rf_val:.4f}")
    print(classification_report(y_test, y_pred_rf_test, digits=4))
    print(f"Test AUC (RF): {roc_auc_score(y_test, test_probs_rf):.4f}")
    print(f"Test F1 (RF): {f1_score(y_test, y_pred_rf_test):.4f}")

    #Fusion results
    fused_probs_test = best_alpha * test_probs_bert + (1.0 - best_alpha) * test_probs_rf
    y_pred_fusion_test = (fused_probs_test >= best_thresh_fusion_val).astype(int)

    print("\n-----Fusion results-----")
    print(f"Applied alpha (from VAL): {best_alpha:.1f}")
    print(f"Applied threshold (from VAL): {best_thresh_fusion_val:.4f}")
    print(classification_report(y_test, y_pred_fusion_test, digits=4))
    print(f"Test AUC (Fusion): {roc_auc_score(y_test, fused_probs_test):.4f}")
    print(f"Test F1 (Fusion): {f1_score(y_test, y_pred_fusion_test):.4f}")


if __name__ == "__main__":
    main(bert_path="models/tuned_bert_model", rf_path="models/rf_model.pkl")