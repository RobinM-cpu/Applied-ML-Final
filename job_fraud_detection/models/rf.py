import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score
)

def main(base_path, output_dir):
    X_train = np.load(os.path.join(base_path, "X_train.npy"))
    X_val = np.load(os.path.join(base_path, "X_val.npy"))
    X_test = np.load(os.path.join(base_path, "X_test.npy"))

    y_train = np.load(os.path.join(base_path, "y_train.npy"))
    y_val = np.load(os.path.join(base_path, "y_val.npy"))
    y_test = np.load(os.path.join(base_path, "y_test.npy"))

    with open(os.path.join(base_path, "feature_columns.json")) as f:
        feature_names = json.load(f)


    rf = RandomForestClassifier(class_weight="balanced", random_state=42)

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    y_pred_proba = rf.predict_proba(X_val)[:, 1]
    print(classification_report(y_val, y_pred))
    print("Val AUC:", roc_auc_score(y_val, y_pred_proba))

    importances = rf.feature_importances_
    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    print("\nTop 10 Features by Importance:")
    print(importance_df.sort_values("importance", ascending=False).head(10))

    y_pred_rf_test = rf.predict(X_test)
    y_proba_rf_test = rf.predict_proba(X_test)[:, 1]

    print("Random Forest Test Classification Report:")
    print(classification_report(y_test, y_pred_rf_test))
    print("Test AUC:", roc_auc_score(y_test, y_proba_rf_test))

    joblib.dump(rf, os.path.join(output_dir, "rf_model.pkl"))

if __name__ == "__main__":
    main(base_path="data/processed", output_dir="models")
