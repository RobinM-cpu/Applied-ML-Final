import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from job_fraud_detection.saver import Saver
import sys
sys.path.append('.')

rf_saver = Saver()


def main(base_path: str):
    X_train = pd.read_csv(os.path.join(base_path, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(base_path, "y_train.csv")).squeeze()

    X_train = X_train.drop(columns=["text"])

    rf = RandomForestClassifier(class_weight="balanced", max_features="sqrt",
                                criterion="gini", n_estimators=100,
                                random_state=42)

    rf.fit(X_train, y_train)

    rf_saver.save(rf, 'rf_model.pkl')


if __name__ == "__main__":
    main(base_path="data/processed")
