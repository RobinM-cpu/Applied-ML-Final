import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('.')
from job_fraud_detection.saver import Saver

rf_saver = Saver()


def main(base_path):
    X_train = np.load(os.path.join(base_path, "X_train.npy"))

    y_train = np.load(os.path.join(base_path, "y_train.npy"))

    rf = RandomForestClassifier(class_weight="balanced", max_features="sqrt",
                                criterion="gini", n_estimators=100,
                                random_state=42)

    rf.fit(X_train, y_train)

    rf_saver.save(rf, 'rf_model.pkl')


if __name__ == "__main__":
    main(base_path="data/processed")
