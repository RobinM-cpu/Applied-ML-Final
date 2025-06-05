import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier

def main(base_path, output_dir):
    X_train = np.load(os.path.join(base_path, "X_train.npy"))
    X_val = np.load(os.path.join(base_path, "X_val.npy"))

    y_train = np.load(os.path.join(base_path, "y_train.npy"))
    y_val = np.load(os.path.join(base_path, "y_val.npy"))

    rf = RandomForestClassifier(class_weight="balanced", max_features="sqrt", criterion="gini", n_estimators=100, random_state=42)

    rf.fit(X_train, y_train)

    joblib.dump(rf, os.path.join(output_dir, "rf_model.pkl"))

if __name__ == "__main__":
    main(base_path="data/processed", output_dir="models")
