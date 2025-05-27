# Job Fraud Detection 🛠️

Our repo:
```bash
├───data  # Stores .csv
├───job_fraud_detection
│   ├───data  # For data processing, not storing .csv
│   ├───features # Might use this for multimodality later on
│   └───models  # For model creation, not storing .h5
├───models  # Stores .h5 or other models
├───notebooks  # Contains experimental .ipynbs
├───reports # Plots, and diagrams etc. if we decide to use them
├───tests
│   ├───data # Unit tests for data preprocessing functions
│   ├───features # Tests for multimodality?
│   └───models # Unit tests for baseline and BERT model training/evaluation
├───.gitignore
├───.pre-commit-config.yaml
├───main.py # Api implementation?
├───run_preprocessing.py # Run to generate processed datasets from raw dataset. No need to do this as csv's already in repo
├───train_model_baseline.py # Run to train baseline model
├───train_model_bert.py # Run to train bert model
├───README.md
```