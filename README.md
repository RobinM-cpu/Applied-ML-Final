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
│   ├───features # Not implemented, tests for multimodality?
│   └───models # Unit tests for baseline and BERT model training/evaluation
├───.gitignore
├───.pre-commit-config.yaml
├───main.py # API implementation
├───run_preprocessing.py # Ignore for now; Run to generate processed datasets from raw dataset. No need to do this as csv's already in repo
├───train_model_baseline.py # Ignore for now; Run to train baseline model
├───train_model_bert.py # Ignore for now; Run to train bert model
├───README.md
```
1) to run enter folder Applied-ML-Final, and write in terminal: uvicorn main:app --reload
2) cURL command:
    hint:
        - url => for example http://127.0.0.1:8000
        - **A** => one feature
        - **B** => the next

    1st line: curl -X 'POST' \
    2nd line: 'url/predict?**A**=%20&**B**' \
    3rd line: -H 'accept: application/json' \
    4th line: -d '' -w "\n"

