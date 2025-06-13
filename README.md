# Job Fraud Detection 🛠️

## Welcome!

##### Prevalence of online fraud, specifically regarding job offers has been and continues to be a widespread problem. To tackle this, we created a project that uses Logistic Regression trained on TF-IDF features, BERT, and Multimodal BERT with Random Forest to identify whether a job posting is fraudulent or not; that is, do you need to be worried that the posting you are looking at is not actually leading to you dream job?

##### If you are someone who is new to the job market, does not have many experience with how to identify a legitimate job posting, try inputting the job details through our Streamlit or API app and receive a prediction of whether the job is real or fake!

Our repo:
```bash
├───app # Streamlit application
    └───pages # Streamlit pages
├───data  # Stores .csv
├───job_fraud_detection
│   ├───data  # For data processing, not storing .csv
│   ├───models  # For model creation, not storing .h5
│   └───saver.py  # Class for saving and loading models
├───models  # Stores .h5 and other models
├───tests
│   ├───data # Unit tests for data preprocessing functions
│   └───models # Unit tests for baseline and BERT model training/evaluation
├───.gitignore
├───.gitattributes
├───.pre-commit-config.yaml
├───main.py # API implementation
├───README.md
├───requirements.txt
├───run_preprocessing.py # Run to generate processed datasets from raw dataset
├───train_models.py # Run to train models
```
1) to run the API, enter folder Applied-ML-Final, and write in terminal: uvicorn main:app --reload
2) cURL command:
    hint:
        - url => for example http://127.0.0.1:8000
        - **A** => one feature
        - **B** => the next

    1st line: curl -X 'POST' \
    2nd line: 'url/predict?**A**=%20&**B**' \
    3rd line: -H 'accept: application/json' \
    4th line: -d '' -w "\n"

3) necessary dependencies are found in requirements.txt

4) to run the Streamlit, write in terminal: streamlit run app/Home.py