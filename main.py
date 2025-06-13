from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse
from pydantic import BaseModel
import pandas as pd
import tensorflow as tf
import re
from datasets import Dataset
from job_fraud_detection.models import baseline, rf
from job_fraud_detection.data import preprocessing, preprocess_bert, \
    multimodal_preprocess
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification
)

# errors that are raised in all models

empty_data_error = HTTPException(
            status_code=422,
            detail="Input data was filtered out during preprocessing. "
            "Check that your input is in English and contains valid "
            "characters. If description was provided, make sure it is a "
            "sensible sentence.")

no_description_error = HTTPException(
            status_code=417,
            detail="Description has to be provided "
                   "for a prediction to be made.")

no_input_error = HTTPException(
            status_code=400,
            detail="At least the description "
            "should be provided to make a prediction.")

wrong_salary = HTTPException(
            status_code=400,
            detail="Salary range must be in the format [lower-end]-"
            "[higher-end]. If you do not know the salary range, fill in 0-0.")


def no_binary_feature(binary_feature: str) -> HTTPException:
    return HTTPException(
            status_code=417,
            detail=f"A truth value about the {binary_feature} has to be "
            f"provided for a prediction to be made.")


# pydantic input, same for Logistic Regression and BERT

class ModelInput(BaseModel):
    job_id: int
    title: str
    description: str
    location: str
    department: str
    company_profile: str
    requirements: str
    benefits: str
    employment_type: str
    required_experience: str
    required_education: str
    industry: str
    function: str


class MultimodalityModelInput(BaseModel):
    job_id: int
    title: str
    description: str
    location: str
    department: str
    company_profile: str
    requirements: str
    benefits: str
    employment_type: str
    required_experience: str
    required_education: str
    industry: str
    function: str
    salary_range: str
    telecommuting: str
    has_company_logo: str
    has_questions: str


class ModelOutput(BaseModel):
    prediction: str


app = FastAPI(title="Applied Machine Learning - Group 34",
              summary="Deployment of our Model.")


@app.get("/", description="Root endpoint that redirects to documentation.")
async def root():
    return RedirectResponse(url='/docs')


@app.post("/predict-logistic-regression",
          summary="Fraud detection baseline model endpoint.",
          description="Requires at least one of the input fields to form a "
          "prediction. Details of the input fields are as follows: title is "
          "the title of the job, description is a summary of the job details, "
          "location is the country code, department is the corporate "
          "department (e.g. sales), company profile is a brief description of "
          "the company, requirements are the job requirements, benefits are "
          "the benefits provided by the employer, employment_type is the type "
          "of role (e.g. full-time), required_experience is the level (e.g "
          "entry level), required_education is the education level, industry "
          "is the industry (e.g IT), function (i.e consulting, customer "
          "service, etc.). ",
          response_model=ModelOutput,
          response_description="Prediction of fraud where 'Fraudulent' is a "
          "prediction of 0.5 and above and 'Real' is a prediction less than "
          "0.5.")
async def prediction(title: str = " ", description: str = " ",
                     location: str = " ", department: str = " ",
                     company_profile: str = " ", requirements: str = " ",
                     benefits: str = " ", employment_type: str = " ",
                     required_experience: str = " ",
                     required_education: str = " ", industry: str = " ",
                     function: str = " "):

    model_input = ModelInput(job_id=0, title=title, description=description,
                             location=location, department=department,
                             company_profile=company_profile,
                             requirements=requirements, benefits=benefits,
                             employment_type=employment_type,
                             required_experience=required_experience,
                             required_education=required_education,
                             industry=industry, function=function)

    # turn input into a dict
    input_dict = model_input.model_dump()
    non_job_id = {key: value for key,
                  value in input_dict.items() if key != "job_id"}
    all_empty = all(value == " " for value in non_job_id.values())

    if input_dict['description'] == ' ':
        raise no_description_error

    if all_empty:
        raise no_input_error

    processed_data = preprocessing.main(data=input_dict)

    if processed_data.empty:
        raise empty_data_error

    # load saved models #EDIIIIIIIIIIIIIIIIIIT
    vectorizer = baseline.model_saver.load('vectorizer.pkl')
    # vectorize data into TF-IDF feature for prediction
    tfidf_processed_data = vectorizer.transform(processed_data)
    log_reg_model = baseline.model_saver.load('log_reg_model.pkl')

    prediction = baseline.predict(log_reg_model, tfidf_processed_data)

    if prediction >= 0.5:
        result = "Fraudulent"
    else:
        result = "Real"
    return ModelOutput(prediction=result)


@app.post("/predict-bert",
          summary="Fraud detection BERT model endpoint.",
          description="Requires at least one of the input fields to form a "
          "prediction. Details of the input fields are as follows: title is "
          "the title of the job, description is a summary of the job details, "
          "location is the country code, department is the corporate "
          "department (e.g. sales), company profile is a brief description of "
          "the company, requirements are the job requirements, benefits are "
          "the benefits provided by the employer, employment_type is the type "
          "of role (e.g. full-time), required_experience is the level (e.g "
          "entry level), required_education is the education level, industry "
          "is the industry (e.g IT), function (i.e consulting, customer "
          "service, etc.). ",
          response_model=ModelOutput,
          response_description="Prediction of fraud where 'Fraudulent' is a "
          "prediction of 0.5 and above and 'Real' is a prediction less than "
          "0.5.")
async def prediction(title: str = " ", description: str = " ",
                     location: str = " ", department: str = " ",
                     company_profile: str = " ", requirements: str = " ",
                     benefits: str = " ", employment_type: str = " ",
                     required_experience: str = " ",
                     required_education: str = " ", industry: str = " ",
                     function: str = " "):

    model_input = ModelInput(job_id=0, title=title, description=description,
                             location=location, department=department,
                             company_profile=company_profile,
                             requirements=requirements, benefits=benefits,
                             employment_type=employment_type,
                             required_experience=required_experience,
                             required_education=required_education,
                             industry=industry, function=function)

    input_dict = model_input.model_dump()
    non_job_id = {key: value for key,
                  value in input_dict.items() if key != "job_id"}
    all_empty = all(value == " " for value in non_job_id.values())

    if input_dict['description'] == ' ':
        raise no_description_error

    if all_empty:
        raise no_input_error

    processed_data = preprocess_bert.main(data=input_dict)

    if processed_data.empty:
        raise empty_data_error

    # load the BERT model and the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('models/tuned_bert_model')
    bert_model = TFAutoModelForSequenceClassification.from_pretrained(
        'models/tuned_bert_model')

    def preprocess_for_bert(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512
        )

    # preprocess user input for the BERT model
    bert_val_ds = Dataset.from_pandas(processed_data)
    tokenized_val = bert_val_ds.map(preprocess_for_bert)
    val_tf_ds = bert_model.prepare_tf_dataset(
        tokenized_val,
        shuffle=False,
        batch_size=8
    )

    prediction_logits = bert_model.predict(val_tf_ds).logits
    prediction = tf.nn.softmax(prediction_logits, axis=1).numpy()[:, 1]

    if prediction >= 0.5:
        result = "Fraudulent"
    else:
        result = "Real"
    return ModelOutput(prediction=result)


@app.post("/predict-multimodal-bert",
          summary="Fraud detection BERT model endpoint.",
          description="Requires at least one of the input fields to form a "
          "prediction. Details of the input fields are as follows: title is "
          "the title of the job, description is a summary of the job details, "
          "location is the country code, department is the corporate "
          "department (e.g. sales), company profile is a brief description of "
          "the company, requirements are the job requirements, benefits are "
          "the benefits provided by the employer, employment_type is the type "
          "of role (e.g. full-time), required_experience is the level (e.g "
          "entry level), required_education is the education level, industry "
          "is the industry (e.g IT), function (i.e consulting, customer "
          "service, etc.), salary_range is the expected salary range "
          "(e.g. 20000-25000), telecommuting is a binary feature regarding if "
          "a job requires commuting, company_logo is a binary feature "
          "regarding if a posting has a company logo, and questions is "
          "a binary feature regarding if screening questions are present. For "
          "all binary features, input either 0 or 1; 0 if the feature is "
          "not present in your posting, 1 if otherwise.",
          response_model=ModelOutput,
          response_description="Prediction of fraud where 'Fraudulent' is a "
          "prediction of 0.5 and above and 'Real' is a prediction less than "
          "0.5.")
async def prediction(title: str = " ", description: str = " ",
                     location: str = " ", department: str = " ",
                     company_profile: str = " ", requirements: str = " ",
                     benefits: str = " ", employment_type: str = " ",
                     required_experience: str = " ",
                     required_education: str = " ", industry: str = " ",
                     function: str = " ", salary_range: str = " ",
                     telecommuting: str = " ", has_company_logo: str = " ",
                     has_questions: str = " "):

    model_input = MultimodalityModelInput(job_id=0, title=title,
                                          description=description,
                                          location=location,
                                          department=department,
                                          company_profile=company_profile,
                                          requirements=requirements,
                                          benefits=benefits,
                                          employment_type=employment_type,
                                          required_experience=(
                                              required_experience),
                                          required_education=(
                                              required_education),
                                          industry=industry, function=function,
                                          salary_range=salary_range,
                                          telecommuting=telecommuting,
                                          has_company_logo=has_company_logo,
                                          has_questions=has_questions)

    input_dict = model_input.model_dump()
    non_job_id = {key: value for key,
                  value in input_dict.items() if key != "job_id"}
    all_empty = all(value == " " for value in non_job_id.values())

    if input_dict['description'] == ' ':
        raise no_description_error

    if not bool(re.fullmatch(r"\d+-\d+", salary_range.strip())):
        raise wrong_salary

    processed_data = multimodal_preprocess.main(data=input_dict)

    if input_dict['telecommuting'] == ' ':
        raise no_binary_feature('telecommuting')

    if input_dict['has_company_logo'] == ' ':
        raise no_binary_feature('has_company_logo')

    if input_dict['has_questions'] == ' ':
        raise no_binary_feature('has_questions')

    for flag in ("telecommuting", "has_company_logo", "has_questions"):
        val = input_dict.get(flag, "").strip()
        if val not in ("0", "1"):
            raise HTTPException(
                status_code=400,
                detail=f"`{flag}` must be '0' or '1', but got '{val}'."
            )

    if all_empty:
        raise no_input_error

    if processed_data.empty:
        raise empty_data_error

    # load encoder
    enc = multimodal_preprocess.encoder_saver.load('ohe_encoder.pkl')

    cat_columns = ["employment_type", "required_education",
                   "required_experience", "salary_category",
                   "function", "location"]

    # one hot encode user input
    one_hot_encoded = enc.transform(processed_data[cat_columns])
    one_hot_processed_data = pd.DataFrame(one_hot_encoded,
                                          columns=enc.get_feature_names_out(
                                              cat_columns),
                                          index=processed_data.index)
    processed_data = pd.concat([processed_data.drop(cat_columns, axis=1),
                                one_hot_processed_data], axis=1)

    if processed_data.empty:
        raise empty_data_error

    bert_path = ('models/tuned_bert_model')
    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = TFAutoModelForSequenceClassification.from_pretrained(
        bert_path
    )

    def preprocess_for_bert(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512
        )

    bert_val_ds = Dataset.from_pandas(processed_data)
    tokenized_val = bert_val_ds.map(preprocess_for_bert)
    val_tf_ds = bert_model.prepare_tf_dataset(
        tokenized_val,
        shuffle=False,
        batch_size=8
    )

    if (salary_range != " " or telecommuting != " " or has_company_logo != " "
       or has_questions != " "):
        # drop the text column for random forest
        processed_data = processed_data.drop(columns='text')
        # load random forest and predict
        rf_model = rf.rf_saver.load('rf_model.pkl')
        rf_prediction = rf_model.predict(processed_data)

        # predict on text data
        bert_prediction = bert_model.predict(val_tf_ds).logits
        bert_prediction = tf.nn.softmax(bert_prediction, axis=1).numpy()[:, 1]

        # make a fused predictin, threshold=0.4939 and alpha=0.5 calculated
        # in multimodality.py
        fused_probs_test = 0.5 * bert_prediction + (1.0 - 0.5) * rf_prediction
        prediction = (fused_probs_test >= 0.4939).astype(int)
    else:
        prediction_logits = bert_model.predict(val_tf_ds).logits
        prediction = tf.nn.softmax(prediction_logits, axis=1).numpy()[:, 1]

    if prediction >= 0.5:
        result = "Fraudulent"
    else:
        result = "Real"
    return ModelOutput(prediction=result)
