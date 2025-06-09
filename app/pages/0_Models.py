import sys
import streamlit as st
import pandas as pd
import tensorflow as tf
from datasets import Dataset
from job_fraud_detection.models import baseline, rf
from job_fraud_detection.data import preprocessing, preprocess_bert, \
    multimodal_preprocess
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification
)
sys.path.append(".")

st.set_page_config(
    page_title="Models"
)

st.title("Deployment")
st.write("In this page a user can select a model, input data to make "
         "predictions on, and view the results.")

# select model
model_list = ("Logistic Regression", "BERT")
chosen_model = st.selectbox("Select a model", model_list)

form = st.form("Input Data")

# insert data to make predictions on
if chosen_model == "Logistic Regression":
    title = form.text_area("Title", value=" ", help="the title of the job")
    description = form.text_area("Description", value=" ",
                                 help="a summary of the job details")
    location = form.text_area("Location", value=" ",
                              help="the country code where the job is "
                              "located")
    department = form.text_area("Department", value=" ",
                                help="corporate department "
                                "(e.g. sales)")
    company_profile = form.text_area("Company Profile", value=" ",
                                     help="brief description of the "
                                     "company")
    requirements = form.text_area("Requirements", value=" ",
                                  help="the job requirements")
    benefits = form.text_area("Benefits", value=" ",
                              help="the benefits provided by "
                              "the employer")
    employment_type = form.text_area("Employment Type", value=" ",
                                     help="the type of role "
                                     "(e.g. full-time)")
    required_experience = form.text_area("Required Experience", value=" ",
                                         help="the level "
                                         "(e.g entry level)")
    required_education = form.text_area("Required Education", value=" ",
                                        help="the education level")
    industry = form.text_area("Industry", value=" ",
                              help="the industry (e.g IT)")
    function = form.text_area("Function", value=" ",
                              help="the function (i.e consulting, "
                              "customer service, etc.)")
elif chosen_model == "BERT":
    title = form.text_area("Title", value=" ", help="the title of the job")
    description = form.text_area("Description", value=" ",
                                 help="a summary of the job details")
    location = form.text_area("Location", value=" ",
                              help="the country code where the job is "
                              "located")
    department = form.text_area("Department", value=" ",
                                help="corporate department "
                                "(e.g. sales)")
    company_profile = form.text_area("Company Profile", value=" ",
                                     help="brief description of the "
                                     "company")
    requirements = form.text_area("Requirements",  value=" ",
                                  help="the job requirements")
    benefits = form.text_area("Benefits", value=" ",
                              help="the benefits provided by "
                              "the employer")
    employment_type = form.text_area("Employment Type", value=" ",
                                     help="the type of role "
                                     "(e.g. full-time)")
    required_experience = form.text_area("Required Experience", value=" ",
                                         help="the level "
                                         "(e.g entry level)")
    required_education = form.text_area("Required Education", value=" ",
                                        help="the education level")
    industry = form.text_area("Industry", value=" ", help="the industry (e.g IT)")
    function = form.text_area("Function", value=" ",
                              help="the function (i.e consulting, "
                              "customer service, etc.)")
    salary_range = form.text_area("Salary Range", value=" ",
                                  help="the expected salary range "
                                  "(e.g. 20000-25000)")
    telecommuting = form.text_area("Telecommuting", value=" ",
                                   help="a boolean, true for "
                                   "telecommuting positions, false otherwise")
    has_company_logo = form.text_area("Company Logo", value=" ",
                                      help="a boolean, true if the "
                                      "company has a logo, false otherwise")
    has_questions = form.text_area("Questions", value=" ",
                                   help="a boolean, true if there are "
                                   "screening questions, false otherwise")

    # error check and convert to int
    if telecommuting.lower() == "true":
        telecommuting = 1
    elif telecommuting.lower() == "false":
        telecommuting = 0
    elif telecommuting != " ":
        st.write("'Telecommuting' needs to be either true or false")

    if has_company_logo.lower() == "true":
        has_company_logo = 1
    elif has_company_logo.lower() == "false":
        has_company_logo = 0
    elif has_company_logo != " ":
        st.write("'Telecommuting' needs to be either true or false")

    if has_questions.lower() == "true":
        has_questions = 1
    elif has_questions.lower() == "false":
        has_questions = 0
    elif has_questions != " ":
        st.write("'Telecommuting' needs to be either true or false")

# error check and run models
if form.form_submit_button():
    # add data to dictionary
    if chosen_model == "Logistic Regression":
        input_dict = {
            "job_id": 0,
            "title": title,
            "description": description,
            "location": location,
            "department": department,
            "company_profile": company_profile,
            "requirements": requirements,
            "benefits": benefits,
            "employment_type": employment_type,
            "required_experience": required_experience,
            "required_education": required_education,
            "industry": industry,
            "function": function,
            }
    else:
        input_dict = {
            "job_id": 0,
            "title": title,
            "description": description,
            "location": location,
            "department": department,
            "company_profile": company_profile,
            "requirements": requirements,
            "benefits": benefits,
            "employment_type": employment_type,
            "required_experience": required_experience,
            "required_education": required_education,
            "industry": industry,
            "function": function,
            "salary_range": salary_range,
            "telecommuting": telecommuting,
            "has_company_logo": has_company_logo,
            "has_questions": has_questions,
            }

    non_job_id = {key: value for key, value in input_dict.items() if key !=
                  "job_id"}
    all_empty = all(value == " " for value in non_job_id.values())
    prediction = None
    if all_empty:
        st.write("At least the description should be provided to make a "
                 "prediction.")
    else:
        if chosen_model == "Logistic Regression":
            processed_data = preprocessing.main(input_dict)
            if processed_data.empty:
                st.write("Input data was filtered out during preprocessing. "
                         "Check that your input is in English and contains "
                         "valid characters.")
            else:
                vectorizer = baseline.vectorizer_saver.load('vectorizer.pkl')
                tfidf_processed_data = vectorizer.transform(processed_data)
                log_reg_model = baseline.model_saver.load('log_reg_model.pkl')
                prediction = baseline.predict(log_reg_model,
                                              tfidf_processed_data)
                threshold = 0.5
        elif chosen_model == "BERT":
            if salary_range != ' ':
                processed_data = multimodal_preprocess.main(data=input_dict)
                enc = multimodal_preprocess.encoder_saver.load('ohe_encoder.'
                                                               'pkl')

                cat_columns = ["employment_type", "required_education",
                               "required_experience", "salary_category",
                               "function", "location"]

                one_hot_encoded = enc.transform(processed_data[cat_columns])
                one_hot_processed_data = \
                    pd.DataFrame(one_hot_encoded,
                                 columns=enc.get_feature_names_out(
                                     cat_columns),
                                 index=processed_data.index)
                processed_data = pd.concat(
                    [processed_data.drop(cat_columns, axis=1),
                     one_hot_processed_data], axis=1)

            else:
                processed_data = preprocess_bert.main(data=input_dict)

                if processed_data.empty:
                    st.write("Input data was filtered out "
                             "during preprocessing. Check that your input is "
                             "in English and contains valid characters.")

                bert_path = ('models/tuned_bert_model')
                tokenizer = AutoTokenizer.from_pretrained(bert_path)
                bert_model = \
                    TFAutoModelForSequenceClassification.from_pretrained(
                        bert_path)

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

                if salary_range != ' ' or telecommuting != ' ' or has_company_logo != ' ' or has_questions != ' ':
                    rf_model = rf.rf_saver.load('rf_model.pkl')
                    rf_prediction = rf_model.predict(processed_data)

                    bert_prediction = bert_model.predict(processed_data)

                    fused_probs_test = 0.5 * bert_prediction + (1.0 - 0.5) * \
                        rf_prediction
                    prediction = (fused_probs_test >= 0.486).astype(int)
                else:
                    prediction_logits = bert_model.predict(val_tf_ds).logits
                    prediction = tf.nn.softmax(prediction_logits,
                                               axis=1).numpy()[:, 1]
                threshold = 0.486

        # provide prediction
        if prediction is not None:
            if prediction >= threshold:
                st.write("Result: Fraudulent")
            else:
                st.write("Result: Real")
