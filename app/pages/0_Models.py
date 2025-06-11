import sys
import streamlit as st
import pandas as pd
import tensorflow as tf
from job_fraud_detection.data.multimodal_preprocess import encoder_saver
from job_fraud_detection.models.rf import rf_saver
from datasets import Dataset
from job_fraud_detection.models import baseline, rf
from job_fraud_detection.data import preprocessing, preprocess_bert, \
    multimodal_preprocess
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification
)
from joblib import load
import re
sys.path.append(".")

st.set_page_config(
    page_title="Models"
)

st.title("Deployment")
st.write("In this page a user can select a model, input data to make "
         "predictions on, and view the results.")

# select model
model_list = ("Logistic Regression", "BERT", "BERT + Random Forest (Multimodality)")
chosen_model = st.selectbox("Select a model", model_list)

form = st.form("Input Data")

# insert data to make predictions on
if chosen_model == "Logistic Regression":
    title = form.text_area("Title", value=" ", help="the title of the job")
    description = form.text_area("Description", value=" ",
                                 help="a summary of the job details")
    location_options = [
        'missing', 'US', 'NZ', 'DE', 'GB', 'AU', 'SG', 'IL', 'AE', 'CA', 'IN',
        'EG', 'PL', 'GR', 'BE', 'BR', 'SA', 'DK', 'RU', 'ZA', 'CY', 'HK', 'TR',
        'IE', 'LT', 'JP', 'NL', 'AT', 'KR', 'FR', 'EE', 'TH', 'KE', 'MU', 'MX',
        'RO', 'MY', 'FI', 'CN', 'ES', 'PK', 'SE', 'CL', 'UA', 'QA', 'IT', 'LV',
        'IQ', 'BG', 'PH', 'CZ', 'VI', 'MT', 'HU', 'BD', 'KW', 'LU', 'NG', 'RS',
        'BY', 'ID', 'ZM', 'NO', 'BH', 'UG', 'CH', 'VN', 'TT', 'SD', 'SK', 'AR',
        'TW', 'PT', 'PE', 'CO', 'IS', 'SI', 'MA', 'AM', 'TN', 'GH', 'AL', 'HR',
        'CM', 'SV', 'PA', 'NI', 'LK', 'JM', 'KZ', 'KH'
    ]

    location = form.selectbox("Location", location_options)
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
    employment_options = ["missing", "Full-time", "Part-time", "Contract", "Temporary", "Other"]
    employment_type = form.selectbox("Employment Type", employment_options)
    required_experience_options = [
        'missing', 'Internship', 'Not Applicable', 'Mid-Senior level',
        'Associate', 'Entry level', 'Executive', 'Director'
    ]

    required_experience = form.selectbox("Required Experience", required_experience_options)
    required_education_options = [
        'missing', "Bachelor's Degree", "Master's Degree",
        'High School or equivalent', 'Unspecified',
        'Some College Coursework Completed', 'Vocational', 'Certification',
        'Associate Degree', 'Professional', 'Doctorate',
        'Some High School Coursework', 'Vocational - Degree',
        'Vocational - HS Diploma'
    ]
    required_education = form.selectbox("Required Education", required_education_options)
    industry = form.text_area("Industry", value=" ",
                              help="the industry (e.g IT)")
    function_options = [
        'missing', 'Marketing', 'Customer Service', 'Sales', 'Health Care Provider',
        'Management', 'Information Technology', 'Other', 'Engineering',
        'Administrative', 'Design', 'Production', 'Education', 'Supply Chain',
        'Business Development', 'Product Management', 'Financial Analyst',
        'Consulting', 'Human Resources', 'Project Management', 'Manufacturing',
        'Public Relations', 'Strategy/Planning', 'Advertising', 'Finance',
        'General Business', 'Research', 'Accounting/Auditing', 'Art/Creative',
        'Quality Assurance', 'Data Analyst', 'Business Analyst', 'Writing/Editing',
        'Distribution', 'Science', 'Training', 'Purchasing', 'Legal'
    ]

    function = form.selectbox("Function", function_options)

elif chosen_model == "BERT":
    title = form.text_area("Title", value=" ", help="the title of the job")
    description = form.text_area("Description", value=" ",
                                 help="a summary of the job details")
    company_profile = form.text_area("Company Profile", value=" ",
                                     help="brief description of the "
                                     "company")
    requirements = form.text_area("Requirements",  value=" ",
                                  help="the job requirements")
    benefits = form.text_area("Benefits", value=" ",
                              help="the benefits provided by "
                              "the employer")

elif chosen_model == "BERT + Random Forest (Multimodality)":
    title = form.text_area("Title", value=" ", help="the title of the job")
    description = form.text_area("Description", value=" ",
                                 help="a summary of the job details")
    location_options = ['missing'] + sorted([
        'US', 'NZ', 'DE', 'GB', 'AU', 'SG', 'IL', 'AE', 'CA', 'IN', 'EG', 'PL',
        'GR', 'BE', 'BR', 'SA', 'DK', 'RU', 'ZA', 'CY', 'HK', 'TR', 'IE', 'LT',
        'JP', 'NL', 'AT', 'KR', 'FR', 'EE', 'TH', 'KE', 'MU', 'MX', 'RO', 'MY',
        'FI', 'CN', 'ES', 'PK', 'SE', 'CL', 'UA', 'QA', 'IT', 'LV', 'IQ', 'BG',
        'PH', 'CZ', 'VI', 'MT', 'HU', 'BD', 'KW', 'LU', 'NG', 'RS', 'BY', 'ID',
        'ZM', 'NO', 'BH', 'UG', 'CH', 'VN', 'TT', 'SD', 'SK', 'AR', 'TW', 'PT',
        'PE', 'CO', 'IS', 'SI', 'MA', 'AM', 'TN', 'GH', 'AL', 'HR', 'CM', 'SV',
        'PA', 'NI', 'LK', 'JM', 'KZ', 'KH'
    ])

    location = form.selectbox("Location", location_options)
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
    employment_options = ["missing", "Full-time", "Part-time", "Contract", "Temporary", "Other"]
    employment_type = form.selectbox("Employment Type", employment_options)
    required_experience_options = [
        'missing', 'Internship', 'Not Applicable', 'Mid-Senior level',
        'Associate', 'Entry level', 'Executive', 'Director'
    ]

    required_experience = form.selectbox("Required Experience", required_experience_options)
    required_education_options = [
        'missing', "Bachelor's Degree", "Master's Degree",
        'High School or equivalent', 'Unspecified',
        'Some College Coursework Completed', 'Vocational', 'Certification',
        'Associate Degree', 'Professional', 'Doctorate',
        'Some High School Coursework', 'Vocational - Degree',
        'Vocational - HS Diploma'
    ]
    required_education = form.selectbox("Required Education", required_education_options)

    industry = form.text_area("Industry", value=" ", help="the industry (e.g IT)")
    function_options = [
        'missing', 'Marketing', 'Customer Service', 'Sales', 'Health Care Provider',
        'Management', 'Information Technology', 'Other', 'Engineering',
        'Administrative', 'Design', 'Production', 'Education', 'Supply Chain',
        'Business Development', 'Product Management', 'Financial Analyst',
        'Consulting', 'Human Resources', 'Project Management', 'Manufacturing',
        'Public Relations', 'Strategy/Planning', 'Advertising', 'Finance',
        'General Business', 'Research', 'Accounting/Auditing', 'Art/Creative',
        'Quality Assurance', 'Data Analyst', 'Business Analyst', 'Writing/Editing',
        'Distribution', 'Science', 'Training', 'Purchasing', 'Legal'
    ]

    function = form.selectbox("Function", function_options)
    salary_range = form.text_area("Salary Range", value=" ",
                                  help="the expected salary range "
                                  "(e.g. 20000-25000)")
    telecommuting = form.selectbox("Telecommuting", ["False", "True"], index=0)
    has_company_logo = form.selectbox("Company Logo", ["False", "True"], index=0)
    has_questions = form.selectbox("Questions", ["False", "True"], index=0)

    telecommuting = 1 if telecommuting == "True" else 0
    has_company_logo = 1 if has_company_logo == "True" else 0
    has_questions = 1 if has_questions == "True" else 0

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
    elif chosen_model == "BERT":
        input_dict = {
            "job_id": 0,
            "title": title,
            "description": description,
            "company_profile": company_profile,
            "requirements": requirements,
            "benefits": benefits,
            "location": "",
            "department": "",
            "industry": ""
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
            print(processed_data)
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
            processed_data = preprocess_bert.main(data=input_dict)
            if processed_data.empty:
                    st.write("Input data was filtered out "
                             "during preprocessing. Check that your input is "
                             "in English and contains valid characters.")
            else:
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

                print(processed_data)

                bert_val_ds = Dataset.from_pandas(processed_data)
                tokenized_val = bert_val_ds.map(preprocess_for_bert)
                val_tf_ds = bert_model.prepare_tf_dataset(
                    tokenized_val,
                    shuffle=False,
                    batch_size=8
                )

                print(bert_val_ds)

                test_logits = bert_model.predict(val_tf_ds).logits
                predictions = tf.nn.softmax(test_logits, axis=1).numpy()[:, 1]

                prediction = predictions[0]
                print(prediction)
                threshold = 0.9878


        elif chosen_model == "BERT + Random Forest (Multimodality)":
            processed_data = preprocess_bert.main(data=input_dict)
            print(telecommuting)
            if salary_range == " ":
                input_dict["salary_range"] = "-"
            print(list(salary_range))
            if salary_range and salary_range != " ":
                m = re.match(r"^(\d+)-(\d+)$", salary_range.strip())
                if not m:
                    st.error("Salary must be `min-max` (e.g. `20000-25000`).")
                    st.stop()
                min_sal, max_sal = map(int, m.groups())
                if min_sal > max_sal:
                    st.error("Minimum salary cannot exceed maximum salary.")
                    st.stop()
            if telecommuting != 1:
                input_dict["telecommuting"] = 0
            if has_company_logo != 1:
                input_dict["has_company_logo"] = 0
            if has_questions != 1:
                input_dict["has_questions"] = 0
            processed_rf = multimodal_preprocess.main(input_dict)
            processed_rf = processed_rf.replace(" ", "missing")
            processed_rf = processed_rf.replace("", "missing")
            st.write(processed_rf)

            if processed_data.empty:
                    st.write("Input data was filtered out "
                             "during preprocessing. Check that your input is "
                             "in English and contains valid characters.")
            else:
                print("OK")
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

                print(processed_data)

                bert_val_ds = Dataset.from_pandas(processed_data)
                tokenized_val = bert_val_ds.map(preprocess_for_bert)
                val_tf_ds = bert_model.prepare_tf_dataset(
                    tokenized_val,
                    shuffle=False,
                    batch_size=8
                )

                print(bert_val_ds)

                test_logits = bert_model.predict(val_tf_ds).logits
                predictions = tf.nn.softmax(test_logits, axis=1).numpy()[:, 1]

                bert_prediction = predictions[0]

                rf_model = rf_saver.load(name='rf_model.pkl')
                print("nice")
                enc = encoder_saver.load(name='ohe_encoder.pkl')
                print("ok")
                processed_rf = processed_rf.drop(columns=["text"])
                st.write(processed_rf)
                cat_columns = ["employment_type", "required_education",
                   "required_experience", "salary_category",
                   "function", "location"]
                one_hot_encoded = enc.transform(processed_rf[cat_columns])
                one_hot_df = pd.DataFrame(one_hot_encoded,
                                        columns=enc.get_feature_names_out(cat_columns),
                                        index=processed_rf.index)
                X_train = pd.concat([processed_rf.drop(cat_columns, axis=1), one_hot_df],
                                    axis=1)
                st.write(X_train)
                probs_rf = rf_model.predict_proba(X_train)[0, 1]
                prediction = 0.5 * bert_prediction + (1.0 - 0.5) * probs_rf
                st.write(probs_rf, bert_prediction)
                st.write(prediction)

            threshold = 0.4939

        # provide prediction
        if prediction is not None:
            if prediction >= threshold:
                st.write("Result: Fraudulent")
            else:
                st.write("Result: Real")
