from fastapi import FastAPI, HTTPException
from starlette.responses import RedirectResponse
from pydantic import BaseModel

from starlette.responses import RedirectResponse
from fastapi import FastAPI
from job_fraud_detection.models import baseline
from job_fraud_detection.data import preprocessing

import re


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
    salary_range: str


class ModelOutput(BaseModel):
    prediction: str


app = FastAPI(title="Applied Machine Learning - Group 34",
              summary="Deployment of our AML Baseline Model: a Logistic Regression model trained on Real and Fake job-postings and TF-IDF features.")


@app.get("/", description="Root endpoint that redirects to documentation.")
async def root():
    return RedirectResponse(url='/docs')


@app.post("/predict", summary="Fraud detection baseline model endpoint.", 
          description="Requires at least one of the input fields to form a prediction. " \
"Details of the input fields are as follows: title is the title of the job, " \
"description is a summary of the job details, location is the country code, " \
"department is the corporate department (e.g. sales), company profile is a brief description of the company, "
"requirements are the job requirements, benefits are the benefits provided by the employer, " \
"employment_type is the type of role (e.g. full-time), " \
"required_experience is the level (e.g entry level), required_education is the education level, " \
"industry is the industry (e.g IT), function (i.e consulting, customer service, etc.), " \
"salary_range is the expected salary range (e.g. 20000-25000). Returns a prediction of job fradulence.",
          response_model=ModelOutput, response_description="Prediction of fraud where 'Fraudulent' is a prediction of 0.5 and above and 'Real' is a prediction less than 0.5.")
async def prediction(title: str = " ", description: str = " ",
                     location: str = " ", department: str = " ",
                     company_profile: str = " ", requirements: str = " ",
                     benefits: str = " ", employment_type: str = " ",
                     required_experience: str = " ",
                     required_education: str = " ", industry: str = " ",
                     function: str = " ", salary_range: str = " "):

    if not bool(re.fullmatch(r"\d+-\d+", salary_range.strip())):
        raise HTTPException(status_code=400, detail="Salary range must be in the format [lower-end]-[higher-end].")

    model_input = ModelInput(job_id=0, title=title, description=description,
                             location=location, department=department,
                             company_profile=company_profile,
                             requirements=requirements, benefits=benefits,
                             employment_type=employment_type,
                             required_experience=required_experience,
                             required_education=required_education,
                             industry=industry, function=function,
                             salary_range=salary_range)

    input_dict = model_input.model_dump()
    non_job_id = {key: value for key, value in input_dict.items() if key != "job_id"}
    all_empty = all(value == " " for value in non_job_id.values())

    if all_empty:
        raise HTTPException(status_code=400, detail="At least one input should be provided to make a prediction.")

    processed_data = preprocessing.main(input_dict)
    prediction = baseline.main(processed_data, user_input=True,
                               return_metrics=False)
    if prediction >= 0.5:
        result = "Fraudulent"
    else:
        result = "Real"
    return ModelOutput(prediction=result)
