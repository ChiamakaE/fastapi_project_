from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

class model_input(BaseModel):
    gender: int
    marital_status: int
    date_of_birth: int
    employment: int
    income_per_month: int
    loan_type: int
    applicants_job_role_sector: int
    repayment_type: int
    collateral_type: int
    collateral_value: int
    guarantor_dob: int
    guarantor_relationship: int
    guarantor_employment: int
    guarantor_other_sources_of_income: int
    guarantor_income_per_month: int
    loan_amount: int
    applicant_job_role: int
    applicant_job_sector: int
    age: int
    guarantor_age: int
    applicant_street: int
    applicant_zone: int
    applicant_lga: int
    applicant_state: int
    guarantor_street: int
    guarantor_zone: int
    guarantor_lga: int
    guarantor_state: int

# Loading the saved model
loan_prediction_model = joblib.load("loan_prediction_model.pkl")

@app.post('/loan_default_prediction')
def loan_prediction(input_parameters: model_input):
    input_data = input_parameters.dict()

    # Extract input values from input_data dictionary
    gen = input_data['gender']
    mar = input_data['marital_status']
    dob = input_data['date_of_birth']
    emp = input_data['employment']
    incom = input_data['income_per_month']
    loant = input_data['loan_type']
    ajrs = input_data['applicants_job_role_sector']
    repayt = input_data['repayment_type']
    collt = input_data['collateral_type']
    collv = input_data['collateral_value']
    gdob = input_data['guarantor_dob']
    grel = input_data['guarantor_relationship']
    gemp = input_data['guarantor_employment']
    gosi = input_data['guarantor_other_sources_of_income']
    gipm = input_data['guarantor_income_per_month']
    loana = input_data['loan_amount']
    ajr = input_data['applicant_job_role']
    ajs = input_data['applicant_job_sector']
    ag = input_data['age']
    gage = input_data['guarantor_age']
    astr = input_data['applicant_street']
    azone = input_data['applicant_zone']
    alga = input_data['applicant_lga']
    astate = input_data['applicant_state']
    gstreet = input_data['guarantor_street']
    gzone = input_data['guarantor_zone']
    glga = input_data['guarantor_lga']
    gstate = input_data['guarantor_state']

    input_list = [[gen, mar, dob, emp, incom, loant, ajrs, repayt, collt, collv, gdob, grel, gemp, gosi, gipm, loana, ajr, ajs, ag, gage, astr, azone, alga, astate, gstreet, gzone, glga, gstate]]

    prediction = loan_prediction_model.predict(input_list)

    if prediction[0] == 0:
        return 'Applicant will not default'
    else:
        return 'Applicant will default'
