from fastapi import FastAPI
from pydantic import BaseModel
import cloudpickle
import json
# import joblib
# from sklearn.utils import _joblib
import numpy as np



app = FastAPI()

def _randomstate_ctor(seed=None, **kwargs):
    return np.random.mtrand._rand

np.random.RandomState = _randomstate_ctor


class model_input(BaseModel):
    
    gender : int
    marital_status : int
    date_of_birth : int
    employment : int
    income_per_month : int
    loan_type : int
    applicants_job_role_sector : int
    repayment_type : int
    collateral_type : int
    collateral_value : int
    guarantor_dob : int
    guarantor_relationship : int
    guarantor_employment : int
    guarantor_other_sources_of_income : int
    guarantor_income_per_month : int
    loan_amount : int
    applicant_job_role : int
    applicant_job_sector : int
    age : int
    guarantor_age : int
    applicant_street : int
    applicant_zone : int
    applicant_lga : int
    applicant_state : int
    guarantor_street : int
    guarantor_zone : int
    guarantor_lga : int
    guarantor_state : int



# loading the saved model
# loan_prediction_model = pickle.load(open('loan_prediction_model.sav', 'rb'))
# loan_prediction_model = joblib.load('loan_prediction_model.sav')

# Load the saved model
with open('loan_prediction_model.pkl', 'rb') as file:
    loan_prediction_model = cloudpickle.load(file)



@app.post('/loan_default_prediction')
def loan_prediction(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    gen = input_dictionary['gender']
    mar = input_dictionary['marital_status']
    dob = input_dictionary['date_of_birth']
    emp = input_dictionary['employment']
    incom = input_dictionary['income_per_month']
    loant = input_dictionary['loan_type']
    ajrs = input_dictionary['applicants_job_role_sector']
    repayt = input_dictionary['repayment_type']
    collt = input_dictionary['collateral_type']
    collv = input_dictionary['collateral_value']
    gdob = input_dictionary['guarantor_dob']
    grel = input_dictionary['guarantor_relationship']
    gemp = input_dictionary['guarantor_employment']
    gosi = input_dictionary['guarantor_other_sources_of_income']
    gipm = input_dictionary['guarantor_income_per_month']
    loana = input_dictionary['loan_amount']
    ajr = input_dictionary['applicant_job_role']
    ajs = input_dictionary['applicant_job_sector']
    ag = input_dictionary['age']
    gage = input_dictionary['guarantor_age']
    astr = input_dictionary['applicant_street']
    azone = input_dictionary['applicant_zone']
    alga = input_dictionary['applicant_lga']
    astate = input_dictionary['applicant_state']
    gstreet = input_dictionary['guarantor_street']
    gzone = input_dictionary['guarantor_zone']
    glga = input_dictionary['guarantor_lga']
    gstate = input_dictionary['guarantor_state']
    
    input_list = [gen, mar, dob, emp, incom, loant, ajrs,repayt, collt, collv, gdob, grel, gemp, gosi, gipm, loana, ajr, ajs, ag, gage, astr, azone, alga, astate, gstreet, gzone, glga, gstate]
    
    prediction = loan_prediction_model.predict([input_list])
    
    if (prediction[0] == 0):
        return 'Applicant will not default'
    else:
        return 'Applicant will default'
