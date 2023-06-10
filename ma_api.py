from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

class ModelInput(BaseModel):
    gender: str
    marital_status: str
    employment: str
    income_per_month: int
    loan_type: str
    applicants_job_role_sector: str
    collateral_type: str
    collateral_value: int
    guarantor_relationship: str
    guarantor_employment: str
    guarantor_other_sources_of_income: str
    guarantor_income_per_month: int
    loan_amount: int
    applicant_job_role: str
    applicant_job_sector: str
    age: int
    guarantor_age: int
    applicant_lga: str
    applicant_state: str
    guarantor_lga: str
    guarantor_state: str

# Load the trained model
loan_prediction_model = joblib.load("predict_model.pkl")



@app.post('/loan_default_prediction')
def predict_loan_default(model_input: ModelInput):
    input_dict = model_input.dict()
    input_df = pd.DataFrame([input_dict])  # Convert input_dict to a DataFrame
    input_list = input_df.values  # Get the column values as a NumPy array

      # Get the column values excluding the first column

    input_list = [[
        input_dict['gender'],
        input_dict['marital_status'],
        input_dict['employment'],
        input_dict['income_per_month'],
        input_dict['loan_type'],
        input_dict['applicants_job_role_sector'],
        input_dict['collateral_type'],
        input_dict['collateral_value'],
        input_dict['guarantor_relationship'],
        input_dict['guarantor_employment'],
        input_dict['guarantor_other_sources_of_income'],
        input_dict['guarantor_income_per_month'],
        input_dict['loan_amount'],
        input_dict['applicant_job_role'],
        input_dict['applicant_job_sector'],
        input_dict['age'],
        input_dict['guarantor_age'],
        input_dict['applicant_lga'],
        input_dict['applicant_state'],
        input_dict['guarantor_lga'],
        input_dict['guarantor_state'],
    ]]

    prediction = loan_prediction_model.predict(input_df)

    if prediction[0] == 0:
        return 'Applicant will not default'
    else:
        return 'Applicant will default'

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
