from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from credit_score_calculator import CreditScoreTransformer

app = FastAPI()

class ModelInput(BaseModel):
    gender: str
    marital_status: str
    employment: str
    income_per_month: float
    loan_type: str
    collateral_type: str
    collateral_value: float
    guarantor_relationship: str
    guarantor_employment: str
    guarantor_other_sources_of_income: str
    guarantor_income_per_month: float
    loan_amount: float
    applicant_job_role: str
    applicant_job_sector: str
    age: float
    guarantor_age: float


@app.post('/loan_default_prediction')
def predict_loan_default(model_input: ModelInput):
    input_dict = model_input.dict()
    input_df = pd.DataFrame([input_dict])  # Convert input_dict to a DataFrame

    # Load the trained model
    loan_prediction_model = joblib.load("predict_model.pkl")


    # Calculate credit score
    credit_score_transformer = CreditScoreTransformer(model=loan_prediction_model)
    input_df_with_credit_score = credit_score_transformer.transform(input_df)

    # Extract the calculated credit score
    credit_score = int(input_df_with_credit_score["credit_score"].iloc[0])

    # Prepare the response
    if input_df_with_credit_score["loan_default"].iloc[0] == 0:
        loan_eligibility = True
    else:
        loan_eligibility = False

    result = {
        "status": "success",
        "loanEligibility": loan_eligibility,
        "creditScore": credit_score
    }

    return result

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
