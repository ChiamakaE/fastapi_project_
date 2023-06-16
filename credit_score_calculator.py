from sklearn.base import BaseEstimator, TransformerMixin

class CreditScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        prediction = self.model.predict(X)  # Predict loan default using the trained model
        X["loan_default"] = prediction.tolist()  # Add loan_default column to DataFrame
        X["credit_score"] = X.apply(self.calculate_credit_score, axis=1)
        return X

    def calculate_credit_score(self, row):
        income_per_month = row["income_per_month"]
        loan_amount = row["loan_amount"]
        loan_default = row["loan_default"]

        # Calculate the credit score based on the provided conditions
        if income_per_month >= 500000 and loan_default == 0:
            credit_score = 700
        elif income_per_month >= 300000 and loan_default == 0:
            credit_score = 600
        elif income_per_month >= 50000 and loan_default == 1:
            credit_score = 500
        elif income_per_month >= 10000 and loan_default == 1:
            credit_score = 400
        else:
            credit_score = 300

        # Adjust the credit score based on loan amount and income per month
        if loan_amount > (income_per_month * 2):
            credit_score -= 50
        elif loan_amount > (income_per_month * 1.5):
            credit_score -= 25

        # Ensure the credit score is within the range of 300 to 850
        credit_score = max(300, min(850, credit_score))

        return credit_score
