import pandas as pd
import joblib

# Load trained model
model = joblib.load("outputs/xgboost_churn_model.pkl")

# Load dataset
df = pd.read_csv("data/telco_churn_cleaned.csv")

# Remove target column
X = df.drop("Churn", axis=1)

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True, dtype=int)

# Keep the exact feature order used during training
feature_columns = X.columns

# Example customer
customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": 425.0
}])

# Encode customer
customer = pd.get_dummies(
    customer,
    drop_first=True,
    dtype=int
)

# Make sure customer has exactly the same columns as training data
customer = customer.reindex(
    columns=feature_columns,
    fill_value=0
)

# Prediction
prediction = model.predict(customer)[0]

# Churn probability
probability = model.predict_proba(customer)[0][1]

# Risk level
if prediction == 1:
    risk = "High Churn Risk"
else:
    risk = "Low Churn Risk"

print("Predicted Churn:", "Yes" if prediction == 1 else "No")
print("Churn Probability:", round(probability * 100, 2), "%")
print("Risk Level:", risk)

if prediction == 1:
    print("Recommended Action: Consider targeted retention outreach.")
else:
    print("Recommended Action: Continue regular customer engagement.")