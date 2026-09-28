
print("===== THIS IS THE FILE I AM RUNNING =====")
import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/telco_churn_cleaned.csv")

# Display basic information
print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

print("\nNumerical Summary:")
print(df.describe())

print("\nCategorical Columns and Number of Categories:")

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    print(f"{column}: {df[column].nunique()} categories")

print("\nChurn Rate by Contract:")

contract_churn = df.groupby("Contract")["Churn"].mean() * 100

print(contract_churn)

print("\nChurn Rate by Payment Method:")
payment_churn = df.groupby("PaymentMethod")["Churn"].mean() * 100
print(payment_churn)


print("\nChurn Rate by Internet Service:")
internet_churn = df.groupby("InternetService")["Churn"].mean() * 100
print(internet_churn)


print("\nChurn Rate by Senior Citizen:")
senior_churn = df.groupby("SeniorCitizen")["Churn"].mean() * 100
print(senior_churn)


print("\nChurn Rate by Tenure Group:")

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
)

tenure_churn = df.groupby("TenureGroup", observed=False)["Churn"].mean() * 100
print(tenure_churn)
print("REACHED FINAL SECTION")
print("\n========== FINAL FEATURE CHURN ANALYSIS ==========")
print("\nPayment Method:")
print(payment_churn)

print("\nInternet Service:")
print(internet_churn)

print("\nSenior Citizen:")
print(senior_churn)

print("\nTenure Group:")
print(tenure_churn)