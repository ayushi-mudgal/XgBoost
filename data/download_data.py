import pandas as pd

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

df = pd.read_csv(url)

# Clean column names
df.columns = df.columns.str.strip()

# Remove unnecessary customer ID
df = df.drop(columns=["customerID"])

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing TotalCharges
df = df.dropna()

# Convert target to 0/1
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Save cleaned dataset
df.to_csv("data/telco_churn_cleaned.csv", index=False)

print("Dataset cleaned successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nSaved as: data/telco_churn_cleaned.csv")