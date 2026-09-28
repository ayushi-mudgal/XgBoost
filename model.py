import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/telco_churn_cleaned.csv")

# Display the shape of the dataset
print("Dataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display data types
print("\nData Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

print("\nEncoded Features Shape:")
print(X.shape)

# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Features Shape:")
print(X_train.shape)

print("\nTesting Features Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)


# Import XGBoost
from xgboost import XGBClassifier

# Create the XGBoost classification model
model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)

print("\nXGBoost model created successfully!")

# Train the XGBoost model
model.fit(X_train, y_train)

print("\nXGBoost model trained successfully!")
# Make predictions on the test data
y_pred = model.predict(X_test)

print("\nPredictions made successfully!")

print("\nFirst 10 Predictions:")
print(y_pred[:10])

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

from sklearn.metrics import confusion_matrix

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision
precision = precision_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

print("\nPrecision:")
print(precision)

print("\nRecall:")
print(recall)

print("\nF1 Score:")
print(f1)

print("\nChurn Distribution:")
print(y.value_counts())

from sklearn.metrics import roc_auc_score

# Get churn probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:")
print(roc_auc)



print("\nFirst 10 Churn Probabilities:")
print(y_prob[:10])

# Try a lower churn threshold
threshold = 0.30

y_pred_30 = (y_prob >= threshold).astype(int)

recall_30 = recall_score(y_test, y_pred_30)
precision_30 = precision_score(y_test, y_pred_30)

print("\nResults at 0.30 Threshold:")
print("Precision:", precision_30)
print("Recall:", recall_30)

print("\nThreshold Comparison:")

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60]:

    y_pred_threshold = (y_prob >= threshold).astype(int)

    precision_t = precision_score(y_test, y_pred_threshold)
    recall_t = recall_score(y_test, y_pred_threshold)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision_t:.4f} | "
        f"Recall: {recall_t:.4f}"
    )

    # Calculate class imbalance weight
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nScale Pos Weight:")
print(scale_pos_weight)


model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)


# Tuned XGBoost model
tuned_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

# Train tuned model
tuned_model.fit(X_train, y_train)

# Make predictions
y_pred_tuned = tuned_model.predict(X_test)

# Calculate tuned model metrics
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)

# Calculate tuned ROC-AUC
y_prob_tuned = tuned_model.predict_proba(X_test)[:, 1]
roc_auc_tuned = roc_auc_score(y_test, y_prob_tuned)

print("\nTuned Model Results:")
print("Accuracy:", accuracy_tuned)
print("Precision:", precision_tuned)
print("Recall:", recall_tuned)
print("F1 Score:", f1_tuned)
print("ROC-AUC:", roc_auc_tuned)

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    tuned_model,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc"
)

print("\nCross-Validation ROC-AUC Scores:")
print(cv_scores)

print("\nMean Cross-Validation ROC-AUC:")
print(cv_scores.mean())

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": tuned_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

print("\nAverage Tenure by Churn Status:")
print(df.groupby("Churn")["tenure"].mean())

print("\nChurn Rate by Tenure Group:")

df["tenure_group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, float("inf")],
    labels=["0-12", "13-24", "25-48", "49+"]
)

tenure_churn = df.groupby("tenure_group", observed=True)["Churn"].mean()

print(tenure_churn)

print("\nChurn Rate by Contract Type:")

contract_churn = df.groupby("Contract")["Churn"].mean()

print(contract_churn)


print("\nChurn Rate by Contract Type:")

contract_churn = df.groupby("Contract")["Churn"].mean()

print(contract_churn)


print("\nAverage Tenure by Contract Type:")

contract_tenure = df.groupby("Contract")["tenure"].mean()

print(contract_tenure)


print("\nChurn Rate by Internet Service:")

internet_churn = df.groupby("InternetService")["Churn"].mean()

print(internet_churn)

print("\nChurn Rate by Payment Method:")

payment_churn = df.groupby("PaymentMethod")["Churn"].mean()

print(payment_churn)

print("\nChurn Rate by Contract and Internet Service:")

contract_internet_churn = df.groupby(
    ["Contract", "InternetService"]
)["Churn"].mean()

print(contract_internet_churn)



print("\nChurn Rate by Contract and Payment Method:")

contract_payment_churn = df.groupby(
    ["Contract", "PaymentMethod"]
)["Churn"].mean()

print(contract_payment_churn)


print("\nChurn Rate by Contract and Payment Method:")

contract_payment_churn = pd.crosstab(
    df["Contract"],
    df["PaymentMethod"],
    values=df["Churn"],
    aggfunc="mean"
)

print(contract_payment_churn)

print("\nChurn Rate by Contract, Internet Service and Payment Method:")

three_way_churn = df.groupby(
    ["Contract", "InternetService", "PaymentMethod"]
)["Churn"].mean()

print(three_way_churn.sort_values(ascending=False).head(10))


print("\nTop 10 Highest-Churn Customer Segments:")

three_way_churn = (
    df.groupby(
        ["Contract", "InternetService", "PaymentMethod"]
    )["Churn"]
    .mean()
    .reset_index(name="ChurnRate")
    .sort_values("ChurnRate", ascending=False)
    .head(10)
)

print(three_way_churn.to_string(index=False))


print("\nTop 10 Segments by Churn Rate with Customer Count:")

segment_analysis = (
    df.groupby(
        ["Contract", "InternetService", "PaymentMethod"]
    )
    .agg(
        Customers=("Churn", "size"),
        Churn_Rate=("Churn", "mean")
    )
    .reset_index()
    .sort_values("Churn_Rate", ascending=False)
    .head(10)
)

segment_analysis["Churn_Rate"] = (
    segment_analysis["Churn_Rate"] * 100
)

print(segment_analysis.to_string(index=False))


print("\nTop 10 Segments by Actual Churned Customers:")

segment_churn_count = (
    df.groupby(
        ["Contract", "InternetService", "PaymentMethod"]
    )["Churn"]
    .sum()
    .reset_index(name="Churned_Customers")
    .sort_values("Churned_Customers", ascending=False)
    .head(10)
)

print(segment_churn_count.to_string(index=False))

print("\nTop 10 XGBoost Features:")

print(
    feature_importance.head(10).to_string(index=False)
)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(
    y_test,
    tuned_model.predict(X_test)
)

plt.title("XGBoost Confusion Matrix")

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(
    y_test,
    tuned_model.predict_proba(X_test)[:, 1]
)

plt.title("XGBoost ROC Curve")
plt.tight_layout()

plt.savefig(
    "outputs/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_predictions(
    y_test,
    tuned_model.predict_proba(X_test)[:, 1]
)

plt.title("XGBoost Precision-Recall Curve")
plt.tight_layout()

plt.savefig(
    "outputs/precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Churn Rate by Tenure Group

df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 100],
    labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"]
)

tenure_churn = (
    df.groupby("TenureGroup", observed=False)["Churn"]
    .mean()
)

plt.figure(figsize=(8, 5))

plt.bar(
    tenure_churn.index,
    tenure_churn.values * 100
)

plt.xlabel("Tenure Group")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Customer Tenure")

plt.tight_layout()

plt.savefig(
    "outputs/tenure_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Contract × Internet Service Churn

contract_internet_churn = (
    df.groupby(["Contract", "InternetService"])["Churn"]
    .mean()
    .unstack()
)

contract_internet_churn.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Contract and Internet Service")

plt.xticks(rotation=0)
plt.legend(title="Internet Service")

# Convert y-axis to percentage
plt.gca().set_ylim(0, 0.65)
plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, pos: f"{x*100:.0f}%")
)


plt.tight_layout()

plt.savefig(
    "outputs/contract_internet_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Churn Rate by Payment Method

payment_churn = (
    df.groupby("PaymentMethod")["Churn"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

plt.bar(
    payment_churn.index,
    payment_churn.values * 100
)

plt.xlabel("Payment Method")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Payment Method")

plt.xticks(rotation=30, ha="right")

plt.tight_layout()

plt.savefig(
    "outputs/payment_method_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# High-Risk Customer Segment

segment_data = (
    df.groupby(
        ["Contract", "InternetService", "PaymentMethod"]
    )
    .agg(
        Customers=("Churn", "count"),
        ChurnRate=("Churn", "mean")
    )
    .reset_index()
)

segment_data["Segment"] = (
    segment_data["Contract"]
    + " | "
    + segment_data["InternetService"]
    + " | "
    + segment_data["PaymentMethod"]
)

# Calculate number of churned customers
segment_data["ChurnedCustomers"] = (
    segment_data["Customers"] * segment_data["ChurnRate"]
)

top_segments = (
    segment_data
    .sort_values("ChurnedCustomers", ascending=False)
    .head(5)
)

plt.figure(figsize=(12, 6))

plt.barh(
    top_segments["Segment"],
    top_segments["ChurnedCustomers"]
)

plt.xlabel("Number of Churned Customers")
plt.ylabel("Customer Segment")
plt.title("Top Customer Segments by Number of Churned Customers")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "outputs/top_segments_by_churned_customers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# Feature Importance Plot

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 XGBoost Feature Importances")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# Feature Importance Plot

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 XGBoost Feature Importances")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# Churn Rate by Contract

contract_churn = (
    df.groupby("Contract")["Churn"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

plt.bar(
    contract_churn.index,
    contract_churn.values * 100
)

plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Contract Type")

plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

# Churn Rate by Contract Type

contract_churn = (
    df.groupby("Contract")["Churn"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

plt.bar(
    contract_churn.index,
    contract_churn.values * 100
)

plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Contract Type")

plt.tight_layout()
plt.savefig("outputs/contract_churn.png", dpi=300, bbox_inches="tight")
plt.show()

# Top 5 Highest-Churn Customer Segments

segment_churn = (
    df.groupby(
        ["Contract", "InternetService", "PaymentMethod"]
    )["Churn"]
    .mean()
    .reset_index(name="ChurnRate")
    .sort_values("ChurnRate", ascending=False)
    .head(5)
)

segment_churn["Segment"] = (
    segment_churn["Contract"]
    + " | "
    + segment_churn["InternetService"]
    + " | "
    + segment_churn["PaymentMethod"]
)

plt.figure(figsize=(12, 6))

plt.barh(
    segment_churn["Segment"],
    segment_churn["ChurnRate"] * 100
)

plt.xlabel("Churn Rate (%)")
plt.ylabel("Customer Segment")
plt.title("Top 5 Highest-Churn Customer Segments")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig(
    "outputs/top_churn_segments.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

import joblib

joblib.dump(
    tuned_model,
    "outputs/xgboost_churn_model.pkl"
)

print("Model saved successfully!")