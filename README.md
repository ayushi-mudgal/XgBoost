# XGBoost Customer Churn Prediction

A machine learning project that predicts customer churn using XGBoost and identifies high-risk customer segments through exploratory data analysis and customer segmentation.

The project combines predictive modeling with business-focused analysis to help identify customers who may be at higher risk of churn and support targeted retention strategies.

## Business Problem

Customer churn can lead to lost revenue and increased customer acquisition costs. Identifying customers who are more likely to churn allows businesses to prioritize retention efforts and investigate the factors associated with customer attrition.

## Objective

The objective of this project is to:

- Predict whether a customer is likely to churn.
- Estimate the probability of customer churn.
- Evaluate the performance of an XGBoost classification model.
- Identify important customer attributes used by the model.
- Analyze churn patterns across customer segments.
- Identify high-risk and high-volume customer segments.
- Demonstrate how predictions can support targeted retention analysis.

## Key Business Questions

- Which customers are more likely to churn?
- Which customer characteristics are associated with higher churn?
- Which customer segments have the highest observed churn rates?
- How well can XGBoost distinguish between churners and non-churners?
- How can churn predictions support customer retention decisions?

## Dataset

The project uses a Telco Customer Churn dataset containing customer demographics, account information, services, payment details, and churn status.

### Target Variable

- `Churn` — indicates whether the customer churned.
  - `0` = No churn
  - `1` = Churn

### Main Feature Categories

**Customer Information**
- Gender
- Senior Citizen
- Partner
- Dependents

**Account Information**
- Tenure
- Contract
- Monthly Charges
- Total Charges
- Paperless Billing
- Payment Method

**Services**
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

### Data Preparation

The categorical variables were converted into numerical features using one-hot encoding with `drop_first=True`.

The final dataset contained **30 encoded features** used by the XGBoost model.

## Tools & Technologies

### Programming & Data Analysis
- Python
- Pandas
- NumPy

### Machine Learning
- XGBoost
- Scikit-learn
- Classification
- Cross-validation
- Model evaluation

### Data Visualization
- Matplotlib

### Model Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

### Development
- Visual Studio Code
- Python Virtual Environment
- Git & GitHub

## Project Workflow

The project follows an end-to-end machine learning workflow:

1. **Data Loading**
   - Load the customer churn dataset using Pandas.

2. **Data Inspection**
   - Examine dataset dimensions, columns, data types, and missing values.

3. **Data Preprocessing**
   - Separate features and target variable.
   - Convert categorical variables using one-hot encoding.

4. **Exploratory Data Analysis**
   - Analyze churn patterns across:
     - Customer tenure
     - Contract type
     - Internet service
     - Payment method
     - Customer segments

5. **Model Development**
   - Train an initial XGBoost classification model.
   - Tune model parameters to improve performance.

6. **Model Evaluation**
   - Evaluate the model using:
     - Accuracy
     - Precision
     - Recall
     - F1-score
     - ROC-AUC
     - Confusion Matrix
     - ROC Curve
     - Precision-Recall Curve

7. **Cross-Validation**
   - Perform 5-fold cross-validation using ROC-AUC.

8. **Feature Importance**
   - Identify the features most influential to the model's predictions.

9. **Customer Segmentation**
   - Analyze combinations of contract type, internet service, and payment method.
   - Identify high-churn and high-volume segments.

10. **Prediction**
    - Save the trained XGBoost model.
    - Use the saved model to generate churn predictions and churn probabilities for individual customers.

    ## Model Performance

The tuned XGBoost model was evaluated on a held-out test set.

| Metric | Score |
|---|---:|
| Accuracy | 79.96% |
| Precision | 66.31% |
| Recall | 50.00% |
| F1-Score | 57.01% |
| ROC-AUC | 83.35% |

### Cross-Validation

A 5-fold cross-validation was performed using ROC-AUC as the evaluation metric.

- Mean Cross-Validation ROC-AUC: **84.90%**
- Cross-validation scores:
  - Fold 1: 85.63%
  - Fold 2: 84.41%
  - Fold 3: 82.25%
  - Fold 4: 84.98%
  - Fold 5: 87.22%

The cross-validation results show that model performance remained relatively consistent across the five folds.

## Key Business Insights

### 1. Tenure and Churn

Observed churn was highest among newer customers:

| Tenure Group | Churn Rate |
|---|---:|
| 0–12 months | 47.68% |
| 13–24 months | 28.71% |
| 25–48 months | 20.39% |
| 49+ months | 9.51% |

Customers with shorter observed tenure had substantially higher churn rates.

---

### 2. Contract Type

| Contract Type | Churn Rate |
|---|---:|
| Month-to-month | 42.71% |
| One year | 11.28% |
| Two year | 2.85% |

Month-to-month customers showed substantially higher observed churn than customers on longer contracts.

---

### 3. Internet Service

| Internet Service | Churn Rate |
|---|---:|
| DSL | 19.00% |
| Fiber optic | 41.89% |
| No internet service | 7.43% |

Fiber-optic customers had a higher observed churn rate in this dataset.

---

### 4. Payment Method

Electronic-check customers showed the highest observed churn rate:

- Electronic check: **45.29%**
- Mailed check: **19.20%**
- Bank transfer (automatic): **16.73%**
- Credit card (automatic): **15.25%**

---

### 5. High-Risk Customer Segment

The highest observed churn rate was found among:

**Month-to-month + Fiber optic + Electronic check**

- Customers: **1,307**
- Churn rate: **60.37%**
- Churned customers: **789**

This segment is both relatively large and associated with a high observed churn rate, making it a useful segment for further retention analysis.

> Note: These findings represent observed associations in the dataset and should not be interpreted as proof that any individual feature causes churn.

## Feature Importance

The XGBoost model was used to identify the features with the highest model-based importance.

### Top 10 Features

| Rank | Feature |
|---:|---|
| 1 | InternetService_Fiber optic |
| 2 | Contract_Two year |
| 3 | Contract_One year |
| 4 | InternetService_No |
| 5 | StreamingMovies_Yes |
| 6 | tenure |
| 7 | OnlineSecurity_Yes |
| 8 | TechSupport_Yes |
| 9 | PaymentMethod_Electronic check |
| 10 | PaperlessBilling_Yes |

The feature importance analysis helps identify which variables the model relied on most when making predictions.

> Important: Model feature importance indicates predictive contribution within this model. It does not establish causation or indicate that a feature directly causes customer churn.

## Project Visualizations

The project includes several visual analyses to communicate model results and customer churn patterns.

### Model Analysis

- Feature Importance
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

### Business Analysis

- Churn Rate by Contract Type
- Churn Rate by Customer Tenure
- Churn Rate by Payment Method
- Churn Rate by Contract and Internet Service
- Top 5 Highest-Churn Customer Segments
- Top Customer Segments by Number of Churned Customers

These visualizations help connect machine learning results with customer and business-level insights.

## Customer Churn Prediction

The project includes a `predict.py` script that loads the trained XGBoost model and generates a churn prediction for an individual customer.

The prediction pipeline:

1. Accepts customer attributes.
2. Applies the same categorical encoding used during model training.
3. Aligns the customer data with the model's expected features.
4. Generates a churn prediction.
5. Calculates the customer's churn probability.
6. Assigns a simple risk level.

### Example Output

```text
Predicted Churn: Yes
Churn Probability: 54.09 %
Risk Level: High Churn Risk
Recommended Action: Consider targeted retention outreach.

## Project Structure

```text
XGBoost_Customer_Churn/
│
├── data/
│   ├── download_data.py
│   └── telco_churn_cleaned.csv
│
├── outputs/
│   ├── feature_importance.png
│   ├── contract_churn.png
│   ├── top_churn_segments.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── tenure_churn.png
│   ├── contract_internet_churn.png
│   ├── payment_method_churn.png
│   ├── top_segments_by_churned_customers.png
│   └── xgboost_churn_model.pkl
│
├── eda.py
├── model.py
├── predict.py
├── README.md
└── .venv/

## How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
cd XGBoost_Customer_Churn

### 2. Create and Activate Virtual Environment

```bash
```bash
python -m venv .venv
```

### Activate the environment

```powershell
.venv\Scripts\activate
```

```powershell
.venv\Scripts\activate

## Project Overview

This project predicts customer churn using an **XGBoost classification model**.

The goal is to identify customers who are likely to churn and assign a churn risk level so that businesses can take targeted retention actions.

### Key Objectives

* Analyze customer churn patterns using Exploratory Data Analysis (EDA)
* Identify customer segments with higher churn rates
* Train an XGBoost classification model
* Evaluate model performance using classification metrics
* Predict churn probability for individual customers
* Convert predictions into actionable customer retention recommendations

## Dataset

The project uses a telecommunications customer dataset containing customer demographics, account information, subscribed services, payment details, tenure, and churn status.

### Target Variable

* **Churn** — indicates whether a customer left the service.

### Key Information Used

* Customer tenure
* Contract type
* Monthly charges
* Total charges
* Payment method
* Internet and phone services
* Customer demographics
* Churn status
## Exploratory Data Analysis

The analysis identified clear differences in churn across customer segments.

### Key Findings

* Customers with **0–12 months of tenure** had the highest churn rate at approximately **47.7%**.
* Customers with **49–72 months of tenure** had a much lower churn rate of approximately **9.5%**.
* Churn patterns were also analyzed across **payment methods and customer segments**.
* These findings suggest that **early-tenure customers require particular attention from retention teams**.

### EDA Visualizations

The project includes visualizations for:

* Churn by payment method
* Churn by tenure group
* Top customer segments by churn rate
* Top customer segments by number of churned customers
## Machine Learning Model

An **XGBoost Classifier** was used to predict whether a customer is likely to churn.

### Model Workflow

1. Load the cleaned customer dataset
2. Separate features and target variable
3. Prepare categorical and numerical features
4. Train the XGBoost classification model
5. Evaluate model performance
6. Save the trained model as `xgboost_churn_model.pkl`
7. Use the saved model to generate predictions for new customers

### Model Output

For an individual customer, the prediction script provides:

* **Predicted Churn:** Yes / No
* **Churn Probability:** Probability of churn
* **Risk Level:** Customer churn risk classification
* **Recommended Action:** Suggested retention action based on risk
## Model Evaluation

The XGBoost model was evaluated using classification performance metrics.

### Evaluation Metrics

* **Accuracy** — overall proportion of correct predictions
* **Precision** — proportion of predicted churn customers who actually churned
* **Recall** — proportion of actual churn customers correctly identified
* **F1 Score** — balance between precision and recall
* **ROC-AUC** — measures the model's ability to distinguish between churn and non-churn customers

### Evaluation Visualizations

The `outputs` folder contains:

* `roc_curve.png`
* `precision_recall_curve.png`

These visualizations help assess the model's ability to identify customers at risk of churn.
## Prediction Example

The `predict.py` script uses the trained XGBoost model to predict churn for an individual customer.

Example output:

```text
Predicted Churn: Yes
Churn Probability: 54.09 %
Risk Level: High Churn Risk
Recommended Action: Consider targeted retention outreach.
```

The prediction combines the model's churn probability with a risk classification and a suggested business action.
## Business Recommendations

The churn analysis can support targeted customer retention strategies.

* Prioritize **early-tenure customers**, who show higher observed churn rates.
* Use churn probability to identify customers who may require proactive retention outreach.
* Investigate high-churn customer segments to understand the underlying drivers.
* Design targeted retention offers or service interventions based on customer characteristics.
* Monitor churn rates over time to measure whether retention initiatives are improving outcomes.

The model should be used as a **decision-support tool**, with business teams validating the underlying customer context before taking action.

