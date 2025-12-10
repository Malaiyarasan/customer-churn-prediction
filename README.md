# Customer Churn Prediction – Machine Learning

This project predicts which customers are likely to churn using classical
machine learning models. It focuses on **EDA, feature engineering, and model
interpretability** – suitable for Data Analyst / ML Engineer roles.

---

## 🔍 Problem

A telecom-like company wants to identify customers who might stop using the
service (churn) so that the business team can take preventive actions such as:
discounts, personalized offers, or support calls.

---

## 🧠 Approach

1. **Data Understanding & Cleaning**
   - Loaded CSV data (one row = one customer).
   - Handled missing values and inconsistent categories.
   - Encoded categorical variables (e.g., gender, contract type).

2. **Exploratory Data Analysis (EDA)**
   - Checked churn rate by tenure, contract type, and monthly charges.
   - Visualized correlations and churn patterns.

3. **Feature Engineering**
   - Derived features like:
     - tenure buckets (0–6, 6–12, 12+ months)
     - total_spend = monthly_charges * tenure
   - One-hot encoded categorical variables.

4. **Modeling**
   - Split dataset into train/test.
   - Trained:
     - Logistic Regression (baseline)
     - Random Forest Classifier (main model)
   - Used **accuracy, precision, recall, F1-score** to evaluate performance.

5. **Insights**
   - Identified high-risk churn segments:
     - Short-tenure customers with high charges.
     - Month-to-month contract users.
   - These segments can be targeted with retention campaigns.

---

## 🧰 Tech Stack

- Python, Pandas, NumPy
- Scikit-Learn
- Matplotlib / Seaborn
- Jupyter Notebook / VS Code

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── churn_data.csv        # (example dataset placeholder)
│
├── notebooks/
│   └── churn_eda_and_model.ipynb  # EDA + model training (optional)
│
├── src/
│   └── train_model.py        # main training script
│
└── README.md
