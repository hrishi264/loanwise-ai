# Flask
from flask import Flask, render_template, request

# Data manipulation
import pandas as pd
import numpy as np

# Logging
import logging

# ML model
import joblib

# JSON
import json

# Utilities
import sys
import os

# Current directory
current_dir = os.path.dirname(__file__)

# Flask app
app = Flask(__name__, static_folder='static', template_folder='.')

# Logging
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


# =========================
# MODEL PREDICTION FUNCTION
# =========================
def ValuePredictor(data: pd.DataFrame):
    model_path = os.path.join(current_dir, "bin", "model.pkl")
    model = joblib.load(model_path)

    # 🔒 Align features with training
    required_features = model.feature_names_in_
    data = data.reindex(columns=required_features, fill_value=0)

    # 🎯 Probability-based prediction
    probability = model.predict_proba(data)[0][1]  # approval probability
    prediction = 1 if probability >= 0.65 else 0   # bank-style threshold

    return prediction, probability


# =========
# HOME PAGE
# =========
@app.route('/')
def home():
    return render_template('index.html')


# =================
# PREDICTION ROUTE
# =================
@app.route('/prediction', methods=['POST'])
def predict():
    if request.method == 'POST':

        # -------- FORM DATA --------
        name = request.form['name']
        gender = float(request.form['gender'])
        education = float(request.form['education'])
        self_employed = float(request.form['self_employed'])
        marital_status = float(request.form['marital_status'])
        dependents = request.form['dependents']
        applicant_income = float(request.form['applicant_income'])
        coapplicant_income = float(request.form['coapplicant_income'])
        loan_amount = float(request.form['loan_amount'])
        loan_term = float(request.form['loan_term'])
        credit_history = float(request.form['credit_history'])
        property_area = request.form['property_area']

        # 🚨 OUTLIER / INVALID DATA GUARD
        if applicant_income <= 0 or loan_amount <= 0:
            return render_template(
                'prediction.html',
                prediction="Invalid financial data detected.",
                status="error",
                probability=0
            )

        # -------- LOAD SCHEMA --------
        schema_path = os.path.join(current_dir, "data", "columns_set.json")
        with open(schema_path, 'r') as f:
            cols = json.load(f)

        schema_cols = cols['data_columns']

        # -------- CATEGORICAL FEATURES --------
        dep_col = f"Dependents_{dependents}"
        if dep_col in schema_cols:
            schema_cols[dep_col] = 1

        prop_col = f"Property_Area_{property_area}"
        if prop_col in schema_cols:
            schema_cols[prop_col] = 1

        # -------- NUMERICAL / BINARY FEATURES --------
        schema_cols['ApplicantIncome'] = applicant_income
        schema_cols['CoapplicantIncome'] = coapplicant_income
        schema_cols['LoanAmount'] = loan_amount
        schema_cols['Loan_Amount_Term'] = loan_term
        schema_cols['Gender_Male'] = gender
        schema_cols['Married_Yes'] = marital_status
        schema_cols['Education_Not Graduate'] = education
        schema_cols['Self_Employed_Yes'] = self_employed
        schema_cols['Credit_History_1.0'] = credit_history

        # -------- DATAFRAME --------
        df = pd.DataFrame(
            data={k: [v] for k, v in schema_cols.items()},
            dtype=float
        )

        df.fillna(0, inplace=True)

        # -------- PREDICTION --------
        result, probability = ValuePredictor(df)
        confidence = round(probability * 100, 2)

        # -------- DECISION LOGIC (BANK STYLE) --------
        if result == 1 and confidence >= 80:
            prediction = f"High chance of approval ({confidence}%)"
            status = "approved"
        elif result == 1:
            prediction = f"Moderate chance of approval ({confidence}%)"
            status = "review"
        else:
            prediction = f"Low chance of approval ({confidence}%)"
            status = "rejected"

        return render_template(
            'prediction.html',
            prediction=prediction,
            status=status,
            probability=confidence
        )

    return render_template('error.html')


# =======
# RUN APP
# =======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
