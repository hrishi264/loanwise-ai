import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_train.csv")
MODEL_PATH = os.path.join(BASE_DIR, "bin", "model.pkl")
SCHEMA_PATH = os.path.join(BASE_DIR, "data", "columns_set.json")

FEATURE_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Gender_Male",
    "Married_Yes",
    "Education_Not Graduate",
    "Self_Employed_Yes",
    "Credit_History_1.0",
    "Dependents_0",
    "Dependents_1",
    "Dependents_2",
    "Dependents_3+",
    "Property_Area_Rural",
    "Property_Area_Semiurban",
    "Property_Area_Urban",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]:
        data[col] = data[col].fillna(data[col].mode().iloc[0])

    data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())
    data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].median())
    data["Credit_History"] = data["Credit_History"].fillna(data["Credit_History"].mode().iloc[0])

    # The public dataset stores loan amount in thousands. The form collects rupees.
    data["LoanAmount"] = data["LoanAmount"] * 1000

    features = pd.DataFrame(0.0, index=data.index, columns=FEATURE_COLUMNS)
    features["ApplicantIncome"] = data["ApplicantIncome"].astype(float)
    features["CoapplicantIncome"] = data["CoapplicantIncome"].astype(float)
    features["LoanAmount"] = data["LoanAmount"].astype(float)
    features["Loan_Amount_Term"] = data["Loan_Amount_Term"].astype(float)
    features["Gender_Male"] = (data["Gender"] == "Male").astype(float)
    features["Married_Yes"] = (data["Married"] == "Yes").astype(float)
    features["Education_Not Graduate"] = (data["Education"] == "Not Graduate").astype(float)
    features["Self_Employed_Yes"] = (data["Self_Employed"] == "Yes").astype(float)
    features["Credit_History_1.0"] = data["Credit_History"].astype(float)

    for value in ["0", "1", "2", "3+"]:
        features[f"Dependents_{value}"] = (data["Dependents"].astype(str) == value).astype(float)
    for value in ["Rural", "Semiurban", "Urban"]:
        features[f"Property_Area_{value}"] = (data["Property_Area"] == value).astype(float)

    return features


def main():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X = build_features(df)
    y = (df["Loan_Status"] == "Y").astype(int)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    schema = {"data_columns": {col: 0 for col in FEATURE_COLUMNS}}
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print("Model saved to", MODEL_PATH)
    print("Schema saved to", SCHEMA_PATH)


if __name__ == "__main__":
    main()
