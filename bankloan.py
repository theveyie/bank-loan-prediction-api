
from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score
)

import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="Bank Loan Response Prediction API", version="1.1")

class LoanInput(BaseModel):
    AGE: float
    EMPLOY: float
    ADDRESS: float
    DEBTINC: float
    CREDDEBT: float
    OTHDEBT: float

# ---------------------------------------------------------------------------
# Load data and train model once at startup
# ---------------------------------------------------------------------------
df = pd.read_csv("BANK LOAN.csv")

print(df.head())

y = df['DEFAULTER']
X = df.drop(['DEFAULTER', 'SN'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

rf_model = RandomForestClassifier(
    n_estimators=500,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

print('OOB Score (baseline RF):', round(rf_model.oob_score_, 3))

y_pred = rf_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print('Confusion Matrix:\n', cm)
print('\nClassification Report:\n')
print(classification_report(y_test, y_pred, digits=3))
print('Test Accuracy:', round(acc, 3))

#--------------------------------
#
#--------------------------------

@app.get("/")
def root():
    return {"message": "Bankloan Default Prediction API", "version": "1.2", "Accuracy": round(acc, 3)}

@app.post("/predict")
def predict(input_data: LoanInput):
    input_df = pd.DataFrame([input_data.model_dump()])
    input_scaled = scaler.transform(input_df)
    probability = rf_model.predict_proba(input_scaled)[:, 1][0]
    return {"default_probability": round(float(probability), 4)}
