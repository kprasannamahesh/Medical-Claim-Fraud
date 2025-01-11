from fastapi import FastAPI

import joblib

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
# Load the pre-trained logistic regression model
model = joblib.load('logistic_regression_model.joblib')


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers = ["*"]
)

# Define the endpoint for testing the model with the POST method
@app.post("/predict_fraud")
def predict_fraud(data:Dict):
    # Convert the input data into a DataFrame

    # Convert to DataFrame
    single_data_df = pd.DataFrame(data)


    encoder = LabelEncoder()

    # Apply encoding to all categorical columns
    single_data_df['person_gender'] = encoder.fit_transform(single_data_df['person_gender'])
    single_data_df['person_education'] = encoder.fit_transform(single_data_df['person_education'])
    single_data_df['Plan_Type'] = encoder.fit_transform(single_data_df['Plan_Type'])
    single_data_df['Previous_Fraud'] = encoder.fit_transform(single_data_df['Previous_Fraud'])

    prediction = model.predict(single_data_df)

    # Return the prediction result
    result = "Fraud" if prediction[0] == 1 else "Not Fraud"
    return {"prediction": result,"data age":data["person_age"]}
