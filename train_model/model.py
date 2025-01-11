# Import necessary libraries
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
# Load the saved model
model = joblib.load('logistic_regression_model.joblib')

# Load the test dataset
df = pd.read_excel('claim_data_v1.xlsx')  # Adjust the path as needed
df['person_gender'] = df['person_gender'].map({'male': 0, 'female': 1})
df['person_education'] = df['person_education'].map({'Master': 0, 'Bachelor': 1, 'Associate': 2, 'High School': 3})
df['Plan_Type'] = df['Plan_Type'].map({'Individual protection plan': 0, 'Medical': 1,"Life plan":2,"OTHER":3})
df['Previous_Fraud'] = df['Previous_Fraud'].map({'No': 0, 'Yes': 1})
df = df.drop(["Inspected_or_not"],axis=1)


encoder = LabelEncoder()

# Apply encoding to all categorical columns
df['person_gender'] = encoder.fit_transform(df['person_gender'])
df['person_education'] = encoder.fit_transform(df['person_education'])
df['Plan_Type'] = encoder.fit_transform(df['Plan_Type'])
df['Previous_Fraud'] = encoder.fit_transform(df['Previous_Fraud'])


# Make predictions with the loaded model
fraud_predictions = model.predict(df)  # Predict fraud (0 or 1)
fraud_probabilities = model.predict_proba(df)[:, 1]  # Probability of fraud (class 1)
y_pred = model.predict(df)
print(y_pred)
# Evaluate or display the results
print(f'Predictions: {fraud_predictions}')
print(f'Fraud Probabilities: {fraud_probabilities}')
