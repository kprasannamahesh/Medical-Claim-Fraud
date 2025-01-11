from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

train_file_path = '/content/Out_of_time_data.csv'  
test_file_path = '/content/claim_data_v1.xlsx'    

train_data = pd.read_excel(train_file_path)
test_data = pd.read_excel(test_file_path)


def clean_data(data):
    data.columns = data.columns.str.strip().str.replace(" ", "_").str.lower()  
    return data.dropna() 

train_data = clean_data(train_data)
test_data = clean_data(test_data)


columns_to_drop = ['inspected_or_not', 'random']
train_data = train_data.drop(columns=columns_to_drop, errors='ignore')
test_data = test_data.drop(columns=columns_to_drop, errors='ignore')


from sklearn.preprocessing import LabelEncoder
categorical_cols = ['person_gender', 'person_education', 'plan_type', 'previous_fraud']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
   
    le.fit(pd.concat([train_data[col], test_data[col]], axis=0))
    train_data[col] = le.transform(train_data[col])
    test_data[col] = test_data[col].apply(lambda x: x if x in le.classes_ else "Unknown")  
    le.classes_ = np.append(le.classes_, "Unknown")  
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

X_train = train_data.drop(columns=['fraud_status'])  
y_train = train_data['fraud_status']                

X_test = test_data.drop(columns=['fraud_status'])   
y_test = test_data['fraud_status']                


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)


svm_model = SVC(class_weight='balanced', random_state=42)

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
}

grid_search = GridSearchCV(svm_model, param_grid, cv=StratifiedKFold(n_splits=5), scoring='roc_auc', verbose=1)
grid_search.fit(X_train_resampled, y_train_resampled)


best_svm_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")


y_pred = best_svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)


