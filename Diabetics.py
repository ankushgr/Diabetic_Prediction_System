import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
df = pd.read_csv("diabetes.csv")
print(df.head())
print(df.columns)
df.dropna(inplace=True)
x = df.drop('Outcome',axis=1)
y = df['Outcome']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
xtrain,xtest,ytrain,ytest = train_test_split(x_scaled,y,stratify=y,test_size=0.7)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=4, max_features='sqrt', bootstrap=True, n_jobs=-1, random_state=42)
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, min_samples_leaf=4, random_state=42)
rf_cv_scores = cross_val_score(rf_model, x_scaled, y, cv=kf, scoring='accuracy')
dt_cv_scores = cross_val_score(dt_model, x_scaled, y, cv=kf, scoring='accuracy')
rf_model.fit(xtrain, ytrain)
dt_model.fit(xtrain, ytrain)
rf_preds = rf_model.predict(xtest)
dt_preds = dt_model.predict(xtest)
print("Random Forest Accuracy:", accuracy_score(ytest, rf_preds))
print("Decision Tree Accuracy:", accuracy_score(ytest, dt_preds))
# Prediction Of Diabeties #
def predict_custom_input(rf_model, dt_model, scaler):
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    
    print("Enter values for the following features:")
    custom_data = []
    for feature in features:
        value = float(input(f"{feature}: "))
        custom_data.append(value)
    custom_array = np.array(custom_data).reshape(1, -1)
    custom_scaled = scaler.transform(custom_array)
    rf_prediction = rf_model.predict(custom_scaled)[0]
    dt_prediction = dt_model.predict(custom_scaled)[0]
    print("Random Forest Prediction:", "Diabetic" if rf_prediction else "Non-Diabetic")
    print("Decision Tree Prediction:", "Diabetic" if dt_prediction else "Non-Diabetic")
predict_custom_input(rf_model, dt_model, scaler)
