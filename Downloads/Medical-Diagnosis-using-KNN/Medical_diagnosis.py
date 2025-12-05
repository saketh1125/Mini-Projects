import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("cleaned_medical_dataset.csv")

label_encoder = LabelEncoder()
data["condition"] = label_encoder.fit_transform(data["condition"])

X = data.drop("condition", axis=1)
y = data["condition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

new_patient = pd.DataFrame([[45, 1, 0, 26.5, 120, 140]],
columns=['age', 'gender', 'smoking_status', 'bmi', 'blood_pressure', 'glucose_levels'])

new_patient_scaled = scaler.transform(new_patient)
prediction = knn.predict(new_patient_scaled)
diagnosis = label_encoder.inverse_transform(prediction)[0]

print("Predicted Diagnosis:", diagnosis)
