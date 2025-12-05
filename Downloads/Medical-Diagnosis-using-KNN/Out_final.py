import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv("cleaned_medical_dataset.csv")

# Split the data into features and target
X = data.drop("condition", axis=1)
y = data["condition"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=29)
knn.fit(X_train, y_train)

# Prompt the user for each symptom
age = float(input("What is the patient's age? "))
gender = int(input("What is the patient's gender (0 for male, 1 for female)? "))
smoking_status = int(input("What is the patient's smoking status (0 for non-smoker, 1 for smoker)? "))
bmi = float(input("What is the patient's BMI? "))
blood_pressure = float(input("What is the patient's blood pressure? "))
glucose_levels = float(input("What is the patient's glucose level? "))

# Create a DataFrame for the new patient
new_patient = pd.DataFrame([[age, gender, smoking_status, bmi, blood_pressure, glucose_levels]],
                           columns=['age', 'gender', 'smoking_status', 'bmi', 'blood_pressure', 'glucose_levels'])

# Scale the new patient data
new_patient_scaled = scaler.transform(new_patient)

# Predict the condition
prediction = knn.predict(new_patient_scaled)

# Map the predicted condition to a meaningful output
condition_map = {0: "No disease", 1: "Diabetes", 2: "Hypertension"}  # Update this mapping as needed
diagnosis = condition_map[prediction[0]]

print(f"Predicted Diagnosis: {diagnosis}")
