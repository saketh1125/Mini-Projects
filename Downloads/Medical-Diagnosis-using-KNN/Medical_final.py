import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("cleaned_medical_dataset.csv")

# Encode the target variable
label_encoder = LabelEncoder()
data["condition"] = label_encoder.fit_transform(data["condition"])

# Split into features and target
X = data.drop("condition", axis=1)
y = data["condition"]

# Scale the features BEFORE applying SMOTE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to balance the classes (on already scaled data)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_resampled, y_resampled)

# Map numerical condition values to disease names
# Update this mapping based on your actual dataset's conditions
condition_map = {
    0: "No Disease",
    1: "Diabetes",
    2: "High Blood Pressure",
    
}

# Prompt the user for patient details
print("Enter patient details:")
age = float(input("Age: "))
gender = int(input("Gender (0 for male, 1 for female): "))
smoking_status = int(input("Smoking status (0 for non-smoker, 1 for smoker): "))
bmi = float(input("BMI: "))
blood_pressure = float(input("Blood pressure: "))
glucose_levels = float(input("Glucose levels: "))

# Create a DataFrame for the new patient
new_patient = pd.DataFrame([[age, gender, smoking_status, bmi, blood_pressure, glucose_levels]],
                           columns=['age', 'gender', 'smoking_status', 'bmi', 'blood_pressure', 'glucose_levels'])

# Scale the new patient data (using scaler fitted on original data)
new_patient_scaled = scaler.transform(new_patient)

# Predict the condition
prediction = knn.predict(new_patient_scaled)
predicted_condition = prediction[0]
diagnosis = condition_map.get(predicted_condition, "Unknown Disease")

print(f"\nPredicted Diagnosis: {diagnosis}")

# Show prediction probabilities for all classes
probabilities = knn.predict_proba(new_patient_scaled)[0]
print("\nPrediction Confidence:")
for idx, prob in enumerate(probabilities):
    disease_name = condition_map.get(idx, 'Unknown')
    print(f"  {disease_name}: {prob*100:.2f}%")