import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Load Excel File (not CSV)
df = pd.read_excel(r"C:\Users\arnav\FINALDISEASERECOGNITIONPROJECT\disease_symptom_matrix (9).xlsx")

# Step 2: Clean column names
df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True).str.replace('\u200b', '', regex=True)

# Debug: Print column names to verify actual names
print("✅ Column names:")
for col in df.columns:
    print(f">>> '{col}'")

# Step 3: Drop rows with missing values in any of the required target columns
required_columns = ['Disease', 'Treatment', 'Medicine', 'Treatment Duration', 'Medicine Duration']
df = df.dropna(subset=required_columns)

# Step 4: Extract features (X) and targets (y)
X = df.drop(columns=required_columns)
y_disease = df['Disease']
y_treatment = df['Treatment']
y_medicine = df['Medicine']
y_treatmentduration = df['Treatment Duration']
y_medicineduration = df['Medicine Duration']

# Step 5: Train-Test Split (using same X split for all y)
X_train, X_test, y_train_disease, y_test_disease = train_test_split(X, y_disease, test_size=0.2, random_state=42)
_, _, y_train_treatment, y_test_treatment = train_test_split(X, y_treatment, test_size=0.2, random_state=42)
_, _, y_train_medicine, y_test_medicine = train_test_split(X, y_medicine, test_size=0.2, random_state=42)
_, _, y_train_treatdur, y_test_treatdur = train_test_split(X, y_treatmentduration, test_size=0.2, random_state=42)
_, _, y_train_meddur, y_test_meddur = train_test_split(X, y_medicineduration, test_size=0.2, random_state=42)

# Step 6: Train models
model_disease = RandomForestClassifier().fit(X_train, y_train_disease)
model_treatment = RandomForestClassifier().fit(X_train, y_train_treatment)
model_medicine = RandomForestClassifier().fit(X_train, y_train_medicine)
model_treatmentduration = RandomForestClassifier().fit(X_train, y_train_treatdur)
model_medicineduration = RandomForestClassifier().fit(X_train, y_train_meddur)

# Step 7: Save models and symptom list
joblib.dump(model_disease, "model_disease.pkl")
joblib.dump(model_treatment, "model_treatment.pkl")
joblib.dump(model_medicine, "model_medicine.pkl")
joblib.dump(model_treatmentduration, "model_treatmentduration.pkl")
joblib.dump(model_medicineduration, "model_medicineduration.pkl")
joblib.dump(list(X.columns), "symptom_list.pkl")

print("✅ All models trained and saved successfully.")


