import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load real Kaggle dataset
df = pd.read_csv('data/Training.csv')

# Get all unique symptoms
all_symptoms = set()
symptom_cols = [col for col in df.columns if col.startswith('Symptom')]
for col in symptom_cols:
    df[col] = df[col].str.strip().str.replace(' ', '_').str.lower()
    all_symptoms.update(df[col].dropna().unique())

all_symptoms = sorted(list(all_symptoms))
print(f"Total unique symptoms found: {len(all_symptoms)}")

# Convert to binary format
rows = []
for _, row in df.iterrows():
    binary_row = {s: 0 for s in all_symptoms}
    for col in symptom_cols:
        symptom = row[col]
        if pd.notna(symptom) and symptom in binary_row:
            binary_row[symptom] = 1
    binary_row['prognosis'] = row['Disease']
    rows.append(binary_row)

binary_df = pd.DataFrame(rows)
print(f"Dataset shape: {binary_df.shape}")

# Split features and target
X = binary_df.drop('prognosis', axis=1)
y = binary_df['prognosis']

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42)

# Train Random Forest
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")

# Save everything
joblib.dump(model, 'model/model.pkl')
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(list(X.columns), 'model/symptom_columns.pkl')

print("Model saved successfully!")
print(f"Total symptoms: {len(X.columns)}")
print(f"Total diseases: {len(le.classes_)}")