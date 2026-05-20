from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model = joblib.load('model/model.pkl')
le = joblib.load('model/label_encoder.pkl')
symptom_columns = joblib.load('model/symptom_columns.pkl')

desc_df = pd.read_csv('data/symptom_Description.csv')
prec_df = pd.read_csv('data/symptom_precaution.csv')

SEVERITY = {
    'AIDS': 'Serious', 'Heart attack': 'Serious', 'Tuberculosis': 'Serious',
    'Hepatitis B': 'Serious', 'Hepatitis C': 'Serious', 'Hepatitis D': 'Serious',
    'Hepatitis E': 'Serious', 'Alcoholic hepatitis': 'Serious', 'Malaria': 'Serious',
    'Dengue': 'Serious', 'Typhoid': 'Moderate', 'Pneumonia': 'Serious',
    'Diabetes': 'Moderate', 'Hypertension': 'Moderate', 'Hyperthyroidism': 'Moderate',
    'Hypothyroidism': 'Moderate', 'Hypoglycemia': 'Moderate', 'Migraine': 'Moderate',
    'Allergy': 'Mild', 'Common Cold': 'Mild', 'Acne': 'Mild', 'Fungal infection': 'Mild',
    'Chicken pox': 'Moderate', 'GERD': 'Mild', 'Gastroenteritis': 'Mild',
    'Urinary tract infection': 'Mild', 'Psoriasis': 'Mild', 'Impetigo': 'Mild',
    'Arthritis': 'Moderate', 'Osteoarthritis': 'Moderate', 'Vertigo': 'Mild',
    'Bronchial Asthma': 'Moderate', 'Jaundice': 'Moderate', 'Drug Reaction': 'Moderate',
    'Peptic ulcer disease': 'Moderate', 'Chronic cholestasis': 'Moderate',
    'Cervical spondylosis': 'Mild', 'Paralysis (brain hemorrhage)': 'Serious',
    'Varicose veins': 'Mild', 'hepatitis A': 'Moderate',
    'Dimorphic hemmorhoids(piles)': 'Mild'
}

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptom_columns)

@app.route('/predict', methods=['POST'])
def predict():
    selected = request.form.getlist('symptoms')
    duration = request.form.get('duration', 'Not specified')

    input_vector = [1 if s in selected else 0 for s in symptom_columns]
    input_df = pd.DataFrame([input_vector], columns=symptom_columns)

    proba = model.predict_proba(input_df)[0]
    top3_idx = np.argsort(proba)[::-1][:3]

    predictions = []
    for i, idx in enumerate(top3_idx):
        disease = le.inverse_transform([idx])[0]
        confidence = round(proba[idx] * 100, 1)

        desc_row = desc_df[desc_df['Disease'] == disease]
        description = desc_row['Description'].values[0] if len(desc_row) > 0 else "Consult a doctor for proper diagnosis."

        prec_row = prec_df[prec_df['Disease'] == disease]
        precautions = []
        if len(prec_row) > 0:
            for col in ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']:
                if col in prec_row.columns:
                    val = prec_row[col].values[0]
                    if pd.notna(val):
                        precautions.append(val)

        severity = SEVERITY.get(disease, 'Moderate')
        predictions.append({
            'rank': i + 1,
            'disease': disease,
            'confidence': confidence,
            'description': description,
            'precautions': precautions,
            'severity': severity
        })

    return render_template('result.html',
                           predictions=predictions,
                           symptoms_selected=selected,
                           duration=duration)

if __name__ == '__main__':
    app.run(debug=True)