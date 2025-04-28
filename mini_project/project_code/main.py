import pandas as pd
import numpy as np
import sys
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'

def glucose_risk(glucose):
    if glucose < 140:
        return 'Normal'
    elif 140 <= glucose < 200:
        return 'Prediabetic'
    else:
        return 'Diabetic'

def main(csv_path):
    df_test = pd.read_csv(csv_path)

    df_test.fillna(df_test.median(numeric_only=True), inplace=True)
    df_test.fillna("Unknown", inplace=True)

    df_test['bmi_category'] = df_test['bmi'].apply(bmi_category)
    df_test['age_group'] = pd.cut(df_test['age'], bins=[0, 30, 50, 70, 100],
                                  labels=['Young', 'Middle', 'Senior', 'Elderly'])
    df_test['comorbidity_score'] = df_test['hypertension'] + df_test['heart_disease']
    df_test['glucose_level'] = df_test['plasma_glucose'].apply(glucose_risk)
    df_test['bmi_glucose_interaction'] = df_test['bmi'] * df_test['plasma_glucose']
    df_test['age_risk_combo'] = df_test['age'] * df_test['comorbidity_score']

    numerical_features = [
        'age', 'blood_pressure', 'cholesterol', 'max_heart_rate',
        'plasma_glucose', 'skin_thickness', 'insulin', 'bmi',
        'diabetes_pedigree', 'comorbidity_score', 'bmi_glucose_interaction', 'age_risk_combo'
    ]

    categorical_features = [
        'gender', 'residence_type', 'smoking_status',
        'chest_pain_type', 'exercise_angina',
        'bmi_category', 'age_group', 'glucose_level'
    ]

    binary_features = ['hypertension', 'heart_disease']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features),
        ],
        remainder='passthrough'
    )

    X_test = preprocessor.fit_transform(df_test[numerical_features + categorical_features + binary_features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df_test['Cluster'] = kmeans.fit_predict(X_test)

    submission_df = df_test[['patient_id', 'Cluster']]
    submission_df.rename(columns={'Cluster': 'cluster_label'}, inplace=True)
    submission_df.to_csv('output.csv', index=False)

    print("Clustering complete. Output saved to submission.csv")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python cluster_patients.py <path_to_csv>")
    else:
        main(sys.argv[1])
