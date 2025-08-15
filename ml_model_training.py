#Trainingspipeline ohne graphische Darstellung und Auswertung zur Ausfwahl der Features:
# Primärmodell und Strategie zur Datenaufbereitung, Preprocessing und Featureengineering
# auf Colab entwickelt und trainiert.
#Code hier wesentlich gekürzt um graphischen Output und Statistiken sowie
# "Prints", da hierdurch Performance reduziert wird, aber die "prints" niemand sieht.

import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import joblib
import json
import os
from datetime import datetime


def clean_up(df):
    """Bereinigt DataFrame - robust gegen verschiedene Datentypen"""
    df = df.copy()
    drop_indices_Audit = []

    for i, row in df.iterrows():
        location_id = df.loc[i, 'LOCATION_ID']

        # Prüfe nur wenn es ein String ist
        if isinstance(location_id, str) and location_id.isalpha():
            drop_indices_Audit.append(i)

    print(f"Entferne {len(drop_indices_Audit)} Zeilen mit alphabetischen LOCATION_IDs")

    if drop_indices_Audit:
        clean_df = df.drop(drop_indices_Audit)
    else:
        clean_df = df

    clean_df = clean_df.drop_duplicates().dropna()
    return clean_df

def export_model_and_update_performance(model, accuracy, training_samples):

    # Performance History laden oder erstellen
    performance_file = 'model_performance.json'
    if os.path.exists(performance_file):
        with open(performance_file, 'r') as f:
            perf_data = json.load(f)
    else:
        perf_data = {"models": [], "latest_version": "v0"}

    # Neue Version bestimmen
    current_version_num = len(perf_data['models']) + 1
    new_version = f"v{current_version_num}"

    # Modell-Dateien speichern
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_filename = f"audit_risk_model_{new_version}.joblib"
    model_path = os.path.join(models_dir, model_filename)
    latest_path = os.path.join(models_dir, "audit_risk_model_latest.joblib")

    # Modell speichern
    joblib.dump(model, model_path)
    joblib.dump(model, latest_path)

    # Performance-Eintrag erstellen
    new_model_entry = {
        "version": new_version,
        "accuracy": float(accuracy),
        "timestamp": datetime.now().isoformat(),
        "training_samples": training_samples,
        "model_file": model_filename
    }

    # Performance-Historie updaten
    perf_data['models'].append(new_model_entry)
    perf_data['latest_version'] = new_version

    # Performance-Datei speichern
    with open(performance_file, 'w') as f:
        json.dump(perf_data, f, indent=2)

    print(f"Modell {new_version} exportiert mit Accuracy: {accuracy:.4f}")
    print(f"Gespeichert als: {model_path}")

    return new_version


audit_data = pd.read_csv('audit_data.csv')

clean_Audit_df = clean_up(audit_data)
clean_Audit_df['LOCATION_ID'] = clean_Audit_df['LOCATION_ID'].astype(int)

drop_Audit_features_lst = ['LOCATION_ID', 'PARA_A', 'Risk_A', 'PARA_B',
                           'TOTAL', 'numbers', 'Score_B.1', 'Risk_C',
                           'Money_Value', 'Risk_D', 'PROB', 'Risk_B',
                           'History', 'Prob', 'Risk_F', 'Inherent_Risk', 'Detection_Risk', 'Audit_Risk']
train_audit_set = clean_Audit_df.drop(drop_Audit_features_lst, axis=1)
X, y = train_audit_set.drop('Risk', axis=1), train_audit_set['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

clf = RandomForestClassifier(criterion='gini', max_depth=None, bootstrap=True, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = clf.score(X_test, y_test)

# Model Export - Lokale Dateien
training_samples = len(clean_Audit_df)
model_version = export_model_and_update_performance(clf, score, training_samples)
