import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
import joblib
import mlflow
import mlflow.sklearn
import json
import os
from datetime import datetime

mlflow.set_tracking_uri("http://localhost:5000")


def assess_dups_nans(df_audit):
    audit_data = df_audit
    print("\nNullwerte:")
    print(audit_data.isnull().sum())
    print("\nNaN:")
    print(audit_data.isna().sum())
    print("\nDuplikate:")
    print(audit_data.duplicated().sum())


def clean_up(df):
    df = df
    drop_indices_Audit = []
    for i, row in df.iterrows():
        if df.loc[i, 'LOCATION_ID'].isalpha():
            drop_indices_Audit.append(i)
    print(f"Es werden folgende Zeilen aus den Auditdaten entfernt: {drop_indices_Audit}")
    clean_df = df.drop(drop_indices_Audit)
    clean_df = clean_df.dropna()
    #clean_df = clean_df.drop_duplicates().dropna()
    return clean_df


def export_model_and_update_performance(model, accuracy, training_samples):
    """Exportiert Modell und updated Performance-Historie"""

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


print("Audit-Datensatz")
audit_data = pd.read_csv('audit_data.csv')
print(audit_data.head(5))
print("Indexwerte Auditdaten")
print(audit_data.columns)

print("\nDatensatzinfo:")
print(audit_data.info())

print("\nStatistik:")
print(audit_data.describe())
assess_dups_nans(audit_data)

clean_Audit_df = clean_up(audit_data)
assess_dups_nans(clean_Audit_df)

print("\nNach Konversion von LOCATION_ID zu int und Bereinigung")
clean_Audit_df['LOCATION_ID'] = clean_Audit_df['LOCATION_ID'].astype(int)
print(clean_Audit_df.info())

print("\nKorrelation:")
print(clean_Audit_df.corr())

drop_Audit_features_lst = ['LOCATION_ID', 'PARA_A', 'Risk_A', 'PARA_B',
                           'TOTAL', 'numbers', 'Score_B.1', 'Risk_C',
                           'Money_Value', 'Risk_D', 'PROB', 'Risk_B',
                           'History', 'Prob', 'Risk_F', 'Inherent_Risk', 'Detection_Risk', 'Audit_Risk']
train_audit_set = clean_Audit_df.drop(drop_Audit_features_lst, axis=1)
print(train_audit_set.columns)

print(train_audit_set.head(5))
print(round(train_audit_set.describe(), 2))

X, y = train_audit_set.drop('Risk', axis=1), train_audit_set['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

clf = RandomForestClassifier(criterion='gini', max_depth=None, bootstrap=True, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
clf.predict_proba(X_test)
score = clf.score(X_test, y_test)

metric = precision_recall_fscore_support(y_test, y_pred, zero_division='warn')
print(classification_report(y_test, y_pred))

start_time = time.time()
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(importances, index=X_train.columns)

# Model Export - Lokale Dateien
training_samples = len(clean_Audit_df)
model_version = export_model_and_update_performance(clf, score, training_samples)

# MLflow - falls verfügbar
try:
    with mlflow.start_run():
        mlflow.sklearn.log_model(clf, "audit_risk_model")
        mlflow.log_metric("accuracy", score)

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/audit_risk_model"
        mlflow.register_model(model_uri, "audit_risk_model")

        print("Modell wurde unter 'audit_risk_model' auf MLFlow gespeichert UND registriert")
except Exception as e:
    print(f"MLflow speichern fehlgeschlagen: {e}")
    print("Modell wurde trotzdem lokal gespeichert")

print(f"Training abgeschlossen! Model Version: {model_version}, Accuracy: {score:.4f}")