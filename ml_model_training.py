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

def assess_dups_nans(df_audit):
    audit_data = df_audit
    print("\nNullwerte:----------------------------------\n")
    print("Nullwerte Auditdata\n")
    print(audit_data.isnull().sum())
    print("\nNaN:----------------------------------\n")
    print("NaN-Werte Auditdata\n")
    print(audit_data.isna().sum())
    print("\nDuplikate:----------------------------\n")
    print("Duplikate Auditdata\n")
    print(audit_data.duplicated().sum())

def clean_up(df):
    df = df
    drop_indices_Audit = []
    for i, row in df.iterrows():
        if df.loc[i, 'LOCATION_ID'].isalpha():
            drop_indices_Audit.append(i)
    print(f"Es werden folgende Zeilen aus den Auditdaten entfernt: {drop_indices_Audit}")
    clean_df = df.drop(drop_indices_Audit)
    clean_df = clean_df.drop_duplicates().dropna()
    return clean_df

print("Audit-Datensatz-----------------------------\n")
audit_data = pd.read_csv('audit_data.csv')
print(audit_data.head(5))
print("Indexwerte Auditdaten\n")
print(audit_data.columns)

print("\n")

print("\nDatensatzinfo:-----------------------------\n")
print("Info Auditdata\n")
print(audit_data.info())

print("\nStatistik:\n")
print("Stats Auditdata\n")
print(audit_data.describe())
print("\n")
assess_dups_nans(audit_data)

print("\n")

clean_Audit_df = clean_up(audit_data)
assess_dups_nans(clean_Audit_df)

print("\nNach Konversion von LOCATION_ID zu int und Bereinigung-----------------------\n")
clean_Audit_df['LOCATION_ID'] = clean_Audit_df['LOCATION_ID'].astype(int)
print(clean_Audit_df.info())

print("\nKorrelation:--------------------\n")
print("Korrelation Audit_Data\n")
print(clean_Audit_df.corr())
print("\n")

drop_Audit_features_lst = ['LOCATION_ID', 'PARA_A', 'Risk_A', 'PARA_B',
                          'TOTAL', 'numbers', 'Score_B.1', 'Risk_C',
                          'Money_Value', 'Risk_D', 'PROB', 'Risk_B',
                          'History', 'Prob', 'Risk_F', 'Inherent_Risk', 'Detection_Risk', 'Audit_Risk']
train_audit_set = clean_Audit_df.drop(drop_Audit_features_lst, axis=1)
print(train_audit_set.columns)

print(train_audit_set.head(5))

print(train_audit_set['District_Loss'].head(100))
print(train_audit_set['RiSk_E'].describe())

print(round(train_audit_set.describe(),2))

X, y = train_audit_set.drop('Risk', axis=1), train_audit_set['Risk']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

clf = RandomForestClassifier(criterion='gini', max_depth=None, bootstrap=True, random_state=42)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
clf.predict_proba(X_test)
score = clf.score(X_test,y_test)

metric = precision_recall_fscore_support(y_test, y_pred, zero_division='warn')
print(classification_report(y_test, y_pred))

start_time = time.time()
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(importances,index=X_train.columns)

print("Modell wird gespeichert...")
joblib.dump(clf, 'ml_model.joblib')
print("Modell wurde unter ml_model.joblib gespeichert")
# Nach dem Training:
mlflow.sklearn.log_model(clf, "audit_risk_model")
print("Modell wurde unter 'audit_risk_model' auf MLFlow gespeichert")

print(f"Training completed! Model accuracy: {score:.4f}")