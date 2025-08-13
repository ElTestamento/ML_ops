#Imports-----------------------------------
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
import joblib
import mlflow
import mlflow.sklearn

#Funktionen-----------------------------

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
  clean_df=df.drop(drop_indices_Audit)
  clean_df=clean_df.drop_duplicates().dropna()
  return clean_df

#Einfache Analyse zum Clean_up---------------------------------------------------
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

#clean-up des Dataframe--------------------------------------------
clean_Audit_df=clean_up(audit_data)
#Kontrolle ob alles bereinigt---------------------------------------
assess_dups_nans(clean_Audit_df)

#Location-ID konvertieren zu int-Werte
print("\nNach Konversion von LOCATION_ID zu int und Bereinigung-----------------------\n")
clean_Audit_df['LOCATION_ID'] = clean_Audit_df['LOCATION_ID'].astype(int)
print(clean_Audit_df.info())
#Einfache Korrelation-----------------------------------------------------------
print("\nKorrelation:--------------------\n")
print("Korrelation Audit_Data\n")
print(clean_Audit_df.corr())
print("\n")

'''Auskommentiert, da EDA abgeschlossen und exportiert!!!

#ydata-profiling für die ausführliche Datenanalyse-------------------------------
EDA_Report_Audit = ProfileReport(clean_Audit_df)
EDA_Report_Audit.to_file(output_file='EDA_Report_Audit.html')

EDA_Report_Trial = ProfileReport(clean_Trial_df)
EDA_Report_Trial.to_file(output_file='EDA_Report_Trial.html')
'''

#Featureengineering-------------------------------------------------------------
#Das Modell wird erstmal trainiert auf die Parameter, die mit dem Zielparameter "risk 0/1"
#am stärksten korrelieren. Hier Identifizieren wir 8 Features:
#(8 Features: Score, Score_MV, Score_B, Score_A, Sector_score, CONTROL_RISK, District_Loss, RiSk_E
#Target: Risk (bleibt binär 0/1). Auf das Engineering neuer
#Parameter wird auf Grund der bereits trainierbaren Wertpräsentation verzichtet.
#Modell wird anhand des clean_Audit_df trainiert.

drop_Audit_features_lst = ['LOCATION_ID', 'PARA_A', 'Risk_A', 'PARA_B',
       'TOTAL', 'numbers', 'Score_B.1', 'Risk_C',
       'Money_Value', 'Risk_D', 'PROB', 'Risk_B',
       'History', 'Prob', 'Risk_F', 'Inherent_Risk', 'Detection_Risk', 'Audit_Risk']
train_audit_set = clean_Audit_df.drop(drop_Audit_features_lst, axis=1)
print(train_audit_set.columns)

print(train_audit_set.head(5))

print(train_audit_set['District_Loss'].head(100))
print(train_audit_set['RiSk_E'].describe())

#Verbleibende Label: ['Sector_score', 'Score_A', 'Score_B', 'Score_MV', 'District_Loss', 'RiSk_E', 'Score', 'CONTROL_RISK', 'Risk'

fig, axs = plt.subplots(3, 3, figsize=(12, 10))
axs[0,0].hist(train_audit_set['RiSk_E'], bins=20)
axs[0,0].set_xlabel('RiSk_E')
axs[0,0].set_ylabel('Anzahl')
axs[0,0].set_title('Histogramm RiSk_E')

axs[0,1].hist(train_audit_set['Score'], bins=20)
axs[0,1].set_xlabel('Score')
axs[0,1].set_ylabel('Anzahl')
axs[0,1].set_title('Histogramm Score')


axs[0,2].hist(train_audit_set['Score_A'], bins=20)
axs[0,2].set_xlabel('Score_A')
axs[0,2].set_ylabel('Anzahl')
axs[0,2].set_title('Histogramm Score_A')

axs[1,0].hist(train_audit_set['Score_B'], bins=20)
axs[1,0].set_xlabel('Score_B')
axs[1,0].set_ylabel('Anzahl')
axs[1,0].set_title('Histogramm Score_B')

axs[1,1].hist(train_audit_set['Score_MV'], bins=20)
axs[1,1].set_xlabel('Score_MV')
axs[1,1].set_ylabel('Anzahl')
axs[1,1].set_title('Histogramm Score_MV')

axs[1,2].hist(train_audit_set['Sector_score'], bins=20)
axs[1,2].set_xlabel('Sector_score')
axs[1,2].set_ylabel('Anzahl')
axs[1,2].set_title('Histogramm Sector_score')

axs[2,0].hist(train_audit_set['CONTROL_RISK'], bins=20)
axs[2,0].set_xlabel('CONTROL_RISK')
axs[2,0].set_ylabel('Anzahl')
axs[2,0].set_title('Histogramm CONTROL_RISK')

axs[2,1].hist(train_audit_set['District_Loss'], bins=20)
axs[2,1].set_xlabel('District_Loss')
axs[2,1].set_ylabel('Anzahl')
axs[2,1].set_title('Histogramm District_Loss')

axs[2,2].hist(train_audit_set['Risk'], bins=20)
axs[2,2].set_xlabel('Risk')
axs[2,2].set_ylabel('Anzahl')
axs[2,2].set_title('Histogramm Risk')

plt.tight_layout()
plt.show()

print(round(train_audit_set.describe(),2))

#Random-Forrest-Classifier:
#Training/Test-Set
X, y = train_audit_set.drop('Risk', axis=1), train_audit_set['Risk']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8,test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

clf = RandomForestClassifier(criterion='gini', max_depth=None, bootstrap=True, random_state=42)
clf.fit(X_train,y_train)

#Predictions-----------------
y_pred = clf.predict(X_test)
clf.predict_proba(X_test)
clf.score(X_test,y_test)

#Metriken:------------------
metric =precision_recall_fscore_support(y_test, y_pred, zero_division='warn')
print(classification_report(y_test, y_pred))

#Visualisierung--------------------
title = 'Confusion Matrix'
normalize = 'true'

confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap=plt.cm.Blues,
        display_labels=clf.classes_,
        normalize=normalize)

disp.ax_.set_title(title)
print(title)
print(disp.confusion_matrix)
plt.show()

#Featuer-Importance--------------------
'''Feature importances are provided by the fitted attribute
feature_importances_ and they are computed as the mean and standard deviation of accumulation of
the impurity decrease within each tree.'''

start_time = time.time()
importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(importances,index=X_train.columns)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

#Modell speichern mit joblib (ist scikit-learn optimiert):--------------------------
print("Modell wird gespeichert...")
joblib.dump(clf, 'ml_model.joblib')
print("wurde unter ml_model.joblib gespeichert")

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.sklearn.log_model(clf, "audit-risk-model")