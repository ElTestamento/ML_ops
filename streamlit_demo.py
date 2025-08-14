import streamlit as st
import requests
import pandas as pd
import subprocess
import os
import shutil

st.title("MLOps Audit Risk System")

# Tab 1: Prediction
tab1, tab2 = st.tabs(["Risk Prediction", "Training Trigger"])

with tab1:
    st.header("Audit Risk Prediction")

    # Input Form
    sector_score = st.number_input("Sector Score", value=3.89)
    score_a = st.number_input("Score A", value=0.6)
    score_b = st.number_input("Score B", value=0.2)
    score_mv = st.number_input("Score MV", value=0.2)
    district_loss = st.number_input("District Loss", value=2.0)
    risk_e = st.number_input("Risk E", value=0.4)
    score = st.number_input("Score", value=2.4)
    control_risk = st.number_input("Control Risk", value=0.4)

    if st.button("Predict Risk"):
        payload = {
            "Sector_score": sector_score,
            "Score_A": score_a,
            "Score_B": score_b,
            "Score_MV": score_mv,
            "District_Loss": district_loss,
            "RiSk_E": risk_e,
            "Score": score,
            "CONTROL_RISK": control_risk
        }

        response = requests.post("http://localhost:8000/calc", json=payload)
        result = response.json()

        if result["Ergebnis"] == 1:
            st.error("RISIKO ERKANNT")
        else:
            st.success("KEIN RISIKO")

with tab2:
    st.header("Training Trigger")

    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())

        if st.button("Trigger Retraining"):
            # Backup original
            shutil.copy("audit_data.csv", "audit_data_backup.csv")

            # Append new data
            existing_df = pd.read_csv("audit_data.csv")
            combined_df = pd.concat([existing_df, df])
            combined_df.to_csv("audit_data.csv", index=False)

            # Git commit and push
            subprocess.run(["git", "add", "audit_data.csv"])
            subprocess.run(["git", "commit", "-m", "New audit data uploaded"])
            subprocess.run(["git", "push"])

            st.success("GitHub Actions triggered!")

    if st.button("Restore Original Data"):
        shutil.copy("audit_data_backup.csv", "audit_data.csv")
        st.success("Original data restored!")