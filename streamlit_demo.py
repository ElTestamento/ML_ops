import streamlit as st
import requests

st.title("MLOps Audit Risk System")

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