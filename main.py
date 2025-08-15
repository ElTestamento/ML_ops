from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
from fastapi.responses import HTMLResponse

app = FastAPI()

def load_model():
    # 1. Files checken (neueste GitHub Models im Ordner models in der IDE)
    if os.path.exists('models/audit_risk_model_latest.joblib'):
        ml_model = joblib.load('models/audit_risk_model_latest.joblib')
        print("Modell aus Git Files geladen!")
        return ml_model

    print("FEHLER: Kein Modell gefunden - DEMO_MODE")
    return None

def inference(model, input_dict):
    if model is None:
        return [0]
    values = list(input_dict.values())
    input_array = np.array(values).reshape(1, -1)
    return model.predict(input_array)


class Check_class(BaseModel):
    Sector_score: float
    Score_A: float
    Score_B: float
    Score_MV: float
    District_Loss: float
    RiSk_E: float
    Score: float
    CONTROL_RISK: float


# Modell beim Start laden
print("Das Modell wird geladen....")
ml_model = load_model()

if ml_model is not None:
    print("Modell erfolgreich geladen!")
else:
    print("FEHLER: Kein Modell gefunden - DEMO_MODE")


@app.get("/")
def root():
    return {"status": "Audit Risk API", "model_loaded": ml_model is not None}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
    }

@app.post("/calc")
def predict_risk(check_input: Check_class):
    """Hauptendpoint für Risk Predictions"""
    if ml_model is None:
        return {"Ergebnis": "DEMO_MODE", "message": "Kein Modell geladen - nur Testing"}

    # Prediction durchführen
    input_dict = check_input.dict()
    result = inference(ml_model, input_dict)
    prediction = int(result[0])

    # Prediction für Reporting speichern
    prediction_data = {
        'timestamp': [pd.Timestamp.now()],
        'sector_score': [check_input.Sector_score],
        'score_a': [check_input.Score_A],
        'score_b': [check_input.Score_B],
        'score_mv': [check_input.Score_MV],
        'district_loss': [check_input.District_Loss],
        'risk_e': [check_input.RiSk_E],
        'score': [check_input.Score],
        'control_risk': [check_input.CONTROL_RISK],
        'prediction': [prediction]
    }

    prediction_df = pd.DataFrame(prediction_data)
    prediction_df.to_csv('predictions.csv', mode='a', header=False, index=False)

    return {"Ergebnis": prediction}


def create_chart(chart_type, df):
    plt.figure(figsize=(6, 4))

    if chart_type == "risk_distribution":
        risk_counts = df['prediction'].value_counts()
        plt.bar(['Kein Risiko', 'Risiko'], [risk_counts.get(0, 0), risk_counts.get(1, 0)],
                color=['green', 'red'])
        plt.title('Risiko-Verteilung')

    elif chart_type == "timeline":
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        plt.scatter(df['timestamp'], df['prediction'], c=df['prediction'], cmap='RdYlGn_r', s=100)
        plt.title('Predictions über Zeit')
        plt.ylim(-0.5, 1.5)

    elif chart_type == "score_distribution":
        plt.hist(df['score'], bins=5, color='blue', alpha=0.7)
        plt.title('Score-Verteilung')
        plt.xlabel('Score')

    elif chart_type == "feature_averages":
        features = ['sector_score', 'score_a', 'score_b', 'score_mv',
                    'district_loss', 'risk_e', 'score', 'control_risk']
        means = [df[f].mean() for f in features]
        plt.bar(range(len(features)), means)
        plt.xticks(range(len(features)), features, rotation=45)
        plt.title('Feature-Durchschnitte')
        plt.tight_layout()

    # Chart zu Base64 konvertieren
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    chart_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return chart_base64


@app.get("/report", response_class=HTMLResponse)
def generate_report():

    # CSV laden
    df = pd.read_csv('predictions.csv', names=[
        'timestamp', 'sector_score', 'score_a', 'score_b', 'score_mv',
        'district_loss', 'risk_e', 'score', 'control_risk', 'prediction'
    ])

    # Letzte Eingabe für Report
    last_row = df.iloc[-1]
    input_text = f"""
    Sector Score: {last_row['sector_score']}, Score A: {last_row['score_a']}, 
    Score B: {last_row['score_b']}, Score MV: {last_row['score_mv']}, 
    District Loss: {last_row['district_loss']}, Risk E: {last_row['risk_e']}, 
    Score: {last_row['score']}, Control Risk: {last_row['control_risk']}
    """

    prediction_text = "RISIKO ERKANNT" if last_row['prediction'] == 1 else "KEIN RISIKO"

    # Charts erstellen
    chart1 = create_chart("risk_distribution", df)
    chart2 = create_chart("timeline", df)
    chart3 = create_chart("score_distribution", df)
    chart4 = create_chart("feature_averages", df)

    # HTML Report
    html = f"""
    <html>
    <body style="font-family: Arial; margin: 40px;">
        <h1>Audit Risk Report</h1>

        <h2>Letzte Eingabe:</h2>
        <p>{input_text}</p>

        <h2>Prediction:</h2>
        <p style="font-size: 24px;">{prediction_text}</p>

        <h2>Charts:</h2>
        <img src="data:image/png;base64,{chart1}" style="margin: 10px;">
        <img src="data:image/png;base64,{chart2}" style="margin: 10px;"><br>
        <img src="data:image/png;base64,{chart3}" style="margin: 10px;">
        <img src="data:image/png;base64,{chart4}" style="margin: 10px;">
    </body>
    </html>
    """

    return html