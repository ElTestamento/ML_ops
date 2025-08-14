from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import mlflow.pyfunc
import os
import pandas as pd

#Imports für Visualisierung:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
from fastapi.responses import HTMLResponse

# APP erstellen
app = FastAPI()

# Modell laden - Container-ready!
print("Das Modell wird aus MLFlow geladen....")

# Verwende Environment Variable oder Default
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_uri)

print(f"MLflow URI: {mlflow_uri}")

try:
    ml_model = mlflow.pyfunc.load_model('models:/audit_risk_model/latest')
    print("Das Modell wurde erfolgreich geladen!")
except Exception as e:
    print(f"FEHLER beim Model Loading: {e}")
    print("Container läuft ohne Modell - nur für Testing!")
    ml_model = None

def inference(model, input_dict):
    if model is None:
        return [0]  # Dummy Response wenn kein Modell
    values = list(input_dict.values())
    arr = np.array(values)
    input_array = arr.reshape(1, -1)
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

@app.get("/")
def pred():
    return {"app.get.Route": "Derzeit keine Verwendung", "model_loaded": ml_model is not None}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "mlflow_uri": mlflow.get_tracking_uri()
    }

@app.post("/calc")
def give_and_check(check_input: Check_class):
    if ml_model is None:
        return {"Ergebnis": "DEMO_MODE", "message": "Kein Modell geladen - nur Testing"}

    # Pydantic-Instanz zu Dictionary konvertieren
    input_dict = check_input.dict()
    result = inference(ml_model, input_dict)
    # Konvertiere numpy array zu Python int
    prediction = int(result[0])

    # 🎯 Logging einfügen:
    prediction_data = {#alle relevanten Variablen werden geloggt
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
    prediction_df.to_csv('predictions.csv', mode='a', header=False, index=False)#erstelle DF

    return {"Ergebnis": prediction}

#Basis-Visualisierung------------------------------------------
# Zu main.py hinzufügen - nur die nötigsten Zeilen!
def create_charts():
    """4 einfache Charts aus predictions.csv"""
    df = pd.read_csv('predictions.csv', names=[
        'timestamp', 'sector_score', 'score_a', 'score_b', 'score_mv',
        'district_loss', 'risk_e', 'score', 'control_risk', 'prediction'
    ])

    charts = []

    # Chart 1: Risiko-Verteilung
    plt.figure(figsize=(6, 4))
    risk_counts = df['prediction'].value_counts()
    plt.bar(['Kein Risiko', 'Risiko'], [risk_counts.get(0, 0), risk_counts.get(1, 0)], color=['green', 'red'])
    plt.title('Risiko-Verteilung')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    charts.append(base64.b64encode(img.getvalue()).decode())
    plt.close()

    # Chart 2: Timeline
    plt.figure(figsize=(8, 4))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    plt.scatter(df['timestamp'], df['prediction'], c=df['prediction'], cmap='RdYlGn_r', s=100)
    plt.title('Predictions über Zeit')
    plt.ylim(-0.5, 1.5)
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    charts.append(base64.b64encode(img.getvalue()).decode())
    plt.close()

    # Chart 3: Score-Verteilung
    plt.figure(figsize=(6, 4))
    plt.hist(df['score'], bins=5, color='blue', alpha=0.7)
    plt.title('Score-Verteilung')
    plt.xlabel('Score')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    charts.append(base64.b64encode(img.getvalue()).decode())
    plt.close()

    # Chart 4: Feature-Durchschnitte
    plt.figure(figsize=(8, 4))
    features = ['sector_score', 'score_a', 'score_b', 'score_mv', 'district_loss', 'risk_e', 'score', 'control_risk']
    means = [df[f].mean() for f in features]
    plt.bar(range(len(features)), means)
    plt.xticks(range(len(features)), features, rotation=45)
    plt.title('Feature-Durchschnitte')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    charts.append(base64.b64encode(img.getvalue()).decode())
    plt.close()

    return charts[0], charts[1], charts[2], charts[3]


@app.get("/report", response_class=HTMLResponse)
def simple_report():
    """Einfacher Report: Input + Prediction + 4 Charts"""

    # Letzte Prediction aus CSV holen
    df = pd.read_csv('predictions.csv', names=[
        'timestamp', 'sector_score', 'score_a', 'score_b', 'score_mv',
        'district_loss', 'risk_e', 'score', 'control_risk', 'prediction'
    ])
    last_row = df.iloc[-1]

    input_text = f"""
    Sector Score: {last_row['sector_score']}, Score A: {last_row['score_a']}, 
    Score B: {last_row['score_b']}, Score MV: {last_row['score_mv']}, 
    District Loss: {last_row['district_loss']}, Risk E: {last_row['risk_e']}, 
    Score: {last_row['score']}, Control Risk: {last_row['control_risk']}
    """

    prediction_text = "🚨 RISIKO ERKANNT" if last_row['prediction'] == 1 else "✅ KEIN RISIKO"

    # Charts erstellen
    chart1, chart2, chart3, chart4 = create_charts()

    # Einfaches HTML
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
