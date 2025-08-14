from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import mlflow.pyfunc
import os
import pandas as pd

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