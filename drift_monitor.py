import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

def check_model_drift():
    df = pd.read_csv('predictions.csv', names=[
        'timestamp', 'sector_score', 'score_a', 'score_b', 'score_mv',
        'district_loss', 'risk_e', 'score', 'control_risk', 'prediction'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    cutoff_date = datetime.now() - timedelta(days=7)
    recent_df = df[df['timestamp'] >= cutoff_date]

    predictions = recent_df['prediction'].values

    prediction_variance = float(np.var(predictions))
    risk_ratio = float(np.mean(predictions))
    expected_risk_ratio = 0.393

    prediction_changes = np.diff(predictions) if len(predictions) > 1 else [0]
    change_frequency = float(np.mean(np.abs(prediction_changes)))

    baseline_accuracy = 0.98
    performance_threshold = 0.05

    consistency_score = 1.0 - min(prediction_variance, 0.5)
    distribution_score = 1.0 - min(abs(risk_ratio - expected_risk_ratio), 0.3)
    temporal_score = 1.0 - min(change_frequency, 0.5)

    estimated_performance = (consistency_score + distribution_score + temporal_score) / 3.0
    performance_drop = baseline_accuracy - estimated_performance
    drift_detected = performance_drop > performance_threshold

    result = {
        'drift_detected': drift_detected,
        'estimated_performance': estimated_performance,
        'performance_drop': performance_drop,
        'recommendation': 'RETRAIN_MODEL' if drift_detected else 'CONTINUE_MONITORING'
    }

    with open('model_drift_report.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == "__main__":
    result = check_model_drift()

    print(f"Drift: {'DETECTED' if result['drift_detected'] else 'OK'}")
    print(f"Performance: {result['estimated_performance']:.3f}")
    print(f"Drop: {result['performance_drop']:.3f}")
    print(f"Action: {result['recommendation']}")

    exit(1 if result['drift_detected'] else 0)