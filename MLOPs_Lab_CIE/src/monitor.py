import pandas as pd
import json
import os

def run_monitoring():
    # 1. Load Data[cite: 1]
    train_df = pd.read_csv("data/training_data.csv")
    
    logs = []
    with open("logs/predictions.jsonl", "r") as f:
        for line in f:
            logs.append(json.loads(line))
    
    live_df = pd.DataFrame([l['input'] for l in logs])
    predictions = [l['prediction'] for l in logs]

    # 2. Drift Detection Logic[cite: 1]
    def get_status(feature, threshold):
        train_mean = train_df[feature].mean()
        live_mean = live_df[feature].mean()
        shift = abs(live_mean - train_mean)
        status = "ALERT" if shift > threshold else "OK"
        return {
            "feature": feature,
            "train_mean": round(float(train_mean), 2),
            "live_mean": round(float(live_mean), 2),
            "shift": round(float(shift), 2),
            "threshold": threshold,
            "status": status
        }

    alerts = [
        get_status("model_params_millions", 500),
        get_status("queue_depth", 5)
    ]

    # 3. Generate step3_monitoring.json[cite: 1]
    output = {
        "total_predictions": len(logs),
        "mean_prediction": round(float(sum(predictions)/len(predictions)), 2),
        "drift_detected": any(a['status'] == "ALERT" for a in alerts),
        "alerts": alerts
    }

    os.makedirs("results", exist_ok=True)
    with open("results/step3_monitoring.json", "w") as f:
        json.dump(output, f, indent=4)
    
    print("Monitoring Report Generated in results/step3_monitoring.json")

if __name__ == "__main__":
    run_monitoring()