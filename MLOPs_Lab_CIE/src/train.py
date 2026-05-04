import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def train_baseline_comparison():
    # 1. Load Data
    df = pd.read_csv("data/training_data.csv")
    X = df.drop(columns=['job_completion_min'])
    y = df['job_completion_min']
    
    # 2. Split Data (using required random_state=42 and test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Setup MLflow Experiment
    mlflow.set_experiment("gpuforge-job-completion")
    
    model_results = []
    
    # Define models to compare[cite: 1]
    candidates = [
        ("Ridge", Ridge()),
        ("GradientBoosting", GradientBoostingRegressor(random_state=42))
    ]
    
    for name, model_obj in candidates:
        with mlflow.start_run(run_name=name):
            # Train
            model_obj.fit(X_train, y_train)
            preds = model_obj.predict(X_test)
            
            # Compute required metrics[cite: 1]
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            mape = mean_absolute_percentage_error(y_test, preds)
            
            # Log to MLflow[cite: 1]
            mlflow.log_params(model_obj.get_params())
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2, "mape": mape})
            mlflow.set_tag("experiment_type", "baseline_comparison")
            
            # Save for JSON output
            model_results.append({
                "name": name,
                "mae": round(float(mae), 4),
                "rmse": round(float(rmse), 4),
                "r2": round(float(r2), 4),
                "mape": round(float(mape), 4)
            })
            
            # Save model artifact locally
            os.makedirs("models", exist_ok=True)
            joblib.dump(model_obj, f"models/{name}.pkl")

    # 4. Select Best Model by RMSE[cite: 1]
    best_model_data = min(model_results, key=lambda x: x['rmse'])
    
    # 5. Save results/step1_tracking.json[cite: 1]
    os.makedirs("results", exist_ok=True)
    step1_output = {
        "experiment_name": "gpuforge-job-completion",
        "models": model_results,
        "best_model": best_model_data["name"],
        "best_metric_name": "rmse",
        "best_metric_value": best_model_data["rmse"]
    }
    
    with open("results/step1_tracking.json", "w") as f:
        json.dump(step1_output, f, indent=4)
    
    # Save the winner as 'best_model.pkl' for Task 2[cite: 1]
    joblib.dump(joblib.load(f"models/{best_model_data['name']}.pkl"), "models/best_model.pkl")
    
    print(f"Task 1 Complete. Best Model: {best_model_data['name']} (RMSE: {best_model_data['rmse']})")

if __name__ == "__main__":
    train_baseline_comparison()