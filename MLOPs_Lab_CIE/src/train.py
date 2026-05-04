import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train():
    # 1. Load Data
    df = pd.read_csv("data/training_data.csv")
    X = df.drop(columns=['power_output_kwh'])
    y = df['power_output_kwh']
    
    # 2. Train/Test Split (Required parameters)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Setup MLflow Experiment
    mlflow.set_experiment("solaredge-power-output-kwh")
    model_results = []
    
    # Models to compare
    candidates = [
        ("SVR", SVR()), 
        ("Gradient Boosting", GradientBoostingRegressor(random_state=42))
    ]
    
    for name, model_obj in candidates:
        with mlflow.start_run(run_name=name):
            # Train the model
            model_obj.fit(X_train, y_train)
            preds = model_obj.predict(X_test)
            
            # Calculate metrics[cite: 2]
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            
            # --- CRITICAL UPDATES FOR TASK 2 ---
            # Log metrics and hyperparameters[cite: 2]
            mlflow.log_params(model_obj.get_params())
            mlflow.log_metrics({"mae": mae, "rmse": rmse})
            
            # Log the actual model files so Task 2 can find them[cite: 2]
            mlflow.sklearn.log_model(model_obj, "model")
            
            # Set the required domain tag[cite: 2]
            mlflow.set_tag("domain", "solar_energy")
            # -----------------------------------
            
            model_results.append({
                "name": name, 
                "mae": round(mae, 4), 
                "rmse": round(rmse, 4)
            })

    # 4. Determine best model by RMSE[cite: 2]
    best_model_data = min(model_results, key=lambda x: x['rmse'])
    
    # 5. Generate results/step1_s1.json[cite: 2]
    output = {
        "experiment_name": "solaredge-power-output-kwh",
        "models": model_results,
        "best_model": best_model_data["name"],
        "best_metric_name": "rmse",
        "best_metric_value": best_model_data["rmse"]
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/step1_s1.json", "w") as f:
        json.dump(output, f, indent=4)
    
    print(f"Task 1 Complete. Best model: {best_model_data['name']} (RMSE: {best_model_data['rmse']})")

if __name__ == "__main__":
    train()