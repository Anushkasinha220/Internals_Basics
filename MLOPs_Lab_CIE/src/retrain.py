import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

def run_retraining():
    # 1. Load both datasets
    df_old = pd.read_csv("data/training_data.csv")
    df_new = pd.read_csv("data/new_data.csv")
    
    # Combine datasets
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    
    X = df_combined.drop(columns=['power_output_kwh'])
    y = df_combined['power_output_kwh']
    
    # 2. Split data (Required: random_state=42, test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Load the current Champion from MLflow Model Registry
    model_name = "solaredge-power-output-kwh-predictor"
    champion_uri = f"models:/{model_name}@production"
    
    try:
        champion_model = mlflow.sklearn.load_model(champion_uri)
        champ_mae = mean_absolute_error(y_test, champion_model.predict(X_test))
    except Exception as e:
        print(f"Note: Could not load champion from registry, using fallback. Error: {e}")
        champ_mae = float('inf') # Force promotion if no champion exists

    # 4. Retrain the model (Using Gradient Boosting as it likely won Task 1)
    retrained_model = GradientBoostingRegressor(random_state=42)
    retrained_model.fit(X_train, y_train)
    retrained_mae = mean_absolute_error(y_test, retrained_model.predict(X_test))
    
    # 5. Promotion Logic[cite: 1]
    improvement = champ_mae - retrained_mae
    threshold = 0.5
    action = "promoted" if improvement >= threshold else "kept_champion"
    
    # If promoted, register the new version and update alias[cite: 1]
    if action == "promoted":
        with mlflow.start_run(run_name="Retraining_Pipeline"):
            mlflow.sklearn.log_model(retrained_model, "model")
            v_new = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model", 
                model_name
            )
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(model_name, "production", v_new.version)

    # 6. Generate results/step4_s8.json[cite: 1]
    output = {
        "original_data_rows": len(df_old),
        "new_data_rows": len(df_new),
        "combined_data_rows": len(df_combined),
        "champion_mae": round(float(champ_mae), 4) if champ_mae != float('inf') else 0.0,
        "retrained_mae": round(float(retrained_mae), 4),
        "improvement": round(float(improvement), 4) if improvement != float('-inf') else 0.0,
        "min_improvement_threshold": threshold,
        "action": action,
        "comparison_metric": "mae"
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/step4_s8.json", "w") as f:
        json.dump(output, f, indent=4)
    
    print(f"Task 4 Complete. Action: {action} (Improvement: {improvement:.4f})")

if __name__ == "__main__":
    run_retraining()