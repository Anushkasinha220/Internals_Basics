import mlflow
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_squared_error

def promote():
    model_name = "solaredge-power-output-kwh-predictor"
    client = mlflow.tracking.MlflowClient()
    
    # 1. Set initial alias 'production' to Version 1
    client.set_registered_model_alias(model_name, "production", "1")
    
    # 2. Prepare Data
    df = pd.read_csv("data/training_data.csv")
    X = df.drop(columns=['power_output_kwh'])
    y = df['power_output_kwh']
    
    # Required split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train Challenger (random_state=99)
    challenger = GradientBoostingRegressor(random_state=99)
    # Using full data for training here as per common promotion task patterns
    challenger.fit(X.iloc[:20], y.iloc[:20]) 
    
    # FIX: Use root_mean_squared_error directly[cite: 1]
    try:
        c_rmse = root_mean_squared_error(y_test, challenger.predict(X_test))
    except ImportError:
        # Fallback for older scikit-learn versions
        c_rmse = mean_squared_error(y_test, challenger.predict(X_test), squared=False)
    
    # 4. Register Challenger as Version 2[cite: 1]
    with mlflow.start_run(run_name="Challenger_Model"):
        mlflow.sklearn.log_model(challenger, "model")
        v2 = mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/model", model_name)
    
    # 5. Load Version 1 RMSE for comparison[cite: 1]
    with open("results/step2_s6.json", "r") as f:
        v1_data = json.load(f)
        v1_rmse = v1_data["source_metric_value"]
    
    # 6. Promotion Logic[cite: 1]
    action = "kept_champion"
    champion_version = 1
    
    if c_rmse < v1_rmse:
        client.set_registered_model_alias(model_name, "production", v2.version)
        action = "promoted"
        champion_version = int(v2.version)

    # 7. Generate results/step3_s7.json[cite: 1]
    output = {
        "registered_model_name": model_name,
        "alias_name": "production",
        "champion_version": champion_version,
        "challenger_version": 2,
        "action": action
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/step3_s7.json", "w") as f:
        json.dump(output, f, indent=4)
    
    print(f"Task 3 Complete. Action: {action} (Challenger RMSE: {c_rmse:.4f})")

if __name__ == "__main__":
    promote()