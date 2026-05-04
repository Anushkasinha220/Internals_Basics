import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

def run_retraining_pipeline():
    # 1. Load the Task 1 results to identify the champion model type
    with open("results/step1_tracking.json", "r") as f:
        step1_results = json.load(f)
    
    champion_name = step1_results["best_model"]
    
    # 2. Combine Datasets
    df_old = pd.read_csv("data/training_data.csv")
    df_new = pd.read_csv("data/new_data.csv")
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    
    X = df_combined.drop(columns=['job_completion_min'])
    y = df_combined['job_completion_min']
    
    # 3. Split Data (Using required random_state=42 and test_size=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize the same model type that won in Task 1
    if champion_name == "GradientBoosting":
        retrained_model = GradientBoostingRegressor(random_state=42)
    else:
        retrained_model = Ridge()
    
    # 5. Train the new model on combined data
    retrained_model.fit(X_train, y_train)
    
    # 6. Evaluate: Champion vs. Retrained Model
    # Use the existing 'best_model.pkl' as the champion[cite: 1]
    champion_model = joblib.load("models/best_model.pkl")
    
    champ_preds = champion_model.predict(X_test)
    retrain_preds = retrained_model.predict(X_test)
    
    champ_rmse = np.sqrt(mean_squared_error(y_test, champ_preds))
    retrain_rmse = np.sqrt(mean_squared_error(y_test, retrain_preds))
    
    improvement = champ_rmse - retrain_rmse
    threshold = 0.5 # Promotion rule: RMSE improves by at least 0.5[cite: 1]
    
    # 7. Decision Logic[cite: 1]
    action = "promoted" if improvement >= threshold else "kept_champion"
    
    if action == "promoted":
        joblib.dump(retrained_model, "models/best_model_v2.pkl")
        # Promotion essentially makes this the new production-ready model
    
    # 8. Generate results/step4_retraining.json[cite: 1]
    output = {
        "original_data_rows": len(df_old),
        "new_data_rows": len(df_new),
        "combined_data_rows": len(df_combined),
        "champion_rmse": round(float(champ_rmse), 4),
        "retrained_rmse": round(float(retrain_rmse), 4),
        "improvement": round(float(improvement), 4),
        "min_improvement_threshold": threshold,
        "action": action,
        "comparison_metric": "rmse"
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/step4_retraining.json", "w") as f:
        json.dump(output, f, indent=4)
    
    print(f"Task 4 Complete. Action: {action} (Improvement: {improvement:.4f})")

if __name__ == "__main__":
    run_retraining_pipeline()