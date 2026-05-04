import os
import joblib
import pandas as pd
import json
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Constants for the API
MODEL_PATH = "models/best_model.pkl"
LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "predictions.jsonl")
RESULTS_PATH = "results/step2_serving.json"
PORT = 8500

app = FastAPI(
    title="GPUForge Job Completion Predictor",
    description="API with prediction logging and strict validation",
    version="2.0.0"
)

# Load the champion model at startup
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run Task 1 first.")

model = joblib.load(MODEL_PATH)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Schemas ---

class JobRequest(BaseModel):
    # Validation ranges as per Task 2 requirements
    gpu_memory_gb: float = Field(..., ge=8, le=80, description="GPU memory (8-80 GB)")
    batch_size: int = Field(..., ge=8, le=256, description="Batch size (8-256)")
    model_params_millions: float = Field(..., ge=10, le=7000, description="Model params (10-7000M)")
    queue_depth: int = Field(..., ge=1, le=20, description="Queue depth (1-20)")

class PredictionResponse(BaseModel):
    delivery_time_min: float

# --- Endpoints ---

@app.get("/status")
def status():
    """Returns API health and model version."""
    return {
        "status": "running",
        "model": "best_model",
        "version": "1.0"
    }

@app.post("/estimate", response_model=PredictionResponse)
def estimate(request: JobRequest):
    """Predicts completion time and logs the request[cite: 1]."""
    try:
        # 1. Prepare input for model
        input_dict = {
            "gpu_memory_gb": request.gpu_memory_gb,
            "batch_size": request.batch_size,
            "model_params_millions": request.model_params_millions,
            "queue_depth": request.queue_depth
        }
        input_df = pd.DataFrame([input_dict])
        
        # 2. Generate Prediction
        prediction = float(model.predict(input_df)[0])
        
        # 3. Log the prediction for Task 3[cite: 1]
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": input_dict,
            "prediction": prediction,
            "endpoint": "/estimate"
        }
        
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return {"delivery_time_min": round(prediction, 4)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# --- Result Generation for Task 2 ---

def generate_task2_results():
    """Internal helper to generate results/step2_serving.json[cite: 1]."""
    import requests
    import time
    
    test_input = {
        "gpu_memory_gb": 40,
        "batch_size": 64,
        "model_params_millions": 1500,
        "queue_depth": 8
    }

    # Wait for server to be ready
    time.sleep(3)
    try:
        health_resp = requests.get(f"http://127.0.0.1:{PORT}/status").json()
        pred_resp = requests.post(f"http://127.0.0.1:{PORT}/estimate", json=test_input).json()
        
        output = {
            "health_endpoint": "/status",
            "predict_endpoint": "/estimate",
            "port": PORT,
            "health_response": health_resp,
            "test_input": test_input,
            "prediction": pred_resp["delivery_time_min"]
        }

        os.makedirs("results", exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=4)
        print(f"Task 2 output saved to {RESULTS_PATH}")
    except Exception as e:
        print(f"Result generation failed: {e}")

if __name__ == "__main__":
    import uvicorn
    import threading

    # Automatically trigger Task 2 JSON generation once[cite: 1]
    threading.Thread(target=generate_task2_results, daemon=True).start()
    
    # Run server on port 8500[cite: 1]
    uvicorn.run(app, host="127.0.0.1", port=PORT)