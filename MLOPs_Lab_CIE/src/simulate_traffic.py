import requests
import pandas as pd
import time

API_URL = "http://127.0.0.1:8500/estimate"

def simulate():
    # Load data for simulation
    train_df = pd.read_csv("data/training_data.csv").drop(columns=['job_completion_min'])
    new_df = pd.read_csv("data/new_data.csv").drop(columns=['job_completion_min'])

    print("Sending 40 normal requests...")
    # Sample 40 from training data (with replacement since training is small)
    normal_samples = train_df.sample(40, replace=True).to_dict(orient="records")
    for req in normal_samples:
        requests.post(API_URL, json=req)
        time.sleep(0.05)

    print("Sending 10 drifted requests...")
    # Sample 10 from new_data
    drift_samples = new_df.sample(10).to_dict(orient="records")
    for req in drift_samples:
        requests.post(API_URL, json=req)
        time.sleep(0.05)
    
    print("Traffic simulation complete.")

if __name__ == "__main__":
    simulate()