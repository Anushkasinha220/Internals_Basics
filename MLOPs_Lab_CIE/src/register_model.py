import mlflow
import json

def register():
    with open("results/step1_s1.json") as f:
        step1 = json.load(f)
    
    # Retrieve run_id for the best model from MLflow
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("solaredge-power-output-kwh")
    runs = client.search_runs(experiment.experiment_id, order_by=["metrics.rmse ASC"])
    best_run = runs[0]

    model_name = "solaredge-power-output-kwh-predictor"
    result = mlflow.register_model(f"runs:/{best_run.info.run_id}/model", model_name)
    
    output = {
        "registered_model_name": model_name,
        "version": int(result.version),
        "run_id": best_run.info.run_id,
        "source_metric": "rmse",
        "source_metric_value": best_run.data.metrics["rmse"]
    }
    
    with open("results/step2_s6.json", "w") as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    register()