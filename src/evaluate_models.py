models = {
    "XGBoost": {
        "MAE": 8325535.51,
        "RMSE": 18586674.84
    },
    "Prophet": {
        "MAE": 70224299.48,
        "RMSE": 80680534.18
    }
}


best_model = min(models, key=lambda x: models[x]["RMSE"])

print("\nMODEL COMPARISON")
print("-" * 40)

for model_name, metrics in models.items():
    print(f"{model_name}")
    print(f"MAE  : {metrics['MAE']}")
    print(f"RMSE : {metrics['RMSE']}")
    print("-" * 40)

print(f"\nBEST MODEL: {best_model}")