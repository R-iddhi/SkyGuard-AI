from pathlib import Path
import joblib
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "isolation_forest_3features.joblib"

FEATURES = [
    "temperature",
    "pressure",
    "humidity"
]

print("Loading SHAP model...")
model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)
print("SHAP model loaded successfully.")

FEATURE_NAMES = {
    "temperature": "Temperature",
    "pressure": "Atmospheric pressure",
    "humidity": "Relative humidity"
}

def explain_reading(
    temperature,
    pressure,
    humidity
):
    observation = pd.DataFrame(
        [[
            temperature,
            pressure,
            humidity
        ]],
        columns=FEATURES
    )

    shap_values = explainer.shap_values(observation)
    values = shap_values[0]

    explanations = pd.DataFrame({
        "feature": FEATURES,
        "shap_value": values
    })

    explanations["importance"] = explanations["shap_value"].abs()

    explanations = explanations.sort_values(
        "importance",
        ascending=False
    )

    result = []

    for _, row in explanations.iterrows():
        feature = row["feature"]
        shap_value = float(row["shap_value"])
        importance = float(row["importance"])

        if shap_value < 0:
            contribution = "Contributed toward anomaly detection"
        else:
            contribution = "Contributed away from anomaly detection"

        result.append({
            "feature": feature,
            "feature_name": FEATURE_NAMES[feature],
            "shap_value": round(shap_value, 6),
            "importance": round(importance, 6),
            "contribution": contribution,
            "explanation": (
                f"{FEATURE_NAMES[feature]} was an important factor "
                f"in the model decision."
            )
        })

    return result