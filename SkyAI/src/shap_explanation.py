from pathlib import Path

import joblib
import pandas as pd
import shap


# ==================================================
# PROJECT PATHS
# ==================================================

project_root = Path(__file__).resolve().parent.parent

dataset_path = project_root / "data" / "aws_anomaly_dataset.csv"
model_path = project_root / "data" / "isolation_forest_model.joblib"


# ==================================================
# FEATURES
# ==================================================

features = [
    "temperature",
    "pressure",
    "humidity",
    "wind_speed",
    "wind_direction"
]


# ==================================================
# 1. LOAD DATA
# ==================================================

print("Loading dataset...")

df = pd.read_csv(dataset_path)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# ==================================================
# 2. LOAD SAVED ISOLATION FOREST
# ==================================================

print("Loading saved Isolation Forest model...")

model = joblib.load(model_path)

print("Model loaded successfully.")


# ==================================================
# 3. PREPARE DATA
# ==================================================

X = df[features]


# ==================================================
# 4. CREATE SHAP EXPLAINER
# ==================================================

print("Creating SHAP explainer...")

explainer = shap.TreeExplainer(model)


# ==================================================
# 5. SELECT SUSPICIOUS OBSERVATION
# ==================================================

target_time = pd.Timestamp(
    "2016-09-09 17:00:00"
)

matching_rows = df[
    df["timestamp"] == target_time
]


if matching_rows.empty:

    raise ValueError(
        "Target timestamp was not found in the dataset."
    )


target_index = matching_rows.index[0]

observation = X.loc[
    [target_index]
]


# ==================================================
# 6. CALCULATE SHAP VALUES
# ==================================================

print("Calculating SHAP values...")

shap_values = explainer.shap_values(
    observation
)


# ==================================================
# 7. DISPLAY OBSERVATION
# ==================================================

print()
print("=" * 60)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 60)

print("\nObservation being explained:")

print(
    df.loc[
        target_index,
        [
            "timestamp",
            "temperature",
            "pressure",
            "humidity",
            "wind_speed",
            "wind_direction"
        ]
    ]
)


# ==================================================
# 8. DISPLAY SHAP CONTRIBUTIONS
# ==================================================

values = shap_values[0]


print("\nSHAP FEATURE CONTRIBUTIONS")

for feature, value in zip(
    features,
    values
):

    print(
        f"{feature}: {value:.6f}"
    )


# ==================================================
# 9. RANK FEATURES
# ==================================================

contributions = pd.DataFrame({

    "feature": features,

    "shap_value": values

})


contributions["absolute_shap"] = (
    contributions["shap_value"].abs()
)


contributions = contributions.sort_values(
    "absolute_shap",
    ascending=False
)


print(
    "\nFEATURE IMPORTANCE FOR THIS OBSERVATION"
)

print(
    contributions[
        [
            "feature",
            "shap_value",
            "absolute_shap"
        ]
    ].to_string(index=False)
)


# ==================================================
# 10. SHAP BASE VALUE
# ==================================================

print("\nSHAP BASE VALUE:")

print(explainer.expected_value)


# ==================================================
# 11. TOP CONTRIBUTING FEATURE
# ==================================================

top_feature = contributions.iloc[0]

print("\nTOP CONTRIBUTING FEATURE:")

print(
    f"{top_feature['feature']} "
    f"(absolute SHAP = "
    f"{top_feature['absolute_shap']:.6f})"
)


print("\nSHAP analysis completed successfully.")