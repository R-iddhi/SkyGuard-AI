from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_controlled_benchmark.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

previous = df.shift(1)

df["physical_anomaly"] = (
    (df["humidity"] < 0) |
    (df["humidity"] > 100) |
    (df["wind_speed"] < 0) |
    (df["wind_direction"] < 0) |
    (df["wind_direction"] > 360)
)

df["temperature_change"] = (
    df["temperature"] - previous["temperature"]
).abs()

df["pressure_change"] = (
    df["pressure"] - previous["pressure"]
).abs()

df["humidity_change"] = (
    df["humidity"] - previous["humidity"]
).abs()
    
df["temporal_anomaly"] = (
    (df["temperature_change"] > 10) |
    (df["pressure_change"] > 10) |
    (df["humidity_change"] > 20)
)

df["frozen_anomaly"] = (
    df["temperature"]
    .rolling(8)
    .std()
    .fillna(999)
    .eq(0)
)

df["multivariate_anomaly"] = (
    (
        (df["temperature"].abs() > 45) &
        (df["humidity"] < 40) &
        (df["pressure"] < 960)
    ) |
    (
        (df["temperature"] > 40) &
        (df["humidity"] > 90)
    ) |
    (
        (df["temperature"].abs() > 45) &
        df["pressure"].between(960, 1030) &
        df["humidity"].between(20, 80)
    )
)

df["evidence_score"] = 0

df.loc[df["physical_anomaly"], "evidence_score"] += 40
df.loc[df["temporal_anomaly"], "evidence_score"] += 20
df.loc[df["frozen_anomaly"], "evidence_score"] += 30
df.loc[df["multivariate_anomaly"], "evidence_score"] += 30

df["evidence_score"] = df["evidence_score"].clip(upper=100)

def classify(score):
    if score >= 60:
        return "ANOMALY"
    if score >= 30:
        return "SUSPICIOUS"
    return "NORMAL"

df["evidence_status"] = df["evidence_score"].apply(classify)

actual = df["is_anomaly"] == 1
predicted = df["evidence_status"] == "ANOMALY"

true_positive = int((actual & predicted).sum())
false_positive = int((~actual & predicted).sum())
false_negative = int((actual & ~predicted).sum())
true_negative = int((~actual & ~predicted).sum())

precision = (
    true_positive / (true_positive + false_positive)
    if true_positive + false_positive > 0
    else 0
)

recall = (
    true_positive / (true_positive + false_negative)
    if true_positive + false_negative > 0
    else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall > 0
    else 0
)

print("EVIDENCE SCORING DETECTOR")
print()
print("Scoring:")
print("Physical violation      = +40")
print("Temporal anomaly        = +20")
print("Frozen sensor            = +30")
print("Multivariate anomaly    = +30")
print()
print("Status thresholds:")
print("60+  = ANOMALY")
print("30-59 = SUSPICIOUS")
print("0-29 = NORMAL")
print()

print(f"True positives: {true_positive}")
print(f"False positives: {false_positive}")
print(f"False negatives: {false_negative}")
print(f"True negatives: {true_negative}")
print()
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print()

print("=" * 60)
print("DETECTION BY ANOMALY TYPE")
print("=" * 60)

anomaly_df = df[df["is_anomaly"] == 1]

for anomaly_type in anomaly_df["anomaly_type"].unique():
    subset = anomaly_df[
        anomaly_df["anomaly_type"] == anomaly_type
    ]

    total = len(subset)
    detected_count = int(
        (subset["evidence_status"] == "ANOMALY").sum()
    )

    suspicious_count = int(
        (subset["evidence_status"] == "SUSPICIOUS").sum()
    )

    print(
        f"{anomaly_type}: "
        f"ANOMALY={detected_count}/{total}, "
        f"SUSPICIOUS={suspicious_count}/{total}"
    )

print()
print("=" * 60)
print("EVIDENCE SCORE DISTRIBUTION")
print("=" * 60)

print(
    df["evidence_score"]
    .value_counts()
    .sort_index()
    .to_string()
)