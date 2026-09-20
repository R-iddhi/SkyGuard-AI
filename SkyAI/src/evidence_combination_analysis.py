from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_controlled_benchmark.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

previous = df.shift(1)

df["physical"] = (
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

df["temporal"] = (
    (df["temperature_change"] > 10) |
    (df["pressure_change"] > 10) |
    (df["humidity_change"] > 20)
)

df["frozen"] = (
    df["temperature"]
    .rolling(8)
    .std()
    .fillna(999)
    .eq(0)
)

df["multivariate"] = (
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

def get_combination(row):
    evidence = []

    if row["physical"]:
        evidence.append("Physical")

    if row["temporal"]:
        evidence.append("Temporal")

    if row["frozen"]:
        evidence.append("Frozen")

    if row["multivariate"]:
        evidence.append("Multivariate")

    if not evidence:
        return "None"

    return " + ".join(evidence)

df["evidence_combination"] = df.apply(
    get_combination,
    axis=1
)

print("EVIDENCE COMBINATION ANALYSIS")
print()

print("=" * 70)
print("ALL OBSERVATIONS")
print("=" * 70)

print(
    df["evidence_combination"]
    .value_counts()
    .to_string()
)

print()

print("=" * 70)
print("NORMAL OBSERVATIONS")
print("=" * 70)

normal_df = df[df["is_anomaly"] == 0]

print(
    normal_df["evidence_combination"]
    .value_counts()
    .to_string()
)

print()

print("=" * 70)
print("INJECTED ANOMALIES")
print("=" * 70)

anomaly_df = df[df["is_anomaly"] == 1]

for anomaly_type in anomaly_df["anomaly_type"].unique():
    subset = anomaly_df[
        anomaly_df["anomaly_type"] == anomaly_type
    ]

    print()
    print(anomaly_type)

    print(
        subset[
            [
                "timestamp",
                "evidence_combination",
                "temperature_change",
                "pressure_change",
                "humidity_change"
            ]
        ].to_string(index=False)
    )

print()

print("=" * 70)
print("ANOMALY TYPE VS EVIDENCE")
print("=" * 70)

summary = pd.crosstab(
    anomaly_df["anomaly_type"],
    anomaly_df["evidence_combination"]
)

print(summary.to_string())