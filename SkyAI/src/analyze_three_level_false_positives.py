from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_final_benchmark.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

temperature_before = df["temperature"].diff().abs()
temperature_after = df["temperature"].diff(-1).abs()

pressure_before = df["pressure"].diff().abs()
pressure_after = df["pressure"].diff(-1).abs()

humidity_before = df["humidity"].diff().abs()
humidity_after = df["humidity"].diff(-1).abs()

temperature_recovery = (
    (temperature_before > 10)
    & (temperature_after > 10)
    & (
        (
            df["temperature"]
            - (
                df["temperature"].shift(1)
                + df["temperature"].shift(-1)
            ) / 2
        ).abs()
        > 15
    )
)

pressure_recovery = (
    (pressure_before > 10)
    & (pressure_after > 10)
    & (
        (
            df["pressure"]
            - (
                df["pressure"].shift(1)
                + df["pressure"].shift(-1)
            ) / 2
        ).abs()
        > 15
    )
)

humidity_recovery = (
    (humidity_before > 20)
    & (humidity_after > 20)
    & (
        (
            df["humidity"]
            - (
                df["humidity"].shift(1)
                + df["humidity"].shift(-1)
            ) / 2
        ).abs()
        > 25
    )
)

recovery_anomaly = (
    temperature_recovery
    | pressure_recovery
    | humidity_recovery
)

physical_anomaly = (
    (df["humidity"] < 0)
    | (df["humidity"] > 100)
    | (df["wind_speed"] < 0)
    | (df["wind_direction"] < 0)
    | (df["wind_direction"] > 360)
)

frozen_anomaly = (
    df["temperature"]
    .rolling(8)
    .std()
    .fillna(999)
    .eq(0)
)

multivariate_anomaly = (
    (
        (df["temperature"].abs() > 45)
        & (df["humidity"] < 40)
        & (df["pressure"] < 960)
    )
    |
    (
        (df["temperature"] > 40)
        & (df["humidity"] > 90)
    )
    |
    (
        (df["temperature"].abs() > 45)
        & df["pressure"].between(960, 1030)
        & df["humidity"].between(20, 80)
    )
)

strong_evidence = (
    physical_anomaly
    | frozen_anomaly
    | multivariate_anomaly
)

evidence_score = (
    physical_anomaly.astype(int) * 40
    + frozen_anomaly.astype(int) * 40
    + multivariate_anomaly.astype(int) * 40
    + recovery_anomaly.astype(int) * 20
)

evidence_score = evidence_score.clip(upper=100)

status = pd.Series(
    "NORMAL",
    index=df.index
)

status[
    recovery_anomaly
    & ~strong_evidence
] = "SUSPICIOUS"

status[
    strong_evidence
] = "ANOMALY"

df["temperature_recovery"] = temperature_recovery
df["pressure_recovery"] = pressure_recovery
df["humidity_recovery"] = humidity_recovery
df["recovery_anomaly"] = recovery_anomaly
df["physical_anomaly"] = physical_anomaly
df["frozen_anomaly"] = frozen_anomaly
df["multivariate_anomaly"] = multivariate_anomaly
df["evidence_score"] = evidence_score
df["status"] = status

false_positives = df[
    (df["status"] == "ANOMALY")
    & (df["is_anomaly"] == 0)
].copy()

print("=" * 80)
print("SKYGUARD AI FALSE POSITIVE ANALYSIS")
print("=" * 80)

print()
print(f"False positives: {len(false_positives)}")

print()
print("-" * 80)
print("FALSE POSITIVE READINGS")
print("-" * 80)

for index, row in false_positives.iterrows():
    print()
    print(f"INDEX: {index}")
    print(f"Timestamp: {row['timestamp']}")
    print(f"Temperature: {row['temperature']}")
    print(f"Pressure: {row['pressure']}")
    print(f"Humidity: {row['humidity']}")
    print(f"Wind speed: {row['wind_speed']}")
    print(f"Wind direction: {row['wind_direction']}")
    print(f"Evidence score: {row['evidence_score']}")
    print(f"Physical: {row['physical_anomaly']}")
    print(f"Frozen: {row['frozen_anomaly']}")
    print(f"Multivariate: {row['multivariate_anomaly']}")
    print(
        f"Temperature recovery: "
        f"{row['temperature_recovery']}"
    )
    print(
        f"Pressure recovery: "
        f"{row['pressure_recovery']}"
    )
    print(
        f"Humidity recovery: "
        f"{row['humidity_recovery']}"
    )

    print()
    print("NEIGHBORING READINGS")

    start = max(0, index - 2)
    end = min(len(df), index + 3)

    print(
        df.iloc[start:end][
            [
                "timestamp",
                "temperature",
                "pressure",
                "humidity",
                "wind_speed",
                "wind_direction",
                "is_anomaly",
                "event_id"
            ]
        ].to_string(index=True)
    )

print()
print("-" * 80)
print("FALSE POSITIVE RULE SUMMARY")
print("-" * 80)

physical_only = (
    false_positives["physical_anomaly"]
    & ~false_positives["multivariate_anomaly"]
    & ~false_positives["recovery_anomaly"]
)

print(
    f"Physical only: "
    f"{physical_only.sum()}"
)

print(
    f"Multivariate involved: "
    f"{false_positives['multivariate_anomaly'].sum()}"
)

print(
    f"Recovery involved: "
    f"{false_positives['recovery_anomaly'].sum()}"
)

print(
    f"Frozen involved: "
    f"{false_positives['frozen_anomaly'].sum()}"
)

print()
print("-" * 80)
print("FALSE POSITIVE EVENT IDs")
print("-" * 80)

event_counts = (
    false_positives["event_id"]
    .value_counts(dropna=False)
)

print(
    event_counts.to_string()
)

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)