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

improved_anomaly = (
    physical_anomaly
    | recovery_anomaly
    | frozen_anomaly
    | multivariate_anomaly
)

df["improved_anomaly"] = improved_anomaly
df["recovery_anomaly"] = recovery_anomaly
df["physical_anomaly"] = physical_anomaly
df["frozen_anomaly"] = frozen_anomaly
df["multivariate_anomaly"] = multivariate_anomaly

false_positives = df[
    (df["improved_anomaly"])
    & (df["is_anomaly"] == 0)
].copy()

anomaly_rows = df[df["is_anomaly"] == 1].copy()

print("=" * 80)
print("SKYGUARD AI FALSE POSITIVE ANALYSIS")
print("=" * 80)

print()
print(f"Total false positives: {len(false_positives)}")
print(f"Total injected anomaly observations: {len(anomaly_rows)}")

print()
print("-" * 80)
print("FALSE POSITIVE DETAILS")
print("-" * 80)

for index, row in false_positives.iterrows():

    previous_row = df.loc[index - 1] if index > 0 else None
    next_row = df.loc[index + 1] if index < len(df) - 1 else None

    nearby_anomalies = anomaly_rows[
        (anomaly_rows.index >= index - 2)
        & (anomaly_rows.index <= index + 2)
    ]

    if len(nearby_anomalies) > 0:
        nearby_events = nearby_anomalies["event_id"].unique()
        nearby_text = ", ".join(
            str(event) for event in nearby_events
        )
    else:
        nearby_text = "None"

    sources = []

    if row["physical_anomaly"]:
        sources.append("Physical")

    if row["recovery_anomaly"]:
        sources.append("Recovery")

    if row["frozen_anomaly"]:
        sources.append("Frozen")

    if row["multivariate_anomaly"]:
        sources.append("Multivariate")

    source_text = ", ".join(sources)

    print()
    print(f"Row index: {index}")
    print(f"Timestamp: {row['timestamp']}")
    print(
        f"Temperature: {row['temperature']} | "
        f"Pressure: {row['pressure']} | "
        f"Humidity: {row['humidity']}"
    )
    print(f"Detection source: {source_text}")
    print(f"Nearby injected event: {nearby_text}")

    if previous_row is not None:
        print(
            f"Previous: "
            f"T={previous_row['temperature']}, "
            f"P={previous_row['pressure']}, "
            f"H={previous_row['humidity']}"
        )

    if next_row is not None:
        print(
            f"Next: "
            f"T={next_row['temperature']}, "
            f"P={next_row['pressure']}, "
            f"H={next_row['humidity']}"
        )

print()
print("-" * 80)
print("FALSE POSITIVES NEAR INJECTED EVENTS")
print("-" * 80)

near_event = 0
far_from_event = 0

for index in false_positives.index:

    nearby_anomalies = anomaly_rows[
        (anomaly_rows.index >= index - 2)
        & (anomaly_rows.index <= index + 2)
    ]

    if len(nearby_anomalies) > 0:
        near_event += 1
    else:
        far_from_event += 1

print(f"Within ±2 observations of an injected event: {near_event}")
print(f"More than 2 observations away: {far_from_event}")

print()
print("-" * 80)
print("FALSE POSITIVE SOURCE COUNTS")
print("-" * 80)

print(
    f"Recovery:      "
    f"{false_positives['recovery_anomaly'].sum()}"
)

print(
    f"Physical:      "
    f"{false_positives['physical_anomaly'].sum()}"
)

print(
    f"Frozen:        "
    f"{false_positives['frozen_anomaly'].sum()}"
)

print(
    f"Multivariate:  "
    f"{false_positives['multivariate_anomaly'].sum()}"
)

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)