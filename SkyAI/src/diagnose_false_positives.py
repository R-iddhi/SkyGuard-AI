from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_controlled_benchmark.csv"

df = pd.read_csv(DATA_PATH)

previous = df[
    [
        "temperature",
        "pressure",
        "humidity",
        "wind_speed",
        "wind_direction"
    ]
].shift(1)

temperature_change = (
    df["temperature"] - previous["temperature"]
).abs()

pressure_change = (
    df["pressure"] - previous["pressure"]
).abs()

humidity_change = (
    df["humidity"] - previous["humidity"]
).abs()

physical_anomaly = (
    (df["humidity"] < 0) |
    (df["humidity"] > 100) |
    (df["wind_speed"] < 0) |
    (df["wind_direction"] < 0) |
    (df["wind_direction"] > 360)
)

temporal_temperature = temperature_change > 10
temporal_pressure = pressure_change > 10
temporal_humidity = humidity_change > 20

temporal_anomaly = (
    temporal_temperature |
    temporal_pressure |
    temporal_humidity
)

frozen_anomaly = (
    df["temperature"].rolling(8).std().fillna(999) == 0
)

multivariate_anomaly = (
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
        (df["pressure"].between(960, 1030)) &
        (df["humidity"].between(20, 80))
    )
)

rule_prediction = (
    physical_anomaly |
    temporal_anomaly |
    frozen_anomaly |
    multivariate_anomaly
)

false_positives = df[
    (df["is_anomaly"] == 0) &
    rule_prediction
].copy()

false_positives["physical"] = physical_anomaly.loc[
    false_positives.index
].values

false_positives["temp_change"] = temporal_temperature.loc[
    false_positives.index
].values

false_positives["pressure_change"] = temporal_pressure.loc[
    false_positives.index
].values

false_positives["humidity_change"] = temporal_humidity.loc[
    false_positives.index
].values

false_positives["frozen"] = frozen_anomaly.loc[
    false_positives.index
].values

false_positives["multivariate"] = multivariate_anomaly.loc[
    false_positives.index
].values

print("FALSE POSITIVE DIAGNOSTIC")
print()
print(f"Total false positives: {len(false_positives)}")
print()

print("=" * 60)
print("FALSE POSITIVES BY RULE")
print("=" * 60)

print()
print(
    f"Physical validation: "
    f"{int(false_positives['physical'].sum())}"
)

print(
    f"Temperature sudden change: "
    f"{int(false_positives['temp_change'].sum())}"
)

print(
    f"Pressure sudden change: "
    f"{int(false_positives['pressure_change'].sum())}"
)

print(
    f"Humidity sudden change: "
    f"{int(false_positives['humidity_change'].sum())}"
)

print(
    f"Frozen sensor: "
    f"{int(false_positives['frozen'].sum())}"
)

print(
    f"Multivariate inconsistency: "
    f"{int(false_positives['multivariate'].sum())}"
)

print()
print("=" * 60)
print("FALSE POSITIVES WITH MULTIPLE RULES")
print("=" * 60)

rule_columns = [
    "physical",
    "temp_change",
    "pressure_change",
    "humidity_change",
    "frozen",
    "multivariate"
]

false_positives["rules_triggered"] = (
    false_positives[rule_columns].sum(axis=1)
)

print(
    false_positives["rules_triggered"].value_counts()
    .sort_index()
)

print()
print("=" * 60)
print("TOP FALSE POSITIVE EXAMPLES")
print("=" * 60)

columns = [
    "timestamp",
    "temperature",
    "pressure",
    "humidity",
    "wind_speed",
    "wind_direction",
    "physical",
    "temp_change",
    "pressure_change",
    "humidity_change",
    "frozen",
    "multivariate"
]

print(
    false_positives[columns]
    .head(20)
    .to_string(index=False)
)