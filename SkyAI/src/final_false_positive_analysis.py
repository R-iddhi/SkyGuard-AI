from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "aws_clean_final_benchmark.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = (
    df.sort_values("timestamp")
    .reset_index(drop=True)
)


# =========================================================
# PHYSICAL VALIDATION
# =========================================================

df["physical"] = (
    (df["humidity"] < 0)
    |
    (df["humidity"] > 100)
)


# =========================================================
# FROZEN SENSOR
# =========================================================

temperature_same = (
    df["temperature"]
    == df["temperature"].shift(1)
)

group_id = (
    (~temperature_same).cumsum()
)

df["temperature_group"] = group_id

group_sizes = (
    df.groupby(
        "temperature_group"
    )["temperature"]
    .transform("size")
)

df["frozen"] = (
    group_sizes >= 8
)


# =========================================================
# PREVIOUS / NEXT VALUES
# =========================================================

temp_before = (
    df["temperature"].shift(1)
)

temp_after = (
    df["temperature"].shift(-1)
)

pressure_before = (
    df["pressure"].shift(1)
)

pressure_after = (
    df["pressure"].shift(-1)
)

humidity_before = (
    df["humidity"].shift(1)
)

humidity_after = (
    df["humidity"].shift(-1)
)


# =========================================================
# LOCAL NEIGHBORHOOD
# =========================================================

temp_neighbor_avg = (
    df["temperature"]
    .rolling(
        window=5,
        center=True,
        min_periods=3
    )
    .mean()
)

pressure_neighbor_avg = (
    df["pressure"]
    .rolling(
        window=5,
        center=True,
        min_periods=3
    )
    .mean()
)

humidity_neighbor_avg = (
    df["humidity"]
    .rolling(
        window=5,
        center=True,
        min_periods=3
    )
    .mean()
)


# =========================================================
# TEMPERATURE RECOVERY
# =========================================================

df["temperature_recovery"] = (
    (
        abs(
            df["temperature"]
            - temp_before
        ) > 10
    )
    &
    (
        abs(
            df["temperature"]
            - temp_after
        ) > 10
    )
    &
    (
        abs(
            df["temperature"]
            - temp_neighbor_avg
        ) > 15
    )
)


# =========================================================
# PRESSURE RECOVERY
# =========================================================

df["pressure_recovery"] = (
    (
        abs(
            df["pressure"]
            - pressure_before
        ) > 10
    )
    &
    (
        abs(
            df["pressure"]
            - pressure_after
        ) > 10
    )
    &
    (
        abs(
            df["pressure"]
            - pressure_neighbor_avg
        ) > 15
    )
)


# =========================================================
# HUMIDITY RECOVERY
# =========================================================

df["humidity_recovery"] = (
    (
        abs(
            df["humidity"]
            - humidity_before
        ) > 20
    )
    &
    (
        abs(
            df["humidity"]
            - humidity_after
        ) > 20
    )
    &
    (
        abs(
            df["humidity"]
            - humidity_neighbor_avg
        ) > 25
    )
)


# =========================================================
# STRICT RECOVERY LOGIC
#
# Temperature recovery:
#     independently suspicious.
#
# Pressure recovery:
#     independently suspicious.
#
# Humidity recovery:
#     only suspicious when accompanied by
#     temperature or pressure recovery.
# =========================================================

df["recovery"] = (
    df["temperature_recovery"]
    |
    df["pressure_recovery"]
    |
    (
        df["humidity_recovery"]
        &
        (
            df["temperature_recovery"]
            |
            df["pressure_recovery"]
        )
    )
)


# =========================================================
# MULTIVARIATE VALIDATION
# =========================================================

df["multivariate"] = (
    (
        (
            abs(
                df["temperature"]
            ) > 45
        )
        &
        (
            df["humidity"] < 40
        )
        &
        (
            df["pressure"] < 960
        )
    )
    |
    (
        (
            df["temperature"] > 40
        )
        &
        (
            df["humidity"] > 90
        )
    )
    |
    (
        (
            abs(
                df["temperature"]
            ) > 45
        )
        &
        df["pressure"].between(
            960,
            1030
        )
        &
        df["humidity"].between(
            20,
            80
        )
    )
)


# =========================================================
# FINAL BENCHMARK DETECTION
# =========================================================

df["rule_anomaly"] = (
    df["physical"]
    |
    df["frozen"]
    |
    df["multivariate"]
    |
    df["recovery"]
)


# =========================================================
# FALSE POSITIVES
# =========================================================

false_positives = df[
    (
        df["rule_anomaly"]
    )
    &
    (
        df["is_anomaly"] == 0
    )
].copy()


# =========================================================
# OUTPUT
# =========================================================

print("=" * 80)
print(
    "FINAL BENCHMARK FALSE POSITIVE ANALYSIS"
)
print("=" * 80)

print(
    f"Dataset: {DATA_PATH}"
)

print(
    f"Total false positives: "
    f"{len(false_positives)}"
)


# =========================================================
# FALSE POSITIVES BY RULE
# =========================================================

print()
print(
    "FALSE POSITIVES BY RULE"
)
print("-" * 80)

print(
    f"Physical validation:       "
    f"{false_positives['physical'].sum()}"
)

print(
    f"Temperature recovery:      "
    f"{false_positives['temperature_recovery'].sum()}"
)

print(
    f"Pressure recovery:         "
    f"{false_positives['pressure_recovery'].sum()}"
)

print(
    f"Humidity recovery:         "
    f"{false_positives['humidity_recovery'].sum()}"
)

print(
    f"Frozen sensor:              "
    f"{false_positives['frozen'].sum()}"
)

print(
    f"Multivariate:              "
    f"{false_positives['multivariate'].sum()}"
)


# =========================================================
# RULE COUNT
# =========================================================

rule_columns = [
    "physical",
    "temperature_recovery",
    "pressure_recovery",
    "humidity_recovery",
    "frozen",
    "multivariate"
]

false_positives["rule_count"] = (
    false_positives[
        rule_columns
    ]
    .sum(axis=1)
)


print()
print(
    "FALSE POSITIVES BY NUMBER "
    "OF TRIGGERED RULES"
)
print("-" * 80)

print(
    false_positives[
        "rule_count"
    ]
    .value_counts()
    .sort_index()
)


# =========================================================
# FALSE POSITIVE EXAMPLES
# =========================================================

print()
print(
    "FALSE POSITIVE EXAMPLES"
)
print("-" * 80)

display_columns = [
    "timestamp",
    "temperature",
    "pressure",
    "humidity",
    "temperature_recovery",
    "pressure_recovery",
    "humidity_recovery",
    "frozen",
    "multivariate",
    "rule_count"
]

if false_positives.empty:

    print(
        "No false positives detected."
    )

else:

    print(
        false_positives[
            display_columns
        ]
        .to_string(index=False)
    )


# =========================================================
# COMPLETE
# =========================================================

print()
print("=" * 80)
print(
    "ANALYSIS COMPLETE"
)
print("=" * 80)