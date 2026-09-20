from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


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

print("=" * 80)
print("SKYGUARD AI CLEAN BENCHMARK EVALUATION")
print("=" * 80)

print(f"Dataset: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


print(
    f"Total observations: {len(df)}"
)


# =========================================================
# ACTUAL GROUND TRUTH
# =========================================================

actual_anomaly = (
    df["is_anomaly"] == 1
)

actual_normal = (
    df["is_anomaly"] == 0
)

print(
    f"Actual anomaly observations: "
    f"{actual_anomaly.sum()}"
)

print(
    f"Actual normal observations: "
    f"{actual_normal.sum()}"
)


# =========================================================
# PHYSICAL VALIDATION
# =========================================================

physical_anomaly = (
    (df["humidity"] < 0)
    |
    (df["humidity"] > 100)
)


# =========================================================
# FROZEN SENSOR DETECTION
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

frozen_anomaly = (
    group_sizes >= 8
)


# =========================================================
# PREVIOUS / NEXT OBSERVATIONS
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
# NEIGHBOR AVERAGES
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
# RECOVERY DETECTION
#
# Detects a reading that differs substantially from:
#   1. previous reading
#   2. next reading
#   3. local neighborhood
#
# This helps identify isolated sensor/data spikes.
# =========================================================

temperature_recovery = (
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


pressure_recovery = (
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


humidity_recovery = (
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
# Temperature and pressure recovery can independently
# indicate a suspicious observation.
#
# Humidity recovery alone is NOT sufficient because
# humidity can naturally change significantly.
#
# Humidity recovery becomes suspicious when it occurs
# together with temperature or pressure recovery.
# =========================================================

multi_recovery = (
    temperature_recovery
    |
    pressure_recovery
    |
    (
        humidity_recovery
        &
        (
            temperature_recovery
            |
            pressure_recovery
        )
    )
)

recovery_anomaly = (
    multi_recovery
)


# =========================================================
# MULTIVARIATE VALIDATION
# =========================================================

multivariate_anomaly = (
    (
        (
            abs(df["temperature"])
            > 45
        )
        &
        (
            df["humidity"]
            < 40
        )
        &
        (
            df["pressure"]
            < 960
        )
    )
    |
    (
        (
            df["temperature"]
            > 40
        )
        &
        (
            df["humidity"]
            > 90
        )
    )
    |
    (
        (
            abs(df["temperature"])
            > 45
        )
        &
        (
            df["pressure"]
            >= 960
        )
        &
        (
            df["pressure"]
            <= 1030
        )
        &
        (
            df["humidity"]
            >= 20
        )
        &
        (
            df["humidity"]
            <= 80
        )
    )
)


# =========================================================
# STRONG ANOMALY
#
# Strong direct evidence:
#   physical
#   frozen
#   multivariate
#
# These are classified directly as ANOMALY.
# =========================================================

strong_anomaly = (
    physical_anomaly
    |
    frozen_anomaly
    |
    multivariate_anomaly
)


# =========================================================
# STATUS
#
# Strong evidence -> ANOMALY
#
# Strict temporal recovery -> SUSPICIOUS
#
# Otherwise -> NORMAL
# =========================================================

df["predicted_status"] = np.where(
    strong_anomaly,
    "ANOMALY",
    np.where(
        recovery_anomaly,
        "SUSPICIOUS",
        "NORMAL"
    )
)


# =========================================================
# POINT-LEVEL PREDICTION
#
# Both ANOMALY and SUSPICIOUS are considered positive
# detections for point-level evaluation.
# =========================================================

predicted_positive = (
    df["predicted_status"]
    != "NORMAL"
)

actual_positive = (
    df["is_anomaly"] == 1
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

true_positive = (
    predicted_positive
    & actual_positive
).sum()

false_positive = (
    predicted_positive
    & (~actual_positive)
).sum()

false_negative = (
    (~predicted_positive)
    & actual_positive
).sum()

true_negative = (
    (~predicted_positive)
    & (~actual_positive)
).sum()


precision = precision_score(
    actual_positive,
    predicted_positive,
    zero_division=0
)

recall = recall_score(
    actual_positive,
    predicted_positive,
    zero_division=0
)

f1 = f1_score(
    actual_positive,
    predicted_positive,
    zero_division=0
)


# =========================================================
# STATUS DISTRIBUTION
# =========================================================

print()
print("=" * 80)
print("STATUS DISTRIBUTION")
print("=" * 80)

print(
    df["predicted_status"]
    .value_counts()
)


# =========================================================
# POINT LEVEL RESULTS
# =========================================================

print()
print("=" * 80)
print("POINT-LEVEL EVALUATION")
print("=" * 80)

print(
    f"True positives:  {true_positive}"
)

print(
    f"False positives: {false_positive}"
)

print(
    f"False negatives: {false_negative}"
)

print(
    f"True negatives:  {true_negative}"
)

print()

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 Score:  {f1:.4f}"
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

matrix = confusion_matrix(
    actual_positive,
    predicted_positive
)

print()
print("Confusion Matrix:")
print(matrix)


# =========================================================
# EVENT-LEVEL EVALUATION
# =========================================================

print()
print("=" * 80)
print("ANOMALY EVENT EVALUATION")
print("=" * 80)


anomaly_events = (
    df[
        df["event_id"].notna()
        &
        (df["event_id"] != "")
    ]["event_id"]
    .unique()
)


events_detected = 0


for event_id in anomaly_events:

    event_data = df[
        df["event_id"] == event_id
    ]

    actual_rows = len(
        event_data
    )

    anomaly_rows = (
        event_data["predicted_status"]
        == "ANOMALY"
    ).sum()

    suspicious_rows = (
        event_data["predicted_status"]
        == "SUSPICIOUS"
    ).sum()

    detected = (
        anomaly_rows > 0
        or suspicious_rows > 0
    )

    if detected:
        events_detected += 1

    print(
        f"{event_id}: "
        f"{actual_rows} actual rows | "
        f"{anomaly_rows} ANOMALY | "
        f"{suspicious_rows} SUSPICIOUS | "
        f"{'DETECTED' if detected else 'MISSED'}"
    )


total_events = len(
    anomaly_events
)

event_recall = (
    events_detected / total_events
    if total_events > 0
    else 0
)


print()
print(
    f"Events detected: "
    f"{events_detected}/{total_events}"
)

print(
    f"Event recall: "
    f"{event_recall:.4f}"
)


# =========================================================
# EVIDENCE COUNTS
# =========================================================

print()
print("=" * 80)
print("DETECTOR EVIDENCE COUNTS")
print("=" * 80)

print(
    f"Physical anomalies:      "
    f"{physical_anomaly.sum()}"
)

print(
    f"Recovery anomalies:      "
    f"{recovery_anomaly.sum()}"
)

print(
    f"Frozen anomalies:        "
    f"{frozen_anomaly.sum()}"
)

print(
    f"Multivariate anomalies:  "
    f"{multivariate_anomaly.sum()}"
)


# =========================================================
# FINAL MESSAGE
# =========================================================

print()
print("=" * 80)
print("CLEAN BENCHMARK EVALUATION COMPLETE")
print("=" * 80)