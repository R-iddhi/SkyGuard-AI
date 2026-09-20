import pandas as pd

# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/aws_anomaly_dataset.csv"

df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# ==========================================
# 2. CHECK WHETHER TEMPERATURE IS UNCHANGED
# ==========================================

df["temperature_same"] = (
    df["temperature"] == df["temperature"].shift(1)
)


# ==========================================
# 3. IDENTIFY CONTINUOUS GROUPS
# ==========================================

group_id = (
    (~df["temperature_same"]).cumsum()
)

df["temperature_group"] = group_id


# ==========================================
# 4. CALCULATE LENGTH OF EACH GROUP
# ==========================================

group_sizes = (
    df.groupby("temperature_group")["temperature"]
    .transform("size")
)


# ==========================================
# 5. DEFINE FROZEN SENSOR THRESHOLD
# ==========================================

frozen_threshold = 8


# ==========================================
# 6. DETECT COMPLETE FROZEN SEQUENCES
# ==========================================

df["frozen_temperature"] = (
    group_sizes >= frozen_threshold
)


# ==========================================
# 7. DISPLAY RESULTS
# ==========================================

print("FROZEN TEMPERATURE SENSOR DETECTION")

print("\nTotal observations:")
print(len(df))

print("\nFrozen temperature observations:")
print(df["frozen_temperature"].sum())


# ==========================================
# 8. DISPLAY DETECTED OBSERVATIONS
# ==========================================

print("\n\nDETECTED FROZEN TEMPERATURE OBSERVATIONS")

frozen = df[
    df["frozen_temperature"] == True
]

print(
    frozen[
        [
            "timestamp",
            "temperature",
            "pressure",
            "humidity",
            "wind_speed",
            "is_anomaly",
            "frozen_temperature"
        ]
    ].to_string(index=False)
)