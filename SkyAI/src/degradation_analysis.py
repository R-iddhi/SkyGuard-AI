import pandas as pd


# ==========================================
# SKYGUARD AI
# SENSOR DEGRADATION & MAINTENANCE RISK
# ==========================================


# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/aws_anomaly_dataset.csv"

df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# ==========================================
# 2. CALCULATE SENSOR CHANGES
# ==========================================

df["temperature_change"] = df["temperature"].diff()
df["pressure_change"] = df["pressure"].diff()
df["humidity_change"] = df["humidity"].diff()


# ==========================================
# 3. PHYSICAL VALIDATION
# ==========================================

df["physical_anomaly"] = (
    (df["humidity"] < 0) |
    (df["humidity"] > 100) |
    (df["wind_speed"] < 0) |
    (df["wind_direction"] < 0) |
    (df["wind_direction"] > 360)
)


# ==========================================
# 4. TEMPORAL VALIDATION
# ==========================================

df["temporal_anomaly"] = (
    (df["temperature_change"].abs() > 10) |
    (df["pressure_change"].abs() > 10) |
    (df["humidity_change"].abs() > 20)
)


# ==========================================
# 5. FROZEN SENSOR DETECTION
# ==========================================

df["temperature_same"] = (
    df["temperature"] == df["temperature"].shift(1)
)

temperature_group = (
    (~df["temperature_same"]).cumsum()
)

group_sizes = (
    df.groupby(temperature_group)["temperature"]
    .transform("size")
)

df["frozen_sensor"] = group_sizes >= 8


# ==========================================
# 6. MULTIVARIATE VALIDATION
# ==========================================

temperature_anomaly = (
    df["temperature_change"].abs() > 10
)

pressure_anomaly = (
    df["pressure_change"].abs() > 10
)

humidity_anomaly = (
    df["humidity_change"].abs() > 20
)

df["multivariate_anomaly"] = (
    (temperature_anomaly & humidity_anomaly) |
    (temperature_anomaly & pressure_anomaly) |
    (df["humidity"] > 100)
)


# ==========================================
# 7. OVERALL DEGRADATION SIGNAL
# ==========================================

df["degradation_signal"] = (
    df["physical_anomaly"] |
    df["temporal_anomaly"] |
    df["frozen_sensor"] |
    df["multivariate_anomaly"]
)


# ==========================================
# 8. DIVIDE DATA INTO PERIODS
# ==========================================

total = len(df)

historical_start = int(total * 0.40)
historical_end = int(total * 0.60)

recent_start = int(total * 0.80)

historical_df = df.iloc[historical_start:historical_end].copy()
recent_df = df.iloc[recent_start:].copy()


# ==========================================
# 9. CALCULATE ANOMALY RATES
# ==========================================

historical_anomalies = int(
    historical_df["degradation_signal"].sum()
)

recent_anomalies = int(
    recent_df["degradation_signal"].sum()
)

historical_rate = (
    historical_anomalies /
    len(historical_df)
) * 100

recent_rate = (
    recent_anomalies /
    len(recent_df)
) * 100


# ==========================================
# 10. DEGRADATION CHANGE
# ==========================================

rate_change = recent_rate - historical_rate

if historical_rate > 0:

    percentage_increase = (
        rate_change / historical_rate
    ) * 100

else:

    percentage_increase = 0


# ==========================================
# 11. COMPONENT-SPECIFIC RECENT RATES
# ==========================================

recent_physical_rate = (
    recent_df["physical_anomaly"].mean() * 100
)

recent_temporal_rate = (
    recent_df["temporal_anomaly"].mean() * 100
)

recent_frozen_rate = (
    recent_df["frozen_sensor"].mean() * 100
)

recent_multivariate_rate = (
    recent_df["multivariate_anomaly"].mean() * 100
)


# ==========================================
# 12. CALCULATE MAINTENANCE RISK SCORE
# ==========================================

maintenance_risk_score = 0


# ------------------------------------------
# Recent anomaly activity
# ------------------------------------------

if recent_rate >= 5:

    maintenance_risk_score += 30

elif recent_rate >= 2:

    maintenance_risk_score += 20

elif recent_rate >= 1:

    maintenance_risk_score += 10


# ------------------------------------------
# Degradation trend
# ------------------------------------------

if percentage_increase >= 100:

    maintenance_risk_score += 25

elif percentage_increase >= 50:

    maintenance_risk_score += 20

elif percentage_increase >= 25:

    maintenance_risk_score += 10


# ------------------------------------------
# Frozen sensor behavior
# ------------------------------------------

if recent_frozen_rate >= 1:

    maintenance_risk_score += 20

elif recent_frozen_rate > 0:

    maintenance_risk_score += 10


# ------------------------------------------
# Physical failures
# ------------------------------------------

if recent_physical_rate >= 1:

    maintenance_risk_score += 15

elif recent_physical_rate > 0:

    maintenance_risk_score += 8


# ------------------------------------------
# Multivariate inconsistencies
# ------------------------------------------

if recent_multivariate_rate >= 1:

    maintenance_risk_score += 15

elif recent_multivariate_rate > 0:

    maintenance_risk_score += 8


# ------------------------------------------
# Temporal instability
# ------------------------------------------

if recent_temporal_rate >= 5:

    maintenance_risk_score += 10

elif recent_temporal_rate >= 2:

    maintenance_risk_score += 5


# Keep score inside 0-100
maintenance_risk_score = min(
    100,
    maintenance_risk_score
)


# ==========================================
# 13. DETERMINE DEGRADATION TREND
# ==========================================

if percentage_increase >= 50:

    degradation_status = "Increasing"

elif percentage_increase <= -30:

    degradation_status = "Improving"

else:

    degradation_status = "Stable"


# ==========================================
# 14. DETERMINE MAINTENANCE RISK LEVEL
# ==========================================

if maintenance_risk_score >= 70:

    maintenance_risk = "High"

elif maintenance_risk_score >= 40:

    maintenance_risk = "Medium"

else:

    maintenance_risk = "Low"


# ==========================================
# 15. RECOMMENDATION
# ==========================================

if maintenance_risk == "High":

    recommendation = (
        "Immediate sensor inspection and "
        "preventive maintenance recommended."
    )

elif maintenance_risk == "Medium":

    recommendation = (
        "Monitor sensor closely and schedule "
        "preventive maintenance."
    )

else:

    recommendation = (
        "Sensor condition is stable. "
        "No immediate maintenance required."
    )


# ==========================================
# 16. MAINTENANCE RISK FACTORS
# ==========================================

risk_factors = []

if recent_rate >= 1:

    risk_factors.append(
        "Elevated recent anomaly activity"
    )

if percentage_increase >= 25:

    risk_factors.append(
        "Increasing anomaly trend"
    )

if recent_frozen_rate > 0:

    risk_factors.append(
        "Frozen sensor behavior detected"
    )

if recent_physical_rate > 0:

    risk_factors.append(
        "Physical/range violations detected"
    )

if recent_multivariate_rate > 0:

    risk_factors.append(
        "Multivariate inconsistencies detected"
    )

if recent_temporal_rate >= 2:

    risk_factors.append(
        "Frequent temporal changes detected"
    )

if not risk_factors:

    risk_factors.append(
        "No significant maintenance risk factors detected"
    )


# ==========================================
# 17. DISPLAY RESULTS
# ==========================================

print("=" * 55)
print("SKYGUARD AI")
print("SENSOR DEGRADATION & MAINTENANCE RISK ANALYSIS")
print("=" * 55)

print("\nTotal observations:")
print(total)

print("\nHistorical period:")
print(
    historical_df["timestamp"].iloc[0],
    "to",
    historical_df["timestamp"].iloc[-1]
)

print("\nRecent period:")
print(
    recent_df["timestamp"].iloc[0],
    "to",
    recent_df["timestamp"].iloc[-1]
)

print("\nHistorical anomaly observations:")
print(historical_anomalies)

print("\nRecent anomaly observations:")
print(recent_anomalies)

print("\nHistorical anomaly rate:")
print(round(historical_rate, 3), "%")

print("\nRecent anomaly rate:")
print(round(recent_rate, 3), "%")

print("\nChange in anomaly rate:")
print(round(rate_change, 3), "% points")

print("\nRelative change:")
print(round(percentage_increase, 2), "%")

print("\nRecent physical anomaly rate:")
print(round(recent_physical_rate, 3), "%")

print("\nRecent temporal anomaly rate:")
print(round(recent_temporal_rate, 3), "%")

print("\nRecent frozen sensor rate:")
print(round(recent_frozen_rate, 3), "%")

print("\nRecent multivariate anomaly rate:")
print(round(recent_multivariate_rate, 3), "%")

print("\nDegradation trend:")
print(degradation_status)

print("\nMaintenance Risk Score:")
print(
    f"{maintenance_risk_score}/100"
)

print("\nMaintenance Risk:")
print(maintenance_risk)

print("\nRisk Factors:")

for factor in risk_factors:

    print(" -", factor)

print("\nRecommendation:")
print(recommendation)

print("\n" + "=" * 55)
print(
    "Note: Maintenance risk is an analytical indicator, "
    "not a guaranteed future failure prediction."
)
print("=" * 55)