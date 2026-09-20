from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEASONAL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "seasonal_patterns.csv"
)


# ---------------------------------------------------------
# Features used by SkyGuard AI
# ---------------------------------------------------------

FEATURES = [
    "temperature",
    "pressure",
    "humidity"
]


FEATURE_NAMES = {
    "temperature": "Temperature",
    "pressure": "Atmospheric pressure",
    "humidity": "Relative humidity"
}


# ---------------------------------------------------------
# Load seasonal baseline
# ---------------------------------------------------------

print("Loading seasonal baseline...")

seasonal_data = pd.read_csv(
    SEASONAL_DATA_PATH
)

print(
    f"Seasonal baseline loaded: "
    f"{len(seasonal_data)} monthly profiles"
)


# ---------------------------------------------------------
# Robust seasonal anomaly detector
# ---------------------------------------------------------

def detect_seasonal_anomaly(
    timestamp,
    temperature,
    pressure,
    humidity
):
    """
    Compare a new weather observation against
    the expected seasonal pattern for its month.
    """

    timestamp = pd.to_datetime(timestamp)

    month = timestamp.month

    month_data = seasonal_data[
        seasonal_data["month"] == month
    ]

    if month_data.empty:
        return {
            "seasonal_anomaly": False,
            "seasonal_score": 0.0,
            "month": month,
            "details": []
        }

    month_data = month_data.iloc[0]

    values = {
        "temperature": float(temperature),
        "pressure": float(pressure),
        "humidity": float(humidity)
    }

    details = []

    anomaly_features = []

    maximum_score = 0.0

    for feature in FEATURES:

        value = values[feature]

        median = float(
            month_data[f"{feature}_median"]
        )

        mad = float(
            month_data[f"{feature}_mad"]
        )

        p05 = float(
            month_data[f"{feature}_p05"]
        )

        p95 = float(
            month_data[f"{feature}_p95"]
        )

        # -------------------------------------------------
        # Robust z-score
        #
        # 1.4826 converts MAD into a scale comparable
        # to standard deviation for approximately
        # normally distributed data.
        # -------------------------------------------------

        if mad > 0:

            robust_z = abs(
                value - median
            ) / (1.4826 * mad)

        else:

            robust_z = 0.0

        # -------------------------------------------------
        # Determine whether value is outside the normal
        # seasonal percentile range.
        # -------------------------------------------------

        outside_percentile_range = (
            value < p05
            or value > p95
        )

        # -------------------------------------------------
        # Seasonal anomaly decision
        #
        # We require both:
        #
        # 1. Outside the 5th-95th percentile range
        # 2. Robust z-score >= 3
        #
        # This prevents normal seasonal variation from
        # being flagged too aggressively.
        # -------------------------------------------------

        feature_anomaly = (
            outside_percentile_range
            and robust_z >= 3
        )

        if feature_anomaly:

            anomaly_features.append(
                feature
            )

        maximum_score = max(
            maximum_score,
            robust_z
        )

        details.append({
            "feature": feature,
            "feature_name": FEATURE_NAMES[feature],
            "value": round(value, 4),
            "seasonal_median": round(median, 4),
            "seasonal_mad": round(mad, 4),
            "seasonal_p05": round(p05, 4),
            "seasonal_p95": round(p95, 4),
            "robust_z_score": round(
                robust_z,
                4
            ),
            "outside_normal_range":
                outside_percentile_range,
            "seasonal_anomaly":
                feature_anomaly
        })

    # -----------------------------------------------------
    # Overall seasonal decision
    # -----------------------------------------------------

    seasonal_anomaly = (
        len(anomaly_features) > 0
    )

    # -----------------------------------------------------
    # Create human-readable explanation
    # -----------------------------------------------------

    if seasonal_anomaly:

        feature_names = [
            FEATURE_NAMES[feature]
            for feature in anomaly_features
        ]

        summary = (
            "Seasonal deviation detected in: "
            + ", ".join(feature_names)
        )

    else:

        summary = (
            "Reading is consistent with the "
            "expected seasonal pattern."
        )

    return {
        "seasonal_anomaly": seasonal_anomaly,
        "seasonal_score": round(
            maximum_score,
            4
        ),
        "month": month,
        "anomalous_features": anomaly_features,
        "summary": summary,
        "details": details
    }


# ---------------------------------------------------------
# Test cases
# ---------------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SEASONAL DETECTOR TEST")
    print("=" * 70)

    # -----------------------------------------------------
    # Test 1: Normal February reading
    # -----------------------------------------------------

    normal_reading = detect_seasonal_anomaly(
        timestamp="2016-02-15T12:00:00",
        temperature=-3.0,
        pressure=967.0,
        humidity=50.0
    )

    print()
    print("TEST 1: Normal February reading")
    print("-" * 70)

    print(
        f"Seasonal anomaly: "
        f"{normal_reading['seasonal_anomaly']}"
    )

    print(
        f"Seasonal score: "
        f"{normal_reading['seasonal_score']}"
    )

    print(
        f"Summary: "
        f"{normal_reading['summary']}"
    )

    # -----------------------------------------------------
    # Test 2: Extreme February reading
    # -----------------------------------------------------

    extreme_reading = detect_seasonal_anomaly(
        timestamp="2016-02-15T12:00:00",
        temperature=60.0,
        pressure=967.0,
        humidity=50.0
    )

    print()
    print("TEST 2: Extreme February reading")
    print("-" * 70)

    print(
        f"Seasonal anomaly: "
        f"{extreme_reading['seasonal_anomaly']}"
    )

    print(
        f"Seasonal score: "
        f"{extreme_reading['seasonal_score']}"
    )

    print(
        f"Summary: "
        f"{extreme_reading['summary']}"
    )

    # -----------------------------------------------------
    # Test 3: Normal September reading
    # -----------------------------------------------------

    normal_september = detect_seasonal_anomaly(
        timestamp="2016-09-15T12:00:00",
        temperature=-17.0,
        pressure=966.0,
        humidity=40.0
    )

    print()
    print("TEST 3: Normal September reading")
    print("-" * 70)

    print(
        f"Seasonal anomaly: "
        f"{normal_september['seasonal_anomaly']}"
    )

    print(
        f"Seasonal score: "
        f"{normal_september['seasonal_score']}"
    )

    print(
        f"Summary: "
        f"{normal_september['summary']}"
    )

    # -----------------------------------------------------
    # Test 4: Injected humidity anomaly
    # -----------------------------------------------------

    humidity_anomaly = detect_seasonal_anomaly(
        timestamp="2016-09-05T19:00:00",
        temperature=-10.0,
        pressure=970.0,
        humidity=135.0
    )

    print()
    print("TEST 4: Extreme humidity reading")
    print("-" * 70)

    print(
        f"Seasonal anomaly: "
        f"{humidity_anomaly['seasonal_anomaly']}"
    )

    print(
        f"Seasonal score: "
        f"{humidity_anomaly['seasonal_score']}"
    )

    print(
        f"Summary: "
        f"{humidity_anomaly['summary']}"
    )

    # -----------------------------------------------------
    # Detailed output
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("DETAILED FEATURE ANALYSIS — EXTREME HUMIDITY TEST")
    print("=" * 70)

    for detail in humidity_anomaly["details"]:

        print()
        print(
            f"Feature: "
            f"{detail['feature_name']}"
        )

        print(
            f"Value: "
            f"{detail['value']}"
        )

        print(
            f"Seasonal median: "
            f"{detail['seasonal_median']}"
        )

        print(
            f"5th percentile: "
            f"{detail['seasonal_p05']}"
        )

        print(
            f"95th percentile: "
            f"{detail['seasonal_p95']}"
        )

        print(
            f"Robust Z-score: "
            f"{detail['robust_z_score']}"
        )

        print(
            f"Outside normal range: "
            f"{detail['outside_normal_range']}"
        )

        print(
            f"Seasonal anomaly: "
            f"{detail['seasonal_anomaly']}"
        )

    print()
    print("=" * 70)
    print("SEASONAL DETECTOR TEST COMPLETED")
    print("=" * 70)