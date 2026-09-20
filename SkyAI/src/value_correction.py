"""
SkyGuard AI
Sensor Value Correction Module

Purpose:
    Estimate a corrected sensor value when a reading is detected
    as anomalous.

Supported sensors:
    - Temperature
    - Atmospheric Pressure
    - Relative Humidity

Methods:
    Offline:
        Uses surrounding observations to estimate the expected value.

    Real-time:
        Uses previously observed values because future observations
        are not available.

Important:
    The original sensor value is never modified.
    The corrected value is only an estimated value.

    Confidence is an operational confidence score, NOT a calibrated
    probability.
"""

from pathlib import Path
import sys

import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "aws_anomaly_dataset.csv"


# ============================================================
# SUPPORTED FEATURES
# ============================================================

SUPPORTED_FEATURES = {
    "temperature": {
        "minimum": -80.0,
        "maximum": 70.0,
    },
    "pressure": {
        "minimum": 850.0,
        "maximum": 1100.0,
    },
    "humidity": {
        "minimum": 0.0,
        "maximum": 100.0,
    },
}


# ============================================================
# LOAD DATA
# ============================================================

def load_sensor_data():
    """
    Load the cleaned/anomaly dataset.

    Returns:
        pandas.DataFrame
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "timestamp",
        "temperature",
        "pressure",
        "humidity",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ============================================================
# VALIDATE FEATURE
# ============================================================

def validate_feature(feature):
    """
    Check whether the requested sensor feature is supported.
    """

    if feature not in SUPPORTED_FEATURES:
        raise ValueError(
            f"Unsupported feature '{feature}'. "
            f"Supported features: {list(SUPPORTED_FEATURES.keys())}"
        )


# ============================================================
# TEMPORAL MEDIAN
# ============================================================

def calculate_temporal_median(
    df,
    index,
    feature,
    window=5
):
    """
    Calculate a robust expected sensor value using nearby
    observations.

    The current anomalous value is excluded.

    Example:

        Previous values:
        10, 11, 12

        Current:
        60   <-- anomaly

        Next values:
        11, 12, 13

        Median of valid neighbors becomes the estimated value.

    Args:
        df:
            Sensor dataframe.

        index:
            Index of the anomalous observation.

        feature:
            Sensor feature.

        window:
            Number of observations considered on each side.

    Returns:
        float or None
    """

    validate_feature(feature)

    start = max(0, index - window)
    end = min(len(df), index + window + 1)

    values = []

    for i in range(start, end):

        # Never use the anomalous/current observation.
        if i == index:
            continue

        value = df.iloc[i][feature]

        if pd.isna(value):
            continue

        value = float(value)

        # Only use physically valid values.
        limits = SUPPORTED_FEATURES[feature]

        if value < limits["minimum"]:
            continue

        if value > limits["maximum"]:
            continue

        values.append(value)

    if not values:
        return None

    return float(np.median(values))


# ============================================================
# OFFLINE CORRECTION
# ============================================================

def correct_sensor_value(
    df,
    index,
    feature,
    window=5
):
    """
    Estimate a corrected sensor value using surrounding
    observations.

    This method is intended for offline/historical analysis
    where observations before and after the anomaly are available.

    Returns:
        Dictionary containing:

        original_value
        corrected_value
        correction_applied
        method
        confidence
        explanation
    """

    validate_feature(feature)

    if index < 0 or index >= len(df):
        raise IndexError(
            f"Index {index} is outside dataframe range."
        )

    original_value = df.iloc[index][feature]

    if pd.isna(original_value):
        return {
            "feature": feature,
            "original_value": None,
            "corrected_value": None,
            "correction_applied": False,
            "method": "unavailable",
            "confidence": 0.0,
            "explanation": (
                "The original sensor value is missing, so "
                "a corrected value could not be estimated."
            ),
        }

    original_value = float(original_value)

    corrected_value = calculate_temporal_median(
        df=df,
        index=index,
        feature=feature,
        window=window
    )

    if corrected_value is None:
        return {
            "feature": feature,
            "original_value": original_value,
            "corrected_value": None,
            "correction_applied": False,
            "method": "unavailable",
            "confidence": 0.0,
            "explanation": (
                "Insufficient valid neighboring observations "
                "were available to estimate a corrected value."
            ),
        }

    # --------------------------------------------------------
    # Confidence calculation
    # --------------------------------------------------------

    start = max(0, index - window)
    end = min(len(df), index + window + 1)

    valid_neighbors = []

    limits = SUPPORTED_FEATURES[feature]

    for i in range(start, end):

        if i == index:
            continue

        value = df.iloc[i][feature]

        if pd.isna(value):
            continue

        value = float(value)

        if (
            limits["minimum"]
            <= value
            <= limits["maximum"]
        ):
            valid_neighbors.append(value)

    neighbor_count = len(valid_neighbors)

    if neighbor_count >= 8:
        confidence = 0.95

    elif neighbor_count >= 5:
        confidence = 0.90

    elif neighbor_count >= 3:
        confidence = 0.80

    elif neighbor_count >= 2:
        confidence = 0.70

    else:
        confidence = 0.50

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    explanation = (
        f"The estimated {feature} value was calculated using "
        f"a robust temporal median from "
        f"{neighbor_count} valid neighboring observations."
    )

    return {
        "feature": feature,
        "original_value": original_value,
        "corrected_value": round(corrected_value, 3),
        "correction_applied": True,
        "method": "temporal_median",
        "confidence": round(confidence, 2),
        "neighbor_count": neighbor_count,
        "explanation": explanation,
    }


# ============================================================
# REAL-TIME CORRECTION
# ============================================================

def correct_realtime_value(
    previous_values,
    feature,
    current_value,
    window=5
):
    """
    Estimate a corrected sensor value for a real-time reading.

    Because future observations are unavailable in real time,
    only previously observed valid readings are used.

    Args:
        previous_values:
            List of previous sensor values.

        feature:
            temperature / pressure / humidity.

        current_value:
            Current sensor reading.

        window:
            Maximum number of previous readings to use.

    Returns:
        Dictionary containing correction information.
    """

    validate_feature(feature)

    if current_value is None:

        return {
            "feature": feature,
            "original_value": None,
            "corrected_value": None,
            "correction_applied": False,
            "method": "unavailable",
            "confidence": 0.0,
            "explanation": (
                "Current sensor value is missing."
            ),
        }

    current_value = float(current_value)

    valid_values = []

    limits = SUPPORTED_FEATURES[feature]

    for value in previous_values[-window:]:

        if value is None:
            continue

        value = float(value)

        if (
            limits["minimum"]
            <= value
            <= limits["maximum"]
        ):
            valid_values.append(value)

    if not valid_values:

        return {
            "feature": feature,
            "original_value": current_value,
            "corrected_value": None,
            "correction_applied": False,
            "method": "unavailable",
            "confidence": 0.0,
            "explanation": (
                "No valid previous observations are available "
                "for real-time correction."
            ),
        }

    corrected_value = float(
        np.median(valid_values)
    )

    neighbor_count = len(valid_values)

    if neighbor_count >= 5:
        confidence = 0.90

    elif neighbor_count >= 3:
        confidence = 0.80

    elif neighbor_count >= 2:
        confidence = 0.70

    else:
        confidence = 0.50

    return {
        "feature": feature,
        "original_value": current_value,
        "corrected_value": round(corrected_value, 3),
        "correction_applied": True,
        "method": "previous_values_median",
        "confidence": round(confidence, 2),
        "neighbor_count": neighbor_count,
        "explanation": (
            f"The estimated {feature} value was calculated "
            f"using the median of {neighbor_count} valid "
            f"previous sensor observations."
        ),
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

def main():

    print()
    print("=" * 60)
    print("SKYGUARD AI - SENSOR VALUE CORRECTION")
    print("=" * 60)

    try:
        df = load_sensor_data()

    except Exception as error:

        print()
        print("ERROR:")
        print(error)
        print()

        sys.exit(1)

    # --------------------------------------------------------
    # Find a known temperature anomaly
    # --------------------------------------------------------

    anomaly_rows = df[
        df["temperature"] > 50
    ]

    if anomaly_rows.empty:

        print()
        print(
            "No temperature anomaly above 50°C "
            "was found in the dataset."
        )

        return

    index = anomaly_rows.index[0]

    result = correct_sensor_value(
        df=df,
        index=index,
        feature="temperature",
        window=5
    )

    print()
    print(f"Feature:              {result['feature']}")
    print(
        f"Original value:       "
        f"{result['original_value']}"
    )
    print(
        f"Corrected value:      "
        f"{result['corrected_value']}"
    )
    print(
        f"Correction applied:   "
        f"{result['correction_applied']}"
    )
    print(
        f"Method:               "
        f"{result['method']}"
    )
    print(
        f"Confidence:           "
        f"{result['confidence']}"
    )
    print(
        f"Neighbor count:       "
        f"{result.get('neighbor_count', 0)}"
    )
    print(
        f"Explanation:          "
        f"{result['explanation']}"
    )

    print()
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()