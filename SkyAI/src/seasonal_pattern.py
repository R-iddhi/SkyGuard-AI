from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "aws_clean_final_benchmark.csv"
)

OUTPUT_PATH = (
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


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

print("Loading AWS dataset...")

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

print(
    f"Dataset loaded: {len(df)} observations"
)


# ---------------------------------------------------------
# Create month information
# ---------------------------------------------------------

df["month"] = df["timestamp"].dt.month


# ---------------------------------------------------------
# Robust seasonal statistics
#
# Instead of using raw minimum/maximum values,
# we use:
#
# Median
# Median Absolute Deviation (MAD)
# 5th percentile
# 95th percentile
#
# This prevents extreme injected anomalies from
# defining the normal seasonal range.
# ---------------------------------------------------------

seasonal_statistics = []


for month in range(1, 13):

    month_data = df[
        df["month"] == month
    ]

    if month_data.empty:
        continue

    row = {
        "month": month
    }

    for feature in FEATURES:

        values = (
            month_data[feature]
            .dropna()
            .astype(float)
        )

        median = values.median()

        mad = np.median(
            np.abs(values - median)
        )

        percentile_05 = values.quantile(0.05)

        percentile_95 = values.quantile(0.95)

        row[f"{feature}_median"] = round(
            median,
            4
        )

        row[f"{feature}_mad"] = round(
            mad,
            4
        )

        row[f"{feature}_p05"] = round(
            percentile_05,
            4
        )

        row[f"{feature}_p95"] = round(
            percentile_95,
            4
        )

    seasonal_statistics.append(
        row
    )


# ---------------------------------------------------------
# Create seasonal dataframe
# ---------------------------------------------------------

seasonal_df = pd.DataFrame(
    seasonal_statistics
)


# ---------------------------------------------------------
# Save robust seasonal model
# ---------------------------------------------------------

seasonal_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print()
print("=" * 70)
print("ROBUST SEASONAL PATTERN ANALYSIS")
print("=" * 70)

print()

print(
    seasonal_df.to_string(
        index=False
    )
)

print()

print("=" * 70)

print(
    f"Robust seasonal pattern file saved to:\n{OUTPUT_PATH}"
)

print("=" * 70)