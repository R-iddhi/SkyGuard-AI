import time
from pathlib import Path

import pandas as pd
import requests


# ==================================================
# CONFIGURATION
# ==================================================

API_URL = "http://127.0.0.1:8000/detect"

STATION_ID = "FROZEN_SENSOR_TEST"

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "aws_anomaly_dataset.csv"
)


# ==================================================
# LOAD DATA
# ==================================================

print("=" * 70)
print("SKYGUARD AI - FROZEN SENSOR TEST")
print("=" * 70)
print()

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {len(df)} observations")
print()


# ==================================================
# SELECT DATA
# ==================================================

# One normal reading immediately before
# the frozen sensor starts.
normal_reading = df.iloc[8999]

# Ten consecutive frozen readings.
frozen_readings = df.iloc[9000:9010]


print("Normal baseline:")
print(
    f"Timestamp: {normal_reading['timestamp']}"
)
print(
    f"Temperature: {normal_reading['temperature']} °C"
)

print()

print("Frozen sensor period:")
print(
    f"Start: {frozen_readings.iloc[0]['timestamp']}"
)
print(
    f"End: {frozen_readings.iloc[-1]['timestamp']}"
)
print(
    f"Frozen temperature: "
    f"{frozen_readings.iloc[0]['temperature']} °C"
)

print()


# ==================================================
# COMBINE READINGS
# ==================================================

simulation_data = pd.concat(
    [
        df.iloc[[8999]],
        df.iloc[9000:9010]
    ]
)


# ==================================================
# STREAM READINGS
# ==================================================

detected = False

for number, (_, row) in enumerate(
    simulation_data.iterrows(),
    start=1
):

    payload = {
        "station_id": STATION_ID,
        "timestamp": str(row["timestamp"]),
        "temperature": float(row["temperature"]),
        "pressure": float(row["pressure"]),
        "humidity": float(row["humidity"]),
        "wind_speed": float(row["wind_speed"]),
        "wind_direction": float(row["wind_direction"])
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        if response.status_code != 200:

            print(
                f"[{number:02d}] "
                f"API ERROR: {response.status_code}"
            )

            continue

        result = response.json()

        print(
            f"[{number:02d}] "
            f"{payload['timestamp']} | "
            f"Temp: "
            f"{payload['temperature']:6.1f} °C | "
            f"Status: "
            f"{result['status']:7s} | "
            f"Type: "
            f"{result['anomaly_type']} | "
            f"Severity: "
            f"{result['severity']}"
        )

        if (
            result["status"] == "ANOMALY"
            and result["anomaly_type"] == "Frozen Sensor"
        ):

            detected = True

            print()
            print(
                "🚨 FROZEN SENSOR DETECTED"
            )

            print(
                f"   Evidence Score: "
                f"{result['evidence_score']}"
            )

            print(
                f"   Root Cause: "
                f"{result['root_cause']}"
            )

            print()

    except requests.exceptions.ConnectionError:

        print(
            "❌ Could not connect to FastAPI."
        )

        print(
            "Make sure Uvicorn is running."
        )

        break

    except Exception as e:

        print(
            f"❌ ERROR: {e}"
        )

    time.sleep(0.3)


# ==================================================
# FINAL RESULT
# ==================================================

print("=" * 70)

if detected:

    print(
        "✅ FROZEN SENSOR TEST PASSED"
    )

    print(
        "SkyGuard successfully detected the "
        "sensor becoming stuck."
    )

else:

    print(
        "❌ FROZEN SENSOR TEST FAILED"
    )

    print(
        "The frozen sensor was not detected."
    )

print("=" * 70)