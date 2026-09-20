from pathlib import Path
import time

import pandas as pd
import requests


# ==================================================
# PROJECT PATH
# ==================================================

project_root = Path(__file__).resolve().parent.parent

data_path = (
    project_root
    / "data"
    / "aws_anomaly_dataset.csv"
)


# ==================================================
# API CONFIGURATION
# ==================================================

API_URL = "http://127.0.0.1:8000/detect"

STATION_ID = "SIM_AWS_001"

NUMBER_OF_READINGS = 20

DELAY_SECONDS = 0.5


# ==================================================
# LOAD DATASET
# ==================================================

print("=" * 60)
print("SKYGUARD AI - REAL-TIME AWS SIMULATOR")
print("=" * 60)

print()
print("Loading AWS dataset...")

df = pd.read_csv(data_path)

print(
    f"Dataset loaded successfully: {len(df)} observations"
)


# ==================================================
# SELECT FIRST READINGS
# ==================================================

simulation_data = df.head(
    NUMBER_OF_READINGS
).copy()


print(
    f"Simulating {len(simulation_data)} real-time readings..."
)

print()


# ==================================================
# SEND READINGS TO API
# ==================================================

for index, row in simulation_data.iterrows():

    payload = {

        "station_id": STATION_ID,

        "timestamp": str(
            row["timestamp"]
        ),

        "temperature": float(
            row["temperature"]
        ),

        "pressure": float(
            row["pressure"]
        ),

        "humidity": float(
            row["humidity"]
        ),

        "wind_speed": float(
            row["wind_speed"]
        ),

        "wind_direction": float(
            row["wind_direction"]
        )
    }


    # --------------------------------------------------
    # Send reading to FastAPI
    # --------------------------------------------------

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )


        # --------------------------------------------------
        # Check response
        # --------------------------------------------------

        if response.status_code == 200:

            result = response.json()


            status = result["status"]

            anomaly_type = result["anomaly_type"]

            severity = result["severity"]


            print(
                f"[{index + 1:02d}] "
                f"{payload['timestamp']} | "
                f"Temp: {payload['temperature']:6.1f} °C | "
                f"Status: {status:7s} | "
                f"Type: {anomaly_type} | "
                f"Severity: {severity}"
            )


        else:

            print(
                f"[{index + 1:02d}] "
                f"API ERROR: {response.status_code}"
            )


    except requests.exceptions.ConnectionError:

        print(
            "ERROR: Could not connect to SkyGuard API."
        )

        print(
            "Make sure FastAPI is running."
        )

        break


    except requests.exceptions.Timeout:

        print(
            "ERROR: API request timed out."
        )

        break


    except Exception as e:

        print(
            f"ERROR: {e}"
        )

        break


    # --------------------------------------------------
    # Simulate real-time delay
    # --------------------------------------------------

    time.sleep(
        DELAY_SECONDS
    )


# ==================================================
# SIMULATION COMPLETE
# ==================================================

print()

print("=" * 60)

print("REAL-TIME SIMULATION COMPLETED")

print("=" * 60)