import requests
import time
from datetime import datetime, timedelta


API_URL = "http://127.0.0.1:8000/detect"

STATION_ID = "LIVE_AWS_001"


def send_reading(
    timestamp,
    temperature,
    pressure,
    humidity,
    wind_speed,
    wind_direction
):

    payload = {
        "station_id": STATION_ID,
        "timestamp": timestamp,

        "temperature": temperature,
        "pressure": pressure,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction
    }

    response = requests.post(
        API_URL,
        json=payload
    )

    if response.status_code != 200:

        print("API ERROR:", response.text)

        return

    result = response.json()

    print(
        f"\n[{timestamp}]"
    )

    print(
        f"Temperature : {temperature} °C"
    )

    print(
        f"Pressure    : {pressure} hPa"
    )

    print(
        f"Humidity    : {humidity} %"
    )

    print(
        f"Status      : {result['status']}"
    )

    print(
        f"Type        : {result['anomaly_type']}"
    )

    print(
        f"Severity    : {result['severity']}"
    )

    print(
        f"Confidence  : {result['confidence_score']}"
    )

    if result["shap_top_feature"]:

        print(
            f"SHAP Top   : "
            f"{result['shap_top_feature']}"
        )


# ============================================================
# START
# ============================================================

print("=" * 60)
print("SKYGUARD AI — REAL-TIME WEATHER STATION SIMULATOR")
print("=" * 60)

print()
print("Station:", STATION_ID)
print("Starting simulation...")


base_time = datetime.now()


# ============================================================
# PHASE 1 — NORMAL READINGS
# ============================================================

print("\n--- PHASE 1: NORMAL WEATHER ---")

normal_readings = [

    (-3.0, 966.0, 50.0, 12.0, 180.0),

    (-2.8, 966.1, 51.0, 13.0, 182.0),

    (-2.9, 965.9, 50.0, 11.0, 179.0),

    (-3.1, 966.2, 52.0, 12.5, 181.0),

]


for i, reading in enumerate(normal_readings):

    timestamp = (
        base_time
        + timedelta(minutes=i)
    ).isoformat()

    send_reading(
        timestamp,
        *reading
    )

    time.sleep(1)


# ============================================================
# PHASE 2 — TEMPERATURE SPIKE
# ============================================================

print("\n--- PHASE 2: TEMPERATURE SPIKE ---")

timestamp = (
    base_time
    + timedelta(minutes=4)
).isoformat()

send_reading(
    timestamp,
    60.0,
    966.0,
    50.0,
    12.0,
    180.0
)

time.sleep(1)


# ============================================================
# PHASE 3 — RETURN TO NORMAL
# ============================================================

print("\n--- PHASE 3: RETURN TO NORMAL ---")

timestamp = (
    base_time
    + timedelta(minutes=5)
).isoformat()

send_reading(
    timestamp,
    -3.0,
    966.0,
    50.0,
    12.0,
    180.0
)

time.sleep(1)


# ============================================================
# PHASE 4 — FROZEN SENSOR
# ============================================================

print("\n--- PHASE 4: FROZEN SENSOR ---")

for i in range(9):

    timestamp = (
        base_time
        + timedelta(minutes=6 + i)
    ).isoformat()

    send_reading(
        timestamp,
        -2.9,
        966.0,
        50.0,
        12.0,
        180.0
    )

    time.sleep(0.7)


# ============================================================
# PHASE 5 — INVALID HUMIDITY
# ============================================================

print("\n--- PHASE 5: INVALID HUMIDITY ---")

timestamp = (
    base_time
    + timedelta(minutes=15)
).isoformat()

send_reading(
    timestamp,
    -3.0,
    966.0,
    135.0,
    12.0,
    180.0
)

time.sleep(1)


# ============================================================
# PHASE 6 — MULTIVARIATE ANOMALY
# ============================================================

print("\n--- PHASE 6: MULTIVARIATE ANOMALY ---")

timestamp = (
    base_time
    + timedelta(minutes=16)
).isoformat()

send_reading(
    timestamp,
    50.0,
    951.5,
    34.0,
    52.5,
    312.0
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("REAL-TIME SIMULATION COMPLETE")
print("=" * 60)

print()
print("✓ Normal readings processed")
print("✓ Temperature spike tested")
print("✓ Recovery tested")
print("✓ Frozen sensor tested")
print("✓ Invalid humidity tested")
print("✓ Multivariate anomaly tested")

print()
print("SkyGuard AI real-time pipeline is working.")

