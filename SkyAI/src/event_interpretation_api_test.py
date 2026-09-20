import requests


API_URL = "http://127.0.0.1:8000/detect"


def send_reading(payload):
    response = requests.post(
        API_URL,
        json=payload
    )

    print("HTTP status:", response.status_code)

    assert response.status_code == 200

    return response.json()


print()
print("=" * 70)
print("SKYGUARD AI - EVENT INTERPRETATION API TEST")
print("=" * 70)


# ---------------------------------------------------------
# TEST 1: Normal observation
# ---------------------------------------------------------

normal_payload = {
    "station_id": "EVENT-NORMAL",
    "timestamp": "2026-09-19T10:00:00",
    "temperature": -15.0,
    "pressure": 970.0,
    "humidity": 50.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

result = send_reading(normal_payload)

print()
print("TEST 1: Normal Observation")
print("Status:", result["status"])
print(
    "Classification:",
    result["event_interpretation"]["classification"]
)
print(
    "Interpretation:",
    result["event_interpretation"]["interpretation"]
)

assert (
    result["event_interpretation"]["classification"]
    == "LIKELY_NORMAL"
)

print("✅ TEST 1 PASSED")


# ---------------------------------------------------------
# TEST 2: Sudden change
# ---------------------------------------------------------

baseline_payload = {
    "station_id": "EVENT-TEMP",
    "timestamp": "2026-09-19T10:00:00",
    "temperature": -15.0,
    "pressure": 970.0,
    "humidity": 50.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

send_reading(baseline_payload)

sudden_change_payload = {
    "station_id": "EVENT-TEMP",
    "timestamp": "2026-09-19T10:01:00",
    "temperature": -3.0,
    "pressure": 970.0,
    "humidity": 50.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

result = send_reading(sudden_change_payload)

print()
print("TEST 2: Sudden Temperature Change")
print("Status:", result["status"])
print(
    "Classification:",
    result["event_interpretation"]["classification"]
)
print(
    "Evidence strength:",
    result["event_interpretation"]["evidence_strength"]
)

assert (
    result["event_interpretation"]["classification"]
    == "POSSIBLE_METEOROLOGICAL_EVENT"
)

print("✅ TEST 2 PASSED")


# ---------------------------------------------------------
# TEST 3: Spatial inconsistency
# ---------------------------------------------------------

spatial_payload = {
    "station_id": "EVENT-SPATIAL",
    "timestamp": "2026-09-19T10:00:00",
    "temperature": 40.0,
    "pressure": 970.0,
    "humidity": 50.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0,
    "neighbors": [
        {
            "station_id": "NEIGHBOR-1",
            "temperature": 20.0,
            "pressure": 970.0,
            "humidity": 50.0
        },
        {
            "station_id": "NEIGHBOR-2",
            "temperature": 20.0,
            "pressure": 970.0,
            "humidity": 50.0
        },
        {
            "station_id": "NEIGHBOR-3",
            "temperature": 20.0,
            "pressure": 970.0,
            "humidity": 50.0
        }
    ]
}

result = send_reading(spatial_payload)

print()
print("TEST 3: Spatial Inconsistency")
print("Status:", result["status"])
print(
    "Classification:",
    result["event_interpretation"]["classification"]
)
print(
    "Supporting evidence:",
    result["event_interpretation"]["supporting_evidence"]
)

assert (
    result["event_interpretation"]["classification"]
    == "POSSIBLE_SENSOR_OR_LOCAL_EVENT"
)

print("✅ TEST 3 PASSED")


# ---------------------------------------------------------
# TEST 4: Physical/data anomaly
# ---------------------------------------------------------

physical_payload = {
    "station_id": "EVENT-PHYSICAL",
    "timestamp": "2026-09-19T10:00:00",
    "temperature": -15.0,
    "pressure": 970.0,
    "humidity": 135.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

result = send_reading(physical_payload)

print()
print("TEST 4: Physical/Data Anomaly")
print("Status:", result["status"])
print(
    "Classification:",
    result["event_interpretation"]["classification"]
)
print(
    "Evidence strength:",
    result["event_interpretation"]["evidence_strength"]
)
print(
    "Warning:",
    result["event_interpretation"]["warning"]
)

assert (
    result["event_interpretation"]["classification"]
    == "LIKELY_SENSOR_OR_DATA_PROBLEM"
)

print("✅ TEST 4 PASSED")


print()
print("=" * 70)
print("ALL EVENT INTERPRETATION API TESTS PASSED")
print("=" * 70)