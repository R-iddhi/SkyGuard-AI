import requests


API_URL = "http://127.0.0.1:8000/detect"


def send_reading(payload):
    response = requests.post(
        API_URL,
        json=payload
    )

    print("HTTP Status:", response.status_code)

    assert response.status_code == 200

    return response.json()


def print_test_header(number, title):
    print()
    print("=" * 60)
    print(f"TEST {number} — {title}")
    print("=" * 60)


# ============================================================
# TEST 1 — NORMAL READING
# ============================================================

normal_payload = {
    "station_id": "FINAL-NORMAL",
    "timestamp": "2026-09-19T10:00:00",
    "temperature": -15.0,
    "pressure": 970.0,
    "humidity": 50.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

print_test_header(1, "NORMAL READING")

result = send_reading(normal_payload)

print("Status:", result["status"])
print("Anomaly Type:", result["anomaly_type"])
print("Severity:", result["severity"])
print("Root Cause:", result["root_cause"])
print("Evidence Score:", result["evidence_score"])
print("Confidence Score:", result["confidence_score"])
print("ML Detection:", result["ml_detection"])
print("SHAP Top Feature:", result["shap_top_feature"])
print("Reasons:")

for reason in result["reasons"]:
    print("  ✓", reason)

assert result["status"] == "NORMAL"

print("✅ TEST 1 PASSED")


# ============================================================
# TEST 2 — TEMPERATURE SPIKE
# ============================================================

baseline_payload = {
    "station_id": "FINAL-TEMP-SPIKE",
    "timestamp": "2026-09-19T10:00:00",
    "temperature": -15.0,
    "pressure": 970.0,
    "humidity": 50.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

send_reading(baseline_payload)

temperature_spike_payload = {
    "station_id": "FINAL-TEMP-SPIKE",
    "timestamp": "2026-09-19T10:01:00",
    "temperature": 48.0,
    "pressure": 970.0,
    "humidity": 50.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

print_test_header(2, "TEMPERATURE SPIKE")

result = send_reading(temperature_spike_payload)

print("Status:", result["status"])
print("Anomaly Type:", result["anomaly_type"])
print("Severity:", result["severity"])
print("Root Cause:", result["root_cause"])
print("Evidence Score:", result["evidence_score"])
print("Confidence Score:", result["confidence_score"])
print("ML Detection:", result["ml_detection"])
print("SHAP Top Feature:", result["shap_top_feature"])
print("Reasons:")

for reason in result["reasons"]:
    print("  ✓", reason)

assert result["status"] == "ANOMALY"

print("✅ TEST 2 PASSED")


# ============================================================
# TEST 3 — TEMPERATURE DROP
# ============================================================

baseline_payload = {
    "station_id": "FINAL-TEMP-DROP",
    "timestamp": "2026-09-19T10:00:00",
    "temperature": 20.0,
    "pressure": 970.0,
    "humidity": 50.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

send_reading(baseline_payload)

temperature_drop_payload = {
    "station_id": "FINAL-TEMP-DROP",
    "timestamp": "2026-09-19T10:01:00",
    "temperature": -100.0,
    "pressure": 970.0,
    "humidity": 50.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

print_test_header(3, "TEMPERATURE DROP")

result = send_reading(temperature_drop_payload)

print("Status:", result["status"])
print("Anomaly Type:", result["anomaly_type"])
print("Severity:", result["severity"])
print("Root Cause:", result["root_cause"])
print("Evidence Score:", result["evidence_score"])
print("Confidence Score:", result["confidence_score"])
print("ML Detection:", result["ml_detection"])
print("SHAP Top Feature:", result["shap_top_feature"])
print("Reasons:")

for reason in result["reasons"]:
    print("  ✓", reason)

assert result["status"] == "ANOMALY"

print("✅ TEST 3 PASSED")


# ============================================================
# TEST 4 — INVALID HUMIDITY
# ============================================================

baseline_payload = {
    "station_id": "FINAL-HUMIDITY",
    "timestamp": "2026-09-19T10:00:00",
    "temperature": -15.0,
    "pressure": 970.0,
    "humidity": 50.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

send_reading(baseline_payload)

invalid_humidity_payload = {
    "station_id": "FINAL-HUMIDITY",
    "timestamp": "2026-09-19T10:01:00",
    "temperature": -15.0,
    "pressure": 970.0,
    "humidity": 135.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

print_test_header(4, "INVALID HUMIDITY")

result = send_reading(invalid_humidity_payload)

print("Status:", result["status"])
print("Anomaly Type:", result["anomaly_type"])
print("Severity:", result["severity"])
print("Root Cause:", result["root_cause"])
print("Evidence Score:", result["evidence_score"])
print("Confidence Score:", result["confidence_score"])
print("ML Detection:", result["ml_detection"])
print("SHAP Top Feature:", result["shap_top_feature"])
print("Reasons:")

for reason in result["reasons"]:
    print("  ✓", reason)

assert result["status"] == "ANOMALY"

print("✅ TEST 4 PASSED")


# ============================================================
# TEST 5 — MULTIVARIATE ANOMALY
# ============================================================

baseline_payload = {
    "station_id": "FINAL-MULTIVARIATE",
    "timestamp": "2026-09-19T10:00:00",
    "temperature": -15.0,
    "pressure": 970.0,
    "humidity": 50.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

send_reading(baseline_payload)

multivariate_payload = {
    "station_id": "FINAL-MULTIVARIATE",
    "timestamp": "2026-09-19T10:01:00",
    "temperature": 50.0,
    "pressure": 955.0,
    "humidity": 95.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

print_test_header(5, "MULTIVARIATE ANOMALY")

result = send_reading(multivariate_payload)

print("Status:", result["status"])
print("Anomaly Type:", result["anomaly_type"])
print("Severity:", result["severity"])
print("Root Cause:", result["root_cause"])
print("Evidence Score:", result["evidence_score"])
print("Confidence Score:", result["confidence_score"])
print("ML Detection:", result["ml_detection"])
print("SHAP Top Feature:", result["shap_top_feature"])
print("Reasons:")

for reason in result["reasons"]:
    print("  ✓", reason)

assert result["status"] == "ANOMALY"
assert result["anomaly_type"] == "Multivariate Inconsistency"

print("✅ TEST 5 PASSED")


# ============================================================
# TEST 6 — SHAP EXPLANATION + ML ANOMALY
# ============================================================

baseline_payload = {
    "station_id": "FINAL-SHAP",
    "timestamp": "2026-09-19T10:00:00",
    "temperature": -15.0,
    "pressure": 970.0,
    "humidity": 50.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

send_reading(baseline_payload)

# This observation was independently classified as an
# Isolation Forest anomaly by the trained 3-feature model.
#
# Dataset observation:
# Temperature = 41.6 °C
# Pressure    = 946.7 hPa
# Humidity    = 100.0 %

shap_payload = {
    "station_id": "FINAL-SHAP",
    "timestamp": "2026-09-19T10:01:00",
    "temperature": 41.6,
    "pressure": 946.7,
    "humidity": 100.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0
}

print_test_header(6, "SHAP EXPLANATION + ML ANOMALY")

result = send_reading(shap_payload)

print("Status:", result["status"])
print("Anomaly Type:", result["anomaly_type"])
print("Severity:", result["severity"])
print("Root Cause:", result["root_cause"])
print("Evidence Score:", result["evidence_score"])
print("Confidence Score:", result["confidence_score"])
print("ML Detection:", result["ml_detection"])
print("ML Score:", result["ml_score"])
print("SHAP Top Feature:", result["shap_top_feature"])
print("SHAP Summary:", result["shap_summary"])
print("Reasons:")

for reason in result["reasons"]:
    print("  ✓", reason)

assert result["ml_detection"] is True
assert result["shap_top_feature"] is not None
assert result["shap_summary"] is not None

print("✅ TEST 6 PASSED")


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("ALL FINAL BACKEND TESTS PASSED")
print("=" * 60)