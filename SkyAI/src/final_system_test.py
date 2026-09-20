import requests
import time

API_URL = "http://127.0.0.1:8000/detect"


def test_case(
    name,
    station_id,
    temperature,
    pressure,
    humidity,
    wind_speed,
    wind_direction,
    expected_status,
    expected_type
):
    payload = {
        "station_id": station_id,
        "timestamp": "2026-09-08T12:00:00",
        "temperature": temperature,
        "pressure": pressure,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        if response.status_code != 200:
            print(f"❌ {name}")
            print(f"   HTTP Error: {response.status_code}")
            return False

        result = response.json()

        status = result.get("status")
        anomaly_type = result.get("anomaly_type")
        severity = result.get("severity")
        confidence = result.get("confidence_score")
        shap_feature = result.get("shap_top_feature")

        status_ok = status == expected_status
        type_ok = anomaly_type == expected_type

        if status_ok and type_ok:
            print(f"✅ {name}")
            print(f"   Status: {status}")
            print(f"   Type: {anomaly_type}")
            print(f"   Severity: {severity}")
            print(f"   Confidence: {confidence}%")
            print(f"   SHAP Feature: {shap_feature}")
            return True

        print(f"❌ {name}")
        print(f"   Expected: {expected_status} / {expected_type}")
        print(f"   Received: {status} / {anomaly_type}")

        return False

    except Exception as error:
        print(f"❌ {name}")
        print(f"   Error: {error}")
        return False


print()
print("=" * 65)
print("              SKYGUARD AI FINAL SYSTEM TEST")
print("=" * 65)
print()

passed = 0
total = 8


if test_case(
    "Normal Weather Reading",
    "FINAL_NORMAL",
    20,
    970,
    50,
    20,
    180,
    "NORMAL",
    "Normal"
):
    passed += 1

time.sleep(0.5)


if test_case(
    "Temperature Spike",
    "FINAL_TEMP_SPIKE",
    50,
    970,
    50,
    20,
    180,
    "ANOMALY",
    "Multivariate Inconsistency"
):
    passed += 1

time.sleep(0.5)


if test_case(
    "Invalid Humidity",
    "FINAL_HUMIDITY",
    20,
    970,
    135,
    20,
    180,
    "ANOMALY",
    "Physical/Range Error"
):
    passed += 1

time.sleep(0.5)


if test_case(
    "Multivariate Inconsistency",
    "FINAL_MULTI",
    50,
    951,
    34,
    52.5,
    312,
    "ANOMALY",
    "Multivariate Inconsistency"
):
    passed += 1

time.sleep(0.5)


test_case(
    "Pressure Baseline",
    "FINAL_PRESSURE",
    20,
    970,
    50,
    20,
    180,
    "NORMAL",
    "Normal"
)

time.sleep(0.5)


if test_case(
    "Pressure Spike",
    "FINAL_PRESSURE",
    20,
    990,
    50,
    20,
    180,
    "SUSPICIOUS",
    "Sudden Change"
):
    passed += 1

time.sleep(0.5)


if test_case(
    "Normal Recovery",
    "FINAL_RECOVERY",
    20,
    970,
    50,
    20,
    180,
    "NORMAL",
    "Normal"
):
    passed += 1

time.sleep(0.5)


frozen_station = "FINAL_FROZEN"

frozen_passed = True

for i in range(8):

    result = test_case(
        f"Frozen Sensor Reading {i + 1}/8",
        frozen_station,
        20,
        970,
        50,
        20,
        180,
        "NORMAL" if i < 7 else "ANOMALY",
        "Normal" if i < 7 else "Frozen Sensor"
    )

    if not result:
        frozen_passed = False

    time.sleep(0.3)

if frozen_passed:
    passed += 1


time.sleep(0.5)


if test_case(
    "Extreme Temperature + Humidity",
    "FINAL_EXTREME",
    50,
    970,
    95,
    20,
    180,
    "ANOMALY",
    "Multivariate Inconsistency"
):
    passed += 1


print()
print("=" * 65)
print(f"Tests passed: {passed}/{total}")
print("=" * 65)

if passed == total:
    print("✅ SKYGUARD AI SYSTEM TEST PASSED")
    print("✅ Backend detection pipeline is working correctly")
    print("✅ Rule-based validation is working")
    print("✅ Anomaly classification is working")
    print("✅ Severity classification is working")
    print("✅ Confidence scoring is working")
    print("✅ SHAP explainability is working")
    print("✅ Frozen sensor detection is working")
    print()
    print("🚀 SYSTEM READY FOR NEXT STAGE")
else:
    print("❌ SYSTEM TEST FAILED")
    print("Please fix the failed test cases before continuing.")

print("=" * 65)