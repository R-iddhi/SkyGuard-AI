import pandas as pd


# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/aws_anomaly_dataset.csv"

df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# ==========================================
# 2. REAL-TIME DETECTION FUNCTION
# ==========================================

def detect_anomaly(current, previous, temperature_history):

    reasons = []

    # --------------------------------------
    # Physical validation
    # --------------------------------------

    if current["humidity"] < 0 or current["humidity"] > 100:
        reasons.append("Invalid humidity value")

    if current["wind_speed"] < 0:
        reasons.append("Invalid wind speed")

    if current["wind_direction"] < 0 or current["wind_direction"] > 360:
        reasons.append("Invalid wind direction")


    # --------------------------------------
    # Temporal validation
    # --------------------------------------

    if previous is not None:

        temperature_change = (
            current["temperature"] -
            previous["temperature"]
        )

        pressure_change = (
            current["pressure"] -
            previous["pressure"]
        )

        humidity_change = (
            current["humidity"] -
            previous["humidity"]
        )

        if abs(temperature_change) > 10:
            reasons.append("Sudden temperature change")

        if abs(pressure_change) > 10:
            reasons.append("Sudden pressure change")

        if abs(humidity_change) > 20:
            reasons.append("Sudden humidity change")


    # --------------------------------------
    # Frozen sensor detection
    # --------------------------------------

    if len(temperature_history) >= 8:

        recent_temperatures = temperature_history[-8:]

        if len(set(recent_temperatures)) == 1:
            reasons.append("Frozen temperature sensor")


    # --------------------------------------
    # Final decision
    # --------------------------------------

    if reasons:
        status = "ANOMALY"
    else:
        status = "NORMAL"

    return status, reasons


# ==========================================
# 3. TEST CASES
# ==========================================

test_cases = [
    {
        "name": "Temperature Spike",
        "timestamp": "2015-08-13 11:00:00"
    },
    {
        "name": "Temperature Drop",
        "timestamp": "2015-11-10 19:00:00"
    },
    {
        "name": "Frozen Temperature Sensor",
        "timestamp": "2016-02-02 03:00:00"
    },
    {
        "name": "Invalid Humidity",
        "timestamp": "2016-09-05 19:00:00"
    },
    {
        "name": "Multivariate Temperature Anomaly",
        "timestamp": "2016-10-17 11:00:00"
    }
]


# ==========================================
# 4. RESULTS
# ==========================================

print("=" * 60)
print("SKYGUARD AI - REAL-TIME INTEGRATION TEST")
print("=" * 60)

results = []


# ==========================================
# 5. NORMAL TEST CASES
# ==========================================

for test in test_cases:

    # Frozen sensor is handled separately below
    if test["name"] == "Frozen Temperature Sensor":
        continue

    timestamp = test["timestamp"]

    index_list = df.index[
        df["timestamp"] == timestamp
    ].tolist()

    if not index_list:
        print("\nTimestamp not found:", timestamp)
        continue

    index = index_list[0]

    current = df.iloc[index]

    if index > 0:
        previous = df.iloc[index - 1]
    else:
        previous = None

    temperature_history = [
        current["temperature"]
    ]

    status, reasons = detect_anomaly(
        current,
        previous,
        temperature_history
    )

    expected = "ANOMALY"

    passed = status == expected

    results.append({
        "test": test["name"],
        "expected": expected,
        "detected": status,
        "passed": passed
    })

    print("\n" + "-" * 60)

    print("Test:", test["name"])
    print("Timestamp:", current["timestamp"])
    print("Temperature:", current["temperature"], "°C")
    print("Pressure:", current["pressure"], "hPa")
    print("Humidity:", current["humidity"], "%")
    print("Expected:", expected)
    print("Detected:", status)
    print("Result:", "PASS" if passed else "FAIL")

    if reasons:
        print("Reasons:")

        for reason in reasons:
            print(" -", reason)

    else:
        print("Reasons: None")


# ==========================================
# 6. FROZEN SENSOR CONTINUOUS TEST
# ==========================================

print("\n" + "=" * 60)
print("FROZEN SENSOR CONTINUOUS STREAM TEST")
print("=" * 60)


frozen_start = pd.Timestamp("2016-02-02 03:00:00")
frozen_end = pd.Timestamp("2016-02-02 12:00:00")


frozen_df = df[
    (df["timestamp"] >= frozen_start) &
    (df["timestamp"] <= frozen_end)
].copy()


temperature_history = []

previous = None

frozen_detected = False


for _, current in frozen_df.iterrows():

    temperature_history.append(
        current["temperature"]
    )

    status, reasons = detect_anomaly(
        current,
        previous,
        temperature_history
    )

    print(
        current["timestamp"],
        "| Temperature:",
        current["temperature"],
        "|",
        status
    )

    if reasons:

        for reason in reasons:
            print("   -", reason)

    if status == "ANOMALY":
        frozen_detected = True

    previous = current


# ------------------------------------------
# Frozen test result
# ------------------------------------------

results.append({
    "test": "Frozen Temperature Sensor",
    "expected": "ANOMALY",
    "detected": "ANOMALY" if frozen_detected else "NORMAL",
    "passed": frozen_detected
})


print("\nFrozen Sensor Test Result:")

if frozen_detected:
    print("PASS")
else:
    print("FAIL")


# ==========================================
# 7. FINAL SUMMARY
# ==========================================

results_df = pd.DataFrame(results)

passed_tests = results_df["passed"].sum()

total_tests = len(results_df)

print("\n" + "=" * 60)
print("REAL-TIME INTEGRATION TEST SUMMARY")
print("=" * 60)

print("\nTotal tests:")
print(total_tests)

print("\nTests passed:")
print(passed_tests)

print("\nTests failed:")
print(total_tests - passed_tests)

print(
    "\nDetection success rate:",
    round((passed_tests / total_tests) * 100, 2),
    "%"
)

print("\nDetailed results:")
print(results_df.to_string(index=False))

print("\n" + "=" * 60)
print("INTEGRATION TEST COMPLETED")
print("=" * 60)