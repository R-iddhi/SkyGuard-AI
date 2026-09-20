from pathlib import Path
import pandas as pd
from evidence_engine import analyze_reading

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_clean_final_benchmark.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

events = df[df["event_id"].notna()]["event_id"].unique()

print("=" * 80)
print("SKYGUARD AI EVIDENCE ENGINE TEST")
print("=" * 80)

passed = 0
total = len(events)

for event_id in events:
    event_rows = df[df["event_id"] == event_id]

    print("\n" + "-" * 80)
    print(f"EVENT: {event_id}")
    print("-" * 80)

    detected = False

    for index in event_rows.index:
        result = analyze_reading(df, index)

        if result["status"] != "NORMAL":
            detected = True

        print(
            f"Timestamp: {df.loc[index, 'timestamp']}"
        )
        print(
            f"Status: {result['status']}"
        )
        print(
            f"Severity: {result['severity']}"
        )
        print(
            f"Evidence Score: {result['evidence_score']}"
        )

        print(
            f"Physical: {result['physical_anomaly']}"
        )
        print(
            f"Recovery: {result['recovery_anomaly']}"
        )
        print(
            f"Frozen: {result['frozen_anomaly']}"
        )
        print(
            f"Multivariate: {result['multivariate_anomaly']}"
        )

        if result["reasons"]:
            print("Reasons:")
            for reason in result["reasons"]:
                print(f"  - {reason}")

        if result["evidence"]:
            print("Evidence:")
            for evidence in result["evidence"]:
                print(f"  - {evidence}")

        print()

    if detected:
        print(f"RESULT: {event_id} DETECTED")
        passed += 1
    else:
        print(f"RESULT: {event_id} MISSED")

print("\n" + "=" * 80)
print("EVIDENCE ENGINE TEST SUMMARY")
print("=" * 80)

print(f"Events detected: {passed}/{total}")

if passed == total:
    print("ALL EVIDENCE ENGINE TESTS PASSED")
else:
    print("SOME EVIDENCE ENGINE TESTS FAILED")

print("=" * 80)