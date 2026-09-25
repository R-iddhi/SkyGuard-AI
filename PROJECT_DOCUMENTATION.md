# SkyGuard AI
## Intelligent Real-Time Anomaly Detection System for Automatic Weather Stations

---

# 1. Project Overview

SkyGuard AI is an intelligent anomaly detection system designed for Automatic Weather Stations (AWS).

The system monitors:

- Temperature (°C)
- Atmospheric Pressure (hPa)
- Relative Humidity (%)

It combines rule-based validation, temporal analysis, multivariate consistency checking, machine learning, explainable AI, spatial consistency, seasonal patterns, and evidence scoring to identify potentially abnormal sensor observations.

The system is designed to distinguish unusual sensor/data behavior from potentially genuine meteorological events while minimizing false alarms.

---

# 2. Problem Statement

Automatic Weather Stations continuously collect environmental observations used in weather forecasting, climate monitoring, agriculture, disaster management, aviation, and other applications.

Sensor readings can become unreliable because of:

- Sensor malfunction
- Sudden measurement spikes
- Sudden measurement drops
- Frozen sensor values
- Communication failures
- Calibration problems
- Data corruption
- Unusual combinations of sensor measurements

A simple threshold-based system may generate false alarms because extreme weather conditions can also produce unusual measurements.

SkyGuard AI addresses this problem by combining multiple forms of evidence before classifying an observation.

---

# 3. Objectives

The main objectives of SkyGuard AI are:

1. Detect sensor anomalies in real time.
2. Identify sudden abnormal changes.
3. Detect frozen sensor behavior.
4. Detect physically invalid observations.
5. Detect inconsistent combinations of temperature, pressure, and humidity.
6. Analyze temporal and seasonal behavior.
7. Compare observations with neighboring stations when available.
8. Use machine learning for additional anomaly detection.
9. Provide explainable AI information using SHAP.
10. Assign anomaly severity and confidence/evidence scores.
11. Identify possible root causes.
12. Provide sensor health information.
13. Support real-time API-based detection.
14. Provide a visualization dashboard.
15. Support scalable deployment across weather observation networks.

---

# 4. System Architecture

The implemented processing pipeline is:

AWS / Historical Data
        |
        v
Data Ingestion
        |
        v
Data Validation and Cleaning
        |
        v
Feature Preparation
        |
        v
Physical Validation
        |
        +----> Temporal Validation
        |
        +----> Frozen Sensor Detection
        |
        +----> Multivariate Consistency
        |
        +----> Seasonal Analysis
        |
        +----> Spatial Consistency
        |
        +----> Machine Learning
        |
        v
Evidence Combination
        |
        v
Anomaly Classification
        |
        v
Severity + Confidence
        |
        v
Root Cause + Event Interpretation
        |
        v
SHAP Explanation
        |
        v
Sensor Health
        |
        v
FastAPI Real-Time Service
        |
        v
Streamlit Dashboard


---

# 5. Data Pipeline

The system uses a structured data-processing pipeline before anomaly detection.

## 5.1 Data Ingestion

Sensor observations are loaded from the AWS dataset and converted into a standardized tabular format.

The working variables are:

- Timestamp
- Temperature (°C)
- Atmospheric Pressure (hPa)
- Relative Humidity (%)

Additional meteorological variables such as wind speed and wind direction may be retained for analysis but are not part of the primary three-feature ML model.

## 5.2 Data Validation

The system checks:

- Missing values
- Duplicate timestamps
- Invalid physical values
- Timestamp consistency
- Unexpected measurement ranges
- Large temporal gaps

The original raw dataset is preserved, while a cleaned working dataset is used for processing.

## 5.3 Feature Preparation

The system calculates derived temporal features such as:

- Change from previous observation
- Rolling/local statistics
- Frozen-value sequences
- Multivariate relationships
- Seasonal deviations

These features support both rule-based and machine-learning detection.

---

# 6. Anomaly Detection Framework

SkyGuard AI uses multiple complementary detection mechanisms.

## 6.1 Physical Validation

Physical validation checks whether sensor observations fall within acceptable physical ranges.

For example:

- Relative humidity above 100% is considered invalid.
- Other physically impossible measurements are flagged for investigation.

This layer provides deterministic validation before more advanced analysis.

## 6.2 Temporal Validation

Temporal validation analyzes changes between consecutive observations.

Large changes in:

- Temperature
- Atmospheric pressure
- Relative humidity

can indicate sudden sensor/data changes.

Temporal behavior is evaluated together with other evidence to reduce false alarms.

## 6.3 Frozen Sensor Detection

A frozen sensor repeatedly reports the same value for an extended period.

SkyGuard AI detects repeated identical readings and raises an alert only after sufficient consecutive observations are available.

This prevents a single stable reading from being incorrectly classified as a sensor failure.

## 6.4 Multivariate Consistency

The system analyzes temperature, pressure, and humidity together.

A reading can be suspicious when multiple variables change in an inconsistent manner.

This allows SkyGuard AI to identify abnormal combinations that may not be detected by an individual threshold.

## 6.5 Seasonal Analysis

Weather measurements naturally vary according to season.

SkyGuard AI therefore compares observations with seasonal/monthly patterns where applicable.

This helps prevent naturally occurring seasonal conditions from being incorrectly treated as sensor failures.

## 6.6 Spatial Consistency

When neighboring-station information is available, SkyGuard AI compares the target station with nearby observations.

If one station reports an extreme value while neighboring stations remain relatively normal, the observation can receive additional anomaly evidence.

Spatial analysis is optional and depends on the availability of neighboring-station data.

---

# 7. Machine Learning Model

SkyGuard AI uses Isolation Forest as an unsupervised anomaly-detection model.

The primary ML features are:

- Temperature
- Atmospheric Pressure
- Relative Humidity

Isolation Forest attempts to identify observations that are statistically different from normal observations.

The ML result is treated as one source of evidence rather than the sole decision mechanism.

This hybrid approach combines deterministic validation with machine-learning-based anomaly detection.

---

# 8. Evidence Combination

SkyGuard AI combines evidence from multiple detection components.

Possible evidence sources include:

- Physical validation
- Temporal behavior
- Frozen sensor detection
- Multivariate inconsistency
- Seasonal analysis
- Spatial consistency
- Machine-learning detection

The combined evidence is then used to determine the final detection state.

The system supports three major operational states:

- **NORMAL** — no significant anomaly evidence.
- **SUSPICIOUS** — unusual behavior detected but further investigation may be required.
- **ANOMALY** — strong evidence indicates abnormal sensor/data behavior.

This multi-level approach helps avoid treating every unusual weather observation as a confirmed sensor failure.

---

# 9. Anomaly Classification

Detected observations can be classified into categories such as:

- Normal
- Sudden Change
- Physical/Range Error
- Frozen Sensor
- Multivariate Inconsistency

The classification provides an interpretable description of the detected behavior.

---

# 10. Severity and Confidence

SkyGuard AI assigns severity levels based on the available anomaly evidence.

The system supports:

- NONE
- LOW
- MEDIUM
- HIGH

The evidence/confidence score represents the strength of available detection evidence.

**Important:** the current confidence score is an operational evidence-strength score and should not be interpreted as a calibrated statistical probability.

---

# 11. Root Cause Analysis

The system generates a probable root-cause interpretation based on the detected evidence.

Examples include:

- Possible sensor hardware failure
- Possible communication problem
- Possible sensor/data validity problem
- Possible multivariate inconsistency
- Possible unusual meteorological event

These interpretations assist operators in deciding whether an observation requires further investigation.

---

# 12. Explainable AI

SkyGuard AI integrates SHAP to explain the contribution of individual features to the machine-learning decision.

For example, an anomaly may produce:

```text
Top contributing feature:
Temperature

uvicorn src.api:app
streamlit run src\dashboard.py