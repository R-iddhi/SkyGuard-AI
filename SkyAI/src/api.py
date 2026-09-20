from pathlib import Path
from datetime import datetime
from typing import List

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .shap_service import explain_reading
from .seasonal_detector import detect_seasonal_anomaly
from .spatial_consistency import check_spatial_consistency
from .evidence_engine import calculate_evidence
from .event_interpretation import interpret_event
from .value_correction import correct_realtime_value


app = FastAPI(title="SkyGuard AI")


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "isolation_forest_3features.joblib"
)

ANOMALY_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "aws_anomaly_dataset.csv"
)

FEATURES = [
    "temperature",
    "pressure",
    "humidity"
]

station_previous_data = {}
station_temperature_history = {}
station_pressure_history = {}
station_humidity_history = {}
station_previous_timestamp = {}


print("Loading SHAP model...")
print("SHAP model loaded successfully.")

print("Loading saved Isolation Forest...")

model = joblib.load(MODEL_PATH)

print(f"Model: {MODEL_PATH}")
print("Isolation Forest model loaded successfully.")


# =========================================================
# MAINTENANCE MONITORING
# =========================================================

def calculate_maintenance_monitoring():

    try:

        df = pd.read_csv(
            ANOMALY_DATASET_PATH
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        df = (
            df
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # -------------------------------------------------
        # Sensor changes
        # -------------------------------------------------

        df["temperature_change"] = (
            df["temperature"].diff()
        )

        df["pressure_change"] = (
            df["pressure"].diff()
        )

        df["humidity_change"] = (
            df["humidity"].diff()
        )

        # -------------------------------------------------
        # Physical validation
        # -------------------------------------------------

        df["physical_anomaly"] = (
            (df["humidity"] < 0)
            |
            (df["humidity"] > 100)
            |
            (df["wind_speed"] < 0)
            |
            (df["wind_direction"] < 0)
            |
            (df["wind_direction"] > 360)
        )

        # -------------------------------------------------
        # Temporal validation
        # -------------------------------------------------

        df["temporal_anomaly"] = (
            (df["temperature_change"].abs() > 10)
            |
            (df["pressure_change"].abs() > 10)
            |
            (df["humidity_change"].abs() > 20)
        )

        # -------------------------------------------------
        # Frozen sensor detection
        # -------------------------------------------------

        df["temperature_same"] = (
            df["temperature"]
            ==
            df["temperature"].shift(1)
        )

        temperature_group = (
            (~df["temperature_same"]).cumsum()
        )

        group_sizes = (
            df
            .groupby(temperature_group)["temperature"]
            .transform("size")
        )

        df["frozen_sensor"] = (
            group_sizes >= 8
        )

        # -------------------------------------------------
        # Multivariate validation
        # -------------------------------------------------

        temperature_anomaly = (
            df["temperature_change"].abs() > 10
        )

        pressure_anomaly = (
            df["pressure_change"].abs() > 10
        )

        humidity_anomaly = (
            df["humidity_change"].abs() > 20
        )

        df["multivariate_anomaly"] = (
            (
                temperature_anomaly
                &
                humidity_anomaly
            )
            |
            (
                temperature_anomaly
                &
                pressure_anomaly
            )
            |
            (df["humidity"] > 100)
        )

        # -------------------------------------------------
        # Overall degradation signal
        # -------------------------------------------------

        df["degradation_signal"] = (
            df["physical_anomaly"]
            |
            df["temporal_anomaly"]
            |
            df["frozen_sensor"]
            |
            df["multivariate_anomaly"]
        )

        # -------------------------------------------------
        # Historical / recent periods
        # -------------------------------------------------

        total = len(df)

        historical_start = int(
            total * 0.40
        )

        historical_end = int(
            total * 0.60
        )

        recent_start = int(
            total * 0.80
        )

        historical_df = (
            df
            .iloc[
                historical_start:
                historical_end
            ]
            .copy()
        )

        recent_df = (
            df
            .iloc[
                recent_start:
            ]
            .copy()
        )

        # -------------------------------------------------
        # Anomaly rates
        # -------------------------------------------------

        historical_anomalies = int(
            historical_df[
                "degradation_signal"
            ].sum()
        )

        recent_anomalies = int(
            recent_df[
                "degradation_signal"
            ].sum()
        )

        historical_rate = (
            historical_anomalies
            /
            len(historical_df)
        ) * 100

        recent_rate = (
            recent_anomalies
            /
            len(recent_df)
        ) * 100

        rate_change = (
            recent_rate
            -
            historical_rate
        )

        if historical_rate > 0:

            percentage_increase = (
                rate_change
                /
                historical_rate
            ) * 100

        else:

            percentage_increase = 0

        # -------------------------------------------------
        # Component-specific recent rates
        # -------------------------------------------------

        recent_physical_rate = (
            recent_df[
                "physical_anomaly"
            ].mean()
            * 100
        )

        recent_temporal_rate = (
            recent_df[
                "temporal_anomaly"
            ].mean()
            * 100
        )

        recent_frozen_rate = (
            recent_df[
                "frozen_sensor"
            ].mean()
            * 100
        )

        recent_multivariate_rate = (
            recent_df[
                "multivariate_anomaly"
            ].mean()
            * 100
        )

        # -------------------------------------------------
        # Maintenance risk score
        # -------------------------------------------------

        maintenance_risk_score = 0

        if recent_rate >= 5:

            maintenance_risk_score += 30

        elif recent_rate >= 2:

            maintenance_risk_score += 20

        elif recent_rate >= 1:

            maintenance_risk_score += 10

        # -------------------------------------------------
        # Degradation trend
        # -------------------------------------------------

        if percentage_increase >= 100:

            maintenance_risk_score += 25

        elif percentage_increase >= 50:

            maintenance_risk_score += 20

        elif percentage_increase >= 25:

            maintenance_risk_score += 10

        # -------------------------------------------------
        # Frozen sensor
        # -------------------------------------------------

        if recent_frozen_rate >= 1:

            maintenance_risk_score += 20

        elif recent_frozen_rate > 0:

            maintenance_risk_score += 10

        # -------------------------------------------------
        # Physical failures
        # -------------------------------------------------

        if recent_physical_rate >= 1:

            maintenance_risk_score += 15

        elif recent_physical_rate > 0:

            maintenance_risk_score += 8

        # -------------------------------------------------
        # Multivariate inconsistencies
        # -------------------------------------------------

        if recent_multivariate_rate >= 1:

            maintenance_risk_score += 15

        elif recent_multivariate_rate > 0:

            maintenance_risk_score += 8

        # -------------------------------------------------
        # Temporal instability
        # -------------------------------------------------

        if recent_temporal_rate >= 5:

            maintenance_risk_score += 10

        elif recent_temporal_rate >= 2:

            maintenance_risk_score += 5

        maintenance_risk_score = min(
            100,
            maintenance_risk_score
        )

        # -------------------------------------------------
        # Degradation trend
        # -------------------------------------------------

        if percentage_increase >= 50:

            degradation_status = "Increasing"

        elif percentage_increase <= -30:

            degradation_status = "Improving"

        else:

            degradation_status = "Stable"

        # -------------------------------------------------
        # Maintenance risk level
        # -------------------------------------------------

        if maintenance_risk_score >= 70:

            maintenance_risk = "High"

        elif maintenance_risk_score >= 40:

            maintenance_risk = "Medium"

        else:

            maintenance_risk = "Low"

        # -------------------------------------------------
        # Recommendation
        # -------------------------------------------------

        if maintenance_risk == "High":

            recommendation = (
                "Immediate sensor inspection and "
                "preventive maintenance recommended."
            )

        elif maintenance_risk == "Medium":

            recommendation = (
                "Monitor sensor closely and schedule "
                "preventive maintenance."
            )

        else:

            recommendation = (
                "Sensor condition is stable. "
                "No immediate maintenance required."
            )

        # -------------------------------------------------
        # Risk factors
        # -------------------------------------------------

        risk_factors = []

        if recent_rate >= 1:

            risk_factors.append(
                "Elevated recent anomaly activity"
            )

        if percentage_increase >= 25:

            risk_factors.append(
                "Increasing anomaly trend"
            )

        if recent_frozen_rate > 0:

            risk_factors.append(
                "Frozen sensor behavior detected"
            )

        if recent_physical_rate > 0:

            risk_factors.append(
                "Physical/range violations detected"
            )

        if recent_multivariate_rate > 0:

            risk_factors.append(
                "Multivariate inconsistencies detected"
            )

        if recent_temporal_rate >= 2:

            risk_factors.append(
                "Frequent temporal changes detected"
            )

        if not risk_factors:

            risk_factors.append(
                "No significant maintenance risk factors detected"
            )

        return {

            "available": True,

            "risk_score":
                maintenance_risk_score,

            "risk_level":
                maintenance_risk,

            "degradation_trend":
                degradation_status,

            "historical_anomaly_rate":
                round(
                    historical_rate,
                    3
                ),

            "recent_anomaly_rate":
                round(
                    recent_rate,
                    3
                ),

            "change_in_anomaly_rate":
                round(
                    rate_change,
                    3
                ),

            "relative_change":
                round(
                    percentage_increase,
                    2
                ),

            "recent_component_rates": {

                "physical":
                    round(
                        recent_physical_rate,
                        3
                    ),

                "temporal":
                    round(
                        recent_temporal_rate,
                        3
                    ),

                "frozen":
                    round(
                        recent_frozen_rate,
                        3
                    ),

                "multivariate":
                    round(
                        recent_multivariate_rate,
                        3
                    )
            },

            "risk_factors":
                risk_factors,

            "recommendation":
                recommendation,

            "note":
                "Maintenance risk is an analytical indicator, "
                "not a guaranteed future failure prediction."
        }

    except Exception as error:

        return {

            "available": False,

            "risk_score": None,

            "risk_level": "Unavailable",

            "degradation_trend": "Unavailable",

            "historical_anomaly_rate": None,

            "recent_anomaly_rate": None,

            "change_in_anomaly_rate": None,

            "relative_change": None,

            "recent_component_rates": {},

            "risk_factors": [
                "Maintenance monitoring data could not be loaded."
            ],

            "recommendation":
                "Maintenance monitoring is temporarily unavailable.",

            "error":
                str(error)
        }


print("Loading maintenance monitoring...")

maintenance_monitoring = (
    calculate_maintenance_monitoring()
)

if maintenance_monitoring["available"]:

    print(
        "Maintenance monitoring loaded successfully."
    )

    print(
        "Maintenance risk:",
        maintenance_monitoring["risk_level"]
    )

    print(
        "Maintenance risk score:",
        maintenance_monitoring["risk_score"]
    )

else:

    print(
        "Maintenance monitoring could not be loaded."
    )


# =========================================================
# DATA MODELS
# =========================================================

class NeighborData(BaseModel):

    station_id: str

    temperature: float

    pressure: float

    humidity: float


class SensorData(BaseModel):

    station_id: str

    timestamp: str

    temperature: float

    pressure: float

    humidity: float

    wind_speed: float

    wind_direction: float

    neighbors: List[NeighborData] = Field(
        default_factory=list
    )


# =========================================================
# COMMUNICATION CHECK
# =========================================================

def check_communication_gap(
    current_timestamp,
    previous_timestamp
):

    if previous_timestamp is None:
        return False, None

    try:

        current_time = datetime.fromisoformat(
            current_timestamp.replace(
                "Z",
                "+00:00"
            )
        )

        previous_time = datetime.fromisoformat(
            previous_timestamp.replace(
                "Z",
                "+00:00"
            )
        )

        gap_minutes = (
            current_time - previous_time
        ).total_seconds() / 60

        if gap_minutes > 120:

            return True, gap_minutes

    except ValueError:

        return False, None

    return False, gap_minutes


# =========================================================
# SENSOR VALUE CORRECTION
# =========================================================

def calculate_sensor_correction(
    station_id,
    data,
    physical_anomaly,
    temporal_anomaly,
    frozen_anomaly,
    multivariate_anomaly
):
    """
    Estimate corrected values for anomalous sensor readings.

    The original readings are never modified.

    Real-time correction uses only previous observations,
    because future observations are unavailable during
    live monitoring.
    """

    correction_results = {}

    # IMPORTANT:
    # These histories contain ONLY observations that were
    # already received before the current reading.
    #
    # The current reading is added to the histories only
    # AFTER correction has been calculated.

    temperature_history = (
        station_temperature_history.get(
            station_id,
            []
        )
    )

    pressure_history = (
        station_pressure_history.get(
            station_id,
            []
        )
    )

    humidity_history = (
        station_humidity_history.get(
            station_id,
            []
        )
    )

    # -----------------------------------------------------
    # Temperature correction
    # -----------------------------------------------------

    temperature_needs_correction = (
        temporal_anomaly
        or frozen_anomaly
        or multivariate_anomaly
        or abs(data.temperature) > 45
    )

    if temperature_needs_correction:

        correction_results["temperature"] = (
            correct_realtime_value(
                previous_values=temperature_history,
                feature="temperature",
                current_value=data.temperature,
                window=5
            )
        )

    # -----------------------------------------------------
    # Pressure correction
    # -----------------------------------------------------

    pressure_needs_correction = (
        temporal_anomaly
        or multivariate_anomaly
    )

    if pressure_needs_correction:

        correction_results["pressure"] = (
            correct_realtime_value(
                previous_values=pressure_history,
                feature="pressure",
                current_value=data.pressure,
                window=5
            )
        )

    # -----------------------------------------------------
    # Humidity correction
    # -----------------------------------------------------

    humidity_needs_correction = (
        physical_anomaly
        or multivariate_anomaly
    )

    if humidity_needs_correction:

        correction_results["humidity"] = (
            correct_realtime_value(
                previous_values=humidity_history,
                feature="humidity",
                current_value=data.humidity,
                window=5
            )
        )

    return correction_results


# =========================================================
# UPDATE SENSOR HISTORIES
# =========================================================

def update_sensor_histories(
    station_id,
    data
):

    temperature_history = (
        station_temperature_history.setdefault(
            station_id,
            []
        )
    )

    pressure_history = (
        station_pressure_history.setdefault(
            station_id,
            []
        )
    )

    humidity_history = (
        station_humidity_history.setdefault(
            station_id,
            []
        )
    )

    temperature_history.append(
        data.temperature
    )

    pressure_history.append(
        data.pressure
    )

    humidity_history.append(
        data.humidity
    )

    # Keep the most recent 10 observations.
    if len(temperature_history) > 10:

        temperature_history.pop(0)

    if len(pressure_history) > 10:

        pressure_history.pop(0)

    if len(humidity_history) > 10:

        humidity_history.pop(0)


# =========================================================
# ANOMALY DETECTION
# =========================================================

def detect_anomaly(data):

    station_id = data.station_id

    previous = station_previous_data.get(
        station_id
    )

    previous_timestamp = (
        station_previous_timestamp.get(
            station_id
        )
    )

    (
        communication_anomaly,
        gap_minutes
    ) = check_communication_gap(
        data.timestamp,
        previous_timestamp
    )

    temperature_history = (
        station_temperature_history.setdefault(
            station_id,
            []
        )
    )

    physical_anomaly = False
    temporal_anomaly = False
    frozen_anomaly = False
    multivariate_anomaly = False

    reasons = []
    evidence_details = []

    # =====================================================
    # Communication
    # =====================================================

    if communication_anomaly:

        reasons.append(
            f"Communication gap detected: "
            f"{gap_minutes:.1f} minutes between observations."
        )

        evidence_details.append(
            "AWS communication/data stream interruption"
        )

    # =====================================================
    # Physical validation
    # =====================================================

    if (
        data.humidity < 0
        or data.humidity > 100
    ):

        physical_anomaly = True

        reasons.append(
            "Relative humidity is outside "
            f"the valid range: "
            f"{data.humidity:.1f}%"
        )

        evidence_details.append(
            "Physical range violation"
        )

    if data.wind_speed < 0:

        physical_anomaly = True

        reasons.append(
            f"Wind speed is invalid: "
            f"{data.wind_speed:.1f}"
        )

        evidence_details.append(
            "Physical range violation"
        )

    if (
        data.wind_direction < 0
        or data.wind_direction > 360
    ):

        physical_anomaly = True

        reasons.append(
            "Wind direction is outside "
            f"the valid range: "
            f"{data.wind_direction:.1f}°"
        )

        evidence_details.append(
            "Physical range violation"
        )

    # =====================================================
    # Temporal validation
    # =====================================================

    temperature_temporal_anomaly = False
    pressure_temporal_anomaly = False
    humidity_temporal_change = False

    if previous is not None:

        temp_change = abs(
            data.temperature
            - previous["temperature"]
        )

        pressure_change = abs(
            data.pressure
            - previous["pressure"]
        )

        humidity_change = abs(
            data.humidity
            - previous["humidity"]
        )

        if temp_change > 10:

            temperature_temporal_anomaly = True
            temporal_anomaly = True

            reasons.append(
                "Temperature changed suddenly "
                f"by {temp_change:.1f}°C."
            )

            evidence_details.append(
                "Temperature sudden change"
            )

        if pressure_change > 10:

            pressure_temporal_anomaly = True
            temporal_anomaly = True

            reasons.append(
                "Atmospheric pressure changed "
                f"suddenly by {pressure_change:.1f} hPa."
            )

            evidence_details.append(
                "Pressure sudden change"
            )

        if humidity_change > 20:

            humidity_temporal_change = True

            if (
                temperature_temporal_anomaly
                or pressure_temporal_anomaly
            ):

                reasons.append(
                    "Relative humidity changed "
                    f"suddenly by {humidity_change:.1f}%, "
                    "along with another major atmospheric change."
                )

                evidence_details.append(
                    "Humidity change supporting multivariate temporal evidence"
                )

            else:

                reasons.append(
                    "Relative humidity changed "
                    f"by {humidity_change:.1f}%, "
                    "but the change is not treated as a standalone "
                    "temporal anomaly."
                )

                evidence_details.append(
                    "Humidity change monitored without standalone alert"
                )

    # =====================================================
    # Frozen sensor
    # =====================================================

    # IMPORTANT:
    # Do NOT append the current reading to the shared
    # temperature history here.
    #
    # Instead, create a temporary sequence containing
    # previous readings + current reading.
    #
    # This allows frozen-sensor detection to inspect the
    # current observation while keeping the shared history
    # clean for the sensor correction module.

    frozen_temperature_history = (
        temperature_history
        +
        [data.temperature]
    )

    if len(frozen_temperature_history) > 8:

        frozen_temperature_history = (
            frozen_temperature_history[-8:]
        )

    if len(frozen_temperature_history) == 8:

        if len(
            set(frozen_temperature_history)
        ) == 1:

            frozen_anomaly = True

            reasons.append(
                "Temperature remained unchanged "
                "for 8 consecutive observations."
            )

            evidence_details.append(
                "Frozen sensor pattern"
            )

    # =====================================================
    # Multivariate consistency
    # =====================================================

    if (
        abs(data.temperature) > 45
        and data.humidity < 40
        and data.pressure < 960
    ):

        multivariate_anomaly = True

        reasons.append(
            "Temperature, pressure, and humidity "
            "are mutually inconsistent."
        )

        evidence_details.append(
            "Multivariate inconsistency"
        )

    if (
        data.temperature > 40
        and data.humidity > 90
    ):

        multivariate_anomaly = True

        reasons.append(
            "High temperature and very high humidity "
            "form an unusual combination."
        )

        evidence_details.append(
            "Temperature-humidity inconsistency"
        )

    if (
        abs(data.temperature) > 45
        and 960 <= data.pressure <= 1030
        and 20 <= data.humidity <= 80
    ):

        multivariate_anomaly = True

        reasons.append(
            "Temperature is extreme relative "
            "to the observed atmospheric conditions."
        )

        evidence_details.append(
            "Extreme temperature inconsistency"
        )

    # =====================================================
    # Seasonal analysis
    # =====================================================

    seasonal_result = detect_seasonal_anomaly(
        timestamp=data.timestamp,
        temperature=data.temperature,
        pressure=data.pressure,
        humidity=data.humidity
    )

    seasonal_anomaly = (
        seasonal_result["seasonal_anomaly"]
    )

    seasonal_score = (
        seasonal_result["seasonal_score"]
    )

    seasonal_month = (
        seasonal_result["month"]
    )

    seasonal_features = (
        seasonal_result["anomalous_features"]
    )

    seasonal_summary = (
        seasonal_result["summary"]
    )

    if seasonal_anomaly:

        evidence_details.append(
            "Seasonal pattern deviation"
        )

        reasons.append(
            seasonal_summary
        )

    # =====================================================
    # Spatial consistency
    # =====================================================

    neighbors = [
        neighbor.model_dump()
        for neighbor in data.neighbors
    ]

    spatial_result = check_spatial_consistency(
        station_temperature=data.temperature,
        station_pressure=data.pressure,
        station_humidity=data.humidity,
        neighbors=neighbors
    )

    spatial_anomaly = spatial_result.get(
        "spatial_anomaly",
        False
    )

    spatial_summary = spatial_result.get(
        "summary",
        "No neighboring station data available "
        "for spatial comparison."
    )

    spatial_features = spatial_result.get(
        "anomalous_features",
        []
    )

    neighbor_count = spatial_result.get(
        "neighbor_count",
        0
    )

    if spatial_anomaly:

        reasons.append(
            spatial_summary
        )

        evidence_details.append(
            "Spatial inconsistency with neighboring stations"
        )

    # =====================================================
    # Machine learning
    # =====================================================

    observation = pd.DataFrame(
        [[
            data.temperature,
            data.pressure,
            data.humidity
        ]],
        columns=FEATURES
    )

    ml_score = float(
        model.decision_function(
            observation
        )[0]
    )

    ml_anomaly = bool(
        model.predict(
            observation
        )[0] == -1
    )

    # =====================================================
    # Evidence engine
    # =====================================================

    evidence_result = calculate_evidence(

        physical_anomaly=physical_anomaly,

        temporal_anomaly=temporal_anomaly,

        frozen_anomaly=frozen_anomaly,

        multivariate_anomaly=multivariate_anomaly,

        communication_anomaly=communication_anomaly,

        seasonal_anomaly=seasonal_anomaly,

        spatial_anomaly=spatial_anomaly,

        ml_anomaly=ml_anomaly
    )

    evidence_score = (
        evidence_result["evidence_score"]
    )

    evidence_level = (
        evidence_result["evidence_level"]
    )

    evidence_engine_details = (
        evidence_result["evidence_details"]
    )

    evidence_interpretation = (
        evidence_result["interpretation"]
    )

    # =====================================================
    # Confidence
    # =====================================================

    confidence_score = 0

    if physical_anomaly:
        confidence_score += 40

    if temporal_anomaly:
        confidence_score += 20

    if frozen_anomaly:
        confidence_score += 30

    if multivariate_anomaly:
        confidence_score += 30

    if communication_anomaly:
        confidence_score += 40

    if seasonal_anomaly:
        confidence_score += 10

    if spatial_anomaly:
        confidence_score += 20

    if ml_anomaly:
        confidence_score += 20

    confidence_score = min(
        confidence_score,
        100
    )

    # =====================================================
    # Status
    # =====================================================

    if (
        physical_anomaly
        or frozen_anomaly
        or multivariate_anomaly
        or communication_anomaly
    ):

        status = "ANOMALY"

    elif (
        temporal_anomaly
        or spatial_anomaly
    ):

        status = "SUSPICIOUS"

    else:

        status = "NORMAL"

    # =====================================================
    # Anomaly type
    # =====================================================

    if communication_anomaly:

        anomaly_type = "Communication Failure"

    elif frozen_anomaly:

        anomaly_type = "Frozen Sensor"

    elif physical_anomaly:

        anomaly_type = "Physical/Range Error"

    elif multivariate_anomaly:

        anomaly_type = "Multivariate Inconsistency"

    elif spatial_anomaly:

        anomaly_type = "Spatial Inconsistency"

    elif temporal_anomaly:

        anomaly_type = "Sudden Change"

    elif ml_anomaly:

        anomaly_type = "Machine Learning Anomaly"

    else:

        anomaly_type = "Normal"

    # =====================================================
    # Severity
    # =====================================================

    if status == "ANOMALY":

        severity = (
            "HIGH"
            if evidence_score >= 70
            else "MEDIUM"
        )

    elif status == "SUSPICIOUS":

        severity = "LOW"

    else:

        severity = "NONE"

    # =====================================================
    # Root cause
    # =====================================================

    if communication_anomaly:

        root_cause = (
            "Possible AWS communication or "
            "data transmission failure"
        )

    elif physical_anomaly:

        root_cause = (
            "Sensor range or data validity problem"
        )

    elif frozen_anomaly:

        root_cause = (
            "Possible sensor communication "
            "or hardware failure"
        )

    elif multivariate_anomaly:

        root_cause = (
            "Possible sensor fault or "
            "inconsistent sensor readings"
        )

    elif spatial_anomaly:

        root_cause = (
            "Possible station-specific sensor "
            "or data quality problem"
        )

    elif temporal_anomaly:

        root_cause = (
            "Possible sudden sensor/data change"
        )

    else:

        root_cause = "None"

    # =====================================================
    # Event interpretation
    # =====================================================

    event_result = interpret_event(

        status=status,

        evidence_score=evidence_score,

        physical_anomaly=physical_anomaly,

        temporal_anomaly=temporal_anomaly,

        frozen_anomaly=frozen_anomaly,

        multivariate_anomaly=multivariate_anomaly,

        communication_anomaly=communication_anomaly,

        seasonal_anomaly=seasonal_anomaly,

        spatial_anomaly=spatial_anomaly,

        ml_anomaly=ml_anomaly
    )

    event_classification = (
        event_result["classification"]
    )

    event_interpretation = (
        event_result["interpretation"]
    )

    event_evidence_strength = (
        event_result["evidence_strength"]
    )

    event_supporting_evidence = (
        event_result["supporting_evidence"]
    )

    event_warning = (
        event_result["warning"]
    )

    # =====================================================
    # SHAP explanation
    # =====================================================

    shap_result = explain_reading(
        data.temperature,
        data.pressure,
        data.humidity
    )

    shap_top_feature = None
    shap_summary = None

    if shap_result:

        shap_top_feature = (
            shap_result[0]["feature_name"]
        )

        shap_summary = (
            f"{shap_top_feature} was the "
            "strongest contributing feature "
            "to the ML anomaly decision."
        )

    # =====================================================
    # SENSOR VALUE CORRECTION
    # =====================================================

    sensor_correction = {}

    if status != "NORMAL":

        sensor_correction = (
            calculate_sensor_correction(
                station_id=station_id,
                data=data,
                physical_anomaly=physical_anomaly,
                temporal_anomaly=temporal_anomaly,
                frozen_anomaly=frozen_anomaly,
                multivariate_anomaly=multivariate_anomaly
            )
        )

    # =====================================================
    # UPDATE SENSOR HISTORIES
    # =====================================================

    # IMPORTANT:
    # This happens AFTER sensor correction.
    #
    # Therefore the current anomalous reading cannot
    # contaminate its own corrected-value estimate.

    update_sensor_histories(
        station_id=station_id,
        data=data
    )

    # =====================================================
    # Store station history
    # =====================================================

    station_previous_timestamp[
        station_id
    ] = data.timestamp

    station_previous_data[
        station_id
    ] = {

        "temperature":
            data.temperature,

        "pressure":
            data.pressure,

        "humidity":
            data.humidity,

        "wind_speed":
            data.wind_speed,

        "wind_direction":
            data.wind_direction
    }

    # =====================================================
    # Spatial response
    # =====================================================

    spatial_neighbor_average = spatial_result.get(
        "neighbor_average",
        None
    )

    spatial_temperature_difference = spatial_result.get(
        "temperature_difference",
        None
    )

    spatial_pressure_difference = spatial_result.get(
        "pressure_difference",
        None
    )

    spatial_humidity_difference = spatial_result.get(
        "humidity_difference",
        None
    )

    spatial_details = spatial_result.get(
        "details",
        []
    )

    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return {

        "station_id":
            data.station_id,

        "timestamp":
            data.timestamp,

        "sensor_data": {

            "temperature":
                data.temperature,

            "pressure":
                data.pressure,

            "humidity":
                data.humidity,

            "wind_speed":
                data.wind_speed,

            "wind_direction":
                data.wind_direction
        },

        "status":
            status,

        "anomaly_type":
            anomaly_type,

        "severity":
            severity,

        "root_cause":
            root_cause,

        "event_interpretation": {

            "classification":
                event_classification,

            "interpretation":
                event_interpretation,

            "evidence_strength":
                event_evidence_strength,

            "evidence_score":
                evidence_score,

            "supporting_evidence":
                event_supporting_evidence,

            "warning":
                event_warning
        },

        "evidence_score":
            evidence_score,

        "evidence_level":
            evidence_level,

        "evidence_interpretation":
            evidence_interpretation,

        "evidence_engine_details":
            evidence_engine_details,

        "confidence_score":
            confidence_score,

        "ml_detection":
            ml_anomaly,

        "ml_score":
            ml_score,

        "communication_gap":
            communication_anomaly,

        "gap_minutes":
            gap_minutes,

        "seasonal_analysis": {

            "seasonal_anomaly":
                seasonal_anomaly,

            "seasonal_score":
                seasonal_score,

            "month":
                seasonal_month,

            "anomalous_features":
                seasonal_features,

            "summary":
                seasonal_summary
        },

        "spatial_analysis": {

            "spatial_anomaly":
                spatial_anomaly,

            "neighbor_count":
                neighbor_count,

            "anomalous_features":
                spatial_features,

            "summary":
                spatial_summary,

            "neighbor_average":
                spatial_neighbor_average,

            "temperature_difference":
                spatial_temperature_difference,

            "pressure_difference":
                spatial_pressure_difference,

            "humidity_difference":
                spatial_humidity_difference,

            "details":
                spatial_details
        },

        "maintenance_monitoring":
            maintenance_monitoring,

        "sensor_correction":
            sensor_correction,

        "shap_top_feature":
            shap_top_feature,

        "shap_summary":
            shap_summary,

        "reasons":
            reasons,

        "evidence_details":
            evidence_details
    }


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():

    return {

        "system":
            "SkyGuard AI",

        "status":
            "running",

        "message":
            "Intelligent AWS anomaly detection API",

        "maintenance_monitoring":
            maintenance_monitoring
    }


# =========================================================
# DETECTION ENDPOINT
# =========================================================

@app.post("/detect")
def detect(data: SensorData):

    return detect_anomaly(data)