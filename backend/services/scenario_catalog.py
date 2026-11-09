"""
Sensor Scenario Catalog & Test Execution Service
Phase 9 - Step 9.5: "Test normal and abnormal sensor scenarios using documented test cases"
Roadmap Reference: Master Project Roadmap PDF, Section 10, Step 9.5 & Section 1, Page 2

Provides formal definitions, test fixture generators, and verification runners for:
1. Scenario 1 (Nominal Healthy Cruise): Early-life flight cycles (FD001 Unit 1, Cycles 1-15).
2. Scenario 2 (Severe Thermal Degradation): Late-life cycles (FD001 Unit 1, Cycles 185-192, T30 > 1618 °R).
3. Scenario 3 (Isolated Transient Noise Spike): Single-cycle sensor anomaly spike at Cycle 25.
4. Scenario 4 (Sustained Multi-Cycle Degradation): 3 consecutive anomalous cycles (Cycles 188-190).
5. Scenario 5 (Multi-Regime Envelope Invariance): Normal vs abnormal detection in FD002 under dynamic flight envelopes.
6. Scenario 6 (Compound Fault Degradation): Coupled HPC and Fan spool degradation in FD003.

Enforces clear separation between statistical anomaly detection and mechanical fault claims,
providing itemized observational evidence and academic simulation disclaimers.
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.schemas.api_schemas import AnomalyCheckRequest, AnomalyCheckResponse
from backend.api.monitoring import check_anomaly
from backend.services.raw_record_auditor import get_raw_record, SENSOR_NAME_MAP
from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER

DEFAULT_DATA_DIR = os.path.join("dataset", "raw", "C-MAPSS")

SENSOR_UNITS = {
    "T2": "°R", "T24": "°R", "T30": "°R", "T50": "°R",
    "P2": "psia", "P15": "psia", "P30": "psia", "Ps30": "psia",
    "Nf": "rpm", "Nc": "rpm", "NRf": "rpm", "NRc": "rpm",
    "epr": "--", "phi": "pps/psia", "BPR": "--", "farB": "--",
    "htBleed": "BTU/lbm", "W31": "lbm/s", "W32": "lbm/s",
}

NOMINAL_BASELINES_FD001 = {
    "T2": 518.67, "T24": 642.35, "T30": 1585.00, "T50": 1400.00,
    "P2": 14.62, "P15": 21.61, "P30": 553.50, "Ps30": 47.45,
    "Nf": 2388.05, "Nc": 9045.00, "epr": 1.30, "BPR": 8.42,
}

SCENARIO_DEFINITIONS = {
    "scenario_1_nominal_cruise": {
        "id": "scenario_1_nominal_cruise",
        "title": "Scenario 1: Nominal Healthy Cruise",
        "subset": "FD001",
        "split": "train",
        "unit": 1,
        "cycle": 5,
        "description": "Early-life cruise telemetry with baseline gas-path temperatures and pressures.",
        "expected_is_anomaly": False,
        "expected_severity": "NOMINAL",
        "expected_persistence_alarm": False,
        "expected_consecutive_breaches": 0,
    },
    "scenario_2_severe_thermal": {
        "id": "scenario_2_severe_thermal",
        "title": "Scenario 2: Severe Thermal Degradation",
        "subset": "FD001",
        "split": "train",
        "unit": 1,
        "cycle": 192,
        "description": "Late-life operational cycle exhibiting significant HPC thermal creep (T30 > 1618 °R).",
        "expected_is_anomaly": True,
        "expected_severity": "CRITICAL",
        "expected_persistence_alarm": True,
        "expected_consecutive_breaches": 3,
    },
    "scenario_3_transient_noise": {
        "id": "scenario_3_transient_noise",
        "title": "Scenario 3: Isolated Transient Noise Spike",
        "subset": "FD001",
        "split": "train",
        "unit": 1,
        "cycle": 25,
        "description": "Single-cycle sensor anomaly spike surrounded by nominal cruise cycles. Suppressed by persistence filter.",
        "expected_is_anomaly": True,
        "expected_severity": "CAUTION",
        "expected_persistence_alarm": False,
        "expected_consecutive_breaches": 1,
    },
    "scenario_4_sustained_degradation": {
        "id": "scenario_4_sustained_degradation",
        "title": "Scenario 4: Sustained Multi-Cycle Degradation",
        "subset": "FD001",
        "split": "train",
        "unit": 1,
        "cycle": 190,
        "description": "3 consecutive anomalous cycles tripping the temporal persistence filter and locking the alarm.",
        "expected_is_anomaly": True,
        "expected_severity": "CRITICAL",
        "expected_persistence_alarm": True,
        "expected_consecutive_breaches": 3,
    },
    "scenario_5a_multi_regime_nominal": {
        "id": "scenario_5a_multi_regime_nominal",
        "title": "Scenario 5A: Multi-Regime Nominal Envelope Invariance",
        "subset": "FD002",
        "split": "test",
        "unit": 15,
        "cycle": 5,
        "description": "Nominal flight operating across varied altitude and Mach conditions without triggering false alarms.",
        "expected_is_anomaly": False,
        "expected_severity": "NOMINAL",
        "expected_persistence_alarm": False,
        "expected_consecutive_breaches": 0,
    },
    "scenario_5b_multi_regime_abnormal": {
        "id": "scenario_5b_multi_regime_abnormal",
        "title": "Scenario 5B: Multi-Regime Abnormal Stress Detection",
        "subset": "FD002",
        "split": "test",
        "unit": 15,
        "cycle": 165,
        "description": "Abnormal operational cycle in complex multi-regime envelope showing advanced gas-path distress.",
        "expected_is_anomaly": True,
        "expected_severity": "CRITICAL",
        "expected_persistence_alarm": True,
        "expected_consecutive_breaches": 3,
    },
    "scenario_6_compound_fault_fd003": {
        "id": "scenario_6_compound_fault_fd003",
        "title": "Scenario 6: Compound HPC + Fan Spool Degradation",
        "subset": "FD003",
        "split": "test",
        "unit": 8,
        "cycle": 192,
        "description": "Simultaneous high-pressure compressor thermal degradation and low-pressure fan spool deviation in FD003.",
        "expected_is_anomaly": True,
        "expected_severity": "CRITICAL",
        "expected_persistence_alarm": True,
        "expected_consecutive_breaches": 3,
    },
}


def get_scenario_definition(scenario_id: str) -> Dict[str, Any]:
    """Retrieves metadata definition for a documented scenario."""
    if scenario_id not in SCENARIO_DEFINITIONS:
        raise KeyError(f"Unknown scenario ID '{scenario_id}'. Available: {list(SCENARIO_DEFINITIONS.keys())}")
    return SCENARIO_DEFINITIONS[scenario_id]


def generate_scenario_fixture(
    scenario_id: str, data_dir: Optional[str] = None
) -> Tuple[Dict[str, Any], AnomalyCheckRequest]:
    """
    Generates test fixture telemetry and the corresponding AnomalyCheckRequest for a scenario.
    """
    defn = get_scenario_definition(scenario_id)
    base_data = data_dir or DEFAULT_DATA_DIR

    # Load authentic raw record
    raw_record = get_raw_record(
        subset=defn["subset"],
        split=defn["split"],
        unit=defn["unit"],
        cycle=defn["cycle"],
        data_dir=base_data,
    )
    telemetry = dict(raw_record)

    # Apply scenario-specific modifications if required
    req_anomaly_scores = None
    consecutive_breaches = defn["expected_consecutive_breaches"]

    if scenario_id == "scenario_1_nominal_cruise":
        # Pure nominal baseline
        telemetry["T30"] = 1582.85
        telemetry["P30"] = 554.00

    elif scenario_id == "scenario_2_severe_thermal":
        # Elevated thermal creep T30 > 1618 °R
        telemetry["T30"] = 1621.50
        telemetry["sensor_3"] = 1621.50
        telemetry["P30"] = 550.80

    elif scenario_id == "scenario_3_transient_noise":
        # Single cycle spike at Cycle 25: high T30 but consecutive_breaches = 1
        telemetry["T30"] = 1625.00
        telemetry["sensor_3"] = 1625.00
        consecutive_breaches = 1

    elif scenario_id == "scenario_4_sustained_degradation":
        # 3 consecutive anomalous cycles: T30 elevated and consecutive_breaches = 3
        telemetry["T30"] = 1620.00
        telemetry["sensor_3"] = 1620.00
        consecutive_breaches = 3

    elif scenario_id == "scenario_5a_multi_regime_nominal":
        # FD002 Unit 15 Cycle 5 nominal
        telemetry["T30"] = 1260.51

    elif scenario_id == "scenario_5b_multi_regime_abnormal":
        # FD002 Unit 15 Cycle 165 late cycle
        telemetry["T30"] = 1615.00
        telemetry["sensor_3"] = 1615.00
        consecutive_breaches = 3

    elif scenario_id == "scenario_6_compound_fault_fd003":
        # FD003 Unit 8 Cycle 192 compound fault
        telemetry["T30"] = 1618.50
        telemetry["sensor_3"] = 1618.50
        telemetry["Nf"] = 2392.50
        consecutive_breaches = 3

    req = AnomalyCheckRequest(
        subset=defn["subset"],
        unit_number=defn["unit"],
        cycle=defn["cycle"],
        split=defn["split"],
        sensor_readings=telemetry,
        anomaly_scores=req_anomaly_scores,
        consecutive_breaches=consecutive_breaches,
    )

    return telemetry, req


def synthesize_diagnosis_evidence(
    telemetry: Dict[str, Any],
    subset: str = "FD001",
    is_anomaly: bool = False,
    severity: str = "NOMINAL",
) -> List[Dict[str, Any]]:
    """
    Synthesizes itemized observational evidence for anomaly diagnosis.
    Strictly qualifies evidence as statistical outlier observations rather than verified hardware teardowns.
    """
    evidence_list: List[Dict[str, Any]] = []
    baselines = NOMINAL_BASELINES_FD001

    if not is_anomaly:
        return evidence_list

    # Check key indicative channels
    candidate_channels = ["T30", "T50", "T24", "P30", "Ps30", "Nf", "Nc"]

    for ch in candidate_channels:
        if ch in telemetry:
            obs_val = float(telemetry[ch])
            base_val = baselines.get(ch, obs_val)
            delta = obs_val - base_val
            pct_delta = abs(delta) / max(1.0, abs(base_val))

            # If channel deviates meaningfully (> 1.0% or temperature > 15 °R)
            if abs(delta) >= 5.0 or pct_delta >= 0.01:
                direction = "HIGH" if delta > 0 else "LOW"
                unit = SENSOR_UNITS.get(ch, "--")
                evidence_list.append({
                    "sensor_name": ch,
                    "channel_name": f"{ch} ({'HPC Exit Temp' if ch=='T30' else ch})",
                    "observed_value": round(obs_val, 2),
                    "baseline_value": round(base_val, 2),
                    "delta": round(delta, 2),
                    "units": unit,
                    "direction": direction,
                    "severity": severity,
                    "statistical_qualification": (
                        f"Statistical telemetry outlier: {ch} is {abs(delta):.2f} {unit} {direction} relative to nominal baseline. "
                        "Observational statistical evidence only; not a teardown-certified mechanical fault."
                    ),
                })

    # Sort evidence by absolute delta descending
    evidence_list.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return evidence_list


def execute_scenario_test(
    scenario_id: str,
    custom_telemetry: Optional[Dict[str, Any]] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes end-to-end monitoring and diagnosis verification for a documented scenario.
    """
    defn = get_scenario_definition(scenario_id)
    telemetry, req = generate_scenario_fixture(scenario_id, data_dir=data_dir)

    if custom_telemetry is not None:
        telemetry.update(custom_telemetry)
        req.sensor_readings = telemetry

    # 1. Execute anomaly check via monitoring router
    api_resp: AnomalyCheckResponse = check_anomaly(req)

    # 2. Synthesize observational diagnostic evidence
    evidence = synthesize_diagnosis_evidence(
        telemetry=telemetry,
        subset=defn["subset"],
        is_anomaly=api_resp.is_anomaly,
        severity=api_resp.severity,
    )

    # 3. Assemble validated execution record
    execution_result = {
        "scenario_id": scenario_id,
        "title": defn["title"],
        "subset": defn["subset"],
        "unit": defn["unit"],
        "cycle": defn["cycle"],
        "is_anomaly": api_resp.is_anomaly,
        "severity": api_resp.severity,
        "isolation_forest_breached": api_resp.isolation_forest_breached,
        "pca_spe_breached": api_resp.pca_spe_breached,
        "consecutive_breaches": api_resp.consecutive_breaches,
        "persistence_alarm": api_resp.persistence_alarm,
        "summary": api_resp.summary,
        "evidence_count": len(evidence),
        "observational_evidence": evidence,
        "non_authority_disclaimer": ACADEMIC_SIMULATION_DISCLAIMER,
        "pass_criteria_met": (
            api_resp.is_anomaly == defn["expected_is_anomaly"] and
            api_resp.severity == defn["expected_severity"] and
            api_resp.persistence_alarm == defn["expected_persistence_alarm"]
        ),
    }

    return execution_result
