"""
Digital Twin Safety Bridge Module - Phase 6 Step 6.7
Connects autonomous agent outputs back to the Virtual Engine State (Digital Twin)
under strict safety firewalls and thermodynamic invariance constraints.

Roadmap Reference: Phase 6 - Step 6.7:
"Connect agent outputs to the Digital Twin without allowing agents to bypass safety checks"

Core Safety Invariants:
1. Physical Telemetry Invariance:
   - Variables G1–G26 (measured operational settings and gas-path sensors) are strictly read-only and immutable.
   - Any attempt to alter measured telemetry raises SafetyViolationError.
2. Aerothermal Invariant Protection:
   - Variables G27–G29 (derived indicators: hpc_temp_rise, overall_pressure_ratio, spool_speed_ratio)
     must remain bit-for-bit identical before and after agent enrichment.
3. Thermodynamic Consistency Sanity:
   - Enforces check_thermodynamic_sanity(); agent feedback cannot degrade engine aerothermal order.
4. Permitted Agent Feedback Slots:
   - Agents are strictly restricted to updating:
     * G32 (anomaly_score_m1): Isolation Forest score
     * G33 (anomaly_score_m2): PCA-SPE residual energy
     * G34 (anomaly_alert_flag): Binary flag {0, 1} derived from persistence_alert
     * G35 (predicted_rul_xgboost): Point RUL >= 0.0
     * G36 (predicted_rul_lstm): Sequence RUL >= 0.0
     * G37 (prognostic_spread): Epistemic uncertainty |G35 - G36| >= 0.0
     * G38 (virtual_health_index): Normalized health index H(t) in [0.0, 1.0]
     * Non-destructive audit metadata in scenario_audit
5. Bounded Range Validation:
   - Rejects unphysical values (negative RUL, H(t) > 1.0, non-binary flags, NaN, Inf).
"""

import copy
from dataclasses import dataclass
import math
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np

from agents.schemas import (
    AgentExecutionStatus,
    MonitoringOutput,
    DiagnosisOutput,
    PrognosticsOutput,
    OrchestrationResult,
)
from digital_twin.engine_state import EngineState


class SafetyViolationError(Exception):
    """Raised when an agent output violates Digital Twin safety checks or attempts to alter immutable telemetry."""
    pass


# Strict list of immutable physical fields G1 - G29
IMMUTABLE_PHYSICAL_FIELDS: List[str] = [
    # Metadata & Operational Context (G1 - G5)
    "unit_number",
    "time_cycles",
    "altitude",
    "mach_number",
    "throttle_resolver_angle",
    # Gas-Path Thermodynamic Sensors (G6 - G26)
    "T2",
    "T24",
    "T30",
    "T50",
    "P2",
    "P15",
    "P30",
    "Nf",
    "Nc",
    "epr",
    "Ps30",
    "phi",
    "NRf",
    "NRc",
    "BPR",
    "farB",
    "htBleed",
    "Nf_dmd",
    "PCNfR_dmd",
    "W31",
    "W32",
    # Derived Macroscopic Consistency Indicators (G27 - G29)
    "hpc_temp_rise",
    "overall_pressure_ratio",
    "spool_speed_ratio",
]


class DigitalTwinSafetyBridge:
    """
    Safety Firewall Bridge connecting agent outputs back to the Digital Twin.
    Ensures that agent predictions and diagnostics enrich state slots G32–G38 without
    compromising physical sensor telemetry, macroscopic indicators, or thermodynamic sanity.
    """

    def __init__(self, agent_id: str = "twin_safety_bridge_v1"):
        """
        Initializes the Digital Twin Safety Bridge.

        Args:
            agent_id: Provenance identifier for metadata tracking.
        """
        self.agent_id = agent_id

    def validate_feedback_safety(
        self,
        original_state: EngineState,
        candidate_state: EngineState,
    ) -> None:
        """
        Strict Safety Firewall verifying that candidate state does not violate
        physical invariance, aerodynamic bounds, or thermodynamic sanity.

        Args:
            original_state: Baseline EngineState before enrichment.
            candidate_state: Candidate EngineState after agent slot updates.

        Raises:
            SafetyViolationError: If any immutable field was modified or thermodynamic checks fail.
        """
        # 1. Verify that all G1 - G29 immutable fields match bit-for-bit
        for field_name in IMMUTABLE_PHYSICAL_FIELDS:
            orig_val = getattr(original_state, field_name, None)
            cand_val = getattr(candidate_state, field_name, None)

            if orig_val is None or cand_val is None:
                raise SafetyViolationError(
                    f"Safety Firewall: Field '{field_name}' missing in state verification."
                )

            # Floating point exact equality check
            if orig_val != cand_val:
                raise SafetyViolationError(
                    f"CRITICAL SAFETY VIOLATION: Agent output attempted to modify immutable physical field "
                    f"'{field_name}'! Original: {orig_val}, Candidate: {cand_val}."
                )

        # 2. Verify Composite Preprocessing Representations (G30, G31) if present
        if original_state.normalized_feature_vec is not None:
            if candidate_state.normalized_feature_vec is None:
                raise SafetyViolationError("Safety Firewall: G30 (normalized_feature_vec) was cleared.")
            if not np.array_equal(original_state.normalized_feature_vec, candidate_state.normalized_feature_vec):
                raise SafetyViolationError("Safety Firewall: G30 (normalized_feature_vec) was mutated.")

        if original_state.temporal_seq_tensor is not None:
            if candidate_state.temporal_seq_tensor is None:
                raise SafetyViolationError("Safety Firewall: G31 (temporal_seq_tensor) was cleared.")
            if not np.array_equal(original_state.temporal_seq_tensor, candidate_state.temporal_seq_tensor):
                raise SafetyViolationError("Safety Firewall: G31 (temporal_seq_tensor) was mutated.")

        # 3. Verify Thermodynamic Sanity Invariants
        orig_sanity = original_state.check_thermodynamic_sanity()
        cand_sanity = candidate_state.check_thermodynamic_sanity()

        for check_name, passed in orig_sanity.items():
            if passed and not cand_sanity.get(check_name, False):
                raise SafetyViolationError(
                    f"CRITICAL SAFETY VIOLATION: Thermodynamic sanity check '{check_name}' failed "
                    "following agent feedback enrichment!"
                )

        # 4. Range Validation on Updated Agent Slots (G32 - G38)
        # G34: Anomaly Alert Flag {0, 1}
        flag = candidate_state.anomaly_alert_flag
        if flag is not None and flag not in (0, 1):
            raise SafetyViolationError(
                f"Safety Firewall: G34 (anomaly_alert_flag) must be binary 0 or 1, got {flag}."
            )

        # G35: XGBoost RUL >= 0.0
        rul_xgb = candidate_state.predicted_rul_xgboost
        if rul_xgb is not None:
            if not math.isfinite(rul_xgb) or rul_xgb < 0.0:
                raise SafetyViolationError(
                    f"Safety Firewall: G35 (predicted_rul_xgboost) must be finite non-negative, got {rul_xgb}."
                )

        # G36: LSTM RUL >= 0.0
        rul_lstm = candidate_state.predicted_rul_lstm
        if rul_lstm is not None:
            if not math.isfinite(rul_lstm) or rul_lstm < 0.0:
                raise SafetyViolationError(
                    f"Safety Firewall: G36 (predicted_rul_lstm) must be finite non-negative, got {rul_lstm}."
                )

        # G37: Prognostic Spread >= 0.0
        spread = candidate_state.prognostic_spread
        if spread is not None:
            if not math.isfinite(spread) or spread < 0.0:
                raise SafetyViolationError(
                    f"Safety Firewall: G37 (prognostic_spread) must be finite non-negative, got {spread}."
                )

        # G38: Virtual Health Index H(t) in [0.0, 1.0]
        h_idx = candidate_state.virtual_health_index
        if h_idx is not None:
            if not math.isfinite(h_idx) or not (0.0 <= h_idx <= 1.0):
                raise SafetyViolationError(
                    f"Safety Firewall: G38 (virtual_health_index) must be in [0.0, 1.0], got {h_idx}."
                )

    def enrich_state_with_agent_outputs(
        self,
        state: EngineState,
        orchestration_result: OrchestrationResult,
    ) -> EngineState:
        """
        Safely enriches an EngineState instance with verified agent outputs.

        Args:
            state: EngineState instance from DigitalTwinSimulator.
            orchestration_result: OrchestrationResult containing monitoring, diagnosis, and prognostics outputs.

        Returns:
            Enriched EngineState with populated G32-G38 slots and diagnostic metadata.

        Raises:
            SafetyViolationError: If agent feedback violates physical bounds or invariance rules.
            TypeError: If state or orchestration_result are not of expected types.
        """
        if not isinstance(state, EngineState):
            raise TypeError(f"Expected EngineState instance, got {type(state)}")

        if not isinstance(orchestration_result, OrchestrationResult):
            raise TypeError(f"Expected OrchestrationResult instance, got {type(orchestration_result)}")

        try:
            orchestration_result.validate()
        except (ValueError, TypeError) as err:
            raise SafetyViolationError(f"Safety Firewall: Invalid agent output: {str(err)}") from err

        # Create deep copy to protect original state during validation
        enriched = copy.deepcopy(state)

        # ---------------------------------------------------------------------
        # 1. Ingest Monitoring Agent Feedback (G34)
        # ---------------------------------------------------------------------
        mon = orchestration_result.monitoring_result
        if mon is not None:
            # G34: Anomaly Alert Flag (binary 0 or 1)
            enriched.anomaly_alert_flag = 1 if mon.persistence_alert else 0

        # ---------------------------------------------------------------------
        # 2. Ingest Prognostics Agent Feedback (G35, G36, G37, G38)
        # ---------------------------------------------------------------------
        prog = orchestration_result.prognostics_result
        if prog is not None:
            # G35: XGBoost RUL (>= 0.0)
            if prog.predicted_rul_xgboost < 0.0 or not math.isfinite(prog.predicted_rul_xgboost):
                raise SafetyViolationError(
                    f"Agent feedback contains invalid XGBoost RUL: {prog.predicted_rul_xgboost}"
                )
            enriched.predicted_rul_xgboost = float(prog.predicted_rul_xgboost)

            # G36: LSTM RUL (>= 0.0)
            if prog.predicted_rul_lstm < 0.0 or not math.isfinite(prog.predicted_rul_lstm):
                raise SafetyViolationError(
                    f"Agent feedback contains invalid LSTM RUL: {prog.predicted_rul_lstm}"
                )
            enriched.predicted_rul_lstm = float(prog.predicted_rul_lstm)

            # G37: Prognostic Spread (>= 0.0)
            if prog.prognostic_spread < 0.0 or not math.isfinite(prog.prognostic_spread):
                raise SafetyViolationError(
                    f"Agent feedback contains invalid prognostic spread: {prog.prognostic_spread}"
                )
            enriched.prognostic_spread = float(prog.prognostic_spread)

            # G38: Virtual Health Index H(t) in [0.0, 1.0]
            if not (0.0 <= prog.virtual_health_index <= 1.0) or not math.isfinite(prog.virtual_health_index):
                raise SafetyViolationError(
                    f"Agent feedback contains invalid virtual health index: {prog.virtual_health_index}"
                )
            enriched.virtual_health_index = float(prog.virtual_health_index)

        # ---------------------------------------------------------------------
        # 3. Ingest Diagnosis Agent Metadata into Audit Trail (Non-destructive)
        # ---------------------------------------------------------------------
        diag = orchestration_result.diagnosis_result
        audit_dict = enriched.scenario_audit or {}
        audit_dict = dict(audit_dict)

        agent_feedback_record = {
            "timestamp": orchestration_result.metadata.execution_timestamp if orchestration_result.metadata else None,
            "run_id": orchestration_result.run_id,
            "pipeline_status": orchestration_result.status.value,
        }

        if diag is not None:
            agent_feedback_record["diagnosis"] = {
                "primary_fault": diag.primary_fault,
                "confidence_score": float(diag.confidence_score),
                "severity": diag.severity.value,
                "requires_maintenance_action": diag.requires_maintenance_action,
                "diagnostic_summary": diag.diagnostic_summary,
                "candidates_count": len(diag.fault_candidates),
            }

        if mon is not None:
            agent_feedback_record["monitoring"] = {
                "operational_state": mon.operational_state.value,
                "persistence_count": mon.persistence_count,
                "persistence_alert": mon.persistence_alert,
                "threshold_breaches_count": len(mon.threshold_breaches),
            }

        if prog is not None:
            agent_feedback_record["prognostics"] = {
                "ensemble_rul": float(prog.ensemble_rul_estimate),
                "uncertainty_tier": prog.uncertainty_classification,
                "cycles_to_caution": prog.cycles_to_caution_threshold,
            }

        audit_dict["agent_feedback"] = agent_feedback_record
        enriched.scenario_audit = audit_dict

        # ---------------------------------------------------------------------
        # 4. Strict Safety Firewall Validation
        # ---------------------------------------------------------------------
        self.validate_feedback_safety(original_state=state, candidate_state=enriched)

        return enriched
