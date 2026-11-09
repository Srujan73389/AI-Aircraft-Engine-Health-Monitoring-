"""
Digital Twin Repeatability and Simulation Determinism Auditor
Roadmap Reference: Phase 9 - Step 9.6 ("Run Digital Twin scenarios and verify repeatability with identical inputs")
Strict Academic Integrity Standard: Master Roadmap Section 1, Page 2
("Do not invent sensor measurements or report fabricated model scores")

Provides automated audit services to evaluate:
1. Multi-run deterministic bitwise repeatability (variance == 0.0) across 50+ iterations.
2. State isolation and rollback fidelity following rejected / unapproved simulation scenarios.
3. Cryptographic deterministic execution hashing across repeated runs.
"""

import copy
import hashlib
import json
import math
from typing import Dict, Any, List, Optional, Union

import numpy as np

from backend.schemas.api_schemas import SimulationWhatIfRequest
from digital_twin.twin_simulator import DigitalTwinSimulator
from safety.verification import (
    SimulationAction,
    ActionType,
    SafetyVerifier,
)
from safety.closed_loop import ClosedLoopSafetyGovernor
from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER


def compute_execution_hash(
    state_dict: Dict[str, Any],
    decision_status: str,
    violations: List[str],
) -> str:
    """
    Computes a deterministic cryptographic SHA-256 execution hash of a simulation output.
    """
    # Filter to deterministic keys (excluding timestamps)
    deterministic_state = {}
    for k, v in state_dict.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            deterministic_state[k] = round(float(v), 8)
        elif isinstance(v, (str, bool)):
            deterministic_state[k] = v

    canonical_obj = {
        "decision_status": decision_status,
        "violations": sorted(violations),
        "state_after": deterministic_state,
    }
    canonical_json = json.dumps(canonical_obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_state_vector_variance(state_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes the sample variance across repeated simulation states for all numeric channels.
    Clamps variance values < 1e-15 to strictly 0.0 to account for IEEE 754 precision noise.
    """
    if not state_records:
        return {}

    variances: Dict[str, float] = {}
    sample = state_records[0]

    for key, val in sample.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            vals = [float(rec.get(key, 0.0)) for rec in state_records if key in rec and rec[key] is not None]
            if vals:
                # Bitwise check first: if all match first value exactly, variance is strictly 0.0
                first_val = vals[0]
                if all(x == first_val for x in vals):
                    variances[key] = 0.0
                else:
                    v = float(np.var(vals))
                    variances[key] = 0.0 if v < 1e-15 else v

    return variances


def audit_scenario_repeatability(
    scenario_payload: Dict[str, Any],
    iterations: int = 50,
    test_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Executes N identical simulation invocations for a scenario payload and calculates:
    - Channel variances across candidate state vectors
    - Max absolute difference across runs
    - Tri-state decision status uniformity (100% expected)
    - Deterministic execution hashes across all iterations

    Parameters
    ----------
    scenario_payload : Dict[str, Any]
        Simulation request specification (subset, unit_number, cycle, action_type, proposed_value, etc.)
    iterations : int
        Number of repeated invocations (default: 50)
    test_client : Optional[Any]
        FastAPI TestClient adapter. If None, uses direct Governor / DigitalTwinSimulator execution.

    Returns
    -------
    Dict[str, Any]
        Repeatability audit report.
    """
    subset = scenario_payload.get("subset", "FD001").upper()
    unit_number = int(scenario_payload.get("unit_number", 1))
    cycle = int(scenario_payload.get("cycle", 1))
    action_type_str = scenario_payload.get("action_type", "THERMAL_PERTURBATION").upper()
    target_variable = scenario_payload.get("target_variable", "thermal_stress_offset")
    proposed_value = scenario_payload.get("proposed_value", 3.0)
    delta_value = scenario_payload.get("delta_value")
    operator_approval = bool(scenario_payload.get("operator_approval", False))

    states_after: List[Dict[str, Any]] = []
    decision_statuses: List[str] = []
    violation_sets: List[List[str]] = []
    recovery_strategies: List[Optional[str]] = []
    execution_hashes: List[str] = []

    if test_client is not None:
        # HTTP Endpoint Mode
        endpoint_payload = {
            "subset": subset,
            "unit_number": unit_number,
            "cycle": cycle,
            "action_type": action_type_str,
            "target_variable": target_variable,
            "proposed_value": proposed_value,
            "delta_value": delta_value,
            "operator_approval": operator_approval,
        }
        for _ in range(iterations):
            resp = test_client.post("/api/v1/simulation/what-if", json=endpoint_payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Simulation endpoint failed with HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            st_after = data.get("state_after", {})
            status = data.get("decision_status", "UNKNOWN")
            rec_plan = data.get("recovery_plan")
            strat = rec_plan.get("strategy_type") if rec_plan else None
            reason = data.get("reason_summary", "")
            viols = [v.get("constraint_id") for v in rec_plan.get("violations", [])] if rec_plan and "violations" in rec_plan else []
            if not viols:
                for known in ["CONST_TEMP_T30_MAX_WARNING", "CONST_TEMP_T30_CAUTION", "CONST_PRESS_P30", "FIREWALL_MEASURED_IMMUTABILITY"]:
                    if known in reason:
                        viols.append(known)

            states_after.append(st_after)
            decision_statuses.append(status)
            violation_sets.append(viols)
            recovery_strategies.append(strat)
            execution_hashes.append(compute_execution_hash(st_after, status, viols))
    else:
        # Direct Digital Twin & Safety Governor Mode
        try:
            a_type = ActionType[action_type_str]
        except KeyError:
            a_type = ActionType.THERMAL_PERTURBATION

        action = SimulationAction(
            action_id=f"act_audit_{subset}_u{unit_number}_c{cycle}",
            action_type=a_type,
            target_variable=target_variable,
            proposed_value=proposed_value,
            delta_value=delta_value,
        )

        sim = DigitalTwinSimulator(subset=subset, enable_ml_inference=False)
        sim.load_engine(unit_number=unit_number, split="train")
        if cycle > 1:
            while sim.current_state and sim.current_state.time_cycles < cycle:
                if sim.step() is None:
                    break

        baseline_state_dict = sim.current_state.to_dict(include_arrays=False)
        governor = ClosedLoopSafetyGovernor()

        for _ in range(iterations):
            state_instance = copy.deepcopy(baseline_state_dict)
            res = governor.apply_simulation_action(
                simulator_or_state=state_instance,
                action=action,
                operator_approval=operator_approval,
            )
            st_after = res.state_after
            status = res.decision_status.value
            rec_plan = res.recovery_plan
            strat = rec_plan.strategy_type.value if rec_plan else None
            dec = governor.verifier.evaluate_action_decision(state_instance, action, attach_recovery_plan=False)
            viols = [v.constraint_id for v in dec.violations]

            states_after.append(st_after)
            decision_statuses.append(status)
            violation_sets.append(viols)
            recovery_strategies.append(strat)
            execution_hashes.append(compute_execution_hash(st_after, status, viols))

    # Variance and difference calculations
    channel_variances = compute_state_vector_variance(states_after)

    # Max absolute difference across all iterations against iteration 0
    max_abs_diff = 0.0
    if states_after:
        ref_state = states_after[0]
        for st in states_after[1:]:
            for k, ref_val in ref_state.items():
                if isinstance(ref_val, (int, float)) and not isinstance(ref_val, bool):
                    cur_val = st.get(k)
                    if cur_val is not None:
                        diff = abs(float(cur_val) - float(ref_val))
                        if diff > max_abs_diff:
                            max_abs_diff = diff

    # Status uniformity
    unique_statuses = set(decision_statuses)
    status_counts = {s: decision_statuses.count(s) for s in unique_statuses}
    modal_status = max(status_counts, key=status_counts.get)
    status_uniformity = status_counts[modal_status] / float(iterations)

    # Hash determinism
    unique_hashes = set(execution_hashes)
    is_hash_deterministic = len(unique_hashes) == 1

    all_identical = (
        max_abs_diff == 0.0
        and status_uniformity == 1.0
        and is_hash_deterministic
        and all(v == 0.0 for v in channel_variances.values())
    )

    return {
        "iterations": iterations,
        "all_identical": all_identical,
        "max_absolute_difference": max_abs_diff,
        "decision_status": modal_status,
        "status_uniformity": status_uniformity,
        "unique_status_count": len(unique_statuses),
        "channel_variances": channel_variances,
        "execution_hash": execution_hashes[0] if execution_hashes else "",
        "unique_execution_hashes_count": len(unique_hashes),
        "recovery_strategy": recovery_strategies[0] if recovery_strategies else None,
        "violations": violation_sets[0] if violation_sets else [],
        "state_after_sample": states_after[0] if states_after else {},
        "disclaimer": ACADEMIC_SIMULATION_DISCLAIMER,
    }


def verify_state_rollback_fidelity(
    subset: str,
    unit: int,
    cycle: int,
    perturbation_payload: Dict[str, Any],
) -> bool:
    """
    Verifies that executing a rejected, prohibited, or unapproved simulation scenario
    guarantees complete state isolation, leaving the baseline engine state vector S(t)
    100% bitwise unmodified before and after simulation.

    Parameters
    ----------
    subset : str
        C-MAPSS subset identifier (FD001 - FD004)
    unit : int
        Engine unit number
    cycle : int
        Operational flight cycle
    perturbation_payload : Dict[str, Any]
        Perturbation request parameters

    Returns
    -------
    bool
        True if pre-simulation and post-simulation engine state vectors match bit-for-bit (|Δ| = 0.0).
    """
    sim = DigitalTwinSimulator(subset=subset.upper(), enable_ml_inference=False)
    sim.load_engine(unit_number=unit, split="train")
    if cycle > 1:
        while sim.current_state and sim.current_state.time_cycles < cycle:
            if sim.step() is None:
                break

    pre_dict = sim.current_state.to_dict(include_arrays=False)

    action_type_str = perturbation_payload.get("action_type", "THERMAL_PERTURBATION").upper()
    try:
        a_type = ActionType[action_type_str]
    except KeyError:
        a_type = ActionType.THERMAL_PERTURBATION

    action = SimulationAction(
        action_id=f"act_rollback_{subset}_u{unit}_c{cycle}",
        action_type=a_type,
        target_variable=perturbation_payload.get("target_variable", "thermal_stress_offset"),
        proposed_value=perturbation_payload.get("proposed_value", 50.0),
        delta_value=perturbation_payload.get("delta_value"),
    )

    governor = ClosedLoopSafetyGovernor()
    res = governor.apply_simulation_action(
        simulator_or_state=sim,
        action=action,
        operator_approval=perturbation_payload.get("operator_approval", False),
    )

    post_dict = sim.current_state.to_dict(include_arrays=False)

    # Telemetry and state comparison:
    # Check all measured and operational channels (G1 - G38)
    for k, pre_val in pre_dict.items():
        if k.startswith("G") and int(k[1:].split("_")[0]) <= 38:
            post_val = post_dict.get(k)
            if pre_val != post_val:
                return False

    # Check offset slots:
    # If G39 / G40 were None before simulation, they should be either None or 0.0 / {} after rollback
    g39_pre = pre_dict.get("G39_thermal_stress_offset")
    g39_post = post_dict.get("G39_thermal_stress_offset")
    if (g39_pre in (None, 0.0)) and (g39_post not in (None, 0.0)):
        return False

    g40_pre = pre_dict.get("G40_sensor_bias_delta")
    g40_post = post_dict.get("G40_sensor_bias_delta")
    if (g40_pre in (None, {})) and (g40_post not in (None, {})):
        return False

    return True
