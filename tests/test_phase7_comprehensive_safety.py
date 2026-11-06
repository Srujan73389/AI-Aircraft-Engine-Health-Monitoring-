"""
Phase 7 Step 7.9: Master End-to-End Comprehensive Safety Test Suite
Consolidates and validates all Phase 7 safety verification mechanisms across:
- Category A: Allowed States (Nominal Operations)
- Category B: Boundary Breaches (Advisory Caution & Review States)
- Category C: Rejected Operations (Hard Limits, Thermodynamic Invariants & Firewall Violations)
- Cryptographic Immutability & Audit Trail Integrity

Roadmap Reference: Phase 7 - Step 7.9:
"Document safety test cases covering allowed states, boundary breaches and rejected operations"
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from typing import Dict, Any

from safety.constraints import (
    ConstraintSeverity,
    OPERATING_CONSTRAINTS,
    evaluate_constraints,
)
from safety.taxonomy import (
    VariableCategory,
    VARIABLE_TAXONOMY,
    MEASURED_VARIABLE_NAMES,
    classify_state_variables,
    verify_provenance_immutability,
)
from safety.verification import (
    ActionType,
    SimulationAction,
    SafetyVerifier,
    evaluate_simulation_decision,
)
from safety.decision import (
    SafetyDecisionStatus,
    ActionSafetyDecision,
)
from safety.audit import (
    SafetyReasonCode,
    SafetyAuditLogger,
    SafetyAuditRecord,
    infer_reason_code,
)
from safety.recovery import (
    RecoveryStrategyType,
    RecoveryPriority,
    SimulatedRecoveryPlan,
    generate_recovery_recommendation,
)
from safety.disclaimers import (
    ACADEMIC_SIMULATION_DISCLAIMER,
    CommandAuthorityViolationError,
    assert_non_commanding_recommendation,
)
from safety.closed_loop import (
    ClosedLoopSafetyGovernor,
    ClosedLoopUpdateResult,
    safe_simulation_step,
)
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestPhase7ComprehensiveSafety(unittest.TestCase):
    """
    Consolidated Phase 7 Comprehensive Safety Test Suite.
    Validates end-to-end integration of constraints, taxonomy, verification,
    decision classification, audit logging, simulated recovery, non-authority guarantees,
    and closed-loop transactional governor.
    """

    @classmethod
    def setUpClass(cls):
        cls.raw_data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.checkpoint_dir = os.path.join("models", "checkpoints")

        # 1. Capture SHA-256 hashes of all 12 raw C-MAPSS dataset files
        cls.raw_hashes = {}
        if os.path.exists(cls.raw_data_dir):
            for fname in sorted(os.listdir(cls.raw_data_dir)):
                if fname.endswith(".txt") and fname != "readme.txt":
                    fpath = os.path.join(cls.raw_data_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.raw_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 2. Capture SHA-256 hashes of all 4 subset scalers
        cls.scaler_hashes = {}
        if os.path.exists(cls.scalers_dir):
            for fname in sorted(os.listdir(cls.scalers_dir)):
                if fname.endswith(".pkl"):
                    fpath = os.path.join(cls.scalers_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.scaler_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 3. Capture SHA-256 hashes of all 8 champion model checkpoints
        cls.checkpoint_hashes = {}
        if os.path.exists(cls.checkpoint_dir):
            for fname in sorted(os.listdir(cls.checkpoint_dir)):
                if fname.endswith((".json", ".pt")) and not fname.endswith("manifest.json"):
                    fpath = os.path.join(cls.checkpoint_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.checkpoint_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

    def setUp(self):
        """Initialize temporary directory, fresh audit logger, verifier, and governor."""
        self.test_dir = tempfile.mkdtemp(prefix="test_phase7_comprehensive_")
        self.audit_logger = SafetyAuditLogger(
            log_dir=self.test_dir,
            log_filename="phase7_master_audit.jsonl",
        )
        self.verifier = SafetyVerifier(
            strict_firewall=True,
            audit_logger=self.audit_logger,
        )
        self.governor = ClosedLoopSafetyGovernor(verifier=self.verifier)

        # Initialize Digital Twin Simulator with FD001 unit 1
        self.sim = DigitalTwinSimulator(subset="FD001")
        self.sim.load_engine(unit_number=1, split="train")
        self.nominal_state = self.sim.current_state

    def tearDown(self):
        """Clean up temporary test directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # =========================================================================
    # Category A: Allowed States (Nominal Operations)
    # =========================================================================

    def test_01_allowed_nominal_telemetry_evaluation(self):
        """
        Full evaluation of nominal FD001 engine cycle; confirms ACCEPTED status,
        0 violations, and safe commit.
        """
        # 1. Direct aerothermal constraints evaluation
        report = evaluate_constraints(self.sim.current_state.to_dict())
        self.assertTrue(report.is_valid, "Nominal baseline state should satisfy all operating constraints.")
        self.assertEqual(len(report.violations), 0, "Nominal baseline state should have 0 constraint violations.")

        # 2. Closed-loop nominal evaluation action
        nominal_action = SimulationAction(
            action_id="act_allowed_nom_01",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="health_index",
            proposed_value=0.98,
        )
        res = self.governor.apply_simulation_action(self.sim, nominal_action)

        self.assertEqual(res.decision_status, SafetyDecisionStatus.ACCEPTED)
        self.assertTrue(res.update_applied)
        self.assertEqual(self.sim.current_state.virtual_health_index, 0.98)
        self.assertIn("ACCEPTED", res.reason_summary)
        self.assertIsNotNone(res.recovery_plan)
        self.assertEqual(res.recovery_plan.strategy_type, RecoveryStrategyType.CONTINUE_MONITORING)
        self.assertEqual(res.recovery_plan.priority, RecoveryPriority.LOW)

    def test_02_allowed_mild_thermal_perturbation(self):
        """
        Validates mild thermal stress offset (+4 °R) cleanly commits with update_applied=True.
        """
        baseline_t30 = self.sim.current_state.T30
        mild_action = SimulationAction(
            action_id="act_allowed_mild_thermal",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=4.0,  # +4 °R offset (well below 1605 °R caution limit)
        )
        res = self.governor.apply_simulation_action(self.sim, mild_action)

        self.assertEqual(res.decision_status, SafetyDecisionStatus.ACCEPTED)
        self.assertTrue(res.update_applied)
        self.assertEqual(self.sim.current_state.thermal_stress_offset, 4.0)
        # Measured telemetry remains untouched
        self.assertEqual(self.sim.current_state.T30, baseline_t30)

    def test_03_allowed_prognostics_enrichment(self):
        """
        Ingests valid RUL and health index (H=0.95); verifies ACCEPTED decision and parameter commit.
        """
        # 1. Ingest valid predicted RUL
        action_rul = SimulationAction(
            action_id="act_enrich_valid_rul",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="predicted_rul",
            proposed_value=112.5,
        )
        res_rul = self.governor.apply_simulation_action(self.sim, action_rul)
        self.assertEqual(res_rul.decision_status, SafetyDecisionStatus.ACCEPTED)
        self.assertTrue(res_rul.update_applied)
        self.assertEqual(self.sim.current_state.predicted_rul_xgboost, 112.5)

        # 2. Ingest valid health index
        action_health = SimulationAction(
            action_id="act_enrich_valid_health",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="health_index",
            proposed_value=0.95,
        )
        res_h = self.governor.apply_simulation_action(self.sim, action_health)
        self.assertEqual(res_h.decision_status, SafetyDecisionStatus.ACCEPTED)
        self.assertTrue(res_h.update_applied)
        self.assertEqual(self.sim.current_state.virtual_health_index, 0.95)

    # =========================================================================
    # Category B: Boundary Breaches (Advisory Caution & Review States)
    # =========================================================================

    def test_04_boundary_thermal_creep_caution(self):
        """
        Evaluates T30 in caution band (1605 °R < T30 <= 1625 °R); confirms REQUIRES_REVIEW
        and MAINTENANCE_REVIEW plan.
        """
        # Baseline T30 ~ 1589.70 °R; +22.0 °R drives candidate T30 to ~1611.70 °R (caution band)
        caution_action = SimulationAction(
            action_id="act_boundary_creep_caution",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=22.0,
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, caution_action)

        self.assertEqual(decision.status, SafetyDecisionStatus.REQUIRES_REVIEW)
        violated_ids = [v.constraint_id for v in decision.violations]
        self.assertIn("CONST_TEMP_T30_CAUTION", violated_ids)

        # Generate simulated recovery recommendation
        plan = self.verifier.generate_recovery_plan(decision, caution_action)
        self.assertEqual(plan.strategy_type, RecoveryStrategyType.MAINTENANCE_REVIEW)
        self.assertEqual(plan.priority, RecoveryPriority.MEDIUM)
        self.assertIn("review", plan.recommended_actions[0].lower())

    def test_05_boundary_operator_gating_lifecycle(self):
        """
        Verifies caution state blocked without approval (update_applied=False) and
        committed when approved (update_applied=True).
        """
        caution_action = SimulationAction(
            action_id="act_operator_gating_lifecycle",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=22.0,
        )

        # Attempt 1: Without operator affirmative approval
        res_blocked = self.governor.apply_simulation_action(
            self.sim,
            caution_action,
            operator_approval=False,
        )
        self.assertEqual(res_blocked.decision_status, SafetyDecisionStatus.REQUIRES_REVIEW)
        self.assertFalse(res_blocked.update_applied)
        self.assertFalse(res_blocked.operator_approved)
        self.assertIn(self.sim.current_state.thermal_stress_offset, (None, 0.0))

        # Attempt 2: With explicit affirmative operator approval
        res_approved = self.governor.apply_simulation_action(
            self.sim,
            caution_action,
            operator_approval=True,
        )
        self.assertEqual(res_approved.decision_status, SafetyDecisionStatus.REQUIRES_REVIEW)
        self.assertTrue(res_approved.update_applied)
        self.assertTrue(res_approved.operator_approved)
        self.assertEqual(self.sim.current_state.thermal_stress_offset, 22.0)

    # =========================================================================
    # Category C: Rejected Operations (Hard Limits, Invariants & Firewall Violations)
    # =========================================================================

    def test_06_rejected_firewall_direct_telemetry_mutation(self):
        """
        Verifies attempt to mutate measured sensor (T30) is REJECTED with SCENARIO_ROLLBACK.
        """
        baseline_t30 = self.sim.current_state.T30
        illegal_telemetry_action = SimulationAction(
            action_id="act_illegal_direct_sensor_write",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="T30",  # G8 measured sensor
            proposed_value=1620.0,
        )
        res = self.governor.apply_simulation_action(self.sim, illegal_telemetry_action)

        self.assertEqual(res.decision_status, SafetyDecisionStatus.REJECTED)
        self.assertFalse(res.update_applied)
        self.assertEqual(res.recovery_plan.strategy_type, RecoveryStrategyType.SCENARIO_ROLLBACK)
        self.assertEqual(res.recovery_plan.priority, RecoveryPriority.HIGH)
        # Measured telemetry strictly untouched
        self.assertEqual(self.sim.current_state.T30, baseline_t30)

    def test_07_rejected_metallurgical_overtemp_hard_limit(self):
        """
        Verifies T30 > 1625 °R triggers REJECTED, PHYSICAL_INSPECTION, and zero state commit.
        """
        # Offset +50 °R drives candidate T30 to ~1639.7 °R (> 1625 °R hard limit)
        hard_overtemp_action = SimulationAction(
            action_id="act_severe_overtemp_hard_limit",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=50.0,
        )
        res = self.governor.apply_simulation_action(self.sim, hard_overtemp_action)

        self.assertEqual(res.decision_status, SafetyDecisionStatus.REJECTED)
        self.assertFalse(res.update_applied)
        self.assertEqual(res.recovery_plan.strategy_type, RecoveryStrategyType.PHYSICAL_INSPECTION)
        self.assertEqual(res.recovery_plan.priority, RecoveryPriority.CRITICAL)
        # State rolled back: offset reset to 0.0
        self.assertEqual(self.sim.current_state.thermal_stress_offset, 0.0)

    def test_08_rejected_thermodynamic_ordering_inversion(self):
        """
        Verifies stage compression temperature inversion (T30 <= T24) is REJECTED with CRITICAL inspection.
        """
        # Proposed offset -1000 °R drives T30 to ~589.7 °R, which is less than T24 (~642 °R)
        inversion_action = SimulationAction(
            action_id="act_temp_ordering_inversion",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=-1000.0,
        )
        res = self.governor.apply_simulation_action(self.sim, inversion_action)

        self.assertEqual(res.decision_status, SafetyDecisionStatus.REJECTED)
        self.assertFalse(res.update_applied)
        self.assertEqual(res.recovery_plan.strategy_type, RecoveryStrategyType.PHYSICAL_INSPECTION)
        self.assertEqual(res.recovery_plan.priority, RecoveryPriority.CRITICAL)
        self.assertEqual(self.sim.current_state.thermal_stress_offset, 0.0)

    def test_09_rejected_unphysical_prognostic_causality(self):
        """
        Verifies negative RUL triggers REJECTED and parameter clamp to 0.0.
        """
        negative_rul_action = SimulationAction(
            action_id="act_unphysical_negative_rul",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="predicted_rul",
            proposed_value=-15.0,
        )
        res = self.governor.apply_simulation_action(self.sim, negative_rul_action)

        self.assertEqual(res.decision_status, SafetyDecisionStatus.REJECTED)
        self.assertFalse(res.update_applied)
        self.assertEqual(res.recovery_plan.strategy_type, RecoveryStrategyType.PARAMETER_CLAMP)
        # Governed parameter clamped to 0.0
        self.assertEqual(self.sim.current_state.predicted_rul_xgboost, 0.0)

    # =========================================================================
    # Cryptographic Immutability & Audit Trail Integrity
    # =========================================================================

    def test_10_phase7_master_immutability_and_audit_verification(self):
        """
        Verifies append-only audit trail logging across all tests and bit-for-bit SHA-256
        immutability of all 12 raw files, 4 scalers, and 8 champion checkpoints.
        """
        # 1. Execute representative actions across nominal, caution, and rejection categories
        actions_to_audit = [
            SimulationAction("audit_nom", ActionType.THERMAL_PERTURBATION, "thermal_stress_offset", 3.0),
            SimulationAction("audit_caution", ActionType.THERMAL_PERTURBATION, "thermal_stress_offset", 22.0),
            SimulationAction("audit_rejected", ActionType.STATE_ENRICHMENT, "T30", 1700.0),
        ]
        for act in actions_to_audit:
            self.governor.apply_simulation_action(self.sim, act)

        # 2. Verify JSONL audit trail was generated and contains valid JSON records
        audit_file = os.path.join(self.test_dir, "phase7_master_audit.jsonl")
        self.assertTrue(os.path.exists(audit_file), "Audit log file should exist.")

        records = []
        with open(audit_file, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    rec = json.loads(line_str)
                    records.append(rec)

        self.assertEqual(len(records), 3, "Audit logger should contain exactly 3 recorded safety events.")
        statuses = [r["decision_status"] for r in records]
        self.assertIn("ACCEPTED", statuses)
        self.assertIn("REQUIRES_REVIEW", statuses)
        self.assertIn("REJECTED", statuses)

        for r in records:
            self.assertIn("record_id", r)
            self.assertIn("decision_status", r)
            self.assertIn("reason_code", r)
            self.assertIn("timestamp", r)
            self.assertIn("target_variable", r)

        # 3. Verify all 12 raw C-MAPSS dataset files are untouched
        self.assertEqual(len(self.raw_hashes), 12, "Expected 12 raw C-MAPSS text files.")
        for fname, expected_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                expected_hash,
                f"Raw dataset file '{fname}' was modified during Phase 7 execution!",
            )

        # 4. Verify all 4 subset scalers are untouched
        self.assertEqual(len(self.scaler_hashes), 4, "Expected 4 subset scalers.")
        for fname, expected_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                expected_hash,
                f"Scaler file '{fname}' was modified during Phase 7 execution!",
            )

        # 5. Verify all 8 champion model checkpoints are untouched
        self.assertEqual(len(self.checkpoint_hashes), 8, "Expected 8 champion checkpoints.")
        for fname, expected_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                expected_hash,
                f"Champion checkpoint '{fname}' was modified during Phase 7 execution!",
            )


if __name__ == "__main__":
    unittest.main()
