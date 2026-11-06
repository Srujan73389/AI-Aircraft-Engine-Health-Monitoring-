"""
Unit and Aerospace Safety Verification Test Suite for Step 7.6
Verifies Simulated Recovery Recommendations, Strategies, and Parameter Mitigation.

Roadmap Reference: Phase 7 - Step 7.6
"""

import os
import hashlib
import unittest
from typing import Dict, Any

from safety.recovery import (
    RecoveryStrategyType,
    RecoveryPriority,
    SimulatedRecoveryPlan,
    generate_recovery_recommendation,
)
from safety.decision import (
    SafetyDecisionStatus,
    ActionSafetyDecision,
)
from safety.verification import (
    ActionType,
    SimulationAction,
    SafetyVerifier,
)
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestSafetyRecoveryStep7_6(unittest.TestCase):
    """
    Test suite for Phase 7 Step 7.6: Simulated Recovery Strategies and Action Remediation.
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
        self.verifier = SafetyVerifier()
        self.sim = DigitalTwinSimulator(subset="FD001")
        self.sim.load_engine(unit_number=1, split="train")
        self.nominal_state = self.sim.current_state

    def test_01_strategy_type_enumeration(self):
        """Asserts all 5 recovery strategy types and 4 priority levels exist."""
        expected_strategies = [
            "SCENARIO_ROLLBACK",
            "MAINTENANCE_REVIEW",
            "PHYSICAL_INSPECTION",
            "PARAMETER_CLAMP",
            "CONTINUE_MONITORING",
        ]
        self.assertEqual(len(RecoveryStrategyType), 5)
        for strat in expected_strategies:
            self.assertIn(strat, [s.value for s in RecoveryStrategyType])

        expected_priorities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        self.assertEqual(len(RecoveryPriority), 4)
        for prio in expected_priorities:
            self.assertIn(prio, [p.value for p in RecoveryPriority])

    def test_02_plan_nominal_continue_monitoring(self):
        """Verifies CONTINUE_MONITORING for an ACCEPTED action."""
        action = SimulationAction(
            action_id="act_nom_rec",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=3.0,
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        self.assertEqual(decision.status, SafetyDecisionStatus.ACCEPTED)

        plan = generate_recovery_recommendation(decision, action, self.nominal_state)
        self.assertIsInstance(plan, SimulatedRecoveryPlan)
        self.assertEqual(plan.strategy_type, RecoveryStrategyType.CONTINUE_MONITORING)
        self.assertEqual(plan.priority, RecoveryPriority.LOW)
        self.assertTrue(plan.mitigation_parameters.get("action_permitted"))
        self.assertTrue(plan.is_simulation_only)

    def test_03_plan_firewall_rollback(self):
        """Verifies SCENARIO_ROLLBACK for telemetry firewall breaches."""
        action = SimulationAction(
            action_id="act_firewall_violation",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="T30",  # G8 immutable sensor
            proposed_value=1620.0,
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        self.assertEqual(decision.status, SafetyDecisionStatus.REJECTED)

        plan = generate_recovery_recommendation(decision, action, self.nominal_state)
        self.assertEqual(plan.strategy_type, RecoveryStrategyType.SCENARIO_ROLLBACK)
        self.assertEqual(plan.priority, RecoveryPriority.HIGH)
        self.assertTrue(plan.mitigation_parameters.get("rollback_performed"))
        self.assertTrue(plan.mitigation_parameters.get("firewall_enforced"))
        self.assertIn("Safety Firewall", plan.recommended_actions[0])

    def test_04_plan_thermal_creep_maintenance_review(self):
        """Verifies MAINTENANCE_REVIEW for REQUIRES_REVIEW thermal creep."""
        action = SimulationAction(
            action_id="act_creep_rec",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=22.0,  # Caution range: T30 ~ 1611.7 °R
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        self.assertEqual(decision.status, SafetyDecisionStatus.REQUIRES_REVIEW)

        plan = generate_recovery_recommendation(decision, action, self.nominal_state)
        self.assertEqual(plan.strategy_type, RecoveryStrategyType.MAINTENANCE_REVIEW)
        self.assertEqual(plan.priority, RecoveryPriority.MEDIUM)
        self.assertFalse(plan.mitigation_parameters.get("action_permitted"))
        self.assertEqual(plan.mitigation_parameters.get("review_category"), "THERMAL_CREEP_INSPECTION")
        self.assertIn("turnaround", plan.recommended_actions[0].lower())

    def test_05_plan_overtemp_physical_inspection(self):
        """Verifies PHYSICAL_INSPECTION + HIGH/CRITICAL priority for hard limit overtemp."""
        action = SimulationAction(
            action_id="act_overtemp_rec",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=45.0,  # T30 ~ 1634.7 °R (> 1625 °R warning)
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        self.assertEqual(decision.status, SafetyDecisionStatus.REJECTED)

        plan = generate_recovery_recommendation(decision, action, self.nominal_state)
        self.assertEqual(plan.strategy_type, RecoveryStrategyType.PHYSICAL_INSPECTION)
        self.assertIn(plan.priority, (RecoveryPriority.HIGH, RecoveryPriority.CRITICAL))
        self.assertTrue(plan.mitigation_parameters.get("rollback_performed"))
        self.assertEqual(plan.mitigation_parameters.get("inspection_type"), "OPTICAL_BORESCOPE_HPC_COMBUSTOR")
        self.assertIn("borescope", plan.recommended_actions[1].lower())

    def test_06_plan_inversion_physical_inspection(self):
        """Verifies PHYSICAL_INSPECTION for aerodynamic stage inversions."""
        action = SimulationAction(
            action_id="act_inv_rec",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=-1000.0,
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        self.assertEqual(decision.status, SafetyDecisionStatus.REJECTED)

        plan = generate_recovery_recommendation(decision, action, self.nominal_state)
        self.assertEqual(plan.strategy_type, RecoveryStrategyType.PHYSICAL_INSPECTION)
        self.assertEqual(plan.priority, RecoveryPriority.CRITICAL)
        self.assertTrue(plan.mitigation_parameters.get("rollback_performed"))
        self.assertIn("stall/surge", plan.recommended_actions[1].lower())

    def test_07_plan_prognostic_parameter_clamp(self):
        """Verifies PARAMETER_CLAMP for negative RUL or invalid health index."""
        # Negative RUL clamp
        action_rul = SimulationAction(
            action_id="act_neg_rul_rec",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="predicted_rul",
            proposed_value=-15.0,
        )
        dec_rul = self.verifier.evaluate_action_decision(self.nominal_state, action_rul)
        plan_rul = generate_recovery_recommendation(dec_rul, action_rul, self.nominal_state)

        self.assertEqual(plan_rul.strategy_type, RecoveryStrategyType.PARAMETER_CLAMP)
        self.assertEqual(plan_rul.priority, RecoveryPriority.HIGH)
        self.assertEqual(plan_rul.mitigation_parameters.get("clamped_rul"), 0.0)
        self.assertEqual(plan_rul.mitigation_parameters.get("original_value"), -15.0)

        # Health index clamp
        action_hi = SimulationAction(
            action_id="act_hi_rec",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="virtual_health_index",
            proposed_value=1.35,
        )
        dec_hi = self.verifier.evaluate_action_decision(self.nominal_state, action_hi)
        plan_hi = generate_recovery_recommendation(dec_hi, action_hi, self.nominal_state)

        self.assertEqual(plan_hi.strategy_type, RecoveryStrategyType.PARAMETER_CLAMP)
        self.assertEqual(plan_hi.mitigation_parameters.get("clamped_health_index"), 1.0)
        self.assertEqual(plan_hi.mitigation_parameters.get("original_value"), 1.35)

    def test_08_plan_mitigation_parameters_content(self):
        """Verifies mitigation_parameters provides exact rollback target or clamp values."""
        # Test Flight Altitude exceedance rollback parameters
        action_alt = SimulationAction(
            action_id="act_alt_rec",
            action_type=ActionType.OPERATIONAL_RECONFIGURATION,
            target_variable="altitude",
            proposed_value=50000.0,
        )
        dec_alt = self.verifier.evaluate_action_decision(self.nominal_state, action_alt)
        plan_alt = generate_recovery_recommendation(dec_alt, action_alt, self.nominal_state)

        self.assertEqual(plan_alt.strategy_type, RecoveryStrategyType.SCENARIO_ROLLBACK)
        self.assertTrue(plan_alt.mitigation_parameters.get("rollback_performed"))
        self.assertEqual(plan_alt.mitigation_parameters.get("target_variable"), "altitude")
        self.assertEqual(plan_alt.mitigation_parameters.get("certified_envelope_limit"), 45000.0)

    def test_09_verifier_recovery_integration(self):
        """Verifies seamless invocation through SafetyVerifier.generate_recovery_plan."""
        action = SimulationAction(
            action_id="act_int_rec",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=22.0,
        )
        decision = self.verifier.evaluate_action_decision(
            self.nominal_state,
            action,
            attach_recovery_plan=True,
        )
        self.assertIsNotNone(decision.recovery_plan)
        self.assertEqual(decision.recovery_plan.strategy_type, RecoveryStrategyType.MAINTENANCE_REVIEW)

        # Direct verifier helper
        plan = self.verifier.generate_recovery_plan(decision, action, self.nominal_state)
        self.assertIsInstance(plan, SimulatedRecoveryPlan)
        self.assertEqual(plan.strategy_type, RecoveryStrategyType.MAINTENANCE_REVIEW)

        # Check serialization
        data = plan.to_dict()
        self.assertEqual(data["strategy_type"], "MAINTENANCE_REVIEW")
        self.assertEqual(data["priority"], "MEDIUM")
        self.assertTrue(data["is_simulation_only"])

    def test_10_dataset_and_model_immutability(self):
        """Cryptographic SHA-256 verification ensuring zero raw datasets, scalers, or checkpoints are modified."""
        # 1. Verify all 12 raw C-MAPSS dataset files
        self.assertEqual(len(self.raw_hashes), 12, "Expected 12 raw C-MAPSS text files.")
        for fname, expected_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                expected_hash,
                f"Raw dataset file '{fname}' was modified during Step 7.6 execution!",
            )

        # 2. Verify all 4 subset scalers
        self.assertEqual(len(self.scaler_hashes), 4, "Expected 4 subset scalers.")
        for fname, expected_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                expected_hash,
                f"Scaler file '{fname}' was modified during Step 7.6 execution!",
            )

        # 3. Verify all 8 champion model checkpoints
        self.assertEqual(len(self.checkpoint_hashes), 8, "Expected 8 champion checkpoints.")
        for fname, expected_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                expected_hash,
                f"Champion checkpoint '{fname}' was modified during Step 7.6 execution!",
            )


if __name__ == "__main__":
    unittest.main()
