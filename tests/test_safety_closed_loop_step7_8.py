"""
Unit and Aerospace Safety Verification Test Suite for Step 7.8
Verifies Closed-Loop State Gating, Rollback Execution, Parameter Clamping, and Telemetry Immutability.

Roadmap Reference: Phase 7 - Step 7.8
"""

import os
import hashlib
import unittest
from typing import Dict, Any

from safety.closed_loop import (
    ClosedLoopUpdateResult,
    ClosedLoopSafetyGovernor,
    safe_simulation_step,
)
from safety.decision import SafetyDecisionStatus
from safety.recovery import RecoveryStrategyType
from safety.verification import (
    ActionType,
    SimulationAction,
    SafetyVerifier,
)
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestSafetyClosedLoopStep7_8(unittest.TestCase):
    """
    Test suite for Phase 7 Step 7.8: Closed-Loop Simulation Updates and State Gating.
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
        self.governor = ClosedLoopSafetyGovernor()
        self.sim = DigitalTwinSimulator(subset="FD001")
        self.sim.load_engine(unit_number=1, split="train")
        self.nominal_state = self.sim.current_state

    def test_01_closed_loop_governor_initialization(self):
        """Asserts governor initializes with verifier and audit logger."""
        gov = ClosedLoopSafetyGovernor()
        self.assertIsNotNone(gov.verifier)
        self.assertIsInstance(gov.verifier, SafetyVerifier)

        custom_verifier = SafetyVerifier(strict_firewall=True)
        gov_custom = ClosedLoopSafetyGovernor(verifier=custom_verifier)
        self.assertEqual(gov_custom.verifier, custom_verifier)

    def test_02_commit_nominal_accepted_action(self):
        """Asserts ACCEPTED action commits candidate state (update_applied=True)."""
        action = SimulationAction(
            action_id="act_commit_nom",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=3.5,
        )
        res = self.governor.apply_simulation_action(self.sim, action)

        self.assertTrue(res.update_applied)
        self.assertEqual(res.decision_status, SafetyDecisionStatus.ACCEPTED)
        self.assertEqual(self.sim.current_state.thermal_stress_offset, 3.5)
        self.assertIn("ACCEPTED", res.reason_summary)

    def test_03_block_requires_review_without_approval(self):
        """Asserts REQUIRES_REVIEW action is blocked when operator_approval=False."""
        action = SimulationAction(
            action_id="act_block_creep",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=22.0,  # Caution range: T30 ~ 1611.7 °R
        )
        res = self.governor.apply_simulation_action(self.sim, action, operator_approval=False)

        self.assertFalse(res.update_applied)
        self.assertEqual(res.decision_status, SafetyDecisionStatus.REQUIRES_REVIEW)
        self.assertFalse(res.operator_approved)
        self.assertIn("blocked", res.reason_summary.lower())
        self.assertEqual(res.state_before, res.state_after)

    def test_04_allow_requires_review_with_approval(self):
        """Asserts REQUIRES_REVIEW action commits when operator_approval=True."""
        action = SimulationAction(
            action_id="act_allow_creep",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=22.0,
        )
        res = self.governor.apply_simulation_action(self.sim, action, operator_approval=True)

        self.assertTrue(res.update_applied)
        self.assertEqual(res.decision_status, SafetyDecisionStatus.REQUIRES_REVIEW)
        self.assertTrue(res.operator_approved)
        self.assertEqual(self.sim.current_state.thermal_stress_offset, 22.0)
        self.assertIn("operator override", res.reason_summary.lower())

    def test_05_block_and_rollback_firewall_violation(self):
        """Asserts REJECTED firewall breach does NOT modify state and applies rollback."""
        orig_t30 = self.sim.current_state.T30
        illegal_action = SimulationAction(
            action_id="act_illegal_direct_t30",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="T30",  # G8 measured sensor
            proposed_value=1620.0,
        )
        res = self.governor.apply_simulation_action(self.sim, illegal_action)

        self.assertFalse(res.update_applied)
        self.assertEqual(res.decision_status, SafetyDecisionStatus.REJECTED)
        self.assertEqual(self.sim.current_state.T30, orig_t30)  # Unmodified!
        self.assertEqual(res.recovery_plan.strategy_type, RecoveryStrategyType.SCENARIO_ROLLBACK)

    def test_06_block_and_rollback_hard_overtemp(self):
        """Asserts REJECTED overtemp (> 1625 °R) does NOT update state and clamps/rolls back."""
        # Initial offset 5.0
        self.sim.current_state.thermal_stress_offset = 5.0

        # Attempt severe overtemp (+45 °R driving T30 > 1625 °R)
        action = SimulationAction(
            action_id="act_severe_burn",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=45.0,
        )
        res = self.governor.apply_simulation_action(self.sim, action)

        self.assertFalse(res.update_applied)
        self.assertEqual(res.decision_status, SafetyDecisionStatus.REJECTED)
        # Rollback remediation must reset offset to 0.0
        self.assertEqual(self.sim.current_state.thermal_stress_offset, 0.0)
        self.assertEqual(res.recovery_plan.strategy_type, RecoveryStrategyType.PHYSICAL_INSPECTION)

    def test_07_block_and_clamp_negative_rul(self):
        """Asserts REJECTED negative RUL applies parameter clamp remediation."""
        action = SimulationAction(
            action_id="act_negative_rul_clamp",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="predicted_rul",
            proposed_value=-18.0,
        )
        res = self.governor.apply_simulation_action(self.sim, action)

        self.assertFalse(res.update_applied)
        self.assertEqual(res.decision_status, SafetyDecisionStatus.REJECTED)
        self.assertEqual(res.recovery_plan.strategy_type, RecoveryStrategyType.PARAMETER_CLAMP)
        # Clamped to 0.0
        self.assertEqual(self.sim.current_state.predicted_rul_xgboost, 0.0)

    def test_08_measured_telemetry_immutability_throughout_loop(self):
        """Asserts measured sensors (G6-G26) remain bit-for-bit identical before and after rejected actions."""
        # Capture all 21 measured sensors
        measured_slots = [
            "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
            "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
            "Nf_dmd", "PCNfR_dmd", "W31", "W32"
        ]
        baseline_telemetry = {slot: getattr(self.sim.current_state, slot) for slot in measured_slots}

        # Subject simulator to 3 distinct rejected actions
        bad_actions = [
            SimulationAction("bad_1", ActionType.STATE_ENRICHMENT, "P30", 720.0),
            SimulationAction("bad_2", ActionType.THERMAL_PERTURBATION, "thermal_stress_offset", 50.0),
            SimulationAction("bad_3", ActionType.THERMAL_PERTURBATION, "thermal_stress_offset", -1000.0),
        ]
        for act in bad_actions:
            res = self.governor.apply_simulation_action(self.sim, act)
            self.assertFalse(res.update_applied)

        # Assert every single sensor is bit-for-bit unchanged
        for slot in measured_slots:
            current_val = getattr(self.sim.current_state, slot)
            self.assertEqual(
                current_val,
                baseline_telemetry[slot],
                f"Measured telemetry '{slot}' was modified during closed-loop execution!",
            )

    def test_09_digital_twin_simulator_apply_safe_action_integration(self):
        """Tests end-to-end integration on DigitalTwinSimulator instance."""
        # 1. Test nominal action through simulator method
        nom_action = SimulationAction(
            action_id="act_sim_integration_nom",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=2.0,
        )
        res_nom = self.sim.apply_safe_action(nom_action)
        self.assertIsInstance(res_nom, ClosedLoopUpdateResult)
        self.assertTrue(res_nom.update_applied)
        self.assertEqual(self.sim.current_state.thermal_stress_offset, 2.0)

        # 2. Test rejected action through simulator method
        bad_action = SimulationAction(
            action_id="act_sim_integration_bad",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="T30",
            proposed_value=1700.0,
        )
        res_bad = self.sim.apply_safe_action(bad_action)
        self.assertFalse(res_bad.update_applied)
        self.assertEqual(res_bad.decision_status, SafetyDecisionStatus.REJECTED)

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
                f"Raw dataset file '{fname}' was modified during Step 7.8 execution!",
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
                f"Scaler file '{fname}' was modified during Step 7.8 execution!",
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
                f"Champion checkpoint '{fname}' was modified during Step 7.8 execution!",
            )


if __name__ == "__main__":
    unittest.main()
