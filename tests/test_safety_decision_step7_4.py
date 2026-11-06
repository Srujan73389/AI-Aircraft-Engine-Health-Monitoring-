"""
Unit and Aerospace Safety Verification Test Suite for Step 7.4
Verifies Tri-State Action Safety Decisions (ACCEPTED, REJECTED, REQUIRES_REVIEW).

Roadmap Reference: Phase 7 - Step 7.4
"""

import os
import hashlib
import unittest
from typing import Dict, Any

from safety.decision import (
    SafetyDecisionStatus,
    ActionSafetyDecision,
    classify_safety_decision,
)
from safety.verification import (
    ActionType,
    SimulationAction,
    SafetyCheckResult,
    SafetyVerifier,
    evaluate_simulation_decision,
)
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestSafetyDecisionStep7_4(unittest.TestCase):
    """
    Test suite for Phase 7 Step 7.4: Tri-State Safety Decision Classification.
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

    def test_01_decision_status_enumeration(self):
        """Asserts ACCEPTED, REJECTED, and REQUIRES_REVIEW exist in SafetyDecisionStatus."""
        self.assertEqual(len(SafetyDecisionStatus), 3)
        self.assertIn("ACCEPTED", [s.value for s in SafetyDecisionStatus])
        self.assertIn("REJECTED", [s.value for s in SafetyDecisionStatus])
        self.assertIn("REQUIRES_REVIEW", [s.value for s in SafetyDecisionStatus])
        self.assertEqual(SafetyDecisionStatus.ACCEPTED, "ACCEPTED")
        self.assertEqual(SafetyDecisionStatus.REJECTED, "REJECTED")
        self.assertEqual(SafetyDecisionStatus.REQUIRES_REVIEW, "REQUIRES_REVIEW")

    def test_02_classify_nominal_action_accepted(self):
        """Asserts valid nominal actions receive ACCEPTED status."""
        action = SimulationAction(
            action_id="act_nom_01",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=4.0,  # Safe mild thermal stress: T30 becomes ~1593.7 °R
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        self.assertEqual(decision.status, SafetyDecisionStatus.ACCEPTED)
        self.assertTrue(decision.is_accepted)
        self.assertFalse(decision.is_rejected)
        self.assertFalse(decision.requires_review)
        self.assertEqual(decision.caution_count, 0)
        self.assertEqual(decision.critical_count, 0)
        self.assertEqual(decision.hard_limit_count, 0)
        self.assertIn("ACCEPTED", decision.summary_message)

    def test_03_classify_caution_breach_requires_review(self):
        """Asserts thermal creep (T30 in 1605-1625 °R) receives REQUIRES_REVIEW."""
        # Baseline T30 ~ 1589.7 °R. Offset +22.0 °R -> 1611.7 °R (Caution: > 1605, Warning: <= 1625)
        action = SimulationAction(
            action_id="act_caution_creep",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=22.0,
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        self.assertEqual(decision.status, SafetyDecisionStatus.REQUIRES_REVIEW)
        self.assertTrue(decision.requires_review)
        self.assertFalse(decision.is_accepted)
        self.assertFalse(decision.is_rejected)
        self.assertGreaterEqual(decision.caution_count, 1)
        self.assertEqual(decision.critical_count, 0)
        self.assertEqual(decision.hard_limit_count, 0)
        self.assertIn("REQUIRES_REVIEW", decision.summary_message)

    def test_04_classify_firewall_breach_rejected(self):
        """Asserts direct override of measured sensors receives REJECTED status."""
        illegal_action = SimulationAction(
            action_id="act_illegal_telemetry_t30",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="T30",  # G8 measured sensor
            proposed_value=1600.0,
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, illegal_action)
        self.assertEqual(decision.status, SafetyDecisionStatus.REJECTED)
        self.assertTrue(decision.is_rejected)
        self.assertFalse(decision.is_accepted)
        self.assertFalse(decision.requires_review)
        self.assertIn("firewall", decision.summary_message.lower())

    def test_05_classify_hard_limit_rejected(self):
        """Asserts T30 > 1625 °R warning hard limit receives REJECTED status."""
        # Baseline T30 ~ 1589.7 °R. Offset +45.0 °R -> 1634.7 °R (> 1625 °R warning)
        action = SimulationAction(
            action_id="act_hard_overtemp",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=45.0,
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        self.assertEqual(decision.status, SafetyDecisionStatus.REJECTED)
        self.assertTrue(decision.is_rejected)
        self.assertGreaterEqual(decision.hard_limit_count, 1)
        self.assertIn("REJECTED", decision.summary_message)

    def test_06_classify_ordering_inversion_rejected(self):
        """Asserts temperature or pressure inversion receives REJECTED status."""
        # Drive T30 below T24
        action = SimulationAction(
            action_id="act_temp_inv",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=-1000.0,
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        self.assertEqual(decision.status, SafetyDecisionStatus.REJECTED)
        self.assertTrue(decision.is_rejected)
        self.assertGreaterEqual(decision.hard_limit_count, 1)

    def test_07_classify_flight_envelope_exceedance_rejected(self):
        """Asserts altitude/Mach exceedance receives REJECTED status."""
        action_alt = SimulationAction(
            action_id="act_ceiling_breach",
            action_type=ActionType.OPERATIONAL_RECONFIGURATION,
            target_variable="altitude",
            proposed_value=50000.0,  # > 45,000 ft
        )
        decision_alt = self.verifier.evaluate_action_decision(self.nominal_state, action_alt)
        self.assertEqual(decision_alt.status, SafetyDecisionStatus.REJECTED)
        self.assertGreaterEqual(decision_alt.critical_count, 1)

        action_mach = SimulationAction(
            action_id="act_mach_breach",
            action_type=ActionType.OPERATIONAL_RECONFIGURATION,
            target_variable="mach_number",
            proposed_value=0.95,  # > 0.90
        )
        decision_mach = self.verifier.evaluate_action_decision(self.nominal_state, action_mach)
        self.assertEqual(decision_mach.status, SafetyDecisionStatus.REJECTED)
        self.assertGreaterEqual(decision_mach.critical_count, 1)

    def test_08_classify_negative_rul_rejected(self):
        """Asserts negative RUL prediction receives REJECTED status."""
        action = SimulationAction(
            action_id="act_neg_rul",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="predicted_rul",
            proposed_value=-20.0,
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        self.assertEqual(decision.status, SafetyDecisionStatus.REJECTED)
        self.assertTrue(decision.is_rejected)
        self.assertGreaterEqual(decision.hard_limit_count, 1)

    def test_09_verifier_evaluate_action_decision_integration(self):
        """Tests end-to-end verifier integration returning ActionSafetyDecision."""
        action = SimulationAction(
            action_id="act_enrichment_integration",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="prognostics",
            proposed_value={
                "predicted_rul_xgboost": 120.0,
                "virtual_health_index": 0.95,
            },
        )
        decision = evaluate_simulation_decision(self.nominal_state, action, self.verifier)
        self.assertIsInstance(decision, ActionSafetyDecision)
        self.assertEqual(decision.status, SafetyDecisionStatus.ACCEPTED)
        self.assertTrue(decision.is_accepted)

        data = decision.to_dict()
        self.assertEqual(data["action_id"], "act_enrichment_integration")
        self.assertEqual(data["status"], "ACCEPTED")
        self.assertIn("decision_timestamp", data)

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
                f"Raw dataset file '{fname}' was modified during Step 7.4 execution!",
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
                f"Scaler file '{fname}' was modified during Step 7.4 execution!",
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
                f"Champion checkpoint '{fname}' was modified during Step 7.4 execution!",
            )


if __name__ == "__main__":
    unittest.main()
