"""
Unit and Aerospace Safety Verification Test Suite for Step 7.7
Verifies Academic Simulation Boundaries, Non-Authority Disclaimers, and Command Restrictions.

Roadmap Reference: Section 1 Working Rules and Phase 7 - Step 7.7
"""

import os
import hashlib
import unittest
from typing import Dict, Any

from safety.disclaimers import (
    ACADEMIC_SIMULATION_DISCLAIMER,
    FORBIDDEN_COMMAND_PATTERNS,
    CommandAuthorityViolationError,
    assert_non_commanding_recommendation,
    apply_academic_disclaimer,
)
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


class TestSafetyNonCommandingStep7_7(unittest.TestCase):
    """
    Test suite for Phase 7 Step 7.7: Academic Simulation and Non-Authority Enforcement.
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

    def test_01_disclaimer_text_content(self):
        """Asserts disclaimer explicitly mentions academic simulation, NASA C-MAPSS, non-authority."""
        disclaimer = ACADEMIC_SIMULATION_DISCLAIMER
        self.assertIn("ACADEMIC SIMULATION", disclaimer)
        self.assertIn("NASA C-MAPSS", disclaimer)
        self.assertIn("DO NOT issue real-time aircraft commands", disclaimer)
        self.assertIn("AMM/CMM", disclaimer)
        self.assertIn("airworthiness directives", disclaimer)

    def test_02_forbidden_command_patterns_coverage(self):
        """Asserts key prohibited control verbs and authorities exist in FORBIDDEN_COMMAND_PATTERNS."""
        required_patterns = [
            "EXECUTE_FADEC_OVERRIDE",
            "FADEC_COMMAND",
            "AUTHORIZE_FLIGHT_RELEASE",
            "OVERRIDE_MEL",
            "REPLACE_MAINTENANCE_MANUAL",
            "CLEAR_AIRWORTHINESS_FAULT",
            "ISSUE_FLIGHT_CLEARANCE",
        ]
        for pattern in required_patterns:
            self.assertIn(pattern, FORBIDDEN_COMMAND_PATTERNS)

    def test_03_assert_non_commanding_clean_text(self):
        """Asserts valid advisory texts/recommendations pass without raising error."""
        clean_text_1 = "Schedule line maintenance inspection for High-Pressure Compressor thermal creep."
        clean_text_2 = "Apply parameter clamp to enforce predicted Remaining Useful Life non-negativity (RUL >= 0)."
        self.assertTrue(assert_non_commanding_recommendation(clean_text_1))
        self.assertTrue(assert_non_commanding_recommendation(clean_text_2))

    def test_04_assert_non_commanding_detects_fadec_override(self):
        """Asserts scanning text containing 'FADEC_OVERRIDE' raises CommandAuthorityViolationError."""
        bad_text = "Action: EXECUTE_FADEC_OVERRIDE to reset fuel metering valve throttle limit."
        with self.assertRaises(CommandAuthorityViolationError) as ctx:
            assert_non_commanding_recommendation(bad_text)
        self.assertIn("Prohibited pattern", str(ctx.exception))
        self.assertIn("FADEC_OVERRIDE", str(ctx.exception))

    def test_05_assert_non_commanding_detects_maintenance_replacement(self):
        """Asserts scanning text containing 'REPLACE_MAINTENANCE_MANUAL' or 'OVERRIDE_MEL' raises error."""
        bad_text_amm = "Recommendation: REPLACE_MAINTENANCE_MANUAL task 72-00 with AI diagnostic output."
        with self.assertRaises(CommandAuthorityViolationError):
            assert_non_commanding_recommendation(bad_text_amm)

        bad_text_mel = "Directive: OVERRIDE_MEL dispatch criteria for flight release."
        with self.assertRaises(CommandAuthorityViolationError):
            assert_non_commanding_recommendation(bad_text_mel)

    def test_06_apply_academic_disclaimer_metadata(self):
        """Asserts apply_academic_disclaimer injects non-authority metadata."""
        sample_dict = {"scenario": "thermal_creep_simulation", "cycles": 100}
        enriched = apply_academic_disclaimer(sample_dict)

        self.assertTrue(enriched["is_simulation_only"])
        self.assertFalse(enriched["is_maintenance_authority"])
        self.assertFalse(enriched["is_flight_control_command"])
        self.assertEqual(enriched["regulatory_status"], "EXPLORATORY_PROTOTYPE_NOT_FOR_FLIGHT")
        self.assertEqual(enriched["disclaimer_notice"], ACADEMIC_SIMULATION_DISCLAIMER)

    def test_07_recovery_plan_non_authority_defaults(self):
        """Asserts SimulatedRecoveryPlan has non-authority defaults and disclaimer."""
        plan = SimulatedRecoveryPlan(
            plan_id="REC_test_001",
            action_id="act_001",
            decision_status=SafetyDecisionStatus.ACCEPTED,
            strategy_type=RecoveryStrategyType.CONTINUE_MONITORING,
            priority=RecoveryPriority.LOW,
            recommended_actions=["Continue nominal monitoring."],
            mitigation_parameters={"nominal": True},
        )
        self.assertTrue(plan.is_simulation_only)
        self.assertFalse(plan.is_maintenance_authority)
        self.assertFalse(plan.is_flight_control_command)
        self.assertEqual(plan.disclaimer_notice, ACADEMIC_SIMULATION_DISCLAIMER)

        plan_dict = plan.to_dict()
        self.assertFalse(plan_dict["is_maintenance_authority"])
        self.assertFalse(plan_dict["is_flight_control_command"])
        self.assertIn("RESEARCH & ACADEMIC SIMULATION ONLY", plan_dict["disclaimer_notice"])

    def test_08_generate_recovery_plan_passes_non_commanding_check(self):
        """Verifies that plans generated by generate_recovery_recommendation pass non-commanding check."""
        action = SimulationAction(
            action_id="act_creep_check",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=22.0,
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        plan = generate_recovery_recommendation(decision, action, self.nominal_state)

        # Asserts plan verified non-commanding
        self.assertTrue(assert_non_commanding_recommendation(plan))
        self.assertFalse(plan.is_maintenance_authority)
        self.assertFalse(plan.is_flight_control_command)

    def test_09_verifier_rejects_forbidden_simulation_actions(self):
        """Verifies that SafetyVerifier.verify_action blocks actions containing forbidden commanding patterns."""
        forbidden_action = SimulationAction(
            action_id="act_fadec_cmd",
            action_type=ActionType.OPERATIONAL_RECONFIGURATION,
            target_variable="altitude",
            proposed_value=35000.0,
            description="Attempt to EXECUTE_FADEC_OVERRIDE on flight control computer",
        )
        res = self.verifier.verify_action(self.nominal_state, forbidden_action)
        self.assertFalse(res.is_safe)
        self.assertFalse(res.provenance_allowed)
        self.assertEqual(res.violations[0].constraint_id, "COMMAND_AUTHORITY_VIOLATION")
        self.assertIn("Command Authority Rejection", res.violations[0].message)

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
                f"Raw dataset file '{fname}' was modified during Step 7.7 execution!",
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
                f"Scaler file '{fname}' was modified during Step 7.7 execution!",
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
                f"Champion checkpoint '{fname}' was modified during Step 7.7 execution!",
            )


if __name__ == "__main__":
    unittest.main()
