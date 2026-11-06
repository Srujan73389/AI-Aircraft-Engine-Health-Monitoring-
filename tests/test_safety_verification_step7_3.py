"""
Unit and Safety Verification Test Suite for Step 7.3
Verifies Action Verification, What-If Projections, and Physical Firewall Enforcement.

Roadmap Reference: Phase 7 - Step 7.3
"""

import os
import hashlib
import unittest
from typing import Dict, Any

from safety.constraints import (
    ConstraintSeverity,
    ConstraintSource,
    OPERATING_CONSTRAINTS,
)
from safety.taxonomy import (
    VariableCategory,
    VARIABLE_TAXONOMY,
)
from safety.verification import (
    ActionType,
    SimulationAction,
    SafetyCheckResult,
    SafetyVerifier,
    verify_simulation_action,
)
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestSafetyVerificationStep7_3(unittest.TestCase):
    """
    Test suite for Phase 7 Step 7.3: Safety Verification Engine and Action Evaluation.
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

    def test_01_verifier_initialization(self):
        """Confirms verifier initializes with default or custom constraints and taxonomy."""
        v = SafetyVerifier()
        self.assertEqual(len(v.constraints), len(OPERATING_CONSTRAINTS))
        self.assertEqual(len(v.taxonomy), len(VARIABLE_TAXONOMY))
        self.assertTrue(v.strict_firewall)

        # Custom constraints mapping
        subset_constraints = {"CONST_FLIGHT_MACH": OPERATING_CONSTRAINTS["CONST_FLIGHT_MACH"]}
        v_custom = SafetyVerifier(constraints=subset_constraints, strict_firewall=False)
        self.assertEqual(len(v_custom.constraints), 1)
        self.assertFalse(v_custom.strict_firewall)

    def test_02_accept_nominal_simulation_action(self):
        """Verifies valid what-if perturbation within safe thermal limits."""
        action = SimulationAction(
            action_id="act_therm_nom_01",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=5.0,  # +5 °R offset
            description="Mild thermal stress perturbation test",
        )
        res = self.verifier.verify_action(self.nominal_state, action)
        self.assertTrue(res.is_safe)
        self.assertTrue(res.provenance_allowed)
        self.assertEqual(len(res.violations), 0)
        self.assertIsNotNone(res.projected_state)
        self.assertAlmostEqual(res.projected_state["T30"], self.nominal_state.T30 + 5.0, places=2)

    def test_03_reject_immutable_telemetry_action(self):
        """Asserts rejection of any action targeting MEASURED physical sensors directly."""
        illegal_action = SimulationAction(
            action_id="act_direct_t30_override",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="T30",  # G8: MEASURED telemetry
            proposed_value=1600.0,
            description="Illegal direct sensor override",
        )
        res = self.verifier.verify_action(self.nominal_state, illegal_action)
        self.assertFalse(res.is_safe)
        self.assertFalse(res.provenance_allowed)
        self.assertEqual(len(res.violations), 1)
        self.assertEqual(res.violations[0].constraint_id, "FIREWALL_MEASURED_IMMUTABILITY")
        self.assertIn("Safety Firewall Rejection", res.violations[0].message)

        # Test P30 direct override
        illegal_p30 = SimulationAction(
            action_id="act_direct_p30",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="P30",
            proposed_value=560.0,
        )
        res_p30 = self.verifier.verify_action(self.nominal_state, illegal_p30)
        self.assertFalse(res_p30.is_safe)
        self.assertFalse(res_p30.provenance_allowed)

    def test_04_reject_overtemp_thermal_action(self):
        """Asserts detection of proposed thermal actions that breach maximum T30 warnings."""
        # Baseline T30 ~ 1589.7 °R. Offset +45 °R -> 1634.7 °R (> 1625 °R warning hard limit)
        action = SimulationAction(
            action_id="act_severe_overtemp",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=45.0,
        )
        res = self.verifier.verify_action(self.nominal_state, action)
        self.assertFalse(res.is_safe)
        self.assertTrue(res.provenance_allowed)  # Target is thermal_stress_offset (SIMULATED), allowed
        violated_ids = [v.constraint_id for v in res.violations]
        self.assertIn("CONST_TEMP_T30_MAX_WARNING", violated_ids)
        self.assertIn("CONST_TEMP_T30_CAUTION", violated_ids)

    def test_05_reject_overpressure_action(self):
        """Asserts rejection of actions driving P30 beyond casing limits."""
        # Baseline P30 ~ 554.36 psia. Bias +200.0 psia -> 754.36 psia (> 700.0 psia)
        action = SimulationAction(
            action_id="act_casing_burst",
            action_type=ActionType.SENSOR_BIAS_INJECTION,
            target_variable="sensor_bias_delta",
            proposed_value={"P30": 200.0},
        )
        res = self.verifier.verify_action(self.nominal_state, action)
        self.assertFalse(res.is_safe)
        violated_ids = [v.constraint_id for v in res.violations]
        self.assertIn("CONST_PRESS_P30", violated_ids)

    def test_06_reject_flight_envelope_exceedance(self):
        """Asserts rejection of actions setting altitude > 45 kft or Mach > 0.90."""
        # Altitude exceedance (52,000 ft)
        alt_action = SimulationAction(
            action_id="act_unphysical_altitude",
            action_type=ActionType.OPERATIONAL_RECONFIGURATION,
            target_variable="altitude",
            proposed_value=52000.0,
        )
        res_alt = self.verifier.verify_action(self.nominal_state, alt_action)
        self.assertFalse(res_alt.is_safe)
        self.assertIn("CONST_FLIGHT_ALTITUDE", [v.constraint_id for v in res_alt.violations])

        # Mach exceedance (Mach 0.96)
        mach_action = SimulationAction(
            action_id="act_transonic_drag_divergence",
            action_type=ActionType.OPERATIONAL_RECONFIGURATION,
            target_variable="mach_number",
            proposed_value=0.96,
        )
        res_mach = self.verifier.verify_action(self.nominal_state, mach_action)
        self.assertFalse(res_mach.is_safe)
        self.assertIn("CONST_FLIGHT_MACH", [v.constraint_id for v in res_mach.violations])

    def test_07_reject_aerothermal_ordering_inversion(self):
        """Asserts rejection of actions inverting temperature (T24 >= T30) or pressure."""
        # Drastic negative thermal perturbation driving T30 below T24
        # Baseline T30 ~ 1589.7 °R, T24 ~ 641.82 °R. Perturb by -1000 °R -> T30 becomes 589.7 °R (< T24)
        action = SimulationAction(
            action_id="act_temp_inversion",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=-1000.0,
        )
        res = self.verifier.verify_action(self.nominal_state, action)
        self.assertFalse(res.is_safe)
        violated_ids = [v.constraint_id for v in res.violations]
        self.assertIn("CONST_INVARIANT_TEMP_ORDER", violated_ids)

    def test_08_reject_negative_rul_action(self):
        """Asserts rejection of actions attempting to set negative RUL."""
        action = SimulationAction(
            action_id="act_negative_rul",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="predicted_rul",
            proposed_value=-14.0,
        )
        res = self.verifier.verify_action(self.nominal_state, action)
        self.assertFalse(res.is_safe)
        violated_ids = [v.constraint_id for v in res.violations]
        self.assertIn("CONST_PROG_RUL_NON_NEGATIVE", violated_ids)

    def test_09_prognostics_state_enrichment_action(self):
        """Verifies safe state enrichment with valid RUL and health index."""
        action = SimulationAction(
            action_id="act_safe_enrichment",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="prognostics",
            proposed_value={
                "predicted_rul_xgboost": 115.0,
                "predicted_rul_lstm": 112.0,
                "virtual_health_index": 0.92,
                "anomaly_alert_flag": 0,
            },
        )
        res = verify_simulation_action(self.nominal_state, action)
        self.assertTrue(res.is_safe)
        self.assertTrue(res.provenance_allowed)
        self.assertEqual(len(res.violations), 0)
        self.assertEqual(res.projected_state["predicted_rul_xgboost"], 115.0)
        self.assertEqual(res.projected_state["virtual_health_index"], 0.92)

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
                f"Raw dataset file '{fname}' was modified during Step 7.3 execution!",
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
                f"Scaler file '{fname}' was modified during Step 7.3 execution!",
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
                f"Champion checkpoint '{fname}' was modified during Step 7.3 execution!",
            )


if __name__ == "__main__":
    unittest.main()
