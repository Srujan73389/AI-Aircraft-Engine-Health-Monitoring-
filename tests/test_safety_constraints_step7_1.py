"""
Unit and Aerospace Safety Verification Test Suite for Step 7.1
Verifies Operating Constraints, Physical Invariants, and Aerothermal Assumptions.

Roadmap Reference: Phase 7 - Step 7.1
"""

import os
import hashlib
import unittest
from typing import Dict, Any

from safety.constraints import (
    ConstraintSource,
    ConstraintSeverity,
    OperatingConstraint,
    ConstraintViolation,
    ConstraintEvaluationReport,
    OPERATING_CONSTRAINTS,
    evaluate_constraints,
)
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestSafetyConstraintsStep7_1(unittest.TestCase):
    """
    Test suite for Phase 7 Step 7.1: Operating Constraints and Aerospace Assumptions.
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

    def test_01_registry_completeness(self):
        """Asserts all required flight, sensor, and model constraints exist in the registry."""
        self.assertGreaterEqual(len(OPERATING_CONSTRAINTS), 20)

        # Ambient flight envelope
        self.assertIn("CONST_FLIGHT_ALTITUDE", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_FLIGHT_MACH", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_FLIGHT_TRA", OPERATING_CONSTRAINTS)

        # Gas-path temperatures
        self.assertIn("CONST_TEMP_T2", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_TEMP_T24", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_TEMP_T30_CAUTION", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_TEMP_T30_MAX_WARNING", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_TEMP_T50", OPERATING_CONSTRAINTS)

        # Gas-path pressures & derived indicators
        self.assertIn("CONST_PRESS_P2", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_PRESS_P15", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_PRESS_P30", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_PRESS_PS30", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_DERIVED_OPR", OPERATING_CONSTRAINTS)

        # Rotor speeds
        self.assertIn("CONST_SPEED_NF", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_SPEED_NC", OPERATING_CONSTRAINTS)

        # Prognostics & Health Index
        self.assertIn("CONST_PROG_HEALTH_INDEX", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_PROG_RUL_NON_NEGATIVE", OPERATING_CONSTRAINTS)

        # Aerothermal ordering invariants
        self.assertIn("CONST_INVARIANT_TEMP_ORDER", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_INVARIANT_PRESS_ORDER", OPERATING_CONSTRAINTS)
        self.assertIn("CONST_INVARIANT_SPOOL_ORDER", OPERATING_CONSTRAINTS)

    def test_02_bounded_range_validity(self):
        """Asserts min_value < max_value for all bounded numeric constraints."""
        for c_id, constraint in OPERATING_CONSTRAINTS.items():
            if constraint.min_value is not None and constraint.max_value is not None:
                self.assertLess(
                    constraint.min_value,
                    constraint.max_value,
                    f"Constraint {c_id} has invalid bounds: min={constraint.min_value} >= max={constraint.max_value}",
                )

    def test_03_constraint_provenance(self):
        """Asserts valid source, severity, and non-empty assumption strings for every constraint."""
        for c_id, constraint in OPERATING_CONSTRAINTS.items():
            self.assertIsInstance(constraint.source, ConstraintSource)
            self.assertIsInstance(constraint.severity, ConstraintSeverity)
            self.assertGreater(
                len(constraint.aerothermal_assumption.strip()),
                15,
                f"Constraint {c_id} must have a non-empty, detailed aerothermal assumption.",
            )
            self.assertGreater(
                len(constraint.description.strip()),
                10,
                f"Constraint {c_id} must have a descriptive summary.",
            )

    def test_04_temperature_constraint_violations(self):
        """Asserts detection of out-of-range T30 across caution and hard limit thresholds."""
        # 1. Nominal T30
        rep_nominal = evaluate_constraints({"T30": 1585.0})
        self.assertNotIn("CONST_TEMP_T30_CAUTION", rep_nominal.violated_constraints)
        self.assertNotIn("CONST_TEMP_T30_MAX_WARNING", rep_nominal.violated_constraints)

        # 2. Caution breach (1605 °R < T30 <= 1625 °R)
        rep_caution = evaluate_constraints({"T30": 1612.0})
        self.assertIn("CONST_TEMP_T30_CAUTION", rep_caution.violated_constraints)
        self.assertNotIn("CONST_TEMP_T30_MAX_WARNING", rep_caution.violated_constraints)
        caution_viol = [v for v in rep_caution.violations if v.constraint_id == "CONST_TEMP_T30_CAUTION"][0]
        self.assertEqual(caution_viol.severity, ConstraintSeverity.CAUTION)

        # 3. Maximum warning / hard limit breach (T30 > 1625 °R)
        rep_hard = evaluate_constraints({"T30": 1635.0})
        self.assertIn("CONST_TEMP_T30_CAUTION", rep_hard.violated_constraints)
        self.assertIn("CONST_TEMP_T30_MAX_WARNING", rep_hard.violated_constraints)
        hard_viol = [v for v in rep_hard.violations if v.constraint_id == "CONST_TEMP_T30_MAX_WARNING"][0]
        self.assertEqual(hard_viol.severity, ConstraintSeverity.HARD_LIMIT)

        # 4. Severe lower breach
        rep_low = evaluate_constraints({"T30": 950.0})
        self.assertIn("CONST_TEMP_T30_MAX_WARNING", rep_low.violated_constraints)

    def test_05_pressure_constraint_violations(self):
        """Asserts detection of out-of-range P30 and derived OPR."""
        # Upper P30 breach (> 700 psia)
        rep_p30_high = evaluate_constraints({"P30": 745.0})
        self.assertIn("CONST_PRESS_P30", rep_p30_high.violated_constraints)

        # Lower P30 breach (< 100 psia)
        rep_p30_low = evaluate_constraints({"P30": 85.0})
        self.assertIn("CONST_PRESS_P30", rep_p30_low.violated_constraints)

        # OPR breach (> 55)
        rep_opr = evaluate_constraints({"P30": 600.0, "P2": 10.0})  # OPR = 60.0 > 55.0
        self.assertIn("CONST_DERIVED_OPR", rep_opr.violated_constraints)

    def test_06_flight_envelope_violations(self):
        """Asserts detection of unphysical altitude and Mach flight conditions."""
        # Unphysical high altitude (> 45,000 ft or 45 kft)
        rep_alt_high_ft = evaluate_constraints({"altitude": 52000.0})
        self.assertIn("CONST_FLIGHT_ALTITUDE", rep_alt_high_ft.violated_constraints)

        rep_alt_high_kft = evaluate_constraints({"altitude": 48.0})  # 48 kft = 48000 ft
        self.assertIn("CONST_FLIGHT_ALTITUDE", rep_alt_high_kft.violated_constraints)

        # Unphysical negative altitude (<-50 ft)
        rep_alt_neg = evaluate_constraints({"altitude": -500.0})
        self.assertIn("CONST_FLIGHT_ALTITUDE", rep_alt_neg.violated_constraints)

        # Transonic Mach limit exceedance (Mach > 0.90)
        rep_mach_high = evaluate_constraints({"mach_number": 0.95})
        self.assertIn("CONST_FLIGHT_MACH", rep_mach_high.violated_constraints)

        # Negative Mach exceedance (Mach < -0.01)
        rep_mach_neg = evaluate_constraints({"mach_number": -0.15})
        self.assertIn("CONST_FLIGHT_MACH", rep_mach_neg.violated_constraints)

    def test_07_aerothermal_ordering_invariants(self):
        """Asserts evaluation of thermodynamic order (T30 > T24 > T2, P30 > P15 > P2, Nc > Nf)."""
        nominal_core = {
            "T2": 518.67,
            "T24": 641.82,
            "T30": 1589.70,
            "P2": 14.62,
            "P15": 21.61,
            "P30": 554.36,
            "Nf": 2388.06,
            "Nc": 9046.19,
        }
        rep_valid = evaluate_constraints(nominal_core)
        self.assertNotIn("CONST_INVARIANT_TEMP_ORDER", rep_valid.violated_constraints)
        self.assertNotIn("CONST_INVARIANT_PRESS_ORDER", rep_valid.violated_constraints)
        self.assertNotIn("CONST_INVARIANT_SPOOL_ORDER", rep_valid.violated_constraints)

        # Temperature order inversion (T24 > T30)
        inv_temp = dict(nominal_core, T24=1650.0, T30=1550.0)
        rep_inv_temp = evaluate_constraints(inv_temp)
        self.assertIn("CONST_INVARIANT_TEMP_ORDER", rep_inv_temp.violated_constraints)

        # Pressure order inversion (P15 > P30)
        inv_press = dict(nominal_core, P15=580.0, P30=550.0)
        rep_inv_press = evaluate_constraints(inv_press)
        self.assertIn("CONST_INVARIANT_PRESS_ORDER", rep_inv_press.violated_constraints)

        # Spool speed inversion (Nf > Nc)
        inv_spool = dict(nominal_core, Nf=9500.0, Nc=2388.0)
        rep_inv_spool = evaluate_constraints(inv_spool)
        self.assertIn("CONST_INVARIANT_SPOOL_ORDER", rep_inv_spool.violated_constraints)

    def test_08_prognostics_bounds_violations(self):
        """Asserts detection of negative RUL or invalid health index."""
        # Negative RUL
        rep_neg_rul = evaluate_constraints({"predicted_rul": -12.5})
        self.assertIn("CONST_PROG_RUL_NON_NEGATIVE", rep_neg_rul.violated_constraints)

        rep_neg_xgb_rul = evaluate_constraints({"predicted_rul_xgboost": -4.0})
        self.assertIn("CONST_PROG_RUL_NON_NEGATIVE", rep_neg_xgb_rul.violated_constraints)

        # Valid RUL
        rep_pos_rul = evaluate_constraints({"predicted_rul": 45.0, "predicted_rul_xgboost": 42.0})
        self.assertNotIn("CONST_PROG_RUL_NON_NEGATIVE", rep_pos_rul.violated_constraints)

        # Health index bounds [0.0, 1.0]
        rep_hi_high = evaluate_constraints({"virtual_health_index": 1.25})
        self.assertIn("CONST_PROG_HEALTH_INDEX", rep_hi_high.violated_constraints)

        rep_hi_neg = evaluate_constraints({"virtual_health_index": -0.10})
        self.assertIn("CONST_PROG_HEALTH_INDEX", rep_hi_neg.violated_constraints)

        rep_hi_valid = evaluate_constraints({"virtual_health_index": 0.85})
        self.assertNotIn("CONST_PROG_HEALTH_INDEX", rep_hi_valid.violated_constraints)

    def test_09_nominal_state_evaluation(self):
        """Evaluates a nominal Digital Twin EngineState; asserts zero violations."""
        sim = DigitalTwinSimulator(subset="FD001")
        sim.load_engine(unit_number=1, split="train")
        state = sim.current_state

        # Enrich state with nominal prognostics metadata
        state.virtual_health_index = 0.98
        state.predicted_rul_xgboost = 112.0
        state.predicted_rul_lstm = 110.0

        report = evaluate_constraints(state)
        self.assertTrue(report.is_valid, f"Nominal state reported violations: {report.violations}")
        self.assertEqual(report.violation_count, 0)
        self.assertEqual(len(report.violations), 0)
        self.assertGreater(report.passed_count, 15)

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
                f"Raw dataset file '{fname}' was modified during Step 7.1 execution!",
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
                f"Scaler file '{fname}' was modified during Step 7.1 execution!",
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
                f"Champion checkpoint '{fname}' was modified during Step 7.1 execution!",
            )


if __name__ == "__main__":
    unittest.main()
