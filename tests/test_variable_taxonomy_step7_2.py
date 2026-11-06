"""
Unit and Safety Verification Test Suite for Step 7.2
Verifies Variable Taxonomy, Provenance Classifications, and Immutability Firewall.

Roadmap Reference: Phase 7 - Step 7.2
"""

import os
import hashlib
import unittest
from typing import Dict, Any

from safety.taxonomy import (
    VariableCategory,
    VariableProvenance,
    ProvenanceViolationError,
    StateVariableClassificationReport,
    VARIABLE_TAXONOMY,
    get_variable_provenance,
    classify_state_variables,
    verify_provenance_immutability,
)
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestVariableTaxonomyStep7_2(unittest.TestCase):
    """
    Test suite for Phase 7 Step 7.2: Variable Taxonomy and Provenance Definitions.
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

    def test_01_taxonomy_registry_completeness(self):
        """Asserts all G1–G40 and auxiliary configuration variables exist in taxonomy."""
        # 1. Check all canonical G1 through G40 entries
        for i in range(1, 41):
            gid = f"G{i}"
            self.assertIn(
                gid,
                VARIABLE_TAXONOMY,
                f"Taxonomy registry must define entry for state vector variable '{gid}'.",
            )
            prov = VARIABLE_TAXONOMY[gid]
            self.assertEqual(prov.variable_id, gid)
            self.assertTrue(len(prov.canonical_name) > 0)
            self.assertTrue(len(prov.units) > 0)
            self.assertTrue(len(prov.description) > 0)

        # 2. Check auxiliary entries
        self.assertIn("M1", VARIABLE_TAXONOMY)
        self.assertIn("M2", VARIABLE_TAXONOMY)
        self.assertIn("CFG_SAMPLING_FREQ", VARIABLE_TAXONOMY)
        self.assertIn("CFG_SUBSET", VARIABLE_TAXONOMY)
        self.assertIn("CFG_T30_CAUTION", VARIABLE_TAXONOMY)
        self.assertIn("CFG_T30_WARNING", VARIABLE_TAXONOMY)

    def test_02_category_enumeration_coverage(self):
        """Asserts all 4 categories (MEASURED, PREDICTED, SIMULATED, MANUALLY_CONFIGURED) are utilized."""
        all_categories = {prov.category for prov in VARIABLE_TAXONOMY.values()}
        self.assertEqual(len(all_categories), 4)
        self.assertIn(VariableCategory.MEASURED, all_categories)
        self.assertIn(VariableCategory.PREDICTED, all_categories)
        self.assertIn(VariableCategory.SIMULATED, all_categories)
        self.assertIn(VariableCategory.MANUALLY_CONFIGURED, all_categories)

    def test_03_measured_immutability_flags(self):
        """Asserts all G6–G26 measured sensors have is_immutable_in_twin = True."""
        measured_gids = [f"G{i}" for i in range(6, 27)]
        self.assertEqual(len(measured_gids), 21)

        for gid in measured_gids:
            prov = VARIABLE_TAXONOMY[gid]
            self.assertEqual(
                prov.category,
                VariableCategory.MEASURED,
                f"Variable {gid} must be categorized as MEASURED.",
            )
            self.assertTrue(
                prov.is_immutable_in_twin,
                f"Measured variable {gid} ({prov.canonical_name}) must have is_immutable_in_twin = True.",
            )

        # Asserts non-measured variables do NOT have immutable telemetry flag
        for gid in ["G1", "G2", "G3", "G4", "G5", "G27", "G28", "G32", "G35", "G38", "G39", "G40"]:
            prov = VARIABLE_TAXONOMY[gid]
            self.assertFalse(
                prov.is_immutable_in_twin,
                f"Non-measured variable {gid} should not be marked as immutable telemetry.",
            )

    def test_04_predicted_variable_classification(self):
        """Asserts RUL models and anomaly scores are categorized as PREDICTED."""
        predicted_gids = ["G32", "G33", "G34", "G35", "G36", "G37", "G38", "M1", "M2"]
        for gid in predicted_gids:
            prov = VARIABLE_TAXONOMY[gid]
            self.assertEqual(
                prov.category,
                VariableCategory.PREDICTED,
                f"Variable {gid} ({prov.canonical_name}) must be classified as PREDICTED.",
            )
            self.assertFalse(prov.is_immutable_in_twin)
            self.assertIsNotNone(prov.associated_models_or_equations)

    def test_05_simulated_variable_classification(self):
        """Asserts derived aerothermal indicators and what-if modifiers are SIMULATED."""
        simulated_gids = ["G27", "G28", "G29", "G30", "G31", "G39", "G40"]
        for gid in simulated_gids:
            prov = VARIABLE_TAXONOMY[gid]
            self.assertEqual(
                prov.category,
                VariableCategory.SIMULATED,
                f"Variable {gid} ({prov.canonical_name}) must be classified as SIMULATED.",
            )
            self.assertFalse(prov.is_immutable_in_twin)
            self.assertIsNotNone(prov.associated_models_or_equations)

    def test_06_manually_configured_classification(self):
        """Asserts flight conditions and operational thresholds are MANUALLY_CONFIGURED."""
        config_gids = ["G1", "G2", "G3", "G4", "G5", "CFG_SAMPLING_FREQ", "CFG_SUBSET", "CFG_T30_CAUTION", "CFG_T30_WARNING"]
        for gid in config_gids:
            prov = VARIABLE_TAXONOMY[gid]
            self.assertEqual(
                prov.category,
                VariableCategory.MANUALLY_CONFIGURED,
                f"Variable {gid} ({prov.canonical_name}) must be classified as MANUALLY_CONFIGURED.",
            )
            self.assertFalse(prov.is_immutable_in_twin)

    def test_07_classify_state_variables_execution(self):
        """Validates state variable partitioning against an EngineState instance."""
        sim = DigitalTwinSimulator(subset="FD001")
        sim.load_engine(unit_number=1, split="train")
        state = sim.current_state

        report = classify_state_variables(state)
        self.assertIsInstance(report, StateVariableClassificationReport)

        # Check Measured variables (G6-G26)
        self.assertGreaterEqual(len(report.measured), 21)
        self.assertIn("T30", report.measured)
        self.assertIn("P30", report.measured)
        self.assertIn("Nf", report.measured)
        self.assertEqual(report.measured["T30"], state.T30)

        # Check Manually Configured variables (G1-G5)
        self.assertGreaterEqual(len(report.manually_configured), 5)
        self.assertIn("unit_number", report.manually_configured)
        self.assertIn("altitude", report.manually_configured)
        self.assertEqual(report.manually_configured["unit_number"], state.unit_number)

        # Check Simulated / Derived variables (G27-G29)
        self.assertGreaterEqual(len(report.simulated), 3)
        self.assertIn("hpc_temp_rise", report.simulated)
        self.assertIn("overall_pressure_ratio", report.simulated)
        self.assertEqual(report.simulated["hpc_temp_rise"], state.hpc_temp_rise)

        # Check dictionary-style indexing
        self.assertIn("T30", report["measured"])
        self.assertIn("unit_number", report["manually_configured"])

    def test_08_verify_provenance_immutability_pass(self):
        """Confirms that safe state enrichment passes immutability verification."""
        sim = DigitalTwinSimulator(subset="FD001")
        sim.load_engine(unit_number=1, split="train")
        orig_state = sim.current_state

        # Create updated state enriched with ML predictions and simulation perturbations
        updated_dict = orig_state.to_dict()
        updated_dict["predicted_rul_xgboost"] = 112.5
        updated_dict["predicted_rul_lstm"] = 110.0
        updated_dict["virtual_health_index"] = 0.94
        updated_dict["anomaly_alert_flag"] = 0
        updated_dict["thermal_stress_offset"] = 2.0

        is_valid = verify_provenance_immutability(orig_state, updated_dict)
        self.assertTrue(is_valid)

    def test_09_verify_provenance_immutability_fail(self):
        """Confirms that attempting to mutate a MEASURED sensor raises an explicit error."""
        sim = DigitalTwinSimulator(subset="FD001")
        sim.load_engine(unit_number=1, split="train")
        orig_state = sim.current_state

        # Attempt to mutate measured sensor T30
        tampered_dict = orig_state.to_dict()
        tampered_dict["T30"] = orig_state.T30 + 50.0  # Illegal direct sensor override

        with self.assertRaises((ProvenanceViolationError, ValueError)) as ctx:
            verify_provenance_immutability(orig_state, tampered_dict)

        self.assertIn("Provenance immutability violation", str(ctx.exception))
        self.assertIn("T30", str(ctx.exception))

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
                f"Raw dataset file '{fname}' was modified during Step 7.2 execution!",
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
                f"Scaler file '{fname}' was modified during Step 7.2 execution!",
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
                f"Champion checkpoint '{fname}' was modified during Step 7.2 execution!",
            )


if __name__ == "__main__":
    unittest.main()
