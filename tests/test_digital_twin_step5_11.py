"""
Unit and Integration Tests for Phase 5 Step 5.11: Digital Twin Assumptions, Taxonomy & Labeling
Verifies complete 40-variable state taxonomy, strict separation of measured vs simulated values,
scenario provenance and audit metadata tagging, unperturbed baseline defaults, and artifact immutability.

Roadmap Reference: Phase 5 - Step 5.11:
"Document assumptions and clearly label simulated values separately from measured dataset values."
"""

import hashlib
import json
import math
import os
import unittest
import numpy as np
import pandas as pd

from digital_twin.engine_state import EngineState, StateClassification, STATE_TAXONOMY
from digital_twin.twin_simulator import DigitalTwinSimulator
from digital_twin.scenario_interface import (
    create_thermal_stress_scenario,
    create_sensor_bias_scenario,
)
from digital_twin.verify_digital_twin_contracts import (
    DigitalTwinContractVerifier,
    TaxonomyContractResult,
    UnperturbedDefaultsResult,
    ProvenanceLabelingResult,
    ScenarioPerturbationContractResult,
)


class TestDigitalTwinStep5_11(unittest.TestCase):
    """
    Verification suite for Phase 5 Step 5.11: Digital Twin Assumptions, Taxonomy & Labeling.
    """

    @classmethod
    def setUpClass(cls):
        cls.raw_data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.checkpoint_dir = os.path.join("models", "checkpoints")

        # 1. Capture SHA-256 hashes of all 12 raw C-MAPSS dataset files
        cls.raw_hashes = {}
        for fname in sorted(os.listdir(cls.raw_data_dir)):
            if fname.endswith(".txt") and fname != "readme.txt":
                fpath = os.path.join(cls.raw_data_dir, fname)
                with open(fpath, "rb") as f:
                    cls.raw_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 2. Capture SHA-256 hashes of all 4 subset scalers
        cls.scaler_hashes = {}
        for fname in sorted(os.listdir(cls.scalers_dir)):
            if fname.endswith(".pkl"):
                fpath = os.path.join(cls.scalers_dir, fname)
                with open(fpath, "rb") as f:
                    cls.scaler_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 3. Capture SHA-256 hashes of all 8 champion model checkpoints
        cls.checkpoint_hashes = {}
        for fname in sorted(os.listdir(cls.checkpoint_dir)):
            if fname.endswith((".json", ".pt")) and not fname.endswith("manifest.json"):
                fpath = os.path.join(cls.checkpoint_dir, fname)
                with open(fpath, "rb") as f:
                    cls.checkpoint_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        cls.verifier = DigitalTwinContractVerifier(
            raw_data_dir=cls.raw_data_dir,
            scalers_dir=cls.scalers_dir,
            checkpoint_dir=cls.checkpoint_dir,
        )

    def test_01_complete_taxonomy_conformance(self):
        """
        Verify that STATE_TAXONOMY and EngineState cover exactly all 40 variables G1 to G40.
        """
        res = self.verifier.verify_taxonomy_conformance()
        self.assertTrue(res.is_valid)
        self.assertEqual(res.total_variables, 40)
        self.assertEqual(res.measured_count, 26)
        self.assertEqual(res.derived_count, 7)
        self.assertEqual(res.predicted_count, 5)
        self.assertEqual(res.simulated_count, 2)
        self.assertTrue(res.all_attributes_present)
        self.assertEqual(len(res.missing_attributes), 0)

    def test_02_classification_map_integrity(self):
        """
        Verify that EngineState.get_classification_map() correctly maps all 40 variables.
        """
        cmap = EngineState.get_classification_map()
        self.assertEqual(len(cmap), 40)

        # G1 - G26 are Measured
        for i in range(1, 27):
            gid = f"G{i}"
            self.assertEqual(cmap[gid], StateClassification.MEASURED)

        # G27 - G31 are Derived
        for i in range(27, 32):
            gid = f"G{i}"
            self.assertEqual(cmap[gid], StateClassification.DERIVED)

        # G32 - G36 are Predicted
        for i in range(32, 37):
            gid = f"G{i}"
            self.assertEqual(cmap[gid], StateClassification.PREDICTED)

        # G37 and G38 are Derived (Uncertainty / Proposed Health Index)
        self.assertEqual(cmap["G37"], StateClassification.DERIVED)
        self.assertEqual(cmap["G38"], StateClassification.DERIVED)

        # G39 and G40 are Simulated
        self.assertEqual(cmap["G39"], StateClassification.SIMULATED)
        self.assertEqual(cmap["G40"], StateClassification.SIMULATED)

    def test_03_unperturbed_baseline_defaults_contract(self):
        """
        Verify that under unperturbed baseline simulation:
        - G32-G34 (Anomaly tracking) default to None.
        - G35-G36 (ML prognostics) default to None when ML disabled.
        - G37-G38 (Uncertainty / health index) default to None when ML disabled.
        - G39-G40 (What-if scenario parameters) default to None.
        - scenario_id and scenario_audit default to None.
        """
        res = self.verifier.verify_unperturbed_baseline_contract(subset="FD001", unit_number=1)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.g32_to_g34_none)
        self.assertTrue(res.g35_to_g36_none_when_ml_disabled)
        self.assertTrue(res.g37_to_g38_none_when_ml_disabled)
        self.assertTrue(res.g39_to_g40_none)
        self.assertTrue(res.scenario_audit_none)

    def test_04_separation_of_measured_and_simulated_values(self):
        """
        Verify strict separation of measured vs simulated values:
        - is_simulation_only is True exclusively for G39 and G40.
        - All G1-G38 variables have is_simulation_only == False.
        """
        taxonomy = EngineState.get_taxonomy()
        for gid, info in taxonomy.items():
            if gid in ("G39", "G40"):
                self.assertTrue(info["is_simulation_only"], f"{gid} must be marked is_simulation_only=True")
                self.assertEqual(info["classification"], StateClassification.SIMULATED)
            else:
                self.assertFalse(info["is_simulation_only"], f"{gid} must NOT be marked is_simulation_only=True")
                self.assertNotEqual(info["classification"], StateClassification.SIMULATED)

    def test_05_provenance_labeling_in_to_labeled_dict(self):
        """
        Verify that EngineState.to_labeled_dict() explicitly annotates each variable with
        classification, engineering units, category, and simulation flags.
        """
        res = self.verifier.verify_provenance_and_labeling_contract(subset="FD001", unit_number=1)
        self.assertTrue(res.is_valid)
        self.assertTrue(res.provenance_metadata_present)
        self.assertTrue(res.measured_labeled_correctly)
        self.assertTrue(res.derived_labeled_correctly)
        self.assertTrue(res.predicted_labeled_correctly)
        self.assertTrue(res.simulated_labeled_correctly)

    def test_06_scenario_thermal_stress_provenance_and_isolation(self):
        """
        Verify that applying a thermal stress scenario (+15 °R):
        - Populates G39 with exact offset (+15.0).
        - Leaves G40 as None.
        - Attaches non-empty scenario_audit with matching scenario_id and delta.
        - Leaves raw measured telemetry in raw_history unmodified.
        """
        sc = create_thermal_stress_scenario(
            scenario_id="SC_TEST_STEP5_11_THERMAL",
            target="T30",
            offset_degR=15.0,
            name="Step 5.11 Thermal Provenance Test",
        )
        sim = DigitalTwinSimulator(
            subset="FD001",
            data_dir=self.raw_data_dir,
            scenario=sc,
        )
        sim.load_engine(unit_number=1, split="train")
        st = sim.step()

        self.assertEqual(st.thermal_stress_offset, 15.0)
        self.assertIsNone(st.sensor_bias_delta)
        self.assertEqual(st.scenario_id, "SC_TEST_STEP5_11_THERMAL")
        self.assertIsNotNone(st.scenario_audit)
        self.assertEqual(st.scenario_audit["scenario_id"], "SC_TEST_STEP5_11_THERMAL")

        # Verify raw ground truth in raw_history remains unmodified
        orig_t30 = sim.trajectory_df.iloc[0]["sensor_3"]
        raw_hist_t30 = sim.raw_history[0][5]  # sensor_3 is at index 5 in ORDERED_SCALER_FEATURES
        self.assertEqual(raw_hist_t30, orig_t30)

    def test_07_scenario_sensor_bias_provenance_and_isolation(self):
        """
        Verify that applying a sensor bias scenario (+10 psia to P30):
        - Populates G40 with {'P30': 10.0}.
        - Leaves G39 as None.
        - Attaches non-empty scenario_audit with matching scenario_id and parameter.
        - Leaves raw measured telemetry in raw_history unmodified.
        """
        sc = create_sensor_bias_scenario(
            scenario_id="SC_TEST_STEP5_11_BIAS",
            target="P30",
            bias=10.0,
            unit="psia",
            name="Step 5.11 Bias Provenance Test",
        )
        sim = DigitalTwinSimulator(
            subset="FD002",
            data_dir=self.raw_data_dir,
            scenario=sc,
        )
        sim.load_engine(unit_number=1, split="train")
        st = sim.step()

        self.assertIsNone(st.thermal_stress_offset)
        self.assertIsNotNone(st.sensor_bias_delta)
        self.assertEqual(st.sensor_bias_delta.get("P30"), 10.0)
        self.assertEqual(st.scenario_id, "SC_TEST_STEP5_11_BIAS")
        self.assertIsNotNone(st.scenario_audit)
        self.assertEqual(st.scenario_audit["scenario_id"], "SC_TEST_STEP5_11_BIAS")

        # Verify raw ground truth in raw_history remains unmodified
        orig_p30 = sim.trajectory_df.iloc[0]["sensor_7"]
        raw_hist_p30 = sim.raw_history[0][9]  # sensor_7 is at index 9 in ORDERED_SCALER_FEATURES
        self.assertEqual(raw_hist_p30, orig_p30)

    def test_08_contract_verification_reports_exist(self):
        """
        Verify that Step 5.11 contract verification reports exist and validate all contracts.
        """
        out_dir = os.path.join("reports", "digital_twin", "step5_11")
        json_file = os.path.join(out_dir, "contract_verification.json")
        md_file = os.path.join(out_dir, "contract_verification.md")

        self.assertTrue(os.path.exists(json_file))
        self.assertTrue(os.path.exists(md_file))

        with open(json_file, "r") as f:
            data = json.load(f)

        self.assertTrue(data["all_contracts_verified"])
        self.assertTrue(data["taxonomy_contract"]["is_valid"])
        self.assertTrue(data["unperturbed_defaults_contract"]["is_valid"])
        self.assertTrue(data["provenance_and_labeling_contract"]["is_valid"])
        self.assertTrue(data["scenario_perturbation_contract"]["is_valid"])

    def test_09_raw_dataset_before_and_after_immutability(self):
        """
        Verify that raw C-MAPSS dataset files were not modified during the test session.
        Compares SHA-256 hashes captured in setUpClass with current hashes.
        """
        self.assertGreaterEqual(len(self.raw_hashes), 12)
        for fname, initial_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                initial_hash,
                f"Raw dataset file {fname} was modified during test execution!",
            )

    def test_10_model_scalers_and_checkpoints_immutability(self):
        """
        Verify that model scalers and champion checkpoints were not modified during the test session.
        Compares SHA-256 hashes captured in setUpClass with current hashes.
        """
        # Scalers
        self.assertEqual(len(self.scaler_hashes), 4)
        for fname, initial_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                initial_hash,
                f"Scaler file {fname} was modified during test execution!",
            )

        # Checkpoints
        self.assertEqual(len(self.checkpoint_hashes), 8)
        for fname, initial_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                initial_hash,
                f"Checkpoint file {fname} was modified during test execution!",
            )


if __name__ == "__main__":
    unittest.main()
