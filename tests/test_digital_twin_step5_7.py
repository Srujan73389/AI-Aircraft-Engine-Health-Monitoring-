"""
Unit and Integration Tests for Phase 5 Step 5.7: What-If Scenario Interface
Verifies controlled simulation scenarios (Thermal Stress, Sensor Bias, Accelerated Aging).

Roadmap Reference: Phase 5 - Step 5.7
Scope: Academic simulation interface validation without dataset or model modification.
"""

import hashlib
import json
import math
import os
import unittest
import numpy as np
import pandas as pd

from digital_twin.engine_state import EngineState
from digital_twin.scenario_interface import (
    ScenarioCategory,
    ScenarioConfig,
    ScenarioValidator,
    ScenarioModifier,
    create_thermal_stress_scenario,
    create_sensor_bias_scenario,
    create_accelerated_aging_scenario,
    ORDERED_SCALER_FEATURES,
    THERMAL_STRESS_TARGETS,
    SENSOR_BIAS_LIMITS,
    ACCELERATED_AGING_LIMITS,
)
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestDigitalTwinStep5_7(unittest.TestCase):
    """
    Comprehensive verification suite for Digital Twin What-If Scenario Interface.
    Enforces canonical bounds, simulation-only constraints, audit retention, and artifact immutability.
    """

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.ckpt_dir = os.path.join("models", "checkpoints")

        # 1. Capture SHA-256 hashes of all raw C-MAPSS dataset files
        cls.raw_file_hashes = {}
        for fname in sorted(os.listdir(cls.data_dir)):
            if fname.endswith(".txt"):
                fpath = os.path.join(cls.data_dir, fname)
                with open(fpath, "rb") as f:
                    cls.raw_file_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 2. Capture SHA-256 hashes of all four subset scalers
        cls.scaler_hashes = {}
        for fname in sorted(os.listdir(cls.scalers_dir)):
            if fname.endswith(".pkl"):
                fpath = os.path.join(cls.scalers_dir, fname)
                with open(fpath, "rb") as f:
                    cls.scaler_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # 3. Capture SHA-256 hashes of all eight Phase 4 champion checkpoints
        cls.ckpt_hashes = {}
        for fname in sorted(os.listdir(cls.ckpt_dir)):
            if fname.startswith(("xgb_", "lstm_")) and fname.endswith(("_champion.json", "_champion.pt")):
                fpath = os.path.join(cls.ckpt_dir, fname)
                with open(fpath, "rb") as f:
                    cls.ckpt_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        # Verify expected artifact counts are present at session start
        assert len(cls.raw_file_hashes) >= 12, f"Expected >= 12 raw dataset files, found {len(cls.raw_file_hashes)}."
        assert len(cls.scaler_hashes) == 4, f"Expected 4 scaler files, found {len(cls.scaler_hashes)}."
        assert len(cls.ckpt_hashes) == 8, f"Expected 8 champion checkpoint files, found {len(cls.ckpt_hashes)}."

    # -------------------------------------------------------------------------
    # 1. Thermal Stress Scenario Configuration & Canonical Validation
    # -------------------------------------------------------------------------
    def test_01_valid_thermal_stress_config(self):
        """Verify creation and attributes of valid Thermal Stress scenarios."""
        sc_t30 = create_thermal_stress_scenario(
            scenario_id="SCEN_TS_01",
            target="T30",
            offset_degR=35.0,
            description="High pressure compressor exit temperature elevation",
        )
        self.assertEqual(sc_t30.scenario_id, "SCEN_TS_01")
        self.assertEqual(sc_t30.category, ScenarioCategory.THERMAL_STRESS)
        self.assertEqual(sc_t30.target_parameter, "T30")
        self.assertEqual(sc_t30.modification_value, 35.0)
        self.assertEqual(sc_t30.unit, "°R")
        self.assertEqual(sc_t30.validation_range, (-100.0, 200.0))
        self.assertTrue(sc_t30.is_simulation_only)

        # Test T50 negative thermal offset (within [-100, 200])
        sc_t50 = create_thermal_stress_scenario(
            scenario_id="SCEN_TS_02",
            target="T50",
            offset_degR=-25.0,
        )
        self.assertEqual(sc_t50.modification_value, -25.0)

    def test_02_invalid_thermal_offset_out_of_bounds(self):
        """Verify that out-of-bounds thermal offsets raise ValueError and are never clipped."""
        # Upper bound exceeded (+250.0 > +200.0)
        with self.assertRaises(ValueError) as ctx:
            create_thermal_stress_scenario(
                scenario_id="SCEN_ERR_01",
                target="T30",
                offset_degR=250.0,
            )
        self.assertIn("outside canonical validation range", str(ctx.exception))

        # Lower bound exceeded (-150.0 < -100.0)
        with self.assertRaises(ValueError) as ctx:
            create_thermal_stress_scenario(
                scenario_id="SCEN_ERR_02",
                target="T30",
                offset_degR=-150.0,
            )
        self.assertIn("outside canonical validation range", str(ctx.exception))

    def test_03_invalid_thermal_stress_non_temperature_channel(self):
        """Verify rejection when Thermal Stress target is not a temperature sensor."""
        with self.assertRaises(ValueError) as ctx:
            ScenarioConfig(
                scenario_id="SCEN_ERR_03",
                category=ScenarioCategory.THERMAL_STRESS,
                name="Invalid Pressure Target",
                target_parameter="P30",
                modification_value=20.0,
                unit="°R",
                validation_range=(-100.0, 200.0),
                description="Invalid thermal target",
            )
        self.assertIn("Invalid target 'P30' for Thermal Stress", str(ctx.exception))

    def test_04_thermal_stress_rejects_custom_validation_range_bypass(self):
        """Verify that callers cannot broaden canonical thermal bounds via custom validation_range."""
        # Attempt to bypass [-100, 200] with widened custom range [-500, 500]
        with self.assertRaises(ValueError) as ctx:
            ScenarioConfig(
                scenario_id="SCEN_BYPASS_01",
                category=ScenarioCategory.THERMAL_STRESS,
                name="Bypass Thermal Bounds Attempt",
                target_parameter="T30",
                modification_value=300.0,
                unit="°R",
                validation_range=(-500.0, 500.0),
                description="Malicious custom range",
            )
        self.assertIn("Custom validation_range (-500.0, 500.0) is prohibited", str(ctx.exception))
        self.assertIn("Must match canonical range (-100.0, 200.0)", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 2. Sensor Bias Scenario Configuration & Canonical Validation
    # -------------------------------------------------------------------------
    def test_05_valid_sensor_bias_configs(self):
        """Verify valid Sensor Bias configurations across different physical sensor types."""
        # Pressure bias (psia)
        sc_p30 = create_sensor_bias_scenario(
            scenario_id="SCEN_SB_01",
            target="P30",
            bias=12.5,
            unit="psia",
        )
        self.assertEqual(sc_p30.category, ScenarioCategory.SENSOR_BIAS)
        self.assertEqual(sc_p30.target_parameter, "P30")
        self.assertEqual(sc_p30.modification_value, 12.5)

        # Spool speed bias (rpm)
        sc_nc = create_sensor_bias_scenario(
            scenario_id="SCEN_SB_02",
            target="Nc",
            bias=-150.0,
            unit="rpm",
        )
        self.assertEqual(sc_nc.modification_value, -150.0)

        # Altitude setting bias (kft)
        sc_alt = create_sensor_bias_scenario(
            scenario_id="SCEN_SB_03",
            target="altitude",
            bias=3.0,
            unit="kft",
        )
        self.assertEqual(sc_alt.modification_value, 3.0)

        # TRA setting bias (%)
        sc_tra = create_sensor_bias_scenario(
            scenario_id="SCEN_SB_04",
            target="TRA",
            bias=-5.0,
            unit="%",
        )
        self.assertEqual(sc_tra.modification_value, -5.0)

    def test_06_invalid_sensor_selection_bias(self):
        """Verify rejection when target is not a recognized C-MAPSS channel."""
        with self.assertRaises(ValueError) as ctx:
            create_sensor_bias_scenario(
                scenario_id="SCEN_ERR_04",
                target="non_existent_vibration_sensor",
                bias=1.0,
                unit="g",
            )
        self.assertIn("not in known sensor bias limits dictionary", str(ctx.exception))

    def test_07_invalid_sensor_bias_range(self):
        """Verify rejection when bias exceeds validated canonical parameter limits."""
        # Pressure bias exceeding range (-50, 100)
        with self.assertRaises(ValueError) as ctx:
            create_sensor_bias_scenario(
                scenario_id="SCEN_ERR_05",
                target="P30",
                bias=150.0,
                unit="psia",
            )
        self.assertIn("outside canonical validation range", str(ctx.exception))

        # Speed bias exceeding range (-500, 500)
        with self.assertRaises(ValueError) as ctx:
            create_sensor_bias_scenario(
                scenario_id="SCEN_ERR_06",
                target="Nf",
                bias=600.0,
                unit="rpm",
            )
        self.assertIn("outside canonical validation range", str(ctx.exception))

    def test_08_invalid_sensor_bias_unit(self):
        """Verify rejection when sensor bias unit does not match physical channel."""
        with self.assertRaises(ValueError) as ctx:
            create_sensor_bias_scenario(
                scenario_id="SCEN_ERR_07",
                target="P30",
                bias=10.0,
                unit="rpm",
            )
        self.assertIn("Invalid unit 'rpm' for target 'P30'", str(ctx.exception))

    def test_09_sensor_bias_rejects_custom_validation_range_bypass(self):
        """Verify that callers cannot broaden canonical sensor bias limits via custom ranges."""
        # Attempt to bypass P30 canonical range (-50.0, 100.0) with (-500.0, 500.0)
        with self.assertRaises(ValueError) as ctx:
            ScenarioConfig(
                scenario_id="SCEN_BYPASS_02",
                category=ScenarioCategory.SENSOR_BIAS,
                name="Bypass Pressure Bounds Attempt",
                target_parameter="P30",
                modification_value=300.0,
                unit="psia",
                validation_range=(-500.0, 500.0),
                description="Malicious custom pressure range",
            )
        self.assertIn("Custom validation_range (-500.0, 500.0) is prohibited", str(ctx.exception))
        self.assertIn("Must match canonical range (-50.0, 100.0)", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 3. Strict Numerical Safety, Simulation-Only Enforcements, & Finiteness
    # -------------------------------------------------------------------------
    def test_10_nan_and_infinity_rejection(self):
        """Verify that NaN and +/- infinity are strictly rejected in modification values."""
        # NaN modification value
        with self.assertRaises(ValueError) as ctx:
            ScenarioConfig(
                scenario_id="SCEN_ERR_08",
                category=ScenarioCategory.THERMAL_STRESS,
                name="NaN Thermal Stress",
                target_parameter="T30",
                modification_value=float("nan"),
                unit="°R",
                validation_range=(-100.0, 200.0),
                description="NaN test",
            )
        self.assertIn("must be a finite numeric value", str(ctx.exception))

        # Positive infinity modification value
        with self.assertRaises(ValueError) as ctx:
            ScenarioConfig(
                scenario_id="SCEN_ERR_09",
                category=ScenarioCategory.THERMAL_STRESS,
                name="Inf Thermal Stress",
                target_parameter="T30",
                modification_value=float("inf"),
                unit="°R",
                validation_range=(-100.0, 200.0),
                description="Inf test",
            )
        self.assertIn("must be a finite numeric value", str(ctx.exception))

    def test_11_optional_values_finiteness_validation(self):
        """Verify that optional baseline_value and simulated_value must be finite when provided."""
        # Non-finite baseline_value
        with self.assertRaises(ValueError) as ctx:
            ScenarioConfig(
                scenario_id="SCEN_ERR_OPT_01",
                category=ScenarioCategory.THERMAL_STRESS,
                name="Non-finite Baseline Test",
                target_parameter="T30",
                modification_value=10.0,
                unit="°R",
                validation_range=(-100.0, 200.0),
                description="Non-finite baseline test",
                baseline_value=float("nan"),
            )
        self.assertIn("baseline_value must be a finite numeric value", str(ctx.exception))

        # Non-finite simulated_value
        with self.assertRaises(ValueError) as ctx:
            ScenarioConfig(
                scenario_id="SCEN_ERR_OPT_02",
                category=ScenarioCategory.THERMAL_STRESS,
                name="Non-finite Simulated Test",
                target_parameter="T30",
                modification_value=10.0,
                unit="°R",
                validation_range=(-100.0, 200.0),
                description="Non-finite simulated test",
                simulated_value=float("inf"),
            )
        self.assertIn("simulated_value must be a finite numeric value", str(ctx.exception))

    def test_12_is_simulation_only_enforcement(self):
        """Verify that is_simulation_only=False or non-boolean types are strictly rejected in constructor."""
        # False boolean
        with self.assertRaises(ValueError) as ctx:
            ScenarioConfig(
                scenario_id="SCEN_HARDWARE_REJECT",
                category=ScenarioCategory.THERMAL_STRESS,
                name="Hardware Injection Attempt",
                target_parameter="T30",
                modification_value=10.0,
                unit="°R",
                validation_range=(-100.0, 200.0),
                description="Hardware test",
                is_simulation_only=False,
            )
        self.assertIn("is_simulation_only must be True", str(ctx.exception))

        # Integer 0
        with self.assertRaises(ValueError) as ctx:
            ScenarioConfig(
                scenario_id="SCEN_INT_0",
                category=ScenarioCategory.THERMAL_STRESS,
                name="Int 0 Test",
                target_parameter="T30",
                modification_value=10.0,
                unit="°R",
                validation_range=(-100.0, 200.0),
                description="Int 0 test",
                is_simulation_only=0,
            )
        self.assertIn("must be an actual Python boolean", str(ctx.exception))

        # Integer 1
        with self.assertRaises(ValueError) as ctx:
            ScenarioConfig(
                scenario_id="SCEN_INT_1",
                category=ScenarioCategory.THERMAL_STRESS,
                name="Int 1 Test",
                target_parameter="T30",
                modification_value=10.0,
                unit="°R",
                validation_range=(-100.0, 200.0),
                description="Int 1 test",
                is_simulation_only=1,
            )
        self.assertIn("must be an actual Python boolean", str(ctx.exception))

    def test_12b_strict_boolean_from_dict_validation(self):
        """
        Verify strict Boolean deserialization validation in ScenarioConfig.from_dict():
        - Missing flag defaults to True.
        - Actual Boolean True is accepted.
        - Actual Boolean False is rejected.
        - String 'false' is rejected.
        - String 'true' is rejected.
        - String '0' and 'yes' are rejected.
        - Integer 0 and 1 are rejected.
        """
        base_dict = {
            "scenario_id": "SCEN_BOOL_TEST",
            "category": "thermal_stress",
            "name": "Boolean Test",
            "target_parameter": "T30",
            "modification_value": 25.0,
            "unit": "°R",
            "validation_range": [-100.0, 200.0],
            "description": "Boolean test",
        }

        # 1. Missing flag defaults to True
        d1 = dict(base_dict)
        cfg1 = ScenarioConfig.from_dict(d1)
        self.assertIs(cfg1.is_simulation_only, True)

        # 2. Actual Boolean True is accepted
        d2 = dict(base_dict, is_simulation_only=True)
        cfg2 = ScenarioConfig.from_dict(d2)
        self.assertIs(cfg2.is_simulation_only, True)

        # 3. Actual Boolean False is rejected
        d3 = dict(base_dict, is_simulation_only=False)
        with self.assertRaises(ValueError) as ctx3:
            ScenarioConfig.from_dict(d3)
        self.assertIn("is_simulation_only must be True", str(ctx3.exception))

        # 4. String "false" is rejected
        d4 = dict(base_dict, is_simulation_only="false")
        with self.assertRaises(ValueError) as ctx4:
            ScenarioConfig.from_dict(d4)
        self.assertIn("is_simulation_only must be an actual Python boolean", str(ctx4.exception))

        # 5. String "true" is rejected
        d5 = dict(base_dict, is_simulation_only="true")
        with self.assertRaises(ValueError) as ctx5:
            ScenarioConfig.from_dict(d5)
        self.assertIn("is_simulation_only must be an actual Python boolean", str(ctx5.exception))

        # 6. String "0" and "yes" are rejected
        for bad_str in ["0", "yes"]:
            d_str = dict(base_dict, is_simulation_only=bad_str)
            with self.assertRaises(ValueError) as ctx_str:
                ScenarioConfig.from_dict(d_str)
            self.assertIn("is_simulation_only must be an actual Python boolean", str(ctx_str.exception))

        # 7. Integers 0 and 1 are rejected
        for bad_int in [0, 1]:
            d_int = dict(base_dict, is_simulation_only=bad_int)
            with self.assertRaises(ValueError) as ctx_int:
                ScenarioConfig.from_dict(d_int)
            self.assertIn("is_simulation_only must be an actual Python boolean", str(ctx_int.exception))

    def test_13_unsupported_scenario_category_rejection(self):
        """Verify rejection of unsupported scenario categories."""
        with self.assertRaises(ValueError) as ctx:
            ScenarioConfig(
                scenario_id="SCEN_ERR_11",
                category="turbofan_engine_fire",
                name="Unsupported Category",
                target_parameter="T30",
                modification_value=50.0,
                unit="°R",
                validation_range=(-100.0, 200.0),
                description="Unsupported category test",
            )
        self.assertIn("Unsupported scenario category", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 4. Accelerated Aging Deferral & Temporal Sequence Integrity
    # -------------------------------------------------------------------------
    def test_14_accelerated_aging_rejects_custom_validation_range_bypass(self):
        """Verify that callers cannot broaden canonical accelerated aging range [1.0, 3.0]."""
        with self.assertRaises(ValueError) as ctx:
            ScenarioConfig(
                scenario_id="SCEN_BYPASS_03",
                category=ScenarioCategory.ACCELERATED_AGING,
                name="Bypass Aging Bounds Attempt",
                target_parameter="aging_multiplier",
                modification_value=5.0,
                unit="multiplier",
                validation_range=(0.5, 10.0),
                description="Malicious custom aging range",
            )
        self.assertIn("Custom validation_range (0.5, 10.0) is prohibited for Accelerated Aging", str(ctx.exception))

    def test_15_accelerated_aging_deferred_execution_in_simulator(self):
        """
        Verify that accelerated aging execution is formally disabled in the simulator,
        preventing invalid cycle skipping or discontinuous sequences from reaching ML models.
        """
        sc_aging = create_accelerated_aging_scenario(
            scenario_id="SCEN_AA_DEFERRED",
            aging_multiplier=2.0,
        )

        sim = DigitalTwinSimulator(subset="FD001", data_dir=self.data_dir)
        sim.load_engine(unit_number=1, split="train")

        # 1. Setting accelerated aging scenario on simulator must raise NotImplementedError
        with self.assertRaises(NotImplementedError) as ctx:
            sim.set_scenario(sc_aging)
        self.assertIn("Accelerated aging execution is formally deferred", str(ctx.exception))
        self.assertIn("discontinuous sequences", str(ctx.exception))

        # 2. Passing accelerated aging directly to step() must raise NotImplementedError
        with self.assertRaises(NotImplementedError) as ctx:
            sim.step(scenario=sc_aging)
        self.assertIn("Accelerated aging execution is formally deferred", str(ctx.exception))

        # 3. Passing accelerated aging to simulator constructor must raise NotImplementedError
        with self.assertRaises(NotImplementedError) as ctx:
            DigitalTwinSimulator(subset="FD001", data_dir=self.data_dir, scenario=sc_aging)
        self.assertIn("Accelerated aging execution is formally deferred", str(ctx.exception))

    def test_16_accelerated_aging_audit_documents_deferral(self):
        """Verify that standalone accelerated aging modification explicitly documents deferral."""
        sc_aging = create_accelerated_aging_scenario(
            scenario_id="SCEN_AA_AUDIT",
            aging_multiplier=2.5,
        )
        dummy_raw = np.ones(24, dtype=np.float64) * 10.0
        sim_raw, audit = ScenarioModifier.apply_scenario(dummy_raw, sc_aging)

        np.testing.assert_array_equal(dummy_raw, sim_raw)
        self.assertTrue(audit.get("internal_wear_laws_deferred"))
        self.assertIn("Accelerated aging execution is formally deferred", audit.get("limitation_note", ""))

    # -------------------------------------------------------------------------
    # 5. Scenario Serialization & Reproducibility
    # -------------------------------------------------------------------------
    def test_17_scenario_reproducibility_from_dict(self):
        """Verify that a scenario configuration is fully reproducible via serialization."""
        original_config = create_thermal_stress_scenario(
            scenario_id="SCEN_REPRO_01",
            target="T30",
            offset_degR=45.0,
            description="Reproducibility test scenario",
        )

        d = original_config.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["scenario_id"], "SCEN_REPRO_01")
        self.assertEqual(d["modification_value"], 45.0)

        restored_config = ScenarioConfig.from_dict(d)
        self.assertEqual(restored_config.scenario_id, original_config.scenario_id)
        self.assertEqual(restored_config.category, original_config.category)
        self.assertEqual(restored_config.target_parameter, original_config.target_parameter)
        self.assertEqual(restored_config.modification_value, original_config.modification_value)
        self.assertEqual(restored_config.unit, original_config.unit)
        self.assertEqual(restored_config.validation_range, original_config.validation_range)
        self.assertEqual(restored_config.description, original_config.description)

    # -------------------------------------------------------------------------
    # 6. Simulator Integration, Audit Persistence, & Telemetry Preservation
    # -------------------------------------------------------------------------
    def test_18_measured_sensor_values_remain_unchanged_under_scenario(self):
        """
        Verify that original measured telemetry values (G3 - G26) remain 100% untouched
        in the engine state and raw history when a scenario is active.
        """
        sim = DigitalTwinSimulator(subset="FD001", data_dir=self.data_dir)
        sc = create_thermal_stress_scenario(
            scenario_id="SCEN_TS_ISOLATION",
            target="T30",
            offset_degR=50.0,
        )

        sim.load_engine(unit_number=1, split="train")
        sim.set_scenario(sc)
        s2 = sim.step()

        raw_row_cycle2 = sim.trajectory_df.iloc[1]
        raw_t30_cycle2 = float(raw_row_cycle2["sensor_3"])

        # Ground truth preservation: state.T30 MUST match raw C-MAPSS recorded telemetry
        self.assertEqual(s2.T30, raw_t30_cycle2, "state.T30 must reflect uncorrupted measured telemetry.")
        self.assertEqual(sim.raw_history[1][ORDERED_SCALER_FEATURES.index("sensor_3")], raw_t30_cycle2)

        # Hypothetical offset is recorded explicitly in G39
        self.assertEqual(s2.thermal_stress_offset, 50.0)
        self.assertEqual(s2.scenario_id, "SCEN_TS_ISOLATION")

    def test_19_scenario_audit_persistence_and_serialization(self):
        """
        Verify that scenario audit information is completely preserved in EngineState
        and DigitalTwinSimulator, and is properly serialized.
        """
        sim = DigitalTwinSimulator(subset="FD001", data_dir=self.data_dir)
        sc = create_sensor_bias_scenario(
            scenario_id="SCEN_AUDIT_VERIFY",
            target="P30",
            bias=15.0,
            unit="psia",
        )
        sim.load_engine(unit_number=1, split="train")
        sim.set_scenario(sc)
        s2 = sim.step()

        # Audit must be present on state
        self.assertIsNotNone(s2.scenario_audit, "EngineState must retain scenario_audit.")
        audit = s2.scenario_audit

        # Required audit fields
        self.assertEqual(audit["scenario_id"], "SCEN_AUDIT_VERIFY")
        self.assertEqual(audit["category"], "sensor_bias")
        self.assertEqual(audit["target_parameter"], "P30")
        self.assertEqual(audit["unit"], "psia")
        self.assertTrue(audit["is_simulation_only"])
        self.assertEqual(audit["applied_modification"], 15.0)

        raw_p30 = float(sim.trajectory_df.iloc[1]["sensor_7"])
        self.assertAlmostEqual(audit["original_baseline_value"], raw_p30, places=4)
        self.assertAlmostEqual(audit["resulting_simulated_value"], raw_p30 + 15.0, places=4)

        # Simulator instance must retain audit
        self.assertEqual(sim.current_scenario_audit, audit)
        self.assertIn(audit, sim.scenario_audit_history)

        # Serialization check
        d = s2.to_dict()
        self.assertIn("scenario_audit", d)
        self.assertEqual(d["scenario_audit"]["scenario_id"], "SCEN_AUDIT_VERIFY")

    def test_20_baseline_simulation_remains_identical_without_scenario(self):
        """
        Verify that running simulation with scenario=None produces virtual engine states
        numerically and structurally identical to an independent unperturbed baseline run
        across recorded flight cycles (testing both padded and unpadded rolling history windows):
        - G1 - G2 tracking metadata
        - G3 - G5 operating settings
        - G6 - G26 gas-path sensors
        - G27 - G29 macroscopic consistency indicators
        - G30 normalized feature vectors
        - G31 temporal sequence tensors
        - G35 - G37 Phase 4 ML prognostics (XGBoost RUL, LSTM RUL, prognostic spread)
        - G39, G40, and scenario_audit remaining None
        """
        sim_unperturbed = DigitalTwinSimulator(
            subset="FD001", data_dir=self.data_dir, enable_ml_inference=True
        )
        sim_no_scenario = DigitalTwinSimulator(
            subset="FD001", data_dir=self.data_dir, enable_ml_inference=True, scenario=None
        )

        s_unpert = sim_unperturbed.load_engine(unit_number=1, split="train")
        s_no_scen = sim_no_scenario.load_engine(unit_number=1, split="train")

        sensor_fields = [
            "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc",
            "epr", "Ps30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed",
            "Nf_dmd", "PCNfR_dmd", "W31", "W32"
        ]

        # Step through 35 cycles to cover both replication-padded (1-29) and unpadded (30-35) windows
        for step_i in range(35):
            if step_i > 0:
                s_unpert = sim_unperturbed.step()
                s_no_scen = sim_no_scenario.step()

            # G1 - G2 metadata
            self.assertEqual(s_unpert.unit_number, s_no_scen.unit_number)
            self.assertEqual(s_unpert.time_cycles, s_no_scen.time_cycles)

            # G3 - G5 operating settings
            self.assertEqual(s_unpert.altitude, s_no_scen.altitude)
            self.assertEqual(s_unpert.mach_number, s_no_scen.mach_number)
            self.assertEqual(s_unpert.throttle_resolver_angle, s_no_scen.throttle_resolver_angle)

            # G6 - G26 gas-path sensors
            for field in sensor_fields:
                self.assertEqual(getattr(s_unpert, field), getattr(s_no_scen, field))

            # G27 - G29 macroscopic indicators
            self.assertEqual(s_unpert.hpc_temp_rise, s_no_scen.hpc_temp_rise)
            self.assertEqual(s_unpert.overall_pressure_ratio, s_no_scen.overall_pressure_ratio)
            self.assertEqual(s_unpert.spool_speed_ratio, s_no_scen.spool_speed_ratio)

            # G30 & G31 feature vector and sequence tensor
            np.testing.assert_allclose(s_unpert.normalized_feature_vec, s_no_scen.normalized_feature_vec, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(s_unpert.temporal_seq_tensor, s_no_scen.temporal_seq_tensor, rtol=1e-7, atol=1e-7)

            # G35 - G37 ML prognostic predictions
            self.assertAlmostEqual(s_unpert.predicted_rul_xgboost, s_no_scen.predicted_rul_xgboost, places=5)
            self.assertAlmostEqual(s_unpert.predicted_rul_lstm, s_no_scen.predicted_rul_lstm, places=5)
            self.assertAlmostEqual(s_unpert.prognostic_spread, s_no_scen.prognostic_spread, places=5)

            # G39, G40, and audit must remain None
            self.assertIsNone(s_no_scen.thermal_stress_offset)
            self.assertIsNone(s_no_scen.sensor_bias_delta)
            self.assertIsNone(s_no_scen.scenario_id)
            self.assertIsNone(s_no_scen.scenario_audit)

        self.assertIsNone(sim_no_scenario.current_scenario_audit)

    def test_21_clear_scenario_restores_baseline_simulation(self):
        """Verify that calling clear_scenario() cleanly restores baseline simulation and clears audit."""
        sim = DigitalTwinSimulator(subset="FD001", data_dir=self.data_dir)
        sc = create_thermal_stress_scenario(
            scenario_id="SCEN_TEMP",
            target="T30",
            offset_degR=30.0,
        )
        sim.load_engine(unit_number=1, split="train")
        sim.set_scenario(sc)
        s2 = sim.step()
        self.assertEqual(s2.thermal_stress_offset, 30.0)
        self.assertIsNotNone(s2.scenario_audit)

        # Clear scenario
        sim.clear_scenario()
        s3 = sim.step()
        self.assertIsNone(s3.thermal_stress_offset)
        self.assertIsNone(s3.sensor_bias_delta)
        self.assertIsNone(s3.scenario_id)
        self.assertIsNone(s3.scenario_audit)
        self.assertIsNone(sim.current_scenario_audit)

    # -------------------------------------------------------------------------
    # 7. ML Model Prognostics Under What-If Scenarios
    # -------------------------------------------------------------------------
    def test_22_ml_inference_under_what_if_scenario(self):
        """
        Verify that Phase 4 ML prognostics (XGBoost & LSTM) execute cleanly
        on the simulated sequence tensor under what-if scenarios without errors or shape mismatch.
        """
        sim = DigitalTwinSimulator(
            subset="FD001",
            data_dir=self.data_dir,
            enable_ml_inference=True,
        )
        sim.load_engine(unit_number=1, split="train")

        # Baseline step
        s_baseline = sim.step()
        self.assertIsNotNone(s_baseline.predicted_rul_xgboost)
        self.assertIsNotNone(s_baseline.predicted_rul_lstm)
        self.assertTrue(math.isfinite(s_baseline.predicted_rul_xgboost))
        self.assertTrue(math.isfinite(s_baseline.predicted_rul_lstm))

        # Thermal stress scenario step
        sc = create_thermal_stress_scenario(
            scenario_id="SCEN_ML_TEST",
            target="T30",
            offset_degR=40.0,
        )
        sim.set_scenario(sc)
        s_sim = sim.step()

        self.assertIsNotNone(s_sim.predicted_rul_xgboost)
        self.assertIsNotNone(s_sim.predicted_rul_lstm)
        self.assertIsNotNone(s_sim.prognostic_spread)
        self.assertTrue(math.isfinite(s_sim.predicted_rul_xgboost))
        self.assertTrue(math.isfinite(s_sim.predicted_rul_lstm))
        self.assertTrue(math.isfinite(s_sim.prognostic_spread))
        self.assertEqual(s_sim.thermal_stress_offset, 40.0)
        self.assertIsNotNone(s_sim.scenario_audit)

    # -------------------------------------------------------------------------
    # 8. Before-and-After Test-Session Immutability Safeguards
    # -------------------------------------------------------------------------
    def test_23_raw_dataset_files_before_and_after_immutability(self):
        """Verify before-and-after test-session immutability of all raw C-MAPSS dataset files."""
        for fname, original_hash in self.raw_file_hashes.items():
            fpath = os.path.join(self.data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Raw dataset file '{fname}' was modified! SHA-256 mismatch.",
            )

    def test_24_model_scalers_before_and_after_immutability(self):
        """Verify before-and-after test-session immutability of all 4 Phase 3 model scalers."""
        self.assertEqual(len(self.scaler_hashes), 4, "Expected exactly 4 scalers.")
        for fname, original_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Model scaler file '{fname}' was modified! SHA-256 mismatch.",
            )

    def test_25_model_checkpoints_before_and_after_immutability(self):
        """Verify before-and-after test-session immutability of all 8 Phase 4 champion checkpoints."""
        self.assertEqual(len(self.ckpt_hashes), 8, "Expected exactly 8 champion checkpoints.")
        for fname, original_hash in self.ckpt_hashes.items():
            fpath = os.path.join(self.ckpt_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Champion checkpoint file '{fname}' was modified! SHA-256 mismatch.",
            )


if __name__ == "__main__":
    unittest.main()
