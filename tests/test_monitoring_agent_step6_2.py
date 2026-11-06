"""
Unit and Integration Tests for Phase 6 Step 6.2: Autonomous Monitoring Agent
Verifies independent operation of MonitoringAgent including nominal telemetry ingestion,
physical threshold breach detection, statistical anomaly score evaluation, 3-cycle persistence logic,
trajectory state isolation and reset, Digital Twin integration, input validation, and dataset immutability.

Roadmap Reference: Phase 6 - Step 6.2:
"Implement and test the Monitoring Agent independently"
"""

import hashlib
import json
import math
import os
import unittest
import numpy as np

from agents.schemas import (
    AgentExecutionStatus,
    SeverityLevel,
    OperationalState,
    AgentMetadata,
    ThresholdBreach,
    MonitoringInput,
    MonitoringOutput,
)
from agents.monitoring_agent import MonitoringAgent
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestMonitoringAgentStep6_2(unittest.TestCase):
    """
    Test suite for Phase 6 Step 6.2: Autonomous Monitoring Agent.
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
        """Instantiate a fresh MonitoringAgent for each test."""
        self.agent = MonitoringAgent(agent_id="test_monitoring_agent", subset="FD001")

    def _create_nominal_input(
        self,
        unit_number: int = 1,
        time_cycles: int = 25,
        t30: float = 1585.0,
        p30: float = 554.0,
        if_score: float = -0.080,
        pca_score: float = 0.85,
    ) -> MonitoringInput:
        """Helper to create nominal, thermodynamically consistent telemetry."""
        return MonitoringInput(
            unit_number=unit_number,
            time_cycles=time_cycles,
            operating_context={
                "altitude": 0.001,
                "mach_number": 0.0002,
                "throttle_resolver_angle": 100.0,
            },
            raw_telemetry={
                "T2": 518.67,
                "T24": 642.35,
                "T30": t30,
                "T50": 1400.25,
                "P2": 14.62,
                "P15": 21.61,
                "P30": p30,
                "Nf": 2388.05,
                "Nc": 9045.20,
                "epr": 1.30,
                "Ps30": 47.45,
                "phi": 521.80,
                "NRf": 2388.08,
                "NRc": 8130.40,
                "BPR": 8.42,
                "farB": 0.03,
                "htBleed": 392.0,
                "Nf_dmd": 2388.0,
                "PCNfR_dmd": 100.0,
                "W31": 38.85,
                "W32": 23.35,
            },
            derived_indicators={
                "hpc_temp_rise": t30 - 642.35,
                "overall_pressure_ratio": p30 / 14.62,
                "spool_speed_ratio": 9045.20 / 2388.05,
            },
            anomaly_scores={
                "isolation_forest": if_score,
                "pca_spe": pca_score,
            },
        )

    # -------------------------------------------------------------------------
    # Test 1: Nominal Telemetry Ingestion
    # -------------------------------------------------------------------------
    def test_01_nominal_telemetry_ingestion(self):
        """
        Verify that nominal telemetry produces HEALTHY state, zero breaches, and zero persistence.
        """
        inp = self._create_nominal_input()
        output = self.agent.process_input(inp)

        self.assertIsInstance(output, MonitoringOutput)
        self.assertEqual(output.operational_state, OperationalState.HEALTHY)
        self.assertEqual(output.anomaly_severity, SeverityLevel.NOMINAL)
        self.assertEqual(len(output.threshold_breaches), 0)
        self.assertFalse(output.model_anomaly_flag)
        self.assertFalse(output.isolation_forest_breach)
        self.assertFalse(output.pca_spe_breach)
        self.assertEqual(output.persistence_count, 0)
        self.assertFalse(output.persistence_alert)
        self.assertIn("nominal", output.summary_message.lower())

        # Metadata validation
        self.assertIsNotNone(output.metadata)
        self.assertEqual(output.metadata.agent_id, "test_monitoring_agent")
        self.assertEqual(output.metadata.status, AgentExecutionStatus.SUCCESS)
        self.assertGreaterEqual(output.metadata.execution_time_ms, 0.0)

    # -------------------------------------------------------------------------
    # Test 2: Threshold Breach Detection on High Temperature (T30)
    # -------------------------------------------------------------------------
    def test_02_threshold_breach_high_t30(self):
        """
        Verify detection and severity assignment when HPC outlet temperature T30 exceeds caution & warning limits.
        """
        # A. Caution breach: T30 = 1612.0 °R (> caution_high 1605.0, < warning_high 1625.0)
        inp_caution = self._create_nominal_input(t30=1612.0)
        out_caution = self.agent.process_input(inp_caution)

        self.assertEqual(out_caution.operational_state, OperationalState.ANOMALOUS)
        self.assertEqual(out_caution.anomaly_severity, SeverityLevel.CAUTION)
        t30_breaches = [b for b in out_caution.threshold_breaches if b.sensor_name == "T30"]
        self.assertEqual(len(t30_breaches), 1)
        self.assertEqual(t30_breaches[0].severity, SeverityLevel.CAUTION)
        self.assertEqual(t30_breaches[0].direction, "HIGH")
        self.assertAlmostEqual(t30_breaches[0].breach_margin, 7.0, places=3)
        self.assertEqual(out_caution.persistence_count, 1)
        self.assertFalse(out_caution.persistence_alert)

        # B. Warning breach: T30 = 1630.0 °R (> warning_high 1625.0)
        self.agent.reset_trajectory(1)
        inp_warning = self._create_nominal_input(t30=1630.0)
        out_warning = self.agent.process_input(inp_warning)

        self.assertEqual(out_warning.operational_state, OperationalState.CRITICAL)
        self.assertEqual(out_warning.anomaly_severity, SeverityLevel.WARNING)
        t30_warning = [b for b in out_warning.threshold_breaches if b.sensor_name == "T30"]
        self.assertEqual(len(t30_warning), 1)
        self.assertEqual(t30_warning[0].severity, SeverityLevel.WARNING)
        self.assertEqual(t30_warning[0].direction, "HIGH")
        self.assertAlmostEqual(t30_warning[0].breach_margin, 5.0, places=3)

    # -------------------------------------------------------------------------
    # Test 3: Threshold Breach Detection on High Pressure (P30)
    # -------------------------------------------------------------------------
    def test_03_threshold_breach_high_p30(self):
        """
        Verify detection and severity assignment when HPC outlet pressure P30 exceeds caution & warning limits.
        """
        # A. Caution breach: P30 = 565.0 psia (> caution_high 560.0, < warning_high 575.0)
        inp_caution = self._create_nominal_input(p30=565.0)
        out_caution = self.agent.process_input(inp_caution)

        p30_breaches = [b for b in out_caution.threshold_breaches if b.sensor_name == "P30"]
        self.assertEqual(len(p30_breaches), 1)
        self.assertEqual(p30_breaches[0].severity, SeverityLevel.CAUTION)
        self.assertEqual(p30_breaches[0].direction, "HIGH")
        self.assertAlmostEqual(p30_breaches[0].breach_margin, 5.0, places=3)

        # B. Warning breach: P30 = 580.0 psia (> warning_high 575.0)
        self.agent.reset_trajectory(1)
        inp_warning = self._create_nominal_input(p30=580.0)
        out_warning = self.agent.process_input(inp_warning)

        self.assertEqual(out_warning.operational_state, OperationalState.CRITICAL)
        p30_warning = [b for b in out_warning.threshold_breaches if b.sensor_name == "P30"]
        self.assertEqual(len(p30_warning), 1)
        self.assertEqual(p30_warning[0].severity, SeverityLevel.WARNING)
        self.assertAlmostEqual(p30_warning[0].breach_margin, 5.0, places=3)

    # -------------------------------------------------------------------------
    # Test 4: Statistical Anomaly Detection (Isolation Forest & PCA-SPE)
    # -------------------------------------------------------------------------
    def test_04_statistical_anomaly_detection(self):
        """
        Verify that Isolation Forest (M1) and PCA-SPE (M2) scores exceeding calibrated thresholds trigger alerts.
        """
        # FD001 thresholds: iForest tau = -0.031165, PCA-SPE tau = 1.610083
        # Case A: Only Isolation Forest breaches
        inp_if = self._create_nominal_input(if_score=-0.010, pca_score=0.90)  # -0.010 > -0.031165
        out_if = self.agent.process_input(inp_if)

        self.assertTrue(out_if.model_anomaly_flag)
        self.assertTrue(out_if.isolation_forest_breach)
        self.assertFalse(out_if.pca_spe_breach)
        self.assertEqual(out_if.operational_state, OperationalState.ANOMALOUS)
        self.assertEqual(out_if.anomaly_severity, SeverityLevel.ADVISORY)
        self.assertEqual(out_if.persistence_count, 1)

        # Case B: Only PCA-SPE breaches
        self.agent.reset_trajectory(1)
        inp_pca = self._create_nominal_input(if_score=-0.060, pca_score=2.25)  # 2.25 > 1.610083
        out_pca = self.agent.process_input(inp_pca)

        self.assertTrue(out_pca.model_anomaly_flag)
        self.assertFalse(out_pca.isolation_forest_breach)
        self.assertTrue(out_pca.pca_spe_breach)
        self.assertEqual(out_pca.operational_state, OperationalState.ANOMALOUS)
        self.assertEqual(out_pca.persistence_count, 1)

        # Case C: Both models breach simultaneously
        self.agent.reset_trajectory(1)
        inp_both = self._create_nominal_input(if_score=0.05, pca_score=3.10)
        out_both = self.agent.process_input(inp_both)

        self.assertTrue(out_both.model_anomaly_flag)
        self.assertTrue(out_both.isolation_forest_breach)
        self.assertTrue(out_both.pca_spe_breach)
        self.assertIn("iForest", out_both.summary_message)
        self.assertIn("PCA-SPE", out_both.summary_message)

    # -------------------------------------------------------------------------
    # Test 5: 3-Cycle Persistence Logic
    # -------------------------------------------------------------------------
    def test_05_three_cycle_persistence_logic(self):
        """
        Verify that persistence_alert becomes True exactly on the 3rd consecutive anomalous cycle,
        remains True on subsequent anomalous cycles, and resets immediately on a nominal cycle.
        """
        engine_id = 42

        # Cycle 1: Anomalous (Caution breach) -> count=1, alert=False
        inp_c1 = self._create_nominal_input(unit_number=engine_id, time_cycles=101, t30=1610.0)
        out_c1 = self.agent.process_input(inp_c1)
        self.assertEqual(out_c1.persistence_count, 1)
        self.assertFalse(out_c1.persistence_alert)
        self.assertEqual(out_c1.operational_state, OperationalState.ANOMALOUS)

        # Cycle 2: Anomalous (Statistical anomaly) -> count=2, alert=False
        inp_c2 = self._create_nominal_input(unit_number=engine_id, time_cycles=102, if_score=0.02)
        out_c2 = self.agent.process_input(inp_c2)
        self.assertEqual(out_c2.persistence_count, 2)
        self.assertFalse(out_c2.persistence_alert)
        self.assertEqual(out_c2.operational_state, OperationalState.ANOMALOUS)

        # Cycle 3: Anomalous -> count=3, alert=True (DEGRADED state confirmed!)
        inp_c3 = self._create_nominal_input(unit_number=engine_id, time_cycles=103, t30=1612.0)
        out_c3 = self.agent.process_input(inp_c3)
        self.assertEqual(out_c3.persistence_count, 3)
        self.assertTrue(out_c3.persistence_alert)
        self.assertEqual(out_c3.operational_state, OperationalState.DEGRADED)
        self.assertEqual(out_c3.anomaly_severity, SeverityLevel.CAUTION)
        self.assertIn("DEGRADED", out_c3.summary_message)

        # Cycle 4: Continues anomalous -> count=4, alert=True
        inp_c4 = self._create_nominal_input(unit_number=engine_id, time_cycles=104, t30=1614.0)
        out_c4 = self.agent.process_input(inp_c4)
        self.assertEqual(out_c4.persistence_count, 4)
        self.assertTrue(out_c4.persistence_alert)
        self.assertEqual(out_c4.operational_state, OperationalState.DEGRADED)

        # Cycle 5: Healthy cycle arrives -> count resets to 0, alert clears!
        inp_c5 = self._create_nominal_input(unit_number=engine_id, time_cycles=105, t30=1585.0)
        out_c5 = self.agent.process_input(inp_c5)
        self.assertEqual(out_c5.persistence_count, 0)
        self.assertFalse(out_c5.persistence_alert)
        self.assertEqual(out_c5.operational_state, OperationalState.HEALTHY)
        self.assertEqual(out_c5.anomaly_severity, SeverityLevel.NOMINAL)

    # -------------------------------------------------------------------------
    # Test 6: Trajectory State Isolation & Reset
    # -------------------------------------------------------------------------
    def test_06_trajectory_isolation_and_reset(self):
        """
        Verify that multiple engines maintain isolated persistence states, and reset_trajectory works cleanly.
        """
        # Feed 2 anomalous cycles to Engine 1
        self.agent.process_input(self._create_nominal_input(unit_number=1, time_cycles=10, t30=1612.0))
        self.agent.process_input(self._create_nominal_input(unit_number=1, time_cycles=11, t30=1612.0))
        self.assertEqual(self.agent.get_persistence_count(1), 2)

        # Feed 1 anomalous cycle to Engine 2
        self.agent.process_input(self._create_nominal_input(unit_number=2, time_cycles=5, t30=1615.0))
        self.assertEqual(self.agent.get_persistence_count(2), 1)
        self.assertEqual(self.agent.get_persistence_count(1), 2)  # Engine 1 unaffected

        # Reset Engine 1 specifically
        self.agent.reset_trajectory(unit_number=1)
        self.assertEqual(self.agent.get_persistence_count(1), 0)
        self.assertEqual(self.agent.get_persistence_count(2), 1)  # Engine 2 untouched

        # Global reset
        self.agent.reset_trajectory()
        self.assertEqual(self.agent.get_persistence_count(2), 0)

    # -------------------------------------------------------------------------
    # Test 7: Integration with Digital Twin EngineState
    # -------------------------------------------------------------------------
    def test_07_digital_twin_state_integration(self):
        """
        Verify ingestion of real replayed states from DigitalTwinSimulator across early and late cycles.
        """
        # Configure monitoring agent with calibrated baseline thresholds
        agent = MonitoringAgent(
            agent_id="twin_monitoring_agent",
            subset="FD001",
            sensor_thresholds={"T30": {"caution_high": 1600.0, "warning_high": 1620.0}},
        )

        sim = DigitalTwinSimulator(subset="FD001")
        sim.load_engine(unit_number=1, split="train")

        # Step through early healthy cycles (cycles 1 to 30)
        for _ in range(29):
            sim.step()
        state_early = sim.current_state

        out_early = agent.process_engine_state(state_early)
        self.assertIsInstance(out_early, MonitoringOutput)
        self.assertEqual(out_early.unit_number, 1)
        self.assertEqual(out_early.time_cycles, 30)
        self.assertEqual(out_early.operational_state, OperationalState.HEALTHY)
        self.assertEqual(len(out_early.threshold_breaches), 0)

        # Step through until end of trajectory (Unit 1 has 192 cycles, degraded at late life)
        while sim.current_state.time_cycles < sim.total_cycles:
            sim.step()
        state_late = sim.current_state

        # Process late-life state (T30 = 1601.41 deg R > 1600.0 caution threshold)
        out_late = agent.process_engine_state(state_late)
        self.assertIsInstance(out_late, MonitoringOutput)
        self.assertEqual(out_late.unit_number, 1)
        self.assertEqual(out_late.time_cycles, sim.total_cycles)
        self.assertIn(out_late.operational_state, (OperationalState.ANOMALOUS, OperationalState.DEGRADED))
        t30_breaches = [b for b in out_late.threshold_breaches if b.sensor_name == "T30"]
        self.assertGreaterEqual(len(t30_breaches), 1)

    # -------------------------------------------------------------------------
    # Test 8: Macroscopic Consistency & Thermodynamic Inversion Checks
    # -------------------------------------------------------------------------
    def test_08_macroscopic_consistency_checks(self):
        """
        Verify that physical thermodynamic inversions (T30 <= T24, P30 <= P2, Nc <= Nf) are detected as CRITICAL.
        """
        # A. Temperature inversion: T30 (600.0) <= T24 (642.35)
        inp_temp_inv = self._create_nominal_input(t30=600.0)
        out_temp_inv = self.agent.process_input(inp_temp_inv)
        self.assertEqual(out_temp_inv.operational_state, OperationalState.CRITICAL)
        self.assertEqual(out_temp_inv.anomaly_severity, SeverityLevel.WARNING)
        inv_breach = [b for b in out_temp_inv.threshold_breaches if "INVERSION" in b.sensor_name]
        self.assertGreaterEqual(len(inv_breach), 1)

        # B. Pressure inversion: P30 (10.0) <= P2 (14.62)
        self.agent.reset_trajectory(1)
        inp_press_inv = self._create_nominal_input(p30=10.0)
        out_press_inv = self.agent.process_input(inp_press_inv)
        self.assertEqual(out_press_inv.operational_state, OperationalState.CRITICAL)
        p_inv = [b for b in out_press_inv.threshold_breaches if "P30_P2_INVERSION" in b.sensor_name]
        self.assertEqual(len(p_inv), 1)

    # -------------------------------------------------------------------------
    # Test 9: Input Validation & Non-Finite Rejection
    # -------------------------------------------------------------------------
    def test_09_input_validation_and_non_finite_rejection(self):
        """
        Verify graceful rejection of malformed payloads (NaN, Inf, negative unit/cycles, missing attributes).
        """
        # A. NaN in telemetry
        inp_nan = self._create_nominal_input()
        inp_nan.raw_telemetry["T30"] = float("nan")
        with self.assertRaises(ValueError):
            self.agent.process_input(inp_nan)

        # B. Inf in telemetry
        inp_inf = self._create_nominal_input()
        inp_inf.raw_telemetry["P30"] = float("inf")
        with self.assertRaises(ValueError):
            self.agent.process_input(inp_inf)

        # C. Invalid unit_number
        inp_bad_unit = self._create_nominal_input(unit_number=-1)
        with self.assertRaises(ValueError):
            self.agent.process_input(inp_bad_unit)

        # D. Invalid time_cycles
        inp_bad_cycle = self._create_nominal_input(time_cycles=0)
        with self.assertRaises(ValueError):
            self.agent.process_input(inp_bad_cycle)

        # E. Malformed object in process_engine_state
        with self.assertRaises(TypeError):
            self.agent.process_engine_state({"dummy": 123})

    # -------------------------------------------------------------------------
    # Test 10: Immutability of Raw Datasets, Scalers, and Checkpoints
    # -------------------------------------------------------------------------
    def test_10_dataset_and_model_immutability(self):
        """
        Cryptographically verify that zero raw dataset files, scalers, or model checkpoints were altered.
        """
        # Verify raw datasets
        for fname, original_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Raw C-MAPSS dataset file '{fname}' was modified during test execution!",
            )

        # Verify scalers
        for fname, original_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Phase 3 scaler '{fname}' was modified during test execution!",
            )

        # Verify champion checkpoints
        for fname, original_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Phase 4 model checkpoint '{fname}' was modified during test execution!",
            )


if __name__ == "__main__":
    unittest.main()
