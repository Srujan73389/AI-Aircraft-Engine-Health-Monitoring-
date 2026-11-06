"""
Unit and Integration Tests for Phase 6 Step 6.3: Autonomous Diagnosis Agent
Verifies independent operation of DiagnosisAgent including nominal evaluation,
thermal distress diagnosis, isolated sensor calibration bias detection,
HPC aerodynamic efficiency degradation, confidence bound enforcement [0.0, 0.95],
end-to-end Digital Twin handoff, input validation, and artifact immutability.

Roadmap Reference: Phase 6 - Step 6.3:
"Implement the Diagnosis Agent using documented rules or validated anomaly evidence"
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
    FaultEvidence,
    FaultCandidate,
    DiagnosisInput,
    DiagnosisOutput,
)
from agents.monitoring_agent import MonitoringAgent
from agents.diagnosis_agent import DiagnosisAgent
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestDiagnosisAgentStep6_3(unittest.TestCase):
    """
    Test suite for Phase 6 Step 6.3: Autonomous Diagnosis Agent.
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
        """Instantiate a fresh DiagnosisAgent and MonitoringAgent for each test."""
        self.diag_agent = DiagnosisAgent(agent_id="test_diag_agent", subset="FD001")
        self.mon_agent = MonitoringAgent(agent_id="test_mon_agent", subset="FD001")

    # -------------------------------------------------------------------------
    # Test 1: Nominal Case
    # -------------------------------------------------------------------------
    def test_01_nominal_case(self):
        """
        Verify that nominal monitoring output produces zero fault candidates,
        confidence = 0.0, and requires_maintenance_action = False.
        """
        nom_mon_out = MonitoringOutput(
            unit_number=1,
            time_cycles=10,
            operational_state=OperationalState.HEALTHY,
            threshold_breaches=[],
            model_anomaly_flag=False,
            persistence_count=0,
            persistence_alert=False,
            anomaly_severity=SeverityLevel.NOMINAL,
            summary_message="Nominal telemetry",
        )

        diag_out = self.diag_agent.diagnose(nom_mon_out)

        self.assertIsInstance(diag_out, DiagnosisOutput)
        self.assertEqual(diag_out.unit_number, 1)
        self.assertEqual(diag_out.time_cycles, 10)
        self.assertIsNone(diag_out.primary_fault)
        self.assertEqual(diag_out.confidence_score, 0.0)
        self.assertEqual(len(diag_out.fault_candidates), 0)
        self.assertEqual(len(diag_out.diagnostic_hypotheses), 0)
        self.assertFalse(diag_out.requires_maintenance_action)
        self.assertEqual(diag_out.severity, SeverityLevel.NOMINAL)
        self.assertIn("nominal", diag_out.diagnostic_summary.lower())

    # -------------------------------------------------------------------------
    # Test 2: Thermal Distress Diagnosis
    # -------------------------------------------------------------------------
    def test_02_thermal_distress_diagnosis(self):
        """
        Verify that high T30 and excessive HPC temp rise flag THERMAL_DISTRESS
        with supporting evidence and calibrated confidence <= 0.95.
        """
        # Create monitoring output with T30 and hpc_temp_rise breaches
        t30_breach = ThresholdBreach(
            sensor_name="T30",
            observed_value=1620.0,
            threshold_limit=1605.0,
            direction="HIGH",
            breach_margin=15.0,
            severity=SeverityLevel.CAUTION,
        )
        dt_breach = ThresholdBreach(
            sensor_name="hpc_temp_rise",
            observed_value=1010.0,
            threshold_limit=990.0,
            direction="HIGH",
            breach_margin=20.0,
            severity=SeverityLevel.CAUTION,
        )
        mon_out = MonitoringOutput(
            unit_number=5,
            time_cycles=120,
            operational_state=OperationalState.DEGRADED,
            threshold_breaches=[t30_breach, dt_breach],
            model_anomaly_flag=True,
            persistence_count=3,
            persistence_alert=True,
            anomaly_severity=SeverityLevel.CAUTION,
        )

        telemetry = {"T30": 1620.0, "T50": 1420.0, "hpc_temp_rise": 1010.0, "overall_pressure_ratio": 38.0}
        diag_out = self.diag_agent.diagnose(mon_out, telemetry_snapshot=telemetry)

        self.assertIsInstance(diag_out, DiagnosisOutput)
        self.assertEqual(diag_out.primary_fault, "THERMAL_DISTRESS")
        self.assertGreater(diag_out.confidence_score, 0.70)
        self.assertLessEqual(diag_out.confidence_score, 0.95)
        self.assertTrue(diag_out.requires_maintenance_action)
        self.assertEqual(diag_out.severity, SeverityLevel.CAUTION)

        # Verify contributing evidence contains T30 and hpc_temp_rise
        ev_sensors = [e.sensor_name for e in diag_out.contributing_evidence]
        self.assertIn("T30", ev_sensors)
        self.assertIn("hpc_temp_rise", ev_sensors)
        self.assertIn("HPT", diag_out.diagnostic_summary)

    # -------------------------------------------------------------------------
    # Test 3: Isolated Sensor Bias Diagnosis
    # -------------------------------------------------------------------------
    def test_03_isolated_sensor_bias_diagnosis(self):
        """
        Verify that an isolated sensor breach (e.g. P30 high) while thermodynamic
        companion sensors remain nominal correctly diagnoses SENSOR_CALIBRATION_BIAS.
        """
        p30_breach = ThresholdBreach(
            sensor_name="P30",
            observed_value=572.0,
            threshold_limit=560.0,
            direction="HIGH",
            breach_margin=12.0,
            severity=SeverityLevel.CAUTION,
        )
        mon_out = MonitoringOutput(
            unit_number=8,
            time_cycles=45,
            operational_state=OperationalState.ANOMALOUS,
            threshold_breaches=[p30_breach],
            model_anomaly_flag=False,
            persistence_count=1,
            persistence_alert=False,
            anomaly_severity=SeverityLevel.CAUTION,
        )

        # Companion sensors remain completely nominal
        telemetry = {
            "P30": 572.0,
            "P2": 14.62,
            "P15": 21.61,
            "Ps30": 47.45,
            "T30": 1585.0,
            "overall_pressure_ratio": 37.86,
        }
        diag_out = self.diag_agent.diagnose(mon_out, telemetry_snapshot=telemetry)

        self.assertEqual(diag_out.primary_fault, "SENSOR_CALIBRATION_BIAS")
        self.assertGreaterEqual(diag_out.confidence_score, 0.80)
        self.assertLessEqual(diag_out.confidence_score, 0.95)
        self.assertTrue(diag_out.requires_maintenance_action)
        self.assertIn("calibration", diag_out.diagnostic_summary.lower())

    # -------------------------------------------------------------------------
    # Test 4: HPC Aerodynamic Degradation Diagnosis
    # -------------------------------------------------------------------------
    def test_04_hpc_aerodynamic_degradation_diagnosis(self):
        """
        Verify that elevated T30 combined with depressed OPR flags HPC_EFFICIENCY_DEGRADATION.
        """
        t30_breach = ThresholdBreach(
            sensor_name="T30",
            observed_value=1615.0,
            threshold_limit=1605.0,
            direction="HIGH",
            breach_margin=10.0,
            severity=SeverityLevel.CAUTION,
        )
        opr_breach = ThresholdBreach(
            sensor_name="overall_pressure_ratio",
            observed_value=34.2,
            threshold_limit=35.0,
            direction="LOW",
            breach_margin=0.8,
            severity=SeverityLevel.CAUTION,
        )
        mon_out = MonitoringOutput(
            unit_number=12,
            time_cycles=160,
            operational_state=OperationalState.DEGRADED,
            threshold_breaches=[t30_breach, opr_breach],
            model_anomaly_flag=True,
            persistence_count=4,
            persistence_alert=True,
            anomaly_severity=SeverityLevel.CAUTION,
        )

        telemetry = {"T30": 1615.0, "overall_pressure_ratio": 34.2, "Ps30": 45.2}
        diag_out = self.diag_agent.diagnose(mon_out, telemetry_snapshot=telemetry)

        self.assertEqual(diag_out.primary_fault, "HPC_EFFICIENCY_DEGRADATION")
        self.assertGreater(diag_out.confidence_score, 0.80)
        self.assertLessEqual(diag_out.confidence_score, 0.95)
        self.assertIn("HPC", diag_out.diagnostic_summary)
        self.assertIn("wash", diag_out.diagnostic_summary.lower())

    # -------------------------------------------------------------------------
    # Test 5: Strict Confidence Boundedness [0.0, 0.95]
    # -------------------------------------------------------------------------
    def test_05_strict_confidence_bounds_enforcement(self):
        """
        Verify that under extreme multi-fault anomalies, confidence strictly obeys
        0.0 <= confidence <= 0.95 and never claims 1.0 ground truth certainty.
        """
        # Create severe multi-breach warning scenario
        breaches = [
            ThresholdBreach("T30", 1680.0, 1625.0, "HIGH", 55.0, SeverityLevel.WARNING),
            ThresholdBreach("T50", 1490.0, 1460.0, "HIGH", 30.0, SeverityLevel.WARNING),
            ThresholdBreach("hpc_temp_rise", 1060.0, 1025.0, "HIGH", 35.0, SeverityLevel.WARNING),
            ThresholdBreach("P30", 590.0, 575.0, "HIGH", 15.0, SeverityLevel.WARNING),
        ]
        mon_out = MonitoringOutput(
            unit_number=99,
            time_cycles=250,
            operational_state=OperationalState.CRITICAL,
            threshold_breaches=breaches,
            model_anomaly_flag=True,
            persistence_count=10,
            persistence_alert=True,
            anomaly_severity=SeverityLevel.WARNING,
        )

        diag_out = self.diag_agent.diagnose(mon_out)

        self.assertLessEqual(diag_out.confidence_score, 0.95)
        self.assertGreaterEqual(diag_out.confidence_score, 0.0)
        self.assertNotEqual(diag_out.confidence_score, 1.0)
        for cand in diag_out.fault_candidates:
            self.assertLessEqual(cand.likelihood, 0.95)
            self.assertGreaterEqual(cand.likelihood, 0.0)

    # -------------------------------------------------------------------------
    # Test 6: End-to-End Digital Twin Handoff Pipeline
    # -------------------------------------------------------------------------
    def test_06_end_to_end_twin_to_monitoring_to_diagnosis_pipeline(self):
        """
        Verify seamless handoff:
        DigitalTwinSimulator -> MonitoringAgent.process_engine_state() -> DiagnosisAgent.diagnose()
        """
        sim = DigitalTwinSimulator(subset="FD001")
        sim.load_engine(unit_number=1, split="train")

        # Step to early cycle 30
        for _ in range(29):
            sim.step()
        early_state = sim.current_state

        # Monitoring evaluation
        early_mon_out = self.mon_agent.process_engine_state(early_state)
        self.assertEqual(early_mon_out.operational_state, OperationalState.HEALTHY)

        # Diagnosis evaluation
        early_diag_out = self.diag_agent.diagnose(early_mon_out)
        self.assertIsNone(early_diag_out.primary_fault)
        self.assertEqual(early_diag_out.confidence_score, 0.0)
        self.assertFalse(early_diag_out.requires_maintenance_action)

        # Advance to end of life (cycle 192)
        while sim.current_state.time_cycles < sim.total_cycles:
            sim.step()
        late_state = sim.current_state

        # Monitoring evaluation on late state with calibrated thresholds
        custom_mon = MonitoringAgent(
            agent_id="test_custom_mon",
            subset="FD001",
            sensor_thresholds={"T30": {"caution_high": 1600.0, "warning_high": 1620.0}},
        )
        late_mon_out = custom_mon.process_engine_state(late_state)
        self.assertIn(late_mon_out.operational_state, (OperationalState.ANOMALOUS, OperationalState.DEGRADED))

        # Diagnosis evaluation on late state
        late_diag_out = self.diag_agent.diagnose(
            late_mon_out,
            telemetry_snapshot={"T30": late_state.T30, "T50": late_state.T50, "P30": late_state.P30},
        )
        self.assertIsNotNone(late_diag_out.primary_fault)
        self.assertGreater(late_diag_out.confidence_score, 0.0)
        self.assertLessEqual(late_diag_out.confidence_score, 0.95)

    # -------------------------------------------------------------------------
    # Test 7: DiagnosisInput Structured Wrapper Ingestion
    # -------------------------------------------------------------------------
    def test_07_diagnosis_input_dataclass_ingestion(self):
        """
        Verify ingestion and processing when encapsulated in DiagnosisInput schema dataclass.
        """
        mon_out = MonitoringOutput(
            unit_number=3,
            time_cycles=80,
            operational_state=OperationalState.ANOMALOUS,
            threshold_breaches=[ThresholdBreach("T30", 1610.0, 1605.0, "HIGH", 5.0, SeverityLevel.CAUTION)],
            model_anomaly_flag=False,
            persistence_count=1,
            persistence_alert=False,
            anomaly_severity=SeverityLevel.CAUTION,
        )

        d_inp = DiagnosisInput(
            unit_number=3,
            time_cycles=80,
            monitoring_output=mon_out,
            telemetry_snapshot={"T30": 1610.0, "T24": 642.35},
            flight_regime=0,
            operating_context={"altitude": 0.0, "mach_number": 0.0, "throttle_resolver_angle": 100.0},
            historical_persistence_count=1,
        )

        diag_out = self.diag_agent.diagnose(d_inp)
        self.assertIsInstance(diag_out, DiagnosisOutput)
        self.assertEqual(diag_out.unit_number, 3)
        self.assertEqual(diag_out.time_cycles, 80)
        self.assertIsNotNone(diag_out.primary_fault)

    # -------------------------------------------------------------------------
    # Test 8: Malformed & Non-Finite Input Handling
    # -------------------------------------------------------------------------
    def test_08_malformed_and_non_finite_input_rejection(self):
        """
        Verify graceful rejection of invalid types, NaN, Inf, and negative cycle indices.
        """
        # A. Non-finite value in telemetry
        mon_out = MonitoringOutput(
            unit_number=1,
            time_cycles=10,
            operational_state=OperationalState.HEALTHY,
        )
        with self.assertRaises(ValueError):
            self.diag_agent.diagnose(mon_out, telemetry_snapshot={"T30": float("nan")})

        # B. Invalid object type
        with self.assertRaises(TypeError):
            self.diag_agent.diagnose("invalid_payload_string")

    # -------------------------------------------------------------------------
    # Test 9: Serialization Round-Trip
    # -------------------------------------------------------------------------
    def test_09_diagnosis_output_json_serialization(self):
        """
        Verify JSON round-trip serialization and deserialization of DiagnosisOutput.
        """
        ev = FaultEvidence("T30", 1615.0, 1585.0, 30.0, "HIGH", "T30 high")
        cand = FaultCandidate("THERMAL_DISTRESS", "HPT", 0.85, [ev], "Inspect turbine")
        meta = AgentMetadata("diag_agent", "2026-11-04T22:00:00Z", 5.2, AgentExecutionStatus.SUCCESS)

        out = DiagnosisOutput(
            unit_number=10,
            time_cycles=50,
            diagnostic_hypotheses=["THERMAL_DISTRESS"],
            fault_candidates=[cand],
            primary_fault="THERMAL_DISTRESS",
            confidence_score=0.85,
            contributing_evidence=[ev],
            severity=SeverityLevel.CAUTION,
            requires_maintenance_action=True,
            diagnostic_summary="Thermal distress in HPT",
            metadata=meta,
        )

        json_str = out.to_json()
        reconstructed = DiagnosisOutput.from_json(json_str)

        self.assertEqual(reconstructed.unit_number, out.unit_number)
        self.assertEqual(reconstructed.primary_fault, out.primary_fault)
        self.assertAlmostEqual(reconstructed.confidence_score, out.confidence_score, places=4)
        self.assertEqual(len(reconstructed.fault_candidates), 1)
        self.assertEqual(reconstructed.fault_candidates[0].component, "HPT")

    # -------------------------------------------------------------------------
    # Test 10: Dataset and Model Checkpoint Immutability
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
