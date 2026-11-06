"""
Unit and Integration Tests for Phase 6 Step 6.1: Multi-Agent System Input/Output Schemas
Verifies structured typing, JSON serialization round-trips, validation helpers,
data contract pipelines between agents, and dataset/model artifact immutability.

Roadmap Reference: Phase 6 - Step 6.1:
"Define the input and output schema for every agent"
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
    PrognosticsInput,
    PrognosticsOutput,
    OrchestrationRequest,
    OrchestrationResult,
)
from digital_twin.engine_state import EngineState, StateClassification
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestAgentSchemasStep6_1(unittest.TestCase):
    """
    Verification suite for Phase 6 Step 6.1: Multi-Agent System Input/Output Schemas.
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

    def test_01_metadata_instantiation_and_validation(self):
        """
        Verify AgentMetadata instantiation, serialization, and non-negative execution time enforcement.
        """
        meta = AgentMetadata(
            agent_id="monitoring_agent_v1",
            execution_timestamp="2026-11-04T22:00:00Z",
            execution_time_ms=12.4,
            status=AgentExecutionStatus.SUCCESS,
        )
        meta.validate()
        d = meta.to_dict()
        self.assertEqual(d["agent_id"], "monitoring_agent_v1")
        self.assertEqual(d["status"], "SUCCESS")
        self.assertEqual(d["execution_time_ms"], 12.4)

        # Round trip
        meta2 = AgentMetadata.from_dict(d)
        self.assertEqual(meta2.agent_id, meta.agent_id)
        self.assertEqual(meta2.status, AgentExecutionStatus.SUCCESS)

        # Invalid execution time
        bad_meta = AgentMetadata(
            agent_id="bad_agent",
            execution_timestamp="2026-11-04T22:00:00Z",
            execution_time_ms=-5.0,
            status=AgentExecutionStatus.FAILED,
        )
        with self.assertRaises(ValueError):
            bad_meta.validate()

    def test_02_monitoring_input_instantiation_and_json_roundtrip(self):
        """
        Verify MonitoringInput creation, validation against non-finite values, and JSON roundtrip.
        """
        m_in = MonitoringInput(
            unit_number=1,
            time_cycles=10,
            operating_context={"altitude": 10.0, "mach_number": 0.45, "throttle_resolver_angle": 80.0},
            raw_telemetry={"T2": 518.67, "T24": 642.1, "T30": 1588.0, "P30": 553.2},
            derived_indicators={"hpc_temp_rise": 945.9, "overall_pressure_ratio": 37.8, "spool_speed_ratio": 3.79},
            anomaly_scores={"anomaly_score_m1": -0.12, "anomaly_score_m2": 0.045},
        )
        m_in.validate()

        # JSON Round trip
        json_str = m_in.to_json()
        m_in_restored = MonitoringInput.from_json(json_str)
        self.assertEqual(m_in_restored.unit_number, 1)
        self.assertEqual(m_in_restored.time_cycles, 10)
        self.assertAlmostEqual(m_in_restored.derived_indicators["overall_pressure_ratio"], 37.8)

        # Non-finite rejection
        bad_m_in = MonitoringInput(
            unit_number=1,
            time_cycles=10,
            operating_context={"altitude": float("nan")},
            raw_telemetry={"T2": 518.67},
            derived_indicators={"hpc_temp_rise": 900.0},
        )
        with self.assertRaises(ValueError):
            bad_m_in.validate()

    def test_03_monitoring_output_and_threshold_breach_serialization(self):
        """
        Verify MonitoringOutput with threshold breaches, model flags, and severity indicators.
        """
        breach = ThresholdBreach(
            sensor_name="T30",
            observed_value=1620.0,
            threshold_limit=1600.0,
            direction="HIGH",
            breach_margin=20.0,
            severity=SeverityLevel.CAUTION,
        )
        breach.validate()

        meta = AgentMetadata(
            agent_id="monitoring_agent",
            execution_timestamp="2026-11-04T22:10:00Z",
            execution_time_ms=8.5,
            status=AgentExecutionStatus.SUCCESS,
        )

        m_out = MonitoringOutput(
            unit_number=2,
            time_cycles=125,
            operational_state=OperationalState.DEGRADED,
            threshold_breaches=[breach],
            model_anomaly_flag=True,
            isolation_forest_breach=True,
            pca_spe_breach=False,
            persistence_count=3,
            persistence_alert=True,
            anomaly_severity=SeverityLevel.CAUTION,
            summary_message="Elevated T30 detected over 3 consecutive cycles.",
            metadata=meta,
        )
        m_out.validate()

        json_str = m_out.to_json()
        restored = MonitoringOutput.from_json(json_str)
        self.assertEqual(restored.operational_state, OperationalState.DEGRADED)
        self.assertEqual(len(restored.threshold_breaches), 1)
        self.assertEqual(restored.threshold_breaches[0].sensor_name, "T30")
        self.assertEqual(restored.threshold_breaches[0].severity, SeverityLevel.CAUTION)
        self.assertTrue(restored.persistence_alert)

    def test_04_diagnosis_input_and_output_confidence_bounds(self):
        """
        Verify DiagnosisOutput confidence score is strictly bounded in [0.0, 1.0],
        and fault candidate schemas serialize cleanly.
        """
        evidence = FaultEvidence(
            sensor_name="T30",
            observed_value=1620.0,
            baseline_reference=1580.0,
            delta=40.0,
            direction="HIGH",
            description="HPC outlet temperature elevated by 40 °R",
        )
        candidate = FaultCandidate(
            fault_type="THERMAL_DISTRESS",
            component="HPC",
            likelihood=0.85,
            contributing_evidence=[evidence],
            recommended_action="Inspect combustor liner and HPC exit guide vanes.",
        )
        candidate.validate()

        d_out = DiagnosisOutput(
            unit_number=1,
            time_cycles=120,
            diagnostic_hypotheses=["Thermal distress at HPC exit station", "EGT margin degradation"],
            fault_candidates=[candidate],
            primary_fault="THERMAL_DISTRESS",
            confidence_score=0.85,
            contributing_evidence=[evidence],
            severity=SeverityLevel.CAUTION,
            requires_maintenance_action=True,
            diagnostic_summary="HPC thermal distress suspected with 85% confidence.",
        )
        d_out.validate()

        # JSON Round trip
        d_json = d_out.to_json()
        restored = DiagnosisOutput.from_json(d_json)
        self.assertEqual(restored.primary_fault, "THERMAL_DISTRESS")
        self.assertEqual(restored.confidence_score, 0.85)
        self.assertEqual(len(restored.fault_candidates), 1)

        # Invalid confidence > 1.0 must raise ValueError
        d_out.confidence_score = 1.25
        with self.assertRaises(ValueError):
            d_out.validate()

        # Invalid confidence < 0.0 must raise ValueError
        d_out.confidence_score = -0.1
        with self.assertRaises(ValueError):
            d_out.validate()

    def test_05_prognostics_input_tensor_dimension_validation(self):
        """
        Verify PrognosticsInput validates 30-cycle temporal sequence tensors (G31)
        and 14-dim / 24-dim normalized feature vectors (G30).
        """
        # Valid 14-dim (Tier 1 Core FD001)
        norm_vec_14 = [0.5] * 14
        seq_tensor_30x14 = [[0.5] * 14 for _ in range(30)]
        p_in = PrognosticsInput(
            unit_number=1,
            time_cycles=30,
            subset="FD001",
            normalized_feature_vec=norm_vec_14,
            temporal_seq_tensor=seq_tensor_30x14,
            feature_tier="Tier1_Core14",
        )
        p_in.validate()

        # Invalid feature count (e.g. 10 features)
        bad_p_in = PrognosticsInput(
            unit_number=1,
            time_cycles=30,
            subset="FD001",
            normalized_feature_vec=[0.5] * 10,
            temporal_seq_tensor=seq_tensor_30x14,
        )
        with self.assertRaises(ValueError):
            bad_p_in.validate()

        # Invalid temporal length (e.g. 20 cycles instead of 30)
        bad_tensor_p_in = PrognosticsInput(
            unit_number=1,
            time_cycles=30,
            subset="FD001",
            normalized_feature_vec=norm_vec_14,
            temporal_seq_tensor=[[0.5] * 14 for _ in range(20)],
        )
        with self.assertRaises(ValueError):
            bad_tensor_p_in.validate()

    def test_06_prognostics_output_validation_and_epistemic_spread(self):
        """
        Verify PrognosticsOutput non-negativity of RUL predictions, epistemic spread |G35 - G36|,
        and health index bounds [0.0, 1.0].
        """
        p_out = PrognosticsOutput(
            unit_number=1,
            time_cycles=50,
            subset="FD001",
            predicted_rul_xgboost=85.4,
            predicted_rul_lstm=88.2,
            ensemble_rul_estimate=86.8,
            prognostic_spread=2.8,
            virtual_health_index=0.78,
            uncertainty_classification="LOW_UNCERTAINTY",
            cycles_to_caution_threshold=40.0,
            prognostic_summary="Engine healthy, expected RUL: 86.8 cycles.",
        )
        p_out.validate()

        # Roundtrip JSON
        p_json = p_out.to_json()
        restored = PrognosticsOutput.from_json(p_json)
        self.assertAlmostEqual(restored.predicted_rul_xgboost, 85.4)
        self.assertAlmostEqual(restored.prognostic_spread, 2.8)
        self.assertAlmostEqual(restored.virtual_health_index, 0.78)

        # Invalid negative RUL
        p_out.predicted_rul_xgboost = -10.0
        with self.assertRaises(ValueError):
            p_out.validate()

        # Invalid health index > 1.0
        p_out.predicted_rul_xgboost = 85.4
        p_out.virtual_health_index = 1.5
        with self.assertRaises(ValueError):
            p_out.validate()

    def test_07_pipeline_data_contract_passing(self):
        """
        Verify clean data contract passing:
        1. Digital Twin state -> MonitoringInput
        2. MonitoringOutput -> DiagnosisInput
        3. Digital Twin state -> PrognosticsInput
        """
        sim = DigitalTwinSimulator(subset="FD001", enable_ml_inference=False)
        sim.load_engine(unit_number=1, split="train")
        st = sim.step()  # Cycle 2

        # 1. State -> MonitoringInput
        mon_in = MonitoringInput(
            unit_number=st.unit_number,
            time_cycles=st.time_cycles,
            operating_context={
                "altitude": st.altitude,
                "mach_number": st.mach_number,
                "throttle_resolver_angle": st.throttle_resolver_angle,
            },
            raw_telemetry={
                "T2": st.T2, "T24": st.T24, "T30": st.T30, "T50": st.T50,
                "P2": st.P2, "P15": st.P15, "P30": st.P30, "Nf": st.Nf, "Nc": st.Nc,
            },
            derived_indicators={
                "hpc_temp_rise": st.hpc_temp_rise,
                "overall_pressure_ratio": st.overall_pressure_ratio,
                "spool_speed_ratio": st.spool_speed_ratio,
            },
            anomaly_scores={
                "anomaly_score_m1": st.anomaly_score_m1,
                "anomaly_score_m2": st.anomaly_score_m2,
            },
        )
        mon_in.validate()

        # 2. Mock Monitoring Output -> DiagnosisInput
        mon_out = MonitoringOutput(
            unit_number=mon_in.unit_number,
            time_cycles=mon_in.time_cycles,
            operational_state=OperationalState.HEALTHY,
            anomaly_severity=SeverityLevel.NOMINAL,
        )
        diag_in = DiagnosisInput(
            unit_number=mon_out.unit_number,
            time_cycles=mon_out.time_cycles,
            monitoring_output=mon_out,
            telemetry_snapshot=mon_in.raw_telemetry,
            operating_context=mon_in.operating_context,
        )
        diag_in.validate()
        self.assertEqual(diag_in.unit_number, 1)
        self.assertEqual(diag_in.monitoring_output.operational_state, OperationalState.HEALTHY)

        # 3. State -> PrognosticsInput
        prog_in = PrognosticsInput(
            unit_number=st.unit_number,
            time_cycles=st.time_cycles,
            subset=st.subset,
            normalized_feature_vec=st.normalized_feature_vec.tolist(),
            temporal_seq_tensor=st.temporal_seq_tensor.tolist(),
            feature_tier=st.feature_tier,
            is_padded=st.is_padded,
        )
        prog_in.validate()
        self.assertEqual(len(prog_in.normalized_feature_vec), 14)
        self.assertEqual(len(prog_in.temporal_seq_tensor), 30)

    def test_08_orchestration_request_and_result_schemas(self):
        """
        Verify OrchestrationRequest and OrchestrationResult end-to-end schemas.
        """
        req = OrchestrationRequest(
            engine_id=1,
            flight_cycle=50,
            subset="FD001",
            execution_mode="full",
        )
        req.validate()
        self.assertEqual(req.subset, "FD001")

        # Invalid mode
        bad_req = OrchestrationRequest(
            engine_id=1,
            flight_cycle=50,
            subset="FD001",
            execution_mode="invalid_mode",
        )
        with self.assertRaises(ValueError):
            bad_req.validate()

        # OrchestrationResult
        orch_res = OrchestrationResult(
            run_id="RUN_20261104_001",
            engine_id=1,
            cycle=50,
            subset="FD001",
            status=AgentExecutionStatus.SUCCESS,
            execution_timeline_ms={"monitoring_ms": 10.2, "diagnosis_ms": 5.4, "prognostics_ms": 15.8},
        )
        orch_res.validate()
        orch_dict = orch_res.to_dict()
        self.assertEqual(orch_dict["run_id"], "RUN_20261104_001")
        self.assertEqual(orch_dict["status"], "SUCCESS")

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
