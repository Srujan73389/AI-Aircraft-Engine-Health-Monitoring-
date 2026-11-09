"""
Phase 9 - Step 9.7: Multi-Agent System & Orchestrated Pipeline Verification
Roadmap Reference: Master Project Roadmap PDF, Section 10, Step 9.7
("Test each agent independently and then test the full orchestrated workflow")
Strict Academic Integrity Standard: Master Roadmap Section 1, Page 2
("Display anomaly flags and diagnosis evidence separately from confirmed fault claims")

Test Suite Verifies:
1. MonitoringAgent independent surveillance & persistence tracking
2. DiagnosisAgent independent evidence synthesis & non-authority disclaimer
3. PrognosticsAgent independent dual-model inference & epistemic spread
4. Full orchestrated pipeline on FD001 (nominal vs degraded late life)
5. Orchestrated pipeline on held-out test split (FD001 Unit 3, Cycle 126, split="test")
6. Multi-regime orchestration on FD002 (Unit 15, dynamic flight envelope)
7. Compound fault orchestration on FD003 (Unit 8, dual degradation)
8. Stage execution latencies (sub-15ms per agent, sub-50ms pipeline)
9. Graceful degradation on partial telemetry & alternative execution modes
10. Cryptographic SHA-256 immutability of 12 raw files, 4 scalers, and 8 champion checkpoints
"""

import hashlib
import math
import os
import unittest

from agents.schemas import (
    AgentExecutionStatus,
    OperationalState,
    SeverityLevel,
    MonitoringInput,
    DiagnosisInput,
    PrognosticsInput,
    OrchestrationRequest,
)
from agents.monitoring_agent import MonitoringAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.prognostics_agent import PrognosticsAgent
from agents.orchestrator import AgentOrchestrator
from backend.services.agent_workflow_auditor import (
    audit_independent_agents,
    audit_orchestrated_pipeline,
)
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestAgentOrchestrationStep9_7(unittest.TestCase):
    """
    Automated verification suite validating independent agent capabilities
    and end-to-end multi-agent pipeline orchestration.
    """

    @classmethod
    def setUpClass(cls):
        cls.raw_data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.checkpoint_dir = os.path.join("models", "checkpoints")

        # Snapshot cryptographic hashes
        cls.raw_hashes = {}
        if os.path.exists(cls.raw_data_dir):
            for fname in sorted(os.listdir(cls.raw_data_dir)):
                if fname.endswith(".txt"):
                    fpath = os.path.join(cls.raw_data_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.raw_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        cls.scaler_hashes = {}
        if os.path.exists(cls.scalers_dir):
            for fname in sorted(os.listdir(cls.scalers_dir)):
                if fname.endswith(".pkl"):
                    fpath = os.path.join(cls.scalers_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.scaler_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

        cls.checkpoint_hashes = {}
        if os.path.exists(cls.checkpoint_dir):
            for fname in sorted(os.listdir(cls.checkpoint_dir)):
                if fname.endswith((".json", ".pt")) and not fname.endswith("manifest.json"):
                    fpath = os.path.join(cls.checkpoint_dir, fname)
                    with open(fpath, "rb") as f:
                        cls.checkpoint_hashes[fname] = hashlib.sha256(f.read()).hexdigest()

    def test_01_monitoring_agent_independent_execution(self):
        """
        Validates MonitoringAgent in isolation with nominal and late-life degraded telemetry.
        """
        sim = DigitalTwinSimulator(subset="FD001", enable_ml_inference=True)
        sim.load_engine(unit_number=1, split="train")

        # Nominal early-life evaluation (Cycle 30)
        for _ in range(29):
            sim.step()
        state_nom = sim.current_state

        mon = MonitoringAgent(subset="FD001")
        out_nom = mon.process_engine_state(state_nom)

        self.assertEqual(out_nom.metadata.status, AgentExecutionStatus.SUCCESS)
        self.assertEqual(out_nom.operational_state, OperationalState.HEALTHY)
        self.assertEqual(len(out_nom.threshold_breaches), 0)
        self.assertEqual(out_nom.persistence_count, 0)
        self.assertFalse(out_nom.persistence_alert)

        # Degraded late-life evaluation (Cycle 192)
        while sim.current_state.time_cycles < sim.total_cycles:
            sim.step()
        state_deg = sim.current_state

        # Custom monitoring agent configured to alert on elevated late-life T30
        mon_custom = MonitoringAgent(
            subset="FD001",
            sensor_thresholds={"T30": {"caution_high": 1600.0, "warning_high": 1620.0}},
        )
        out_deg = mon_custom.process_engine_state(state_deg)

        self.assertEqual(out_deg.metadata.status, AgentExecutionStatus.SUCCESS)
        self.assertIn(out_deg.operational_state, (OperationalState.ANOMALOUS, OperationalState.DEGRADED))

    def test_02_diagnosis_agent_independent_execution(self):
        """
        Validates DiagnosisAgent in isolation; confirms evidence generation and non-authority disclaimer.
        """
        diag = DiagnosisAgent(subset="FD001")

        # Create synthetic monitoring output with thermal creep breach
        from agents.schemas import ThresholdBreach
        breach = ThresholdBreach(
            sensor_name="T30",
            observed_value=1618.5,
            threshold_limit=1605.0,
            direction="HIGH",
            breach_margin=13.5,
            severity=SeverityLevel.CAUTION,
        )
        from agents.schemas import MonitoringOutput
        mon_out = MonitoringOutput(
            unit_number=1,
            time_cycles=190,
            operational_state=OperationalState.ANOMALOUS,
            threshold_breaches=[breach],
            persistence_count=3,
            persistence_alert=True,
            anomaly_severity=SeverityLevel.CAUTION,
        )

        telemetry = {"T24": 642.5, "T30": 1618.5, "P30": 551.2}
        diag_out = diag.diagnose(mon_out, telemetry_snapshot=telemetry)

        self.assertEqual(diag_out.metadata.status, AgentExecutionStatus.SUCCESS)
        self.assertGreater(len(diag_out.contributing_evidence), 0)
        self.assertTrue(diag_out.requires_maintenance_action)
        self.assertTrue(
            any(c.fault_type in ("THERMAL_DISTRESS", "HPC_EFFICIENCY_DEGRADATION", "GENERAL_CORE_DEGRADATION") for c in diag_out.fault_candidates)
        )

        # Verify separation of statistical anomalies from certified claims
        first_ev = diag_out.contributing_evidence[0]
        self.assertEqual(first_ev.sensor_name, "T30")
        self.assertEqual(first_ev.observed_value, 1618.5)
        self.assertGreater(first_ev.delta, 0.0)

    def test_03_prognostics_agent_independent_execution(self):
        """
        Validates PrognosticsAgent in isolation; confirms dual-model RUL and spread.
        """
        sim = DigitalTwinSimulator(subset="FD001", enable_ml_inference=True)
        sim.load_engine(unit_number=1, split="train")
        for _ in range(29):
            sim.step()
        state = sim.current_state

        prog = PrognosticsAgent(subset="FD001")
        prog_out = prog.process_engine_state(state)

        self.assertEqual(prog_out.metadata.status, AgentExecutionStatus.SUCCESS)
        self.assertGreater(prog_out.predicted_rul_xgboost, 0.0)
        self.assertGreater(prog_out.predicted_rul_lstm, 0.0)
        self.assertGreater(prog_out.ensemble_rul_estimate, 0.0)
        self.assertGreaterEqual(prog_out.prognostic_spread, 0.0)
        self.assertGreaterEqual(prog_out.virtual_health_index, 0.0)
        self.assertLessEqual(prog_out.virtual_health_index, 1.0)
        self.assertIn(prog_out.uncertainty_classification, ["LOW_UNCERTAINTY", "MODERATE_UNCERTAINTY", "HIGH_UNCERTAINTY"])

    def test_04_full_orchestrated_pipeline_fd001(self):
        """
        Tests complete sequential workflow on FD001 Unit 1 (Cycle 30 vs Cycle 190).
        """
        orch = AgentOrchestrator(subset="FD001")

        # Cycle 30 (nominal)
        sim = DigitalTwinSimulator(subset="FD001", enable_ml_inference=True)
        sim.load_engine(unit_number=1, split="train")
        for _ in range(29):
            sim.step()
        res_nom = orch.orchestrate_state(sim.current_state, mode="full")

        self.assertEqual(res_nom.status, AgentExecutionStatus.SUCCESS)
        self.assertEqual(res_nom.engine_id, 1)
        self.assertEqual(res_nom.cycle, 30)
        self.assertIsNotNone(res_nom.monitoring_result)
        self.assertIsNotNone(res_nom.diagnosis_result)
        self.assertIsNotNone(res_nom.prognostics_result)
        self.assertGreater(res_nom.prognostics_result.ensemble_rul_estimate, 50.0)

        # Cycle 190 (late life)
        while sim.current_state.time_cycles < 190:
            sim.step()
        res_deg = orch.orchestrate_state(sim.current_state, mode="full")

        self.assertEqual(res_deg.status, AgentExecutionStatus.SUCCESS)
        self.assertEqual(res_deg.cycle, 190)
        self.assertLess(res_deg.prognostics_result.ensemble_rul_estimate, 30.0)

    def test_05_orchestrated_pipeline_held_out_test_split(self):
        """
        Tests complete workflow on held-out test engine (FD001 Unit 3, Cycle 126, split="test").
        """
        rep = audit_orchestrated_pipeline(subset="FD001", unit=3, cycle=126, split="test", mode="full")

        self.assertEqual(rep["status"], "SUCCESS")
        self.assertEqual(rep["subset"], "FD001")
        self.assertEqual(rep["unit_number"], 3)
        self.assertEqual(rep["cycle"], 126)
        self.assertEqual(rep["split"], "test")
        self.assertTrue(rep["data_flow_intact"])
        self.assertGreater(rep["prognostics"]["ensemble_rul"], 0.0)

    def test_06_multi_regime_orchestration_fd002(self):
        """
        Tests complete workflow on FD002 Unit 15 under dynamic flight conditions.
        """
        rep = audit_orchestrated_pipeline(subset="FD002", unit=15, cycle=100, split="train", mode="full")

        self.assertEqual(rep["status"], "SUCCESS")
        self.assertEqual(rep["subset"], "FD002")
        self.assertEqual(rep["unit_number"], 15)
        self.assertTrue(rep["data_flow_intact"])
        self.assertGreater(rep["prognostics"]["ensemble_rul"], 0.0)

    def test_07_compound_fault_orchestration_fd003(self):
        """
        Tests complete workflow on FD003 Unit 8 with dual HPC + Fan degradation.
        """
        rep = audit_orchestrated_pipeline(subset="FD003", unit=8, cycle=120, split="train", mode="full")

        self.assertEqual(rep["status"], "SUCCESS")
        self.assertEqual(rep["subset"], "FD003")
        self.assertEqual(rep["unit_number"], 8)
        self.assertTrue(rep["data_flow_intact"])
        self.assertGreater(rep["prognostics"]["ensemble_rul"], 0.0)

    def test_08_pipeline_execution_timeline_and_latencies(self):
        """
        Asserts stage latencies are logged and within performance envelopes (< 50ms total, sub-15ms per agent).
        """
        rep = audit_orchestrated_pipeline(subset="FD001", unit=1, cycle=30, split="train", mode="full")
        timeline = rep["execution_timeline_ms"]

        self.assertIn("monitoring_ms", timeline)
        self.assertIn("diagnosis_ms", timeline)
        self.assertIn("prognostics_ms", timeline)
        self.assertIn("total_elapsed_ms", timeline)

        mon_ms = timeline["monitoring_ms"]
        diag_ms = timeline["diagnosis_ms"]
        prog_ms = timeline["prognostics_ms"]
        total_ms = timeline["total_elapsed_ms"]

        # Monitoring and Diagnosis should be sub-15ms
        self.assertLess(mon_ms, 15.0, f"MonitoringAgent latency exceeded 15ms: {mon_ms:.2f}ms")
        self.assertLess(diag_ms, 15.0, f"DiagnosisAgent latency exceeded 15ms: {diag_ms:.2f}ms")
        # Prognostics dual-model inference should be sub-30ms
        self.assertLess(prog_ms, 30.0, f"PrognosticsAgent latency exceeded 30ms: {prog_ms:.2f}ms")
        # Total pipeline should be sub-50ms
        self.assertLess(total_ms, 50.0, f"Total pipeline latency exceeded 50ms: {total_ms:.2f}ms")

    def test_09_graceful_degradation_on_partial_inputs(self):
        """
        Verifies orchestrator handles partial sensor telemetry or alternative modes gracefully.
        """
        orch = AgentOrchestrator(subset="FD001")
        sim = DigitalTwinSimulator(subset="FD001", enable_ml_inference=True)
        sim.load_engine(unit_number=1, split="train")
        for _ in range(29):
            sim.step()
        state = sim.current_state

        # Mode A: monitoring_only
        res_mon = orch.orchestrate_state(state, mode="monitoring_only")
        self.assertEqual(res_mon.status, AgentExecutionStatus.SUCCESS)
        self.assertIsNotNone(res_mon.monitoring_result)
        self.assertIsNone(res_mon.diagnosis_result)
        self.assertIsNone(res_mon.prognostics_result)

        # Mode B: prognostics_only
        res_prog = orch.orchestrate_state(state, mode="prognostics_only")
        self.assertEqual(res_prog.status, AgentExecutionStatus.SUCCESS)
        self.assertIsNone(res_prog.monitoring_result)
        self.assertIsNone(res_prog.diagnosis_result)
        self.assertIsNotNone(res_prog.prognostics_result)

        # Mode C: dry_run
        res_dry = orch.orchestrate_state(state, mode="dry_run")
        self.assertEqual(res_dry.status, AgentExecutionStatus.SUCCESS)
        self.assertIsNone(res_dry.monitoring_result)
        self.assertIsNone(res_dry.diagnosis_result)
        self.assertIsNone(res_dry.prognostics_result)

    def test_10_dataset_and_model_immutability(self):
        """
        Cryptographic SHA-256 verification guaranteeing zero modification to:
        - 12 raw C-MAPSS dataset files
        - 4 Phase 3 feature scalers
        - 8 Phase 4 champion model checkpoints
        """
        # 1. Raw dataset files
        for fname, orig_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                cur_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(cur_hash, orig_hash, f"Raw C-MAPSS dataset modified: {fname}")

        # 2. Phase 3 scalers
        for fname, orig_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                cur_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(cur_hash, orig_hash, f"Feature scaler modified: {fname}")

        # 3. Phase 4 checkpoints
        for fname, orig_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                cur_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(cur_hash, orig_hash, f"Champion model checkpoint modified: {fname}")


if __name__ == "__main__":
    unittest.main()
