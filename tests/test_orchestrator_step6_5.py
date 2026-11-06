"""
Unit and Integration Tests for Phase 6 Step 6.5: Multi-Agent Orchestrator
Verifies master agent coordination, sequenced pipeline execution (Monitoring -> Diagnosis -> Prognostics),
configurable execution modes, latency profiling, fault isolation, Digital Twin integration,
serialization round-trips, and artifact immutability.

Roadmap Reference: Phase 6 - Step 6.5:
"Implement the Orchestrator to invoke agents in the required sequence"
"""

import hashlib
import json
import math
import os
import unittest
from unittest.mock import MagicMock
import numpy as np

from agents.schemas import (
    AgentExecutionStatus,
    SeverityLevel,
    OperationalState,
    AgentMetadata,
    OrchestrationRequest,
    OrchestrationResult,
    MonitoringOutput,
    DiagnosisOutput,
    PrognosticsOutput,
)
from agents.orchestrator import AgentOrchestrator
from agents.monitoring_agent import MonitoringAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.prognostics_agent import PrognosticsAgent
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestOrchestratorStep6_5(unittest.TestCase):
    """
    Test suite for Phase 6 Step 6.5: Multi-Agent Orchestrator.
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
        """Instantiate fresh Orchestrator and Simulator for each test."""
        self.orchestrator = AgentOrchestrator(
            agent_id="test_orchestrator",
            subset="FD001",
            default_mode="full",
        )
        self.sim = DigitalTwinSimulator(subset="FD001")
        self.sim.load_engine(unit_number=1, split="train")
        # Step forward to cycle 30 to form valid 30-cycle temporal window
        for _ in range(29):
            self.sim.step()
        self.nominal_state = self.sim.current_state

    # -------------------------------------------------------------------------
    # Test 1: Full Pipeline Execution on Nominal State
    # -------------------------------------------------------------------------
    def test_01_full_pipeline_nominal_state(self):
        """
        Verify that full pipeline execution on nominal engine state yields SUCCESS,
        valid monitoring result (HEALTHY), nominal diagnosis (confidence 0.0),
        and active prognostics forecast.
        """
        result = self.orchestrator.orchestrate_state(self.nominal_state, mode="full")

        self.assertIsInstance(result, OrchestrationResult)
        self.assertEqual(result.status, AgentExecutionStatus.SUCCESS)
        self.assertEqual(result.engine_id, 1)
        self.assertEqual(result.cycle, 30)

        # 1. Monitoring evaluation
        self.assertIsNotNone(result.monitoring_result)
        self.assertEqual(result.monitoring_result.operational_state, OperationalState.HEALTHY)
        self.assertEqual(len(result.monitoring_result.threshold_breaches), 0)

        # 2. Diagnosis evaluation
        self.assertIsNotNone(result.diagnosis_result)
        self.assertIsNone(result.diagnosis_result.primary_fault)
        self.assertEqual(result.diagnosis_result.confidence_score, 0.0)
        self.assertFalse(result.diagnosis_result.requires_maintenance_action)

        # 3. Prognostics evaluation
        self.assertIsNotNone(result.prognostics_result)
        self.assertGreater(result.prognostics_result.ensemble_rul_estimate, 50.0)
        self.assertGreater(result.prognostics_result.virtual_health_index, 0.5)

        # 4. Latency timeline
        self.assertIn("monitoring_ms", result.execution_timeline_ms)
        self.assertIn("diagnosis_ms", result.execution_timeline_ms)
        self.assertIn("prognostics_ms", result.execution_timeline_ms)
        self.assertIn("total_elapsed_ms", result.execution_timeline_ms)

    # -------------------------------------------------------------------------
    # Test 2: Full Pipeline Execution on Degraded State
    # -------------------------------------------------------------------------
    def test_02_full_pipeline_degraded_state(self):
        """
        Verify that full pipeline execution on degraded late-life engine state
        properly cascades monitoring alerts, diagnosis isolation, and low RUL forecast.
        """
        # Step forward until late life (cycle 192)
        while self.sim.current_state.time_cycles < self.sim.total_cycles:
            self.sim.step()
        degraded_state = self.sim.current_state

        # Use customized monitoring thresholds to detect elevated late-life T30
        custom_mon = MonitoringAgent(
            agent_id="test_custom_mon",
            subset="FD001",
            sensor_thresholds={"T30": {"caution_high": 1600.0, "warning_high": 1620.0}},
        )
        orch = AgentOrchestrator(
            monitoring_agent=custom_mon,
            diagnosis_agent=self.orchestrator.diagnosis_agent,
            prognostics_agent=self.orchestrator.prognostics_agent,
            subset="FD001",
        )

        result = orch.orchestrate_state(degraded_state, mode="full")

        self.assertEqual(result.status, AgentExecutionStatus.SUCCESS)
        self.assertEqual(result.cycle, 192)

        # Monitoring should detect elevated condition
        self.assertIn(result.monitoring_result.operational_state, (OperationalState.ANOMALOUS, OperationalState.DEGRADED))

        # Diagnosis should isolate fault candidate
        self.assertIsNotNone(result.diagnosis_result.primary_fault)
        self.assertGreater(result.diagnosis_result.confidence_score, 0.0)
        self.assertLessEqual(result.diagnosis_result.confidence_score, 0.95)

        # Prognostics should predict depleted RUL and low health index
        self.assertLess(result.prognostics_result.ensemble_rul_estimate, 30.0)
        self.assertLess(result.prognostics_result.virtual_health_index, 0.3)

    # -------------------------------------------------------------------------
    # Test 3: Selective Execution Modes
    # -------------------------------------------------------------------------
    def test_03_selective_execution_modes(self):
        """
        Verify execution behavior under 'monitoring_only', 'prognostics_only', and 'dry_run'.
        """
        # A. Monitoring Only
        res_mon = self.orchestrator.orchestrate_state(self.nominal_state, mode="monitoring_only")
        self.assertEqual(res_mon.status, AgentExecutionStatus.SUCCESS)
        self.assertIsNotNone(res_mon.monitoring_result)
        self.assertIsNone(res_mon.diagnosis_result)
        self.assertIsNone(res_mon.prognostics_result)
        self.assertIn("monitoring_ms", res_mon.execution_timeline_ms)
        self.assertNotIn("diagnosis_ms", res_mon.execution_timeline_ms)

        # B. Prognostics Only
        res_prog = self.orchestrator.orchestrate_state(self.nominal_state, mode="prognostics_only")
        self.assertEqual(res_prog.status, AgentExecutionStatus.SUCCESS)
        self.assertIsNone(res_prog.monitoring_result)
        self.assertIsNone(res_prog.diagnosis_result)
        self.assertIsNotNone(res_prog.prognostics_result)
        self.assertIn("prognostics_ms", res_prog.execution_timeline_ms)
        self.assertNotIn("monitoring_ms", res_prog.execution_timeline_ms)

        # C. Dry Run
        res_dry = self.orchestrator.orchestrate_state(self.nominal_state, mode="dry_run")
        self.assertEqual(res_dry.status, AgentExecutionStatus.SUCCESS)
        self.assertIsNone(res_dry.monitoring_result)
        self.assertIsNone(res_dry.diagnosis_result)
        self.assertIsNone(res_dry.prognostics_result)
        self.assertIn("total_elapsed_ms", res_dry.execution_timeline_ms)

    # -------------------------------------------------------------------------
    # Test 4: Latency Timeline Tracking
    # -------------------------------------------------------------------------
    def test_04_execution_timeline_tracking(self):
        """
        Verify that the execution timeline captures non-negative millisecond durations
        for all invoked stages and total runtime.
        """
        result = self.orchestrator.orchestrate_state(self.nominal_state, mode="full")
        tl = result.execution_timeline_ms

        self.assertGreaterEqual(tl["monitoring_ms"], 0.0)
        self.assertGreaterEqual(tl["diagnosis_ms"], 0.0)
        self.assertGreaterEqual(tl["prognostics_ms"], 0.0)
        self.assertGreaterEqual(tl["total_elapsed_ms"], 0.0)
        self.assertGreaterEqual(
            tl["total_elapsed_ms"],
            tl["monitoring_ms"] + tl["diagnosis_ms"] + tl["prognostics_ms"] - 1.0,
        )

    # -------------------------------------------------------------------------
    # Test 5: Fault Containment on Downstream Exception
    # -------------------------------------------------------------------------
    def test_05_fault_containment_on_agent_exception(self):
        """
        Verify that when a downstream agent throws an exception, the orchestrator
        catches it gracefully, records the error diagnostic, and sets status=WARNING or FAILED.
        """
        # Injected failure in PrognosticsAgent
        failing_prog = PrognosticsAgent(agent_id="failing_prog", subset="FD001")
        failing_prog.process_engine_state = MagicMock(side_effect=RuntimeError("Injected GPU out-of-memory error"))

        orch = AgentOrchestrator(
            monitoring_agent=self.orchestrator.monitoring_agent,
            diagnosis_agent=self.orchestrator.diagnosis_agent,
            prognostics_agent=failing_prog,
            subset="FD001",
        )

        result = orch.orchestrate_state(self.nominal_state, mode="full")

        # Monitoring and Diagnosis should still succeed
        self.assertIsNotNone(result.monitoring_result)
        self.assertIsNotNone(result.diagnosis_result)
        self.assertIsNone(result.prognostics_result)

        # Pipeline should capture WARNING status and error diagnostic
        self.assertEqual(result.status, AgentExecutionStatus.WARNING)
        self.assertIsNotNone(result.metadata)
        self.assertIn("Injected GPU", result.metadata.error_message)

    # -------------------------------------------------------------------------
    # Test 6: Trajectory State Management (Persistence Reset)
    # -------------------------------------------------------------------------
    def test_06_trajectory_persistence_reset(self):
        """
        Verify that reset_engine cleanly resets monitoring agent persistence counts.
        """
        # Set artificial persistence count in monitoring agent
        self.orchestrator.monitoring_agent._persistence_counts[1] = 5
        self.assertEqual(self.orchestrator.monitoring_agent.get_persistence_count(1), 5)

        # Reset specific engine
        self.orchestrator.reset_engine(unit_number=1)
        self.assertEqual(self.orchestrator.monitoring_agent.get_persistence_count(1), 0)

    # -------------------------------------------------------------------------
    # Test 7: OrchestrationRequest Dispatched Run
    # -------------------------------------------------------------------------
    def test_07_orchestration_request_dispatched_run(self):
        """
        Verify that run() accepts an OrchestrationRequest and automatically steps
        the DigitalTwinSimulator to the requested flight cycle.
        """
        req = OrchestrationRequest(
            engine_id=1,
            flight_cycle=30,
            subset="FD001",
            execution_mode="full",
        )

        result = self.orchestrator.run(req)

        self.assertIsInstance(result, OrchestrationResult)
        self.assertEqual(result.status, AgentExecutionStatus.SUCCESS)
        self.assertEqual(result.engine_id, 1)
        self.assertEqual(result.cycle, 30)
        self.assertIsNotNone(result.monitoring_result)
        self.assertIsNotNone(result.prognostics_result)

    # -------------------------------------------------------------------------
    # Test 8: Malformed Request Rejection
    # -------------------------------------------------------------------------
    def test_08_malformed_request_rejection(self):
        """
        Verify that invalid mode, negative engine_id, or non-finite inputs raise ValueError.
        """
        # A. Invalid execution mode
        with self.assertRaises(ValueError):
            self.orchestrator.orchestrate_state(self.nominal_state, mode="invalid_mode_name")

        # B. Negative engine ID in request
        with self.assertRaises(ValueError):
            OrchestrationRequest(engine_id=-1, flight_cycle=30).validate()

        # C. Non-EngineState object passed
        with self.assertRaises(TypeError):
            self.orchestrator.orchestrate_state("invalid_string_state")

    # -------------------------------------------------------------------------
    # Test 9: Serialization Round-Trip
    # -------------------------------------------------------------------------
    def test_09_orchestration_result_json_serialization(self):
        """
        Verify full JSON round-trip serialization and deserialization of OrchestrationResult.
        """
        result = self.orchestrator.orchestrate_state(self.nominal_state, mode="full")
        json_str = result.to_json()
        reconstructed = OrchestrationResult.from_json(json_str)

        self.assertEqual(reconstructed.run_id, result.run_id)
        self.assertEqual(reconstructed.engine_id, result.engine_id)
        self.assertEqual(reconstructed.cycle, result.cycle)
        self.assertEqual(reconstructed.status, result.status)
        self.assertIsNotNone(reconstructed.monitoring_result)
        self.assertIsNotNone(reconstructed.diagnosis_result)
        self.assertIsNotNone(reconstructed.prognostics_result)
        self.assertAlmostEqual(
            reconstructed.prognostics_result.ensemble_rul_estimate,
            result.prognostics_result.ensemble_rul_estimate,
            places=3,
        )

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
