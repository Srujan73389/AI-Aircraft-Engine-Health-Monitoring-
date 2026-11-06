"""
Unit and Integration Tests for Phase 6 Step 6.6: Structured Audit Logger
Verifies structured log record generation, machine-readable JSONL appending,
query & retrieval APIs, markdown summary generation, seamless Orchestrator integration,
fault containment on I/O failures, and artifact immutability.

Roadmap Reference: Phase 6 - Step 6.6:
"Add structured logs showing each agent's input, output, timestamp, and status"
"""

import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import unittest
import numpy as np

from agents.schemas import (
    AgentExecutionStatus,
    SeverityLevel,
    OperationalState,
    AgentMetadata,
    MonitoringOutput,
    DiagnosisOutput,
    PrognosticsOutput,
    OrchestrationResult,
)
from agents.audit_logger import AgentAuditLogger, AuditLogRecord
from agents.orchestrator import AgentOrchestrator
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestAuditLoggerStep6_6(unittest.TestCase):
    """
    Test suite for Phase 6 Step 6.6: Structured Audit Logger.
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
        """Create a temporary directory for hermetic logging tests."""
        self.test_dir = tempfile.mkdtemp(prefix="agent_logs_test_")
        self.logger = AgentAuditLogger(log_dir=self.test_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test 1: Creation of Individual Structured Log Records Across Agent Types
    # -------------------------------------------------------------------------
    def test_01_create_records_across_all_agent_types(self):
        """
        Verify creation and serialization of structured log records for all agent types.
        """
        for a_type in ["MONITORING", "DIAGNOSIS", "PROGNOSTICS", "ORCHESTRATOR"]:
            rec = self.logger.log_record(
                run_id="run_test_01",
                engine_id=1,
                cycle=30,
                agent_id=f"{a_type.lower()}_v1",
                agent_type=a_type,
                input_summary={"input_param": 123},
                output_summary={"output_metric": 456},
                status=AgentExecutionStatus.SUCCESS,
                execution_time_ms=12.5,
            )

            self.assertIsInstance(rec, dict)
            self.assertEqual(rec["agent_type"], a_type)
            self.assertEqual(rec["engine_id"], 1)
            self.assertEqual(rec["cycle"], 30)
            self.assertEqual(rec["status"], "SUCCESS")
            self.assertIn("log_id", rec)
            self.assertIn("timestamp", rec)

        all_logs = self.logger.get_all_logs()
        self.assertEqual(len(all_logs), 4)

    # -------------------------------------------------------------------------
    # Test 2: JSONL File Persistence & Formatting Fidelity
    # -------------------------------------------------------------------------
    def test_02_jsonl_persistence_and_formatting(self):
        """
        Verify that log records are properly appended as valid JSON-Lines to the target file.
        """
        self.logger.log_record(
            run_id="run_persist_1",
            engine_id=5,
            cycle=10,
            agent_id="mon_agent",
            agent_type="MONITORING",
            input_summary={"T30": 1585.0},
            output_summary={"state": "HEALTHY"},
            status=AgentExecutionStatus.SUCCESS,
            execution_time_ms=2.1,
        )
        self.logger.log_record(
            run_id="run_persist_1",
            engine_id=5,
            cycle=10,
            agent_id="prog_agent",
            agent_type="PROGNOSTICS",
            input_summary={"window_len": 30},
            output_summary={"ensemble_rul": 112.4},
            status=AgentExecutionStatus.SUCCESS,
            execution_time_ms=25.4,
        )

        self.assertTrue(os.path.exists(self.logger.jsonl_path))
        with open(self.logger.jsonl_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.assertEqual(len(lines), 2)
        parsed_0 = json.loads(lines[0])
        parsed_1 = json.loads(lines[1])

        self.assertEqual(parsed_0["agent_type"], "MONITORING")
        self.assertEqual(parsed_0["input_summary"]["T30"], 1585.0)
        self.assertEqual(parsed_1["agent_type"], "PROGNOSTICS")
        self.assertEqual(parsed_1["output_summary"]["ensemble_rul"], 112.4)

    # -------------------------------------------------------------------------
    # Test 3: Query & Retrieval APIs
    # -------------------------------------------------------------------------
    def test_03_query_apis_by_run_engine_status(self):
        """
        Verify get_logs_by_run(), get_logs_by_engine(), and get_logs_by_status().
        """
        # Create records across multiple runs and engines
        self.logger.log_record("run_A", 1, 20, "mon", "MONITORING", {}, {}, AgentExecutionStatus.SUCCESS, 1.0)
        self.logger.log_record("run_A", 1, 20, "diag", "DIAGNOSIS", {}, {}, AgentExecutionStatus.SUCCESS, 2.0)
        self.logger.log_record("run_B", 2, 40, "mon", "MONITORING", {}, {}, AgentExecutionStatus.WARNING, 1.5, "Minor drift")
        self.logger.log_record("run_C", 1, 21, "mon", "MONITORING", {}, {}, AgentExecutionStatus.FAILED, 0.5, "Sensor error")

        # Query by run_id
        run_a_logs = self.logger.get_logs_by_run("run_A")
        self.assertEqual(len(run_a_logs), 2)
        self.assertEqual({r["run_id"] for r in run_a_logs}, {"run_A"})

        # Query by engine_id
        eng_1_logs = self.logger.get_logs_by_engine(1)
        self.assertEqual(len(eng_1_logs), 3)

        eng_2_logs = self.logger.get_logs_by_engine(2)
        self.assertEqual(len(eng_2_logs), 1)

        # Query by status
        success_logs = self.logger.get_logs_by_status(AgentExecutionStatus.SUCCESS)
        self.assertEqual(len(success_logs), 2)

        warning_logs = self.logger.get_logs_by_status("WARNING")
        self.assertEqual(len(warning_logs), 1)
        self.assertEqual(warning_logs[0]["error_message"], "Minor drift")

        failed_logs = self.logger.get_logs_by_status(AgentExecutionStatus.FAILED)
        self.assertEqual(len(failed_logs), 1)

    # -------------------------------------------------------------------------
    # Test 4: Markdown Session Summary Generation
    # -------------------------------------------------------------------------
    def test_04_markdown_summary_generation(self):
        """
        Verify that write_markdown_summary() creates a properly formatted markdown table.
        """
        self.logger.log_record(
            "run_md", 3, 50, "mon_v1", "MONITORING", {}, {"state": "ANOMALOUS"}, AgentExecutionStatus.SUCCESS, 3.2
        )
        md_text = self.logger.write_markdown_summary()

        self.assertIn("# Multi-Agent Audit Trail Summary", md_text)
        self.assertIn("| Timestamp | Run ID | Engine ID | Cycle | Agent Type | Status | Latency (ms) | Output Summary |", md_text)
        self.assertIn("`run_md`", md_text)
        self.assertIn("**MONITORING**", md_text)
        self.assertTrue(os.path.exists(self.logger.md_path))

    # -------------------------------------------------------------------------
    # Test 5: Orchestrator Integration (Automated Multi-Agent Logging)
    # -------------------------------------------------------------------------
    def test_05_orchestrator_automated_logging(self):
        """
        Verify that an AgentOrchestrator configured with AgentAuditLogger
        automatically generates records for Monitoring, Diagnosis, Prognostics, and Orchestrator.
        """
        orch = AgentOrchestrator(
            subset="FD001",
            audit_logger=self.logger,
        )

        sim = DigitalTwinSimulator(subset="FD001")
        sim.load_engine(unit_number=1, split="train")
        for _ in range(29):
            sim.step()
        state = sim.current_state

        result = orch.orchestrate_state(state, mode="full", run_id="run_orch_auto_1")

        self.assertEqual(result.status, AgentExecutionStatus.SUCCESS)

        # Retrieve logs generated during this run
        run_logs = self.logger.get_logs_by_run("run_orch_auto_1")
        agent_types = [r["agent_type"] for r in run_logs]

        # All 4 stages must be logged
        self.assertIn("MONITORING", agent_types)
        self.assertIn("DIAGNOSIS", agent_types)
        self.assertIn("PROGNOSTICS", agent_types)
        self.assertIn("ORCHESTRATOR", agent_types)
        self.assertEqual(len(run_logs), 4)

    # -------------------------------------------------------------------------
    # Test 6: Selective Mode Logging via Orchestrator
    # -------------------------------------------------------------------------
    def test_06_orchestrator_selective_mode_logging(self):
        """
        Verify that selective execution modes log only their corresponding stages.
        """
        orch = AgentOrchestrator(subset="FD001", audit_logger=self.logger)
        sim = DigitalTwinSimulator(subset="FD001")
        sim.load_engine(unit_number=1, split="train")
        for _ in range(29):
            sim.step()
        state = sim.current_state

        # A. Monitoring only
        orch.orchestrate_state(state, mode="monitoring_only", run_id="run_mon_only")
        mon_logs = self.logger.get_logs_by_run("run_mon_only")
        mon_types = [r["agent_type"] for r in mon_logs]
        self.assertIn("MONITORING", mon_types)
        self.assertIn("ORCHESTRATOR", mon_types)
        self.assertNotIn("DIAGNOSIS", mon_types)
        self.assertNotIn("PROGNOSTICS", mon_types)

        # B. Dry run
        orch.orchestrate_state(state, mode="dry_run", run_id="run_dry")
        dry_logs = self.logger.get_logs_by_run("run_dry")
        self.assertEqual(len(dry_logs), 1)
        self.assertEqual(dry_logs[0]["agent_type"], "ORCHESTRATOR")

    # -------------------------------------------------------------------------
    # Test 7: Non-Crashing Fault Containment on I/O Errors
    # -------------------------------------------------------------------------
    def test_07_non_blocking_fault_containment_on_io_failure(self):
        """
        Verify that an invalid or unwriteable directory path does not crash the logger
        or the main orchestrator pipeline.
        """
        # Create logger with an invalid file path as directory
        invalid_file_as_dir = os.path.join(self.test_dir, "file_blocking_dir.txt")
        with open(invalid_file_as_dir, "w") as f:
            f.write("I am a file, not a directory")

        failing_logger = AgentAuditLogger(
            log_dir=os.path.join(invalid_file_as_dir, "nested_impossible_dir"),
        )

        # Logging should not throw an exception
        rec = failing_logger.log_record(
            run_id="run_fail_safe",
            engine_id=1,
            cycle=1,
            agent_id="test_agent",
            agent_type="MONITORING",
            input_summary={},
            output_summary={},
            status=AgentExecutionStatus.SUCCESS,
            execution_time_ms=1.0,
        )
        self.assertIsInstance(rec, dict)
        self.assertEqual(rec["run_id"], "run_fail_safe")
        # In-memory record is still preserved
        self.assertEqual(len(failing_logger.get_all_logs()), 1)

    # -------------------------------------------------------------------------
    # Test 8: Thread-Safe Concurrent Record Logging
    # -------------------------------------------------------------------------
    def test_08_thread_safe_concurrent_logging(self):
        """
        Verify that concurrent logging calls from multiple threads don't corrupt logs.
        """
        num_threads = 5
        records_per_thread = 10

        def worker(tid: int):
            for i in range(records_per_thread):
                self.logger.log_record(
                    run_id=f"run_thread_{tid}",
                    engine_id=tid,
                    cycle=i + 1,
                    agent_id=f"agent_{tid}",
                    agent_type="MONITORING",
                    input_summary={"thread": tid, "iter": i},
                    output_summary={"ack": True},
                    status=AgentExecutionStatus.SUCCESS,
                    execution_time_ms=0.5,
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_expected = num_threads * records_per_thread
        all_logs = self.logger.get_all_logs()
        self.assertEqual(len(all_logs), total_expected)

        # Verify disk persistence line count
        with open(self.logger.jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), total_expected)

    # -------------------------------------------------------------------------
    # Test 9: Backward Compatibility Without Logger
    # -------------------------------------------------------------------------
    def test_09_backward_compatibility_without_logger(self):
        """
        Verify that AgentOrchestrator works 100% seamlessly when audit_logger is None (default).
        """
        orch = AgentOrchestrator(subset="FD001", audit_logger=None)
        self.assertIsNone(orch.audit_logger)

        sim = DigitalTwinSimulator(subset="FD001")
        sim.load_engine(unit_number=1, split="train")
        for _ in range(29):
            sim.step()
        state = sim.current_state

        result = orch.orchestrate_state(state, mode="full")
        self.assertEqual(result.status, AgentExecutionStatus.SUCCESS)
        self.assertIsNotNone(result.monitoring_result)
        self.assertIsNotNone(result.prognostics_result)

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
