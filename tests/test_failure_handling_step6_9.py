"""
Unit and Integration Tests for Phase 6 Step 6.9: Failure Handling & Fault Containment
Verifies that corrupted telemetry, sensor dropouts, agent crashes, unwriteable audit logs,
and safety firewall tampering are caught cleanly, contained safely, reported clearly,
and recovered from reliably on subsequent operational cycles.

Roadmap Reference: Phase 6 - Step 6.9:
"Verify that agent failures are handled safely and reported clearly"
"""

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
import unittest

from agents.schemas import (
    AgentExecutionStatus,
    SeverityLevel,
    OperationalState,
    MonitoringInput,
    OrchestrationResult,
)
from agents.failure_handler import AgentFailureHarness, AgentFailureReport, IncidentRecord
from agents.twin_bridge import DigitalTwinSafetyBridge, SafetyViolationError, IMMUTABLE_PHYSICAL_FIELDS


class TestFailureHandlingStep6_9(unittest.TestCase):
    """
    Test suite for Phase 6 Step 6.9: Failure Handling & Fault Containment.
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
        """Set up temporary working directory and fresh failure harness."""
        self.test_dir = tempfile.mkdtemp(prefix="test_failure_")
        self.harness = AgentFailureHarness(
            subset="FD001",
            log_dir=self.test_dir,
        )

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test 1: NaN and Inf Telemetry Rejection
    # -------------------------------------------------------------------------
    def test_01_nan_inf_telemetry_rejection(self):
        """
        Confirms rejection of non-finite inputs with clear ValueError reporting.
        """
        incident = self.harness.test_corrupted_telemetry_nan_inf(engine_id=1)

        self.assertEqual(incident.failure_mode, "corrupted_telemetry_nan_inf")
        self.assertTrue(incident.contained_safely)
        self.assertEqual(incident.status_assigned, "REJECTED_AT_INGESTION")
        self.assertEqual(incident.exception_type, "ValueError")
        self.assertIn("non-finite", incident.exception_message.lower())
        self.assertTrue(incident.details["nan_rejected"])
        self.assertTrue(incident.details["inf_rejected"])

    # -------------------------------------------------------------------------
    # Test 2: Missing Telemetry Fields Handling
    # -------------------------------------------------------------------------
    def test_02_missing_telemetry_fields_handling(self):
        """
        Confirms graceful handling when critical telemetry fields are missing.
        """
        incident = self.harness.test_missing_sensor_channels(engine_id=1)

        self.assertEqual(incident.failure_mode, "missing_sensor_channels")
        self.assertTrue(incident.contained_safely)
        self.assertEqual(incident.status_assigned, "REJECTED_WITH_ERROR")
        self.assertIn("T30", incident.exception_message)
        self.assertTrue(incident.details["missing_attribute_detected"])

    # -------------------------------------------------------------------------
    # Test 3: MonitoringAgent Crash Isolation
    # -------------------------------------------------------------------------
    def test_03_monitoring_crash_isolation(self):
        """
        Confirms pipeline handles MonitoringAgent crash safely (status=FAILED).
        """
        incident = self.harness.test_monitoring_agent_runtime_crash(engine_id=1)

        self.assertEqual(incident.failure_mode, "monitoring_agent_runtime_crash")
        self.assertTrue(incident.contained_safely)
        self.assertEqual(incident.status_assigned, AgentExecutionStatus.FAILED.value)
        self.assertIn("MonitoringAgent failed", incident.exception_message)
        self.assertTrue(incident.telemetry_preserved)

    # -------------------------------------------------------------------------
    # Test 4: DiagnosisAgent Crash Isolation
    # -------------------------------------------------------------------------
    def test_04_diagnosis_crash_isolation(self):
        """
        Confirms pipeline handles DiagnosisAgent crash safely (status=WARNING).
        """
        incident = self.harness.test_diagnosis_agent_runtime_crash(engine_id=1)

        self.assertEqual(incident.failure_mode, "diagnosis_agent_runtime_crash")
        self.assertTrue(incident.contained_safely)
        self.assertEqual(incident.status_assigned, AgentExecutionStatus.WARNING.value)
        self.assertIn("DiagnosisAgent failed", incident.exception_message)
        self.assertTrue(incident.details["monitoring_preserved"])
        self.assertTrue(incident.details["prognostics_preserved"])

    # -------------------------------------------------------------------------
    # Test 5: PrognosticsAgent Crash Isolation
    # -------------------------------------------------------------------------
    def test_05_prognostics_crash_isolation(self):
        """
        Confirms pipeline handles PrognosticsAgent crash safely (status=WARNING).
        """
        incident = self.harness.test_prognostics_agent_runtime_crash(engine_id=1)

        self.assertEqual(incident.failure_mode, "prognostics_agent_runtime_crash")
        self.assertTrue(incident.contained_safely)
        self.assertEqual(incident.status_assigned, AgentExecutionStatus.WARNING.value)
        self.assertIn("PrognosticsAgent failed", incident.exception_message)
        self.assertTrue(incident.details["monitoring_preserved"])
        self.assertTrue(incident.details["diagnosis_preserved"])

    # -------------------------------------------------------------------------
    # Test 6: Safety Firewall Violation Containment
    # -------------------------------------------------------------------------
    def test_06_safety_firewall_violation_containment(self):
        """
        Confirms illegal state mutation is blocked and candidate state rejected.
        """
        incident = self.harness.test_safety_firewall_tampering(engine_id=1)

        self.assertEqual(incident.failure_mode, "safety_firewall_tampering_rejection")
        self.assertTrue(incident.contained_safely)
        self.assertEqual(incident.status_assigned, "BLOCKED_BY_FIREWALL")
        self.assertEqual(incident.exception_type, "SafetyViolationError")
        self.assertIn("CRITICAL SAFETY VIOLATION", incident.exception_message)
        self.assertTrue(incident.details["telemetry_tampering_blocked"])
        self.assertTrue(incident.details["negative_rul_blocked"])

    # -------------------------------------------------------------------------
    # Test 7: Audit Logger I/O Failure Resilience
    # -------------------------------------------------------------------------
    def test_07_audit_logger_io_failure_resilience(self):
        """
        Confirms unwriteable disk/path does not crash pipeline execution.
        """
        incident = self.harness.test_audit_logger_io_resilience(engine_id=1)

        self.assertEqual(incident.failure_mode, "audit_logger_io_failure_resilience")
        self.assertTrue(incident.contained_safely)
        self.assertEqual(incident.status_assigned, AgentExecutionStatus.SUCCESS.value)
        self.assertTrue(incident.details["orchestration_completed"])

    # -------------------------------------------------------------------------
    # Test 8: Failure Report Generation
    # -------------------------------------------------------------------------
    def test_08_failure_report_generation(self):
        """
        Confirms AgentFailureReport creates structured JSON and markdown summaries.
        """
        report = self.harness.run_all_failure_tests()

        self.assertIsInstance(report, AgentFailureReport)
        self.assertTrue(report.all_contained)
        self.assertEqual(report.total_tests_run, 8)
        self.assertEqual(report.passed_tests_count, 8)
        self.assertEqual(report.failed_tests_count, 0)

        # Test JSON serialization
        json_str = report.to_json()
        data = json.loads(json_str)
        self.assertEqual(data["report_id"], report.report_id)
        self.assertEqual(len(data["incidents"]), 8)

        # Test Markdown formatting
        md_str = report.to_markdown()
        self.assertIn("Multi-Agent System Failure & Resilience Incident Report", md_str)
        self.assertIn("Executive Summary", md_str)
        self.assertIn("ALL CONTAINED (PASS)", md_str)

        # Test file persistence
        json_path, md_path = report.save_report(self.test_dir)
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(md_path))

    # -------------------------------------------------------------------------
    # Test 9: Recovery After Failure
    # -------------------------------------------------------------------------
    def test_09_recovery_after_failure(self):
        """
        Confirms system immediately recovers to full operational capability on subsequent cycle.
        """
        incident = self.harness.test_recovery_after_failure(engine_id=1)

        self.assertEqual(incident.failure_mode, "recovery_after_failure")
        self.assertTrue(incident.contained_safely)
        self.assertEqual(incident.status_assigned, AgentExecutionStatus.SUCCESS.value)
        self.assertTrue(incident.details["cycle_t_failed_safely"])
        self.assertTrue(incident.details["cycle_t_plus_1_recovered"])

    # -------------------------------------------------------------------------
    # Test 10: Cryptographic Immutability of Datasets and Champion Artifacts
    # -------------------------------------------------------------------------
    def test_10_dataset_and_model_immutability(self):
        """
        Verify that all raw data files, scalers, and champion checkpoints are untouched.
        """
        # 1. Verify 12 C-MAPSS raw files
        for fname, orig_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                curr_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                curr_hash,
                orig_hash,
                f"Raw dataset file {fname} was mutated!",
            )

        # 2. Verify 4 scalers
        for fname, orig_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                curr_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                curr_hash,
                orig_hash,
                f"Scaler file {fname} was mutated!",
            )

        # 3. Verify 8 champion checkpoints
        for fname, orig_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                curr_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                curr_hash,
                orig_hash,
                f"Champion checkpoint {fname} was mutated!",
            )


if __name__ == "__main__":
    unittest.main()
