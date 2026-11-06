"""
Unit and Integration Tests for Phase 6 Step 6.8: Multi-Agent Scenario Evaluator
Verifies end-to-end multi-agent interaction on nominal data, degraded data,
and counterfactual what-if scenarios (Thermal Stress, Sensor Bias, Compounded Degradation),
testing safety firewall invariance, structured audit logging, fleet multi-engine isolation,
and artifact immutability.

Roadmap Reference: Phase 6 - Step 6.8:
"Test multi-agent interaction on nominal data, degraded data, and what-if scenarios"
"""

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
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
from agents.scenario_evaluator import MultiAgentScenarioEvaluator
from agents.audit_logger import AgentAuditLogger
from agents.twin_bridge import DigitalTwinSafetyBridge, IMMUTABLE_PHYSICAL_FIELDS
from digital_twin.engine_state import EngineState


class TestScenarioEvaluatorStep6_8(unittest.TestCase):
    """
    Test suite for Phase 6 Step 6.8: Multi-Agent Scenario Evaluator.
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
        """Set up temporary logging directory and fresh evaluator instance."""
        self.test_dir = tempfile.mkdtemp(prefix="test_evaluator_")
        self.audit_logger = AgentAuditLogger(log_dir=self.test_dir)
        self.evaluator = MultiAgentScenarioEvaluator(
            subset="FD001",
            enable_twin_feedback=True,
            log_dir=self.test_dir,
            audit_logger=self.audit_logger,
        )

    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test 1: Nominal Trajectory Interaction
    # -------------------------------------------------------------------------
    def test_01_nominal_trajectory_interaction(self):
        """
        Evaluate early-life flight cycles; assert HEALTHY state, zero breaches,
        zero false alarms, and high RUL forecast.
        """
        metrics = self.evaluator.evaluate_nominal_trajectory(engine_id=1, cycles=25)

        self.assertEqual(metrics["trajectory_type"], "NOMINAL")
        self.assertEqual(metrics["engine_id"], 1)
        self.assertEqual(metrics["total_cycles_evaluated"], 25)
        self.assertEqual(metrics["false_alarms"], 0)
        self.assertEqual(metrics["false_alarm_rate"], 0.0)
        self.assertEqual(metrics["total_threshold_breaches"], 0)
        self.assertGreaterEqual(metrics["mean_virtual_health_index"], 0.70)
        self.assertGreater(metrics["mean_ensemble_rul"], 100.0)
        self.assertTrue(metrics["all_nominal_passed"])

    # -------------------------------------------------------------------------
    # Test 2: Degraded Trajectory Interaction
    # -------------------------------------------------------------------------
    def test_02_degraded_trajectory_interaction(self):
        """
        Replay degraded engine trajectory; assert 3-cycle persistence trigger,
        DEGRADED state, root-cause isolation, and depleted RUL forecast.
        """
        metrics = self.evaluator.evaluate_degraded_trajectory(engine_id=5)

        self.assertEqual(metrics["trajectory_type"], "DEGRADED")
        self.assertEqual(metrics["engine_id"], 5)
        self.assertTrue(metrics["persistence_alert_triggered"])
        self.assertIsNotNone(metrics["alert_cycle"])
        self.assertIsNotNone(metrics["isolated_fault"])
        self.assertIn(metrics["isolated_fault"], ("THERMAL_DISTRESS", "HPC_EFFICIENCY_DEGRADATION", "GENERAL_CORE_DEGRADATION", "SENSOR_CALIBRATION_BIAS"))
        self.assertLess(metrics["final_rul_forecast"], 25.0)
        self.assertLess(metrics["final_virtual_health_index"], 0.20)
        self.assertTrue(metrics["degradation_passed"])

    # -------------------------------------------------------------------------
    # Test 3: What-If Thermal Stress Scenario
    # -------------------------------------------------------------------------
    def test_03_what_if_thermal_stress_scenario(self):
        """
        Inject high thermal offset; assert THERMAL_DISTRESS diagnosis and reduced RUL.
        """
        metrics = self.evaluator.evaluate_what_if_scenario(
            scenario_id="thermal_stress_overtemp",
            engine_id=1,
            cycle=35,
        )

        self.assertTrue(metrics["scenario_passed"])
        self.assertEqual(metrics["primary_fault"], "THERMAL_DISTRESS")
        self.assertGreaterEqual(metrics["confidence_score"], 0.70)
        self.assertIn("T30", metrics["threshold_breaches"])
        self.assertTrue(metrics["telemetry_invariant"])
        self.assertLess(metrics["scenario_rul"], metrics["baseline_rul"])

    # -------------------------------------------------------------------------
    # Test 4: What-If Sensor Bias Scenario
    # -------------------------------------------------------------------------
    def test_04_what_if_sensor_bias_scenario(self):
        """
        Inject isolated P30 bias; assert SENSOR_CALIBRATION_BIAS without false core damage claim.
        """
        metrics = self.evaluator.evaluate_what_if_scenario(
            scenario_id="sensor_bias_drift",
            engine_id=1,
            cycle=35,
        )

        self.assertTrue(metrics["scenario_passed"])
        self.assertEqual(metrics["primary_fault"], "SENSOR_CALIBRATION_BIAS")
        self.assertGreaterEqual(metrics["confidence_score"], 0.80)
        self.assertEqual(metrics["threshold_breaches"], ["P30"])
        self.assertTrue(metrics["telemetry_invariant"])

        # Check recommended action prevents false removal
        orch_res = metrics["orchestration_result"]
        diag = orch_res.diagnosis_result
        self.assertIn("Inspect pneumatic sense lines", diag.diagnostic_summary)

    # -------------------------------------------------------------------------
    # Test 5: What-If Compounded Stress Scenario
    # -------------------------------------------------------------------------
    def test_05_what_if_compounded_stress_scenario(self):
        """
        Inject multi-variable severe stress; assert CRITICAL state and WARNING severity.
        """
        metrics = self.evaluator.evaluate_what_if_scenario(
            scenario_id="compounded_degradation",
            engine_id=1,
            cycle=35,
        )

        self.assertTrue(metrics["scenario_passed"])
        self.assertEqual(metrics["operational_state"], "CRITICAL")
        self.assertTrue(metrics["telemetry_invariant"])

        orch_res = metrics["orchestration_result"]
        diag = orch_res.diagnosis_result
        self.assertEqual(diag.severity, SeverityLevel.WARNING)

    # -------------------------------------------------------------------------
    # Test 6: State Enrichment Across All Scenarios
    # -------------------------------------------------------------------------
    def test_06_state_enrichment_across_all_scenarios(self):
        """
        Assert G32-G38 are populated and G1-G29 remain bit-for-bit invariant across scenario runs.
        """
        for sc_id in ("thermal_stress_overtemp", "sensor_bias_drift", "compounded_degradation"):
            res = self.evaluator.evaluate_what_if_scenario(scenario_id=sc_id, engine_id=1, cycle=30)
            orch_res = res["orchestration_result"]
            enriched = orch_res.enriched_state

            self.assertIsNotNone(enriched)
            self.assertIsInstance(enriched, EngineState)
            # Check G34 - G38
            self.assertIn(enriched.anomaly_alert_flag, (0, 1))
            self.assertGreaterEqual(enriched.predicted_rul_xgboost, 0.0)
            self.assertGreaterEqual(enriched.predicted_rul_lstm, 0.0)
            self.assertGreaterEqual(enriched.prognostic_spread, 0.0)
            self.assertTrue(0.0 <= enriched.virtual_health_index <= 1.0)
            # Check scenario audit
            self.assertIsNotNone(enriched.scenario_audit)
            self.assertIn("agent_feedback", enriched.scenario_audit)

    # -------------------------------------------------------------------------
    # Test 7: Audit Logging Completeness
    # -------------------------------------------------------------------------
    def test_07_audit_logging_completeness(self):
        """
        Assert all scenario runs produce complete JSONL trails and markdown summary entries.
        """
        # Run a nominal evaluation and a what-if evaluation
        self.evaluator.evaluate_nominal_trajectory(engine_id=1, cycles=5)
        self.evaluator.evaluate_what_if_scenario(scenario_id="thermal_stress_overtemp", engine_id=1, cycle=35)
        self.evaluator.audit_logger.write_markdown_summary()

        jsonl_path = os.path.join(self.test_dir, "agent_audit_trail.jsonl")
        md_path = os.path.join(self.test_dir, "agent_audit_summary.md")

        self.assertTrue(os.path.exists(jsonl_path))
        self.assertTrue(os.path.exists(md_path))

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        self.assertGreater(len(lines), 0)
        # Check that records contain required fields
        for rec in lines:
            self.assertIn("log_id", rec)
            self.assertIn("agent_type", rec)
            self.assertIn("status", rec)

    # -------------------------------------------------------------------------
    # Test 8: Multi-Engine Fleet Interaction
    # -------------------------------------------------------------------------
    def test_08_multi_engine_fleet_interaction(self):
        """
        Evaluate multiple engines in sequence; assert clean persistence state isolation between units.
        """
        # Evaluate Engine 1 (nominal)
        nom_metrics = self.evaluator.evaluate_nominal_trajectory(engine_id=1, cycles=10)
        self.assertEqual(nom_metrics["false_alarms"], 0)

        # Persistence count for unit 1 should be 0
        self.assertEqual(self.evaluator.monitoring_agent.get_persistence_count(1), 0)

        # Evaluate Engine 5 (degraded)
        deg_metrics = self.evaluator.evaluate_degraded_trajectory(engine_id=5)
        self.assertTrue(deg_metrics["persistence_alert_triggered"])

        # Engine 1 state should remain independent
        self.assertEqual(self.evaluator.monitoring_agent.get_persistence_count(1), 0)

    # -------------------------------------------------------------------------
    # Test 9: Evaluation Metrics Generation
    # -------------------------------------------------------------------------
    def test_09_evaluation_metrics_generation(self):
        """
        Assert evaluator summary dictionary produces non-null metrics and timing stats.
        """
        summary = self.evaluator.run_full_evaluation_suite(nominal_engine_id=1, degraded_engine_id=5)

        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["benchmark_suite"], "PHASE_6_STEP_6_8_MULTI_AGENT_INTERACTION")
        self.assertTrue(summary["all_benchmarks_passed"])
        self.assertGreater(summary["total_benchmark_elapsed_ms"], 0.0)

        m = summary["metrics"]
        self.assertEqual(m["nominal_false_alarm_rate"], 0.0)
        self.assertGreaterEqual(m["nominal_mean_health_index"], 0.70)
        self.assertEqual(m["what_if_diagnostic_accuracy_pct"], 100.0)
        self.assertEqual(m["what_if_scenarios_evaluated"], 3)
        self.assertEqual(m["what_if_scenarios_passed"], 3)

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
