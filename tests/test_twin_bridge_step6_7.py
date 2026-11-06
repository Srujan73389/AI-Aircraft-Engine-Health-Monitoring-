"""
Unit and Integration Tests for Phase 6 Step 6.7: Digital Twin Safety Bridge
Verifies safe state enrichment, strict telemetry and thermodynamic invariance,
rejection of unphysical agent feedback, non-destructive audit trail attachment,
end-to-end multi-agent orchestrator integration, and artifact immutability.

Roadmap Reference: Phase 6 - Step 6.7:
"Connect agent outputs to the Digital Twin without allowing agents to bypass safety checks"
"""

import copy
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
    MonitoringOutput,
    DiagnosisOutput,
    PrognosticsOutput,
    OrchestrationResult,
)
from agents.twin_bridge import DigitalTwinSafetyBridge, SafetyViolationError, IMMUTABLE_PHYSICAL_FIELDS
from agents.orchestrator import AgentOrchestrator
from digital_twin.twin_simulator import DigitalTwinSimulator
from digital_twin.engine_state import EngineState


class TestTwinBridgeStep6_7(unittest.TestCase):
    """
    Test suite for Phase 6 Step 6.7: Digital Twin Safety Bridge.
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
        """Instantiate fresh Bridge, Orchestrator, and Simulator for each test."""
        self.bridge = DigitalTwinSafetyBridge(agent_id="test_bridge_v1")
        self.sim = DigitalTwinSimulator(subset="FD001")
        self.sim.load_engine(unit_number=1, split="train")
        # Step forward to cycle 30 so that normalized features and sequence tensors exist
        for _ in range(29):
            self.sim.step()
        self.nominal_state = self.sim.current_state

    # -------------------------------------------------------------------------
    # Helper: Build Valid OrchestrationResult
    # -------------------------------------------------------------------------
    def _create_mock_orchestration_result(
        self,
        persistence_alert: bool = False,
        xgb_rul: float = 112.5,
        lstm_rul: float = 110.0,
        health_idx: float = 0.88,
        status: AgentExecutionStatus = AgentExecutionStatus.SUCCESS,
    ) -> OrchestrationResult:
        mon = MonitoringOutput(
            unit_number=1,
            time_cycles=30,
            operational_state=OperationalState.HEALTHY,
            threshold_breaches=[],
            model_anomaly_flag=False,
            isolation_forest_breach=False,
            pca_spe_breach=False,
            persistence_count=0,
            persistence_alert=persistence_alert,
            anomaly_severity=SeverityLevel.NOMINAL,
            summary_message="Nominal telemetry",
            metadata=AgentMetadata(
                agent_id="monitoring_agent",
                execution_timestamp="2026-11-06T11:00:00Z",
                execution_time_ms=5.0,
                status=AgentExecutionStatus.SUCCESS,
            ),
        )

        diag = DiagnosisOutput(
            unit_number=1,
            time_cycles=30,
            primary_fault=None,
            confidence_score=0.0,
            severity=SeverityLevel.NOMINAL,
            fault_candidates=[],
            diagnostic_summary="Nominal aero-thermal operation.",
            requires_maintenance_action=False,
            metadata=AgentMetadata(
                agent_id="diagnosis_agent",
                execution_timestamp="2026-11-06T11:00:00Z",
                execution_time_ms=2.0,
                status=AgentExecutionStatus.SUCCESS,
            ),
        )

        prog = PrognosticsOutput(
            unit_number=1,
            time_cycles=30,
            subset="FD001",
            predicted_rul_xgboost=xgb_rul,
            predicted_rul_lstm=lstm_rul,
            ensemble_rul_estimate=(xgb_rul + lstm_rul) / 2.0,
            prognostic_spread=abs(xgb_rul - lstm_rul),
            virtual_health_index=health_idx,
            uncertainty_classification="LOW_UNCERTAINTY",
            cycles_to_caution_threshold=max(0.0, (xgb_rul + lstm_rul) / 2.0 - 45.0),
            prognostic_summary="Nominal prognosis",
            metadata=AgentMetadata(
                agent_id="prognostics_agent",
                execution_timestamp="2026-11-06T11:00:00Z",
                execution_time_ms=10.0,
                status=AgentExecutionStatus.SUCCESS,
            ),
        )

        return OrchestrationResult(
            run_id="test_run_123",
            engine_id=1,
            cycle=30,
            subset="FD001",
            status=status,
            monitoring_result=mon,
            diagnosis_result=diag,
            prognostics_result=prog,
            execution_timeline_ms={"total_elapsed_ms": 17.0},
            metadata=AgentMetadata(
                agent_id="orchestrator",
                execution_timestamp="2026-11-06T11:00:00Z",
                execution_time_ms=17.0,
                status=status,
            ),
        )

    # -------------------------------------------------------------------------
    # Test 1: Successful State Enrichment
    # -------------------------------------------------------------------------
    def test_01_successful_state_enrichment(self):
        """Verify G32-G38 populated from complete OrchestrationResult without error."""
        orch_res = self._create_mock_orchestration_result(
            persistence_alert=True,
            xgb_rul=95.0,
            lstm_rul=91.0,
            health_idx=0.76,
        )

        enriched = self.bridge.enrich_state_with_agent_outputs(self.nominal_state, orch_res)

        self.assertIsInstance(enriched, EngineState)
        # G34 (anomaly_alert_flag)
        self.assertEqual(enriched.anomaly_alert_flag, 1)
        # G35 (predicted_rul_xgboost)
        self.assertEqual(enriched.predicted_rul_xgboost, 95.0)
        # G36 (predicted_rul_lstm)
        self.assertEqual(enriched.predicted_rul_lstm, 91.0)
        # G37 (prognostic_spread)
        self.assertEqual(enriched.prognostic_spread, 4.0)
        # G38 (virtual_health_index)
        self.assertEqual(enriched.virtual_health_index, 0.76)

    # -------------------------------------------------------------------------
    # Test 2: Firewall Telemetry Immutability (G1 - G29)
    # -------------------------------------------------------------------------
    def test_02_firewall_telemetry_immutability(self):
        """Verify G1-G26 and G27-G29 remain 100% bit-for-bit identical before and after enrichment."""
        orch_res = self._create_mock_orchestration_result()
        enriched = self.bridge.enrich_state_with_agent_outputs(self.nominal_state, orch_res)

        for field_name in IMMUTABLE_PHYSICAL_FIELDS:
            orig_val = getattr(self.nominal_state, field_name)
            cand_val = getattr(enriched, field_name)
            self.assertEqual(
                orig_val,
                cand_val,
                f"Immutable field {field_name} was modified! Orig: {orig_val}, Enriched: {cand_val}",
            )

        # Also check G30, G31 arrays
        np.testing.assert_array_equal(
            self.nominal_state.normalized_feature_vec,
            enriched.normalized_feature_vec,
        )
        np.testing.assert_array_equal(
            self.nominal_state.temporal_seq_tensor,
            enriched.temporal_seq_tensor,
        )

    # -------------------------------------------------------------------------
    # Test 3: Thermodynamic Sanity Consistency
    # -------------------------------------------------------------------------
    def test_03_thermodynamic_sanity_consistency(self):
        """Verify check_thermodynamic_sanity() passes before and after enrichment."""
        orig_sanity = self.nominal_state.check_thermodynamic_sanity()
        self.assertTrue(orig_sanity["overall_sanity_pass"])

        orch_res = self._create_mock_orchestration_result()
        enriched = self.bridge.enrich_state_with_agent_outputs(self.nominal_state, orch_res)

        cand_sanity = enriched.check_thermodynamic_sanity()
        self.assertTrue(cand_sanity["overall_sanity_pass"])
        self.assertEqual(orig_sanity, cand_sanity)

    # -------------------------------------------------------------------------
    # Test 4: Safety Violation on Telemetry Tampering
    # -------------------------------------------------------------------------
    def test_04_safety_violation_on_telemetry_tampering(self):
        """Verify modifying any G1-G29 field raises SafetyViolationError."""
        tampered_state = copy.deepcopy(self.nominal_state)
        # Tamper with temperature sensor T30
        tampered_state.T30 = tampered_state.T30 + 10.0

        with self.assertRaises(SafetyViolationError) as ctx:
            self.bridge.validate_feedback_safety(self.nominal_state, tampered_state)
        self.assertIn("CRITICAL SAFETY VIOLATION", str(ctx.exception))
        self.assertIn("T30", str(ctx.exception))

        # Tamper with cycle count G2
        tampered_state2 = copy.deepcopy(self.nominal_state)
        tampered_state2.time_cycles = tampered_state2.time_cycles + 1

        with self.assertRaises(SafetyViolationError) as ctx2:
            self.bridge.validate_feedback_safety(self.nominal_state, tampered_state2)
        self.assertIn("time_cycles", str(ctx2.exception))

    # -------------------------------------------------------------------------
    # Test 5: Safety Violation on Negative RUL
    # -------------------------------------------------------------------------
    def test_05_safety_violation_on_negative_rul(self):
        """Verify ingesting negative RUL raises SafetyViolationError."""
        orch_res = self._create_mock_orchestration_result(xgb_rul=-5.0)

        with self.assertRaises(SafetyViolationError) as ctx:
            self.bridge.enrich_state_with_agent_outputs(self.nominal_state, orch_res)
        self.assertTrue("predicted_rul_xgboost" in str(ctx.exception) or "invalid XGBoost RUL" in str(ctx.exception))

        # Test validation method directly for LSTM negative RUL
        tampered = copy.deepcopy(self.nominal_state)
        tampered.predicted_rul_lstm = -1.0
        with self.assertRaises(SafetyViolationError) as ctx2:
            self.bridge.validate_feedback_safety(self.nominal_state, tampered)
        self.assertIn("predicted_rul_lstm", str(ctx2.exception))

    # -------------------------------------------------------------------------
    # Test 6: Safety Violation on Invalid Health Index
    # -------------------------------------------------------------------------
    def test_06_safety_violation_on_invalid_health_index(self):
        """Verify ingesting H(t) > 1.0 or < 0.0 raises SafetyViolationError."""
        orch_res_high = self._create_mock_orchestration_result(health_idx=1.25)
        with self.assertRaises(SafetyViolationError) as ctx:
            self.bridge.enrich_state_with_agent_outputs(self.nominal_state, orch_res_high)
        self.assertTrue("virtual_health_index" in str(ctx.exception) or "virtual health index" in str(ctx.exception))

        orch_res_low = self._create_mock_orchestration_result(health_idx=-0.05)
        with self.assertRaises(SafetyViolationError) as ctx2:
            self.bridge.enrich_state_with_agent_outputs(self.nominal_state, orch_res_low)
        self.assertTrue("virtual_health_index" in str(ctx2.exception) or "virtual health index" in str(ctx2.exception))

    # -------------------------------------------------------------------------
    # Test 7: Safety Violation on Non-Binary Flag
    # -------------------------------------------------------------------------
    def test_07_safety_violation_on_non_binary_flag(self):
        """Verify non-binary anomaly_alert_flag raises SafetyViolationError."""
        tampered = copy.deepcopy(self.nominal_state)
        tampered.anomaly_alert_flag = 2  # Non-binary value

        with self.assertRaises(SafetyViolationError) as ctx:
            self.bridge.validate_feedback_safety(self.nominal_state, tampered)
        self.assertIn("anomaly_alert_flag", str(ctx.exception))
        self.assertIn("must be binary 0 or 1", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Test 8: End-to-End Orchestrator Twin Feedback
    # -------------------------------------------------------------------------
    def test_08_end_to_end_orchestrator_twin_feedback(self):
        """Run AgentOrchestrator with enable_twin_feedback=True on DigitalTwinSimulator state."""
        orchestrator = AgentOrchestrator(
            agent_id="test_feedback_orch",
            subset="FD001",
            default_mode="full",
            enable_twin_feedback=True,
            twin_bridge=self.bridge,
        )

        result = orchestrator.orchestrate_state(self.nominal_state, mode="full")

        self.assertEqual(result.status, AgentExecutionStatus.SUCCESS)
        self.assertIsNotNone(result.enriched_state)
        self.assertIsInstance(result.enriched_state, EngineState)

        # Check enrichment slots on returned enriched_state
        self.assertIn(result.enriched_state.anomaly_alert_flag, (0, 1))
        self.assertGreaterEqual(result.enriched_state.predicted_rul_xgboost, 0.0)
        self.assertGreaterEqual(result.enriched_state.predicted_rul_lstm, 0.0)
        self.assertGreaterEqual(result.enriched_state.prognostic_spread, 0.0)
        self.assertTrue(0.0 <= result.enriched_state.virtual_health_index <= 1.0)

        # Ensure timeline profiling captured twin_bridge_ms
        self.assertIn("twin_bridge_ms", result.execution_timeline_ms)
        self.assertGreaterEqual(result.execution_timeline_ms["twin_bridge_ms"], 0.0)

    # -------------------------------------------------------------------------
    # Test 9: Scenario Audit Diagnostic Metadata
    # -------------------------------------------------------------------------
    def test_09_scenario_audit_diagnostic_metadata(self):
        """Verify diagnostic summary and metadata attached into scenario_audit without altering telemetry."""
        # Pre-seed some existing scenario audit
        state_with_audit = copy.deepcopy(self.nominal_state)
        state_with_audit.scenario_audit = {"simulation_tag": "pre_run_check"}

        orch_res = self._create_mock_orchestration_result()
        enriched = self.bridge.enrich_state_with_agent_outputs(state_with_audit, orch_res)

        self.assertIn("simulation_tag", enriched.scenario_audit)
        self.assertEqual(enriched.scenario_audit["simulation_tag"], "pre_run_check")
        self.assertIn("agent_feedback", enriched.scenario_audit)

        fb = enriched.scenario_audit["agent_feedback"]
        self.assertEqual(fb["run_id"], "test_run_123")
        self.assertEqual(fb["pipeline_status"], "SUCCESS")
        self.assertIn("diagnosis", fb)
        self.assertIn("monitoring", fb)
        self.assertIn("prognostics", fb)
        self.assertEqual(fb["diagnosis"]["primary_fault"], None)
        self.assertIn("Nominal aero-thermal operation", fb["diagnosis"]["diagnostic_summary"])

    # -------------------------------------------------------------------------
    # Test 10: Cryptographic Immutability of Datasets and Champion Artifacts
    # -------------------------------------------------------------------------
    def test_10_dataset_and_model_immutability(self):
        """Verify that all raw data files, scalers, and champion checkpoints are untouched."""
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
