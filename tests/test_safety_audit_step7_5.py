"""
Unit and Aerospace Safety Verification Test Suite for Step 7.5
Verifies Safety Reason Codes, Structured Audit Trail Records, and Append-Only JSONL Logging.

Roadmap Reference: Phase 7 - Step 7.5
"""

import json
import os
import hashlib
import tempfile
import unittest
from typing import Dict, Any

from safety.audit import (
    SafetyReasonCode,
    SafetyAuditRecord,
    SafetyAuditLogger,
    infer_reason_code,
)
from safety.decision import (
    SafetyDecisionStatus,
    ActionSafetyDecision,
    classify_safety_decision,
)
from safety.verification import (
    ActionType,
    SimulationAction,
    SafetyVerifier,
)
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestSafetyAuditStep7_5(unittest.TestCase):
    """
    Test suite for Phase 7 Step 7.5: Safety Decision Audit Logging and Reason Codes.
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
        self.verifier = SafetyVerifier()
        self.sim = DigitalTwinSimulator(subset="FD001")
        self.sim.load_engine(unit_number=1, split="train")
        self.nominal_state = self.sim.current_state

    def test_01_reason_code_enumeration(self):
        """Confirms all required domain reason codes exist in SafetyReasonCode."""
        expected_codes = [
            "NOMINAL_EXECUTION",
            "TELEMETRY_FIREWALL_BREACH",
            "THERMAL_CREEP_ADVISORY",
            "METALLURGICAL_HARD_LIMIT_BREACH",
            "PNEUMATIC_CASING_OVERPRESSURE",
            "STAGE_ORDERING_INVERSION",
            "AERODYNAMIC_ENVELOPE_EXCEEDANCE",
            "PROGNOSTIC_CAUSALITY_VIOLATION",
            "GENERAL_CONSTRAINT_VIOLATION",
        ]
        for code in expected_codes:
            self.assertIn(code, [c.value for c in SafetyReasonCode])
            self.assertEqual(getattr(SafetyReasonCode, code).value, code)

    def test_02_record_creation_nominal(self):
        """Verifies record structure for an ACCEPTED decision."""
        action = SimulationAction(
            action_id="act_nom_audit",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=4.0,
            initiator_id="flight_controller_01",
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        logger = SafetyAuditLogger(log_dir=tempfile.mkdtemp())
        rec = logger.log_decision(decision, action, self.nominal_state)

        self.assertIsInstance(rec, SafetyAuditRecord)
        self.assertEqual(rec.decision_status, SafetyDecisionStatus.ACCEPTED)
        self.assertEqual(rec.reason_code, SafetyReasonCode.NOMINAL_EXECUTION)
        self.assertEqual(rec.target_variable, "thermal_stress_offset")
        self.assertEqual(rec.initiator_id, "flight_controller_01")
        self.assertEqual(rec.flight_cycle, self.nominal_state.time_cycles)
        self.assertTrue(rec.record_id.startswith("SAR_"))

    def test_03_record_creation_firewall_breach(self):
        """Verifies TELEMETRY_FIREWALL_BREACH reason code for direct sensor mutation."""
        action = SimulationAction(
            action_id="act_illegal_telemetry",
            action_type=ActionType.STATE_ENRICHMENT,
            target_variable="T30",  # G8 immutable sensor
            proposed_value=1620.0,
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        logger = SafetyAuditLogger(log_dir=tempfile.mkdtemp())
        rec = logger.log_decision(decision, action, self.nominal_state)

        self.assertEqual(rec.decision_status, SafetyDecisionStatus.REJECTED)
        self.assertEqual(rec.reason_code, SafetyReasonCode.TELEMETRY_FIREWALL_BREACH)
        self.assertIn("FIREWALL_MEASURED_IMMUTABILITY", rec.violated_constraints)

    def test_04_record_creation_thermal_creep(self):
        """Verifies THERMAL_CREEP_ADVISORY reason code for REQUIRES_REVIEW status."""
        action = SimulationAction(
            action_id="act_creep_audit",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=22.0,  # T30 ~ 1611.7 °R (Caution boundary)
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        logger = SafetyAuditLogger(log_dir=tempfile.mkdtemp())
        rec = logger.log_decision(decision, action, self.nominal_state)

        self.assertEqual(rec.decision_status, SafetyDecisionStatus.REQUIRES_REVIEW)
        self.assertEqual(rec.reason_code, SafetyReasonCode.THERMAL_CREEP_ADVISORY)
        self.assertIn("CONST_TEMP_T30_CAUTION", rec.violated_constraints)

    def test_05_record_creation_hard_limit(self):
        """Verifies METALLURGICAL_HARD_LIMIT_BREACH for severe overtemp."""
        action = SimulationAction(
            action_id="act_hard_overtemp",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=45.0,  # T30 ~ 1634.7 °R (> 1625 °R warning)
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        logger = SafetyAuditLogger(log_dir=tempfile.mkdtemp())
        rec = logger.log_decision(decision, action, self.nominal_state)

        self.assertEqual(rec.decision_status, SafetyDecisionStatus.REJECTED)
        self.assertEqual(rec.reason_code, SafetyReasonCode.METALLURGICAL_HARD_LIMIT_BREACH)
        self.assertIn("CONST_TEMP_T30_MAX_WARNING", rec.violated_constraints)

    def test_06_record_creation_inversion(self):
        """Verifies STAGE_ORDERING_INVERSION for thermodynamic inversions."""
        action = SimulationAction(
            action_id="act_temp_inv",
            action_type=ActionType.THERMAL_PERTURBATION,
            target_variable="thermal_stress_offset",
            proposed_value=-1000.0,
        )
        decision = self.verifier.evaluate_action_decision(self.nominal_state, action)
        logger = SafetyAuditLogger(log_dir=tempfile.mkdtemp())
        rec = logger.log_decision(decision, action, self.nominal_state)

        self.assertEqual(rec.decision_status, SafetyDecisionStatus.REJECTED)
        self.assertEqual(rec.reason_code, SafetyReasonCode.STAGE_ORDERING_INVERSION)
        self.assertIn("CONST_INVARIANT_TEMP_ORDER", rec.violated_constraints)

    def test_07_append_only_jsonl_logging(self):
        """Verifies writing records to disk and parsing valid JSON lines back."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = SafetyAuditLogger(log_dir=tmp_dir, log_filename="test_log.jsonl")

            # Log 3 actions
            actions = [
                SimulationAction("a1", ActionType.THERMAL_PERTURBATION, "thermal_stress_offset", 3.0),
                SimulationAction("a2", ActionType.THERMAL_PERTURBATION, "thermal_stress_offset", 22.0),
                SimulationAction("a3", ActionType.THERMAL_PERTURBATION, "thermal_stress_offset", 50.0),
            ]
            for act in actions:
                dec = self.verifier.evaluate_action_decision(self.nominal_state, act)
                logger.log_decision(dec, act, self.nominal_state)

            log_file = os.path.join(tmp_dir, "test_log.jsonl")
            self.assertTrue(os.path.exists(log_file))

            with open(log_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            self.assertEqual(len(lines), 3)
            parsed_records = [json.loads(line) for line in lines]
            self.assertEqual(parsed_records[0]["action_id"], "a1")
            self.assertEqual(parsed_records[0]["decision_status"], "ACCEPTED")
            self.assertEqual(parsed_records[1]["action_id"], "a2")
            self.assertEqual(parsed_records[1]["decision_status"], "REQUIRES_REVIEW")
            self.assertEqual(parsed_records[2]["action_id"], "a3")
            self.assertEqual(parsed_records[2]["decision_status"], "REJECTED")

    def test_08_logger_query_and_filtering(self):
        """Tests filtering audit trail records by status and retrieving recent entries."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = SafetyAuditLogger(log_dir=tmp_dir)

            act_accept = SimulationAction("act_acc", ActionType.THERMAL_PERTURBATION, "thermal_stress_offset", 2.0)
            act_review = SimulationAction("act_rev", ActionType.THERMAL_PERTURBATION, "thermal_stress_offset", 22.0)
            act_reject = SimulationAction("act_rej", ActionType.THERMAL_PERTURBATION, "thermal_stress_offset", 45.0)

            logger.log_decision(self.verifier.evaluate_action_decision(self.nominal_state, act_acc := act_accept), act_accept)
            logger.log_decision(self.verifier.evaluate_action_decision(self.nominal_state, act_rev := act_review), act_review)
            logger.log_decision(self.verifier.evaluate_action_decision(self.nominal_state, act_rej := act_reject), act_reject)

            # Query recent
            recent_2 = logger.get_recent_records(2)
            self.assertEqual(len(recent_2), 2)
            self.assertEqual(recent_2[0].action_id, "act_rev")
            self.assertEqual(recent_2[1].action_id, "act_rej")

            # Filter by status
            accepted = logger.filter_by_status(SafetyDecisionStatus.ACCEPTED)
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0].action_id, "act_acc")

            review = logger.filter_by_status(SafetyDecisionStatus.REQUIRES_REVIEW)
            self.assertEqual(len(review), 1)
            self.assertEqual(review[0].action_id, "act_rev")

            rejected = logger.filter_by_status(SafetyDecisionStatus.REJECTED)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0].action_id, "act_rej")

            # Clear memory
            logger.clear_in_memory()
            self.assertEqual(len(logger.get_recent_records(10)), 0)

    def test_09_verifier_automatic_audit_integration(self):
        """Verifies that verifier with logger records decisions automatically."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = SafetyAuditLogger(log_dir=tmp_dir)
            verifier = SafetyVerifier(audit_logger=logger)

            action = SimulationAction(
                action_id="act_auto_audit",
                action_type=ActionType.STATE_ENRICHMENT,
                target_variable="prognostics",
                proposed_value={"predicted_rul": 115.0, "virtual_health_index": 0.94},
            )
            decision = verifier.evaluate_action_decision(self.nominal_state, action)
            self.assertEqual(decision.status, SafetyDecisionStatus.ACCEPTED)

            # Verify logger captured the record automatically
            records = logger.get_recent_records(5)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].action_id, "act_auto_audit")
            self.assertEqual(records[0].reason_code, SafetyReasonCode.NOMINAL_EXECUTION)

    def test_10_dataset_and_model_immutability(self):
        """Cryptographic SHA-256 verification ensuring zero raw datasets, scalers, or checkpoints are modified."""
        # 1. Verify all 12 raw C-MAPSS dataset files
        self.assertEqual(len(self.raw_hashes), 12, "Expected 12 raw C-MAPSS text files.")
        for fname, expected_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                expected_hash,
                f"Raw dataset file '{fname}' was modified during Step 7.5 execution!",
            )

        # 2. Verify all 4 subset scalers
        self.assertEqual(len(self.scaler_hashes), 4, "Expected 4 subset scalers.")
        for fname, expected_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                expected_hash,
                f"Scaler file '{fname}' was modified during Step 7.5 execution!",
            )

        # 3. Verify all 8 champion model checkpoints
        self.assertEqual(len(self.checkpoint_hashes), 8, "Expected 8 champion checkpoints.")
        for fname, expected_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                expected_hash,
                f"Champion checkpoint '{fname}' was modified during Step 7.5 execution!",
            )


if __name__ == "__main__":
    unittest.main()
