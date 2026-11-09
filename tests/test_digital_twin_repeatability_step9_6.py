"""
Phase 9 - Step 9.6: Digital Twin Scenario Repeatability & Simulation Determinism Verification
Roadmap Reference: Master Project Roadmap PDF, Section 10, Step 9.6
("Run Digital Twin scenarios and verify repeatability with identical inputs")
Strict Academic Integrity Standard: Master Roadmap Section 1, Page 2
("Do not invent sensor measurements or report fabricated model scores")

Suite verifies:
1. Multi-run bitwise repeatability across 50 iterations (|Δ| = 0.0, variance = 0.0)
2. Safe perturbations (+3.0 °R) deterministically return ACCEPTED
3. Redline breaches (+50.0 °R) deterministically return REJECTED with rollback recommendations
4. Sensor bias perturbations (+0.05 on EPR) deterministically return candidate states
5. Threshold boundary perturbations (+12.0 °R) deterministically return REQUIRES_REVIEW
6. State vector rollback fidelity preserves baseline state bit-for-bit upon rejected runs
7. Multi-regime determinism in FD002 across complex operating conditions
8. Compound fault determinism in FD003 across dual-failure flight profiles
9. Direct HTTP POST /api/v1/simulation/what-if endpoint repeatability
10. Deterministic execution hashing and audit logging
11. Cryptographic SHA-256 immutability across all 12 raw dataset files, 4 scalers, and 8 checkpoints
"""

import copy
import hashlib
import json
import os
import shutil
import tempfile
import unittest

from backend.main import app
from backend.services.digital_twin_repeatability_auditor import (
    audit_scenario_repeatability,
    verify_state_rollback_fidelity,
    compute_execution_hash,
)
from safety.audit import SafetyAuditLogger
from safety.verification import (
    SimulationAction,
    ActionType,
    SafetyVerifier,
)
from safety.decision import SafetyDecisionStatus
from tests.test_backend_endpoints_step8_2 import FastAPITestClientAdapter


class TestDigitalTwinRepeatabilityStep9_6(unittest.TestCase):
    """
    Automated verification suite validating repeatability, state isolation,
    and determinism in the turbofan digital twin closed-loop simulation architecture.
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

        cls.client = FastAPITestClientAdapter(app)

    def test_01_safe_perturbation_repeatability(self):
        """
        Executes 50 repeated runs of mild safe thermal perturbation (+3.0 °R).
        Asserts max absolute difference is 0.0, channel variance is 0.0, and 100% ACCEPTED.
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 3.0,
            "operator_approval": False,
        }
        report = audit_scenario_repeatability(payload, iterations=50)

        self.assertEqual(report["iterations"], 50)
        self.assertTrue(report["all_identical"])
        self.assertEqual(report["max_absolute_difference"], 0.0)
        self.assertEqual(report["decision_status"], "ACCEPTED")
        self.assertEqual(report["status_uniformity"], 1.0)
        self.assertEqual(report["unique_execution_hashes_count"], 1)
        self.assertEqual(len(report["violations"]), 0)

        # Assert all numerical channels have strictly zero variance
        for ch, var in report["channel_variances"].items():
            self.assertEqual(var, 0.0, f"Non-zero variance detected on channel {ch}: {var}")

    def test_02_redline_breach_repeatability(self):
        """
        Executes 50 repeated runs of severe redline breach (+50.0 °R).
        Asserts max absolute difference is 0.0, 100% REJECTED, and identical recovery plan.
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 50.0,
            "operator_approval": False,
        }
        report = audit_scenario_repeatability(payload, iterations=50)

        self.assertEqual(report["iterations"], 50)
        self.assertTrue(report["all_identical"])
        self.assertEqual(report["max_absolute_difference"], 0.0)
        self.assertEqual(report["decision_status"], "REJECTED")
        self.assertEqual(report["status_uniformity"], 1.0)
        self.assertEqual(report["unique_execution_hashes_count"], 1)
        self.assertIn("PHYSICAL_INSPECTION", str(report["recovery_strategy"]))

        # Confirms metallurgical hard limit violation was consistently triggered
        self.assertIn("CONST_TEMP_T30_MAX_WARNING", report["violations"])

    def test_03_sensor_bias_repeatability(self):
        """
        Executes 50 repeated runs of sensor bias perturbation (+0.05 on EPR).
        Asserts bitwise identical candidate state and 0.0 variance.
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "SENSOR_BIAS_INJECTION",
            "target_variable": "overall_pressure_ratio",
            "proposed_value": 0.05,
            "operator_approval": False,
        }
        report = audit_scenario_repeatability(payload, iterations=50)

        self.assertEqual(report["iterations"], 50)
        self.assertTrue(report["all_identical"])
        self.assertEqual(report["max_absolute_difference"], 0.0)
        self.assertEqual(report["decision_status"], "ACCEPTED")
        self.assertEqual(report["status_uniformity"], 1.0)
        self.assertEqual(report["unique_execution_hashes_count"], 1)

    def test_04_review_required_threshold_repeatability(self):
        """
        Executes 50 repeated runs of boundary threshold perturbation (+12.0 °R at Cycle 25).
        Asserts deterministic REQUIRES_REVIEW verdict and 100% status uniformity.
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 25,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 12.0,
            "operator_approval": False,
        }
        report = audit_scenario_repeatability(payload, iterations=50)

        self.assertEqual(report["iterations"], 50)
        self.assertTrue(report["all_identical"])
        self.assertEqual(report["max_absolute_difference"], 0.0)
        self.assertEqual(report["decision_status"], "REQUIRES_REVIEW")
        self.assertEqual(report["status_uniformity"], 1.0)
        self.assertIn("CONST_TEMP_T30_CAUTION", report["violations"])

    def test_05_state_vector_rollback_fidelity(self):
        """
        Verifies pre-simulation vs post-rollback bitwise match on rejected simulations.
        Asserts baseline engine state vector S(t) is completely uncontaminated (|Δ| = 0.0).
        """
        # Scenario A: Redline thermal breach (+50.0 °R)
        ok_redline = verify_state_rollback_fidelity(
            subset="FD001",
            unit=1,
            cycle=1,
            perturbation_payload={
                "action_type": "THERMAL_PERTURBATION",
                "target_variable": "thermal_stress_offset",
                "proposed_value": 50.0,
            },
        )
        self.assertTrue(ok_redline, "Baseline state corrupted following redline breach rejection")

        # Scenario B: Direct firewall breach attempting mutation of measured sensor T30
        ok_firewall = verify_state_rollback_fidelity(
            subset="FD001",
            unit=1,
            cycle=1,
            perturbation_payload={
                "action_type": "THERMAL_PERTURBATION",
                "target_variable": "T30",
                "proposed_value": 1650.0,
            },
        )
        self.assertTrue(ok_firewall, "Baseline state corrupted following firewall violation rejection")

    def test_06_multi_regime_fd002_simulation_determinism(self):
        """
        Verifies repeatability under FD002 multi-regime operating conditions (Mach 0-0.84, Alt 0-42k ft).
        """
        payload = {
            "subset": "FD002",
            "unit_number": 1,
            "cycle": 10,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 3.0,
            "operator_approval": False,
        }
        report = audit_scenario_repeatability(payload, iterations=50)

        self.assertEqual(report["iterations"], 50)
        self.assertTrue(report["all_identical"])
        self.assertEqual(report["max_absolute_difference"], 0.0)
        self.assertEqual(report["decision_status"], "ACCEPTED")
        self.assertEqual(report["status_uniformity"], 1.0)
        self.assertEqual(report["unique_execution_hashes_count"], 1)

    def test_07_compound_fault_fd003_simulation_determinism(self):
        """
        Verifies repeatability under FD003 compound fault conditions (HPC + Fan degradation).
        """
        payload = {
            "subset": "FD003",
            "unit_number": 8,
            "cycle": 100,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 3.0,
            "operator_approval": False,
        }
        report = audit_scenario_repeatability(payload, iterations=50)

        self.assertEqual(report["iterations"], 50)
        self.assertTrue(report["all_identical"])
        self.assertEqual(report["max_absolute_difference"], 0.0)
        self.assertEqual(report["decision_status"], "ACCEPTED")
        self.assertEqual(report["status_uniformity"], 1.0)
        self.assertEqual(report["unique_execution_hashes_count"], 1)

    def test_08_api_what_if_endpoint_repeatability(self):
        """
        Verifies direct HTTP POST /api/v1/simulation/what-if endpoint returns identical JSON payloads.
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 3.0,
            "operator_approval": False,
        }
        report = audit_scenario_repeatability(payload, iterations=10, test_client=self.client)

        self.assertEqual(report["iterations"], 10)
        self.assertTrue(report["all_identical"])
        self.assertEqual(report["max_absolute_difference"], 0.0)
        self.assertEqual(report["decision_status"], "ACCEPTED")
        self.assertEqual(report["status_uniformity"], 1.0)
        self.assertEqual(report["unique_execution_hashes_count"], 1)

    def test_09_audit_log_repeatability_recording(self):
        """
        Verifies safety audit logger captures each simulation decision with deterministic execution hashes.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            logger = SafetyAuditLogger(log_dir=temp_dir, log_filename="repeatability_audit.jsonl")
            verifier = SafetyVerifier(strict_firewall=True, audit_logger=logger)

            action = SimulationAction(
                action_id="act_rep_audit_01",
                action_type=ActionType.THERMAL_PERTURBATION,
                target_variable="thermal_stress_offset",
                proposed_value=3.0,
            )
            state_dict = {
                "unit_number": 1,
                "time_cycles": 1,
                "altitude": 0.0,
                "mach_number": 0.0,
                "throttle_resolver_angle": 100.0,
                "T2": 518.67,
                "T24": 642.4,
                "T30": 1589.7,
                "T50": 1400.6,
                "P2": 14.62,
                "P15": 21.61,
                "P30": 553.8,
                "Nf": 2388.0,
                "Nc": 9050.0,
            }

            hashes = []
            for _ in range(25):
                decision = verifier.evaluate_action_decision(state_dict, action, attach_recovery_plan=True)
                h = compute_execution_hash(
                    state_dict=decision.projected_state or {},
                    decision_status=decision.status.value,
                    violations=[v.constraint_id for v in decision.violations],
                )
                hashes.append(h)

            self.assertEqual(len(hashes), 25)
            self.assertEqual(len(set(hashes)), 1, "Execution hashes diverged across identical runs")

            # Inspect persistent log entries
            records = logger.get_recent_records(100)
            self.assertEqual(len(records), 25)
            reasons = set(r.reason_code for r in records)
            self.assertEqual(len(reasons), 1)
            statuses = set(r.decision_status for r in records)
            self.assertEqual(len(statuses), 1)
            self.assertEqual(list(statuses)[0], SafetyDecisionStatus.ACCEPTED)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

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
