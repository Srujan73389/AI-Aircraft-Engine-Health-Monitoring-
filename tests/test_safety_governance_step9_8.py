"""
Phase 9 - Step 9.8: Safety Acceptance, Rejection and Defensive Missing-Input Verification
Roadmap Reference: Master Project Roadmap PDF, Section 10, Step 9.8
("Test safety acceptance, rejection and missing-input behaviour")
Strict Academic Integrity Standard: Master Roadmap Section 1, Page 2
("Do not invent sensor measurements or report fabricated model scores")

Test Suite Verifies:
1. Safe action acceptance (+2.0 °R thermal perturbation -> ACCEPTED, update_applied = True)
2. Redline breach rejection (+50.0 °R overtemp -> REJECTED, hard limit violation)
3. Safety firewall rejection (direct write to measured sensor T30 -> REJECTED, rollback mandated)
4. Caution limit evaluation without operator approval -> REQUIRES_REVIEW, update_applied = False
5. Caution limit evaluation with operator approval -> REQUIRES_REVIEW, update_applied = True
6. Missing mandatory input defense -> HTTP 422 on omitted required fields
7. Invalid action_type defense -> HTTP 422 on unrecognized action type string
8. NaN and Inf numerical rejection -> HTTP 422 on non-finite proposed or delta values
9. Non-destructive state rollback fidelity across diverse rejection types
10. Cryptographic SHA-256 immutability of 12 raw files, 4 scalers, and 8 champion checkpoints
"""

import hashlib
import math
import os
import unittest

from backend.main import app
from backend.services.safety_governance_auditor import (
    audit_safety_decision_matrix,
    audit_input_boundary_defense,
)
from backend.services.digital_twin_repeatability_auditor import verify_state_rollback_fidelity
from digital_twin.twin_simulator import DigitalTwinSimulator
from safety.verification import (
    SimulationAction,
    ActionType,
    SafetyVerifier,
)
from safety.decision import SafetyDecisionStatus
from safety.closed_loop import ClosedLoopSafetyGovernor
from tests.test_backend_endpoints_step8_2 import FastAPITestClientAdapter


class TestSafetyGovernanceStep9_8(unittest.TestCase):
    """
    Automated verification suite validating safety governance acceptance, rejection,
    human-in-the-loop review gating, and defensive input boundary enforcement.
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

    def test_01_safe_action_acceptance(self):
        """
        Validates ACCEPTED verdict and candidate state update on mild safe perturbation (+2.0 °R).
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 2.0,
            "operator_approval": False,
        }
        status_code, data = audit_input_boundary_defense(payload, client=self.client)

        self.assertEqual(status_code, 200)
        self.assertEqual(data["decision_status"], "ACCEPTED")
        self.assertTrue(data["update_applied"])
        self.assertFalse(data["operator_approved"])
        self.assertEqual(data["state_after"].get("thermal_stress_offset"), 2.0)
        self.assertIn("CONTINUE_MONITORING", str(data.get("recovery_plan", {}).get("strategy_type")))

    def test_02_redline_breach_rejection(self):
        """
        Validates REJECTED verdict on metallurgical redline breach (+50.0 °R).
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
        status_code, data = audit_input_boundary_defense(payload, client=self.client)

        self.assertEqual(status_code, 200)
        self.assertEqual(data["decision_status"], "REJECTED")
        self.assertFalse(data["update_applied"])
        self.assertIn("CONST_TEMP_T30_MAX_WARNING", data["reason_summary"])
        self.assertIsNotNone(data["recovery_plan"])
        self.assertTrue(data["recovery_plan"]["mitigation_parameters"].get("rollback_performed", False))

    def test_03_firewall_measured_immutability_rejection(self):
        """
        Validates rejection when attempting direct write to raw measured telemetry (T30).
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "T30",
            "proposed_value": 1650.0,
            "operator_approval": False,
        }
        status_code, data = audit_input_boundary_defense(payload, client=self.client)

        self.assertEqual(status_code, 200)
        self.assertEqual(data["decision_status"], "REJECTED")
        self.assertFalse(data["update_applied"])
        self.assertIn("safety firewall", data["reason_summary"].lower())
        self.assertEqual(data["recovery_plan"]["strategy_type"], "SCENARIO_ROLLBACK")
        self.assertTrue(data["recovery_plan"]["mitigation_parameters"].get("firewall_enforced", False))

    def test_04_caution_limit_requires_review(self):
        """
        Validates REQUIRES_REVIEW status when operator_approval is False.
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 22.0,
            "operator_approval": False,
        }
        status_code, data = audit_input_boundary_defense(payload, client=self.client)

        self.assertEqual(status_code, 200)
        self.assertEqual(data["decision_status"], "REQUIRES_REVIEW")
        self.assertFalse(data["update_applied"])
        self.assertFalse(data["operator_approved"])
        self.assertIn("blocked", data["reason_summary"].lower())

    def test_05_caution_limit_with_operator_approval(self):
        """
        Validates committed update under manual operator override signoff.
        """
        payload = {
            "subset": "FD001",
            "unit_number": 1,
            "cycle": 1,
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 22.0,
            "operator_approval": True,
        }
        status_code, data = audit_input_boundary_defense(payload, client=self.client)

        self.assertEqual(status_code, 200)
        self.assertEqual(data["decision_status"], "REQUIRES_REVIEW")
        self.assertTrue(data["update_applied"])
        self.assertTrue(data["operator_approved"])
        self.assertEqual(data["state_after"].get("thermal_stress_offset"), 22.0)

    def test_06_missing_mandatory_inputs_defense(self):
        """
        Asserts HTTP 422 on missing required request fields (target_variable, action_type, proposed_value).
        """
        # Missing target_variable
        code_no_var, _ = audit_input_boundary_defense({
            "action_type": "THERMAL_PERTURBATION",
            "proposed_value": 3.0,
        }, client=self.client)
        self.assertEqual(code_no_var, 422)

        # Missing proposed_value
        code_no_val, _ = audit_input_boundary_defense({
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
        }, client=self.client)
        self.assertEqual(code_no_val, 422)

        # Missing action_type
        code_no_act, _ = audit_input_boundary_defense({
            "target_variable": "thermal_stress_offset",
            "proposed_value": 3.0,
        }, client=self.client)
        self.assertEqual(code_no_act, 422)

    def test_07_invalid_action_type_defense(self):
        """
        Asserts HTTP 422 on illegal/unrecognized action_type string.
        """
        payload = {
            "action_type": "ILLEGAL_DESTROY_SYSTEM_COMMAND",
            "target_variable": "thermal_stress_offset",
            "proposed_value": 3.0,
        }
        code, body = audit_input_boundary_defense(payload, client=self.client)

        self.assertEqual(code, 422)
        self.assertIn("Invalid action_type", str(body))

    def test_08_nan_and_inf_payload_rejection(self):
        """
        Asserts HTTP 422 rejection of NaN and Infinite numerical values.
        """
        # Case A: NaN in proposed_value
        payload_nan = {
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": float("nan"),
        }
        code_nan, body_nan = audit_input_boundary_defense(payload_nan, client=self.client)
        self.assertEqual(code_nan, 422)
        self.assertIn("NaN or Infinite", str(body_nan))

        # Case B: Inf in proposed_value
        payload_inf = {
            "action_type": "THERMAL_PERTURBATION",
            "target_variable": "thermal_stress_offset",
            "proposed_value": float("inf"),
        }
        code_inf, body_inf = audit_input_boundary_defense(payload_inf, client=self.client)
        self.assertEqual(code_inf, 422)
        self.assertIn("NaN or Infinite", str(body_inf))

    def test_09_state_rollback_fidelity_across_rejections(self):
        """
        Asserts baseline engine state is completely preserved across diverse rejected actions.
        """
        # 1. Redline breach rollback
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
        self.assertTrue(ok_redline, "Baseline state contaminated after redline breach rollback")

        # 2. Firewall breach rollback
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
        self.assertTrue(ok_firewall, "Baseline state contaminated after firewall breach rollback")

        # 3. Unapproved caution breach rollback
        ok_caution = verify_state_rollback_fidelity(
            subset="FD001",
            unit=1,
            cycle=1,
            perturbation_payload={
                "action_type": "THERMAL_PERTURBATION",
                "target_variable": "thermal_stress_offset",
                "proposed_value": 22.0,
                "operator_approval": False,
            },
        )
        self.assertTrue(ok_caution, "Baseline state contaminated after unapproved caution rollback")

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
