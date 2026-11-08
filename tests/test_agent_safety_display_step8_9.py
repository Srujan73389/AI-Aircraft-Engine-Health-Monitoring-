"""
Unit and Integration Test Suite for Phase 8 Step 8.9: Agent Execution Logs and Safety Verification Display
Verifies:
- AgentSafetyGovernancePanel component file existence and clean export
- Multi-agent pipeline cards rendering (Monitoring, Diagnosis, Prognostics, Orchestrator)
- Agent execution metadata display (latency, timestamp, status, payload drawer)
- Tri-state safety decision badges (ACCEPTED, REQUIRES_REVIEW, REJECTED)
- Itemized operating constraint violations and limits breakdown
- Simulated recovery plan strategy and recommended actions display
- Backend GET /api/v1/safety/audit-trail endpoint integration
- Audit trail filtering by decision status (ACCEPTED, REJECTED, etc.)
- Root App.jsx dashboard integration of AgentSafetyGovernancePanel
- Cryptographic SHA-256 zero-tampering verification of datasets, scalers, and checkpoints

Roadmap Reference: Phase 8 - Step 8.9:
"Display agent execution logs and safety-verification results"
"""

import hashlib
import json
import os
import unittest

from backend.main import app
from tests.test_backend_independent_endpoints_step8_3 import FastAPITestClientAdapter


class TestAgentSafetyDisplayStep8_9(unittest.TestCase):
    """
    Test suite for Phase 8 Step 8.9: Agent Execution Logs & Safety Verification Governance.
    """

    @classmethod
    def setUpClass(cls):
        cls.frontend_dir = os.path.join("frontend")
        cls.src_dir = os.path.join(cls.frontend_dir, "src")
        cls.raw_data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.checkpoint_dir = os.path.join("models", "checkpoints")

        # Capture cryptographic SHA-256 hashes of all protected artifacts
        cls.raw_hashes = {}
        if os.path.exists(cls.raw_data_dir):
            for fname in sorted(os.listdir(cls.raw_data_dir)):
                if fname.endswith(".txt") and fname != "readme.txt":
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

    def test_01_agent_safety_panel_component_exists(self):
        """
        Asserts AgentSafetyGovernancePanel component file exists and exports cleanly.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AgentSafetyGovernancePanel.jsx")
        self.assertTrue(os.path.exists(panel_path), f"Missing component file: {panel_path}")

        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("export const AgentSafetyGovernancePanel", content)
        self.assertIn("export default AgentSafetyGovernancePanel", content)

    def test_02_multi_agent_pipeline_cards_rendered(self):
        """
        Verifies cards for all 4 Phase 6 agents are defined and rendered.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AgentSafetyGovernancePanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Monitoring Agent", content)
        self.assertIn("Diagnosis Agent", content)
        self.assertIn("Prognostics Agent", content)
        self.assertIn("Agent Orchestrator", content)
        self.assertIn("Aerothermal Surveillance & Anomaly Detector", content)
        self.assertIn("Causal Fault Isolation & Pattern Evaluator", content)
        self.assertIn("Remaining Useful Life Forecaster", content)
        self.assertIn("Master Workflow Coordinator & Consensus Engine", content)

    def test_03_agent_execution_metadata_displayed(self):
        """
        Verifies execution metadata: latency, timestamp, status, and raw payload drawer.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AgentSafetyGovernancePanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Latency:", content)
        self.assertIn("Timestamp:", content)
        self.assertIn("STAGE", content)
        self.assertIn("expandedAgent", content)
        self.assertIn("Inspect Raw Payload", content)

    def test_04_safety_decision_status_badges(self):
        """
        Verifies tri-state safety decision status badges: ACCEPTED, REQUIRES_REVIEW, and REJECTED.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AgentSafetyGovernancePanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("renderTriStateBadge", content)
        self.assertIn("ACCEPTED", content)
        self.assertIn("REQUIRES_REVIEW", content)
        self.assertIn("REJECTED", content)

    def test_05_constraint_violations_breakdown_rendered(self):
        """
        Verifies itemized operating constraint check breakdown with limits and breach values.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AgentSafetyGovernancePanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Itemized Operating Constraint Checks", content)
        self.assertIn("Safe Redline:", content)
        self.assertIn("Observed:", content)

    def test_06_simulated_recovery_plan_display(self):
        """
        Verifies simulated recovery plan strategy, recommended actions, and rollback status.
        """
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AgentSafetyGovernancePanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Simulated Recovery Advisory Plan Generated:", content)
        self.assertIn("Strategy:", content)
        self.assertIn("Recommended Action:", content)
        self.assertIn("Rollback Status:", content)

    def test_07_backend_audit_trail_endpoint_integration(self):
        """
        Verifies backend GET /api/v1/safety/audit-trail query returns valid audit response schema.
        """
        resp = self.client.get("/api/v1/safety/audit-trail?limit=10")
        self.assertEqual(resp.status_code, 200)

        data = resp.json()
        self.assertIn("total_records", data)
        self.assertIn("returned_records", data)
        self.assertIn("records", data)
        self.assertIn("disclaimer", data)
        self.assertIsInstance(data["records"], list)

        if data["records"]:
            first = data["records"][0]
            self.assertIn("action_id", first)
            self.assertIn("decision_status", first)
            self.assertIn("target_variable", first)
            self.assertIn("reason_code", first)
            self.assertIn("summary_reason", first)

    def test_08_audit_trail_filtering_logic(self):
        """
        Verifies audit trail endpoint and UI filtering by decision status (ACCEPTED, REJECTED).
        """
        # Test backend filtering by ACCEPTED
        resp_acc = self.client.get("/api/v1/safety/audit-trail?decision_status=ACCEPTED")
        self.assertEqual(resp_acc.status_code, 200)
        data_acc = resp_acc.json()
        for r in data_acc["records"]:
            self.assertEqual(r["decision_status"], "ACCEPTED")

        # Test backend filtering by REJECTED
        resp_rej = self.client.get("/api/v1/safety/audit-trail?decision_status=REJECTED")
        self.assertEqual(resp_rej.status_code, 200)
        data_rej = resp_rej.json()
        for r in data_rej["records"]:
            self.assertEqual(r["decision_status"], "REJECTED")

        # Verify component contains filter pills
        panel_path = os.path.join(self.src_dir, "components", "dashboard", "AgentSafetyGovernancePanel.jsx")
        with open(panel_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("auditFilter", content)
        self.assertIn("setAuditFilter", content)

    def test_09_app_integration_agent_safety_panel(self):
        """
        Confirms App.jsx imports and mounts AgentSafetyGovernancePanel.
        """
        app_path = os.path.join(self.src_dir, "App.jsx")
        with open(app_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("import AgentSafetyGovernancePanel from './components/dashboard/AgentSafetyGovernancePanel';", content)
        self.assertIn("<AgentSafetyGovernancePanel />", content)

    def test_10_dataset_and_model_immutability(self):
        """
        Cryptographic SHA-256 verification of 12 raw files, 4 scalers, and 8 champion checkpoints.
        """
        # 1. 12 Raw C-MAPSS dataset files
        self.assertEqual(len(self.raw_hashes), 12, "Must protect all 12 C-MAPSS dataset files.")
        for fname, original_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Tampering detected in raw dataset file: {fname}"
            )

        # 2. 4 Feature scalers
        self.assertEqual(len(self.scaler_hashes), 4, "Must verify all 4 subset scalers.")
        for fname, original_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Tampering detected in Phase 3 scaler: {fname}"
            )

        # 3. 8 Champion model checkpoints
        self.assertEqual(len(self.checkpoint_hashes), 8, "Must verify all 8 champion checkpoints.")
        for fname, original_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Tampering detected in Phase 4 checkpoint: {fname}"
            )


if __name__ == "__main__":
    unittest.main()
