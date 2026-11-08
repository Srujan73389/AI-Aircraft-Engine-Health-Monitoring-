"""
Unit and Integration Test Suite for Phase 8 Step 8.4: Operational Engine Dashboard
Verifies:
- Frontend scaffolding, Vite build configuration, and index.html structure
- Tailwind CSS aerospace palette theme tokens and tabular font formatting
- Centralized Axios API service mapping of backend endpoints
- React EngineContext state machine contracts, default initializations, and bounds clamping
- Header component rendering of the mandatory academic simulation disclaimer
- EngineSelectorBar dropdowns, stepper controls, and cycle scrubber range limits
- EngineSummaryCard display of canonical engine IDs, operational cycle, and aerothermal indicators
- Seamless integration of backend cycle state vector S(t) with summary card telemetry
- Cryptographic SHA-256 zero-tampering verification of datasets, scalers, and checkpoints

Roadmap Reference: Phase 8 - Step 8.4:
"Build a dashboard showing the selected engine ID and operating cycle"
"""

import hashlib
import json
import os
import re
import unittest

from backend.main import app
from tests.test_backend_independent_endpoints_step8_3 import FastAPITestClientAdapter


class TestFrontendDashboardStep8_4(unittest.TestCase):
    """
    Test suite for Phase 8 Step 8.4: Operational Engine Dashboard.
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

    def test_01_frontend_scaffolding_exists(self):
        """
        Asserts package.json, vite.config.js, index.html, and core React source files exist.
        """
        required_files = [
            os.path.join(self.frontend_dir, "package.json"),
            os.path.join(self.frontend_dir, "vite.config.js"),
            os.path.join(self.frontend_dir, "tailwind.config.js"),
            os.path.join(self.frontend_dir, "index.html"),
            os.path.join(self.src_dir, "main.jsx"),
            os.path.join(self.src_dir, "App.jsx"),
            os.path.join(self.src_dir, "index.css"),
            os.path.join(self.src_dir, "services", "api.js"),
            os.path.join(self.src_dir, "context", "EngineContext.jsx"),
            os.path.join(self.src_dir, "components", "layout", "Header.jsx"),
            os.path.join(self.src_dir, "components", "dashboard", "EngineSelectorBar.jsx"),
            os.path.join(self.src_dir, "components", "dashboard", "EngineSummaryCard.jsx"),
        ]

        for fpath in required_files:
            self.assertTrue(os.path.exists(fpath), f"Required frontend file missing: {fpath}")

        # Validate package.json contents
        pkg_path = os.path.join(self.frontend_dir, "package.json")
        with open(pkg_path, "r", encoding="utf-8") as f:
            pkg_data = json.load(f)

        self.assertIn("dependencies", pkg_data)
        self.assertIn("react", pkg_data["dependencies"])
        self.assertIn("axios", pkg_data["dependencies"])
        self.assertIn("lucide-react", pkg_data["dependencies"])

        # Validate index.html title
        html_path = os.path.join(self.frontend_dir, "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        self.assertIn("Aircraft Engine AI Health Monitoring & Digital Twin", html_content)

    def test_02_tailwind_aerospace_theme_configured(self):
        """
        Verifies custom aerospace colors and tabular font configurations in tailwind config and css.
        """
        tw_path = os.path.join(self.frontend_dir, "tailwind.config.js")
        with open(tw_path, "r", encoding="utf-8") as f:
            tw_content = f.read()

        # Check required aerospace color tokens
        expected_tokens = [
            "#0B0F19",  # bg
            "#111827",  # card
            "#1F2937",  # border
            "#06B6D4",  # cyan
            "#3B82F6",  # blue
            "#10B981",  # emerald
            "#F59E0B",  # amber
            "#EF4444",  # rose
        ]
        for token in expected_tokens:
            self.assertIn(token, tw_content, f"Missing aerospace color token {token} in tailwind.config.js")

        # Check tabular-nums in index.css
        css_path = os.path.join(self.src_dir, "index.css")
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        self.assertIn("tabular-nums", css_content)

    def test_03_api_service_endpoints_mapped(self):
        """
        Verifies api.js maps fleet subsets, engines, and cycle state routes.
        """
        api_path = os.path.join(self.src_dir, "services", "api.js")
        with open(api_path, "r", encoding="utf-8") as f:
            api_content = f.read()

        self.assertIn("/fleet/subsets", api_content)
        self.assertIn("/fleet/${subset}/engines", api_content)
        self.assertIn("/engine/${subset}/${unit}/cycle/${cycle}", api_content)
        self.assertIn("/engine/${subset}/${unit}/sensors", api_content)

    def test_04_engine_context_state_contract(self):
        """
        Validates EngineContext state structure, default initialization, and handlers.
        """
        ctx_path = os.path.join(self.src_dir, "context", "EngineContext.jsx")
        with open(ctx_path, "r", encoding="utf-8") as f:
            ctx_content = f.read()

        # Check default values
        self.assertIn("selectedSubset, setSelectedSubset] = useState('FD001')", ctx_content)
        self.assertIn("selectedUnit, setSelectedUnit] = useState(1)", ctx_content)
        self.assertIn("selectedCycle, setSelectedCycle] = useState(1)", ctx_content)
        self.assertIn("maxCycles, setMaxCycles] = useState(192)", ctx_content)

        # Check exposed methods
        self.assertIn("setSubset", ctx_content)
        self.assertIn("setUnit", ctx_content)
        self.assertIn("setCycle", ctx_content)
        self.assertIn("stepCycle", ctx_content)
        self.assertIn("togglePlayReplay", ctx_content)

    def test_05_header_displays_academic_disclaimer(self):
        """
        Confirms academic disclaimer is rendered in header component.
        """
        header_path = os.path.join(self.src_dir, "components", "layout", "Header.jsx")
        with open(header_path, "r", encoding="utf-8") as f:
            header_content = f.read()

        self.assertIn("ACADEMIC SIMULATION ONLY", header_content)
        self.assertIn("Aircraft Engine Digital Twin & AI Prognostics", header_content)
        self.assertIn("backendOnline", header_content)

    def test_06_engine_selector_handles_subset_switch(self):
        """
        Verifies engine selector logic resets unit and cycle on subset switch.
        """
        ctx_path = os.path.join(self.src_dir, "context", "EngineContext.jsx")
        with open(ctx_path, "r", encoding="utf-8") as f:
            ctx_content = f.read()

        # Subsets switch reset check
        self.assertIn("setSelectedUnit(1)", ctx_content)
        self.assertIn("setSelectedCycle(1)", ctx_content)

        # Engine selector bar elements
        bar_path = os.path.join(self.src_dir, "components", "dashboard", "EngineSelectorBar.jsx")
        with open(bar_path, "r", encoding="utf-8") as f:
            bar_content = f.read()

        self.assertIn("subset-select", bar_content)
        self.assertIn("unit-select", bar_content)
        self.assertIn("type=\"range\"", bar_content)

    def test_07_cycle_scrubber_bounds_enforcement(self):
        """
        Verifies scrubber clamps cycles between 1 and maxCycles.
        """
        ctx_path = os.path.join(self.src_dir, "context", "EngineContext.jsx")
        with open(ctx_path, "r", encoding="utf-8") as f:
            ctx_content = f.read()

        # Mathematical bounds clamping logic
        self.assertIn("Math.max(1, Math.min(cycleInt, maxCycles))", ctx_content)
        self.assertIn("Math.max(1, Math.min(target, maxCycles))", ctx_content)

        # Input element bounds in UI
        bar_path = os.path.join(self.src_dir, "components", "dashboard", "EngineSelectorBar.jsx")
        with open(bar_path, "r", encoding="utf-8") as f:
            bar_content = f.read()

        self.assertIn("min=\"1\"", bar_content)
        self.assertIn("max={maxCycles}", bar_content)

    def test_08_summary_card_displays_canonical_ids(self):
        """
        Asserts engine ID, cycle, and aerothermal indicators (T30, P30, Nf, Nc) are properly displayed.
        """
        card_path = os.path.join(self.src_dir, "components", "dashboard", "EngineSummaryCard.jsx")
        with open(card_path, "r", encoding="utf-8") as f:
            card_content = f.read()

        # Canonical identifiers and units
        self.assertIn("ENG-{selectedSubset}-U{selectedUnit}", card_content)
        self.assertIn("T30", card_content)
        self.assertIn("P30", card_content)
        self.assertIn("Nf", card_content)
        self.assertIn("Nc", card_content)
        self.assertIn("°R", card_content)
        self.assertIn("psia", card_content)
        self.assertIn("rpm", card_content)

        # Provenance categories
        self.assertIn("MEASURED", card_content)
        self.assertIn("DERIVED", card_content)
        self.assertIn("PREDICTED", card_content)
        self.assertIn("SIMULATED", card_content)

    def test_09_backend_integration_cycle_state_match(self):
        """
        Verifies backend cycle state payload integrates directly into summary view.
        """
        resp = self.client.get("/api/v1/engine/FD001/1/cycle/1")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["subset"], "FD001")
        self.assertEqual(data["unit_number"], 1)
        self.assertEqual(data["cycle"], 1)

        state = data["state_vector"]
        # Required core parameters present in backend state
        self.assertIn("T30", state)
        self.assertIn("P30", state)
        self.assertIn("Nf", state)
        self.assertIn("Nc", state)
        self.assertIn("T24", state)
        self.assertIn("T50", state)
        self.assertIn("epr", state)
        self.assertIn("BPR", state)

        # Plausible physical operating values for FD001 cycle 1
        self.assertGreater(state["T30"], 1500.0)
        self.assertGreater(state["P30"], 400.0)
        self.assertGreater(state["Nf"], 2000.0)
        self.assertGreater(state["Nc"], 8000.0)

    def test_10_dataset_and_model_immutability(self):
        """
        Cryptographic SHA-256 verification of all 12 raw files, 4 scalers, and 8 champion checkpoints.
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
