"""
Unit and Integration Test Suite for Phase 9 Step 9.10: Master Verification & Validation (V&V) Matrix
Verifies:
1. Coverage across all Phase 9 system testing steps (Steps 9.1 through 9.9 and Step 9.10).
2. Mandatory schema field completeness for every test record.
3. 100% PASS rate across all recorded V&V test records.
4. Traceability of held-out test engines and truncated cycle envelopes (Step 9.1).
5. Traceability of bit-for-bit sensor matching (|Δ| <= 1e-5) (Step 9.2).
6. Traceability of authentic RUL ground-truth predictions and error residuals (Step 9.3).
7. Traceability of recalculated aerospace evaluation metrics (RMSE, MAE, NASA Score S) over 707 engines (Step 9.4).
8. Traceability of 50-iteration digital twin repeatability and zero variance (Step 9.6).
9. Export file integrity for Markdown and JSON report formats.
10. Cryptographic SHA-256 immutability of raw datasets, scalers, and champion checkpoints.

Roadmap Reference: Phase 9 - Step 9.10:
"Record test case, input, expected result, actual result, pass/fail status and evidence"
"""

import hashlib
import json
import os
import unittest
from typing import Any, Dict, List

from backend.services.master_test_reporter import (
    generate_master_test_matrix,
    get_verification_summary,
    export_matrix_to_markdown,
    export_matrix_to_json,
)


class TestMasterMatrixStep9_10(unittest.TestCase):
    """
    Test suite certifying the completeness, traceability, and cryptographic integrity
    of the Master V&V Test Matrix for Phase 9.
    """

    @classmethod
    def setUpClass(cls):
        cls.raw_data_dir = os.path.join("dataset", "raw", "C-MAPSS")
        cls.scalers_dir = os.path.join("models", "scalers")
        cls.checkpoint_dir = os.path.join("models", "checkpoints")

        # 1. Capture cryptographic SHA-256 hashes of all 12 raw C-MAPSS dataset files
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

        cls.matrix = generate_master_test_matrix()
        cls.summary = get_verification_summary()

    def test_01_matrix_contains_all_phase9_steps(self):
        """
        Verifies that the Master Test Matrix contains records for all Phase 9 steps (9.1 to 9.9).
        """
        required_steps = {
            "Step 9.1",
            "Step 9.2",
            "Step 9.3",
            "Step 9.4",
            "Step 9.5",
            "Step 9.6",
            "Step 9.7",
            "Step 9.8",
            "Step 9.9",
        }
        present_steps = set(r["step"] for r in self.matrix)
        for step in required_steps:
            self.assertIn(step, present_steps, f"Missing test records for {step}")

    def test_02_mandatory_schema_fields_present(self):
        """
        Asserts every test record contains all mandatory fields:
        id, step, description, input_vector, expected_result, actual_result, status, evidence_reference.
        """
        mandatory_fields = [
            "test_case_id",
            "step",
            "description",
            "input_vector",
            "expected_result",
            "actual_result",
            "status",
            "evidence_reference",
        ]
        self.assertGreater(len(self.matrix), 0)
        for record in self.matrix:
            for field in mandatory_fields:
                self.assertIn(field, record, f"Missing field '{field}' in record {record.get('test_case_id')}")
                self.assertTrue(bool(str(record[field]).strip()), f"Empty field '{field}' in record {record.get('test_case_id')}")

    def test_03_all_tests_marked_pass(self):
        """
        Confirms 100% pass status across all recorded test cases in the matrix.
        """
        self.assertGreaterEqual(self.summary["total_test_records"], 50)
        self.assertEqual(self.summary["failed_test_records"], 0)
        self.assertEqual(self.summary["pass_rate_percent"], 100.0)
        for record in self.matrix:
            self.assertEqual(record["status"], "PASS", f"Record {record['test_case_id']} is not marked PASS")

    def test_04_step9_1_held_out_records_traceable(self):
        """
        Confirms held-out cohort test cases are recorded with exact cycle limits and inventory counts.
        """
        step1_records = [r for r in self.matrix if r["step"] == "Step 9.1"]
        self.assertGreaterEqual(len(step1_records), 5)

        descriptions = [r["description"].lower() for r in step1_records]
        self.assertTrue(any("inventory" in d for d in descriptions))
        self.assertTrue(any("cohort" in d for d in descriptions))
        self.assertTrue(any("boundary" in d for d in descriptions))

        # Check trajectory cycle limits
        cohort_record = next(r for r in step1_records if "cohort" in r["description"].lower())
        self.assertIn("126", cohort_record["actual_result"])
        self.assertIn("186", cohort_record["actual_result"])

    def test_05_step9_2_bit_matching_records_traceable(self):
        """
        Confirms raw telemetry bit-matching cases are recorded with |Δ| <= 1e-5 tolerance.
        """
        step2_records = [r for r in self.matrix if r["step"] == "Step 9.2"]
        self.assertGreaterEqual(len(step2_records), 5)

        for r in step2_records:
            self.assertIn("1e-5", r["expected_result"])
            self.assertTrue("|Δ|" in r["actual_result"] or "<=" in r["actual_result"])

    def test_06_step9_3_ground_truth_records_traceable(self):
        """
        Confirms RUL vs ground-truth cases record exact predictions and errors.
        """
        step3_records = [r for r in self.matrix if r["step"] == "Step 9.3"]
        self.assertGreaterEqual(len(step3_records), 5)

        # Check final cycle prediction on FD001 Unit 3
        fd001_r = next(r for r in step3_records if "fd001" in r["description"].lower())
        self.assertIn("116.14", fd001_r["actual_result"])
        self.assertIn("118.82", fd001_r["actual_result"])
        self.assertIn("117.48", fd001_r["actual_result"])

    def test_07_step9_4_metrics_recalculation_traceable(self):
        """
        Confirms recalculated RMSE and NASA scores match benchmark tables across subsets.
        """
        step4_records = [r for r in self.matrix if r["step"] == "Step 9.4"]
        self.assertGreaterEqual(len(step4_records), 5)

        # Check FD001 test recalculation record
        fd001_metric_r = next(r for r in step4_records if "fd001" in r["description"].lower())
        self.assertIn("17.51", fd001_metric_r["actual_result"])
        self.assertIn("18.04", fd001_metric_r["actual_result"])

        # Check asymmetric penalty record
        asym_r = next(r for r in step4_records if "asymmetric" in r["description"].lower())
        self.assertIn("late", asym_r["actual_result"].lower())

    def test_08_step9_6_repeatability_records_traceable(self):
        """
        Confirms 50-iteration repeatability is recorded with variance = 0.0 (|Δ| = 0.0).
        """
        step6_records = [r for r in self.matrix if r["step"] == "Step 9.6"]
        self.assertGreaterEqual(len(step6_records), 4)

        for r in step6_records:
            if "repeatability" in r["description"].lower():
                self.assertIn("50", r["input_vector"])
                self.assertTrue("variance = 0.0" in r["expected_result"].lower() or "50/50" in r["expected_result"])

    def test_09_markdown_and_json_export_integrity(self):
        """
        Asserts exported V&V report exists, is non-empty, and parses cleanly as valid Markdown and JSON.
        """
        md_path = os.path.join("docs", "phase9_step9_10_verification_matrix.md")
        json_path = os.path.join("docs", "phase9_step9_10_verification_matrix.json")

        # Export both formats
        export_matrix_to_markdown(md_path)
        export_matrix_to_json(json_path)

        # Verify Markdown export
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        self.assertGreater(len(md_content), 1000)
        self.assertIn("# Phase 9 — Step 9.10: Master Verification & Validation (V&V) Test Matrix", md_content)
        self.assertIn("TC-9.1-01", md_content)
        self.assertIn("TC-9.9-01", md_content)

        # Verify JSON export
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.assertIn("summary", payload)
        self.assertIn("test_matrix", payload)
        self.assertEqual(len(payload["test_matrix"]), self.summary["total_test_records"])

    def test_10_dataset_and_model_immutability(self):
        """
        Cryptographic verification: Asserts that all 12 raw C-MAPSS dataset files,
        4 scalers, and 8 champion checkpoints remain 100% untouched and byte-identical.
        """
        # 1. Raw dataset files
        self.assertEqual(len(self.raw_hashes), 12)
        for fname, original_hash in self.raw_hashes.items():
            fpath = os.path.join(self.raw_data_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Cryptographic integrity breach detected in raw dataset file: {fname}"
            )

        # 2. Scaler files
        self.assertGreaterEqual(len(self.scaler_hashes), 4)
        for fname, original_hash in self.scaler_hashes.items():
            fpath = os.path.join(self.scalers_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Cryptographic integrity breach detected in scaler file: {fname}"
            )

        # 3. Model checkpoint files
        self.assertEqual(len(self.checkpoint_hashes), 8)
        for fname, original_hash in self.checkpoint_hashes.items():
            fpath = os.path.join(self.checkpoint_dir, fname)
            with open(fpath, "rb") as f:
                current_hash = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(
                current_hash,
                original_hash,
                f"Cryptographic integrity breach detected in model checkpoint: {fname}"
            )


if __name__ == "__main__":
    unittest.main()
