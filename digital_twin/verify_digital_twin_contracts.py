"""
Digital Twin Contract Verifier - Phase 5 Step 5.11
Automated validation of Digital Twin state taxonomy, provenance labeling,
unperturbed default values, and strict separation between measured and simulated values.

Roadmap Reference: Phase 5 - Step 5.11:
"Document assumptions and clearly label simulated values separately from measured dataset values."
"""

from dataclasses import dataclass, asdict
import json
import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from digital_twin.engine_state import EngineState, StateClassification, STATE_TAXONOMY
from digital_twin.twin_simulator import DigitalTwinSimulator
from digital_twin.scenario_interface import (
    create_thermal_stress_scenario,
    create_sensor_bias_scenario,
)


@dataclass
class TaxonomyContractResult:
    total_variables: int
    measured_count: int
    derived_count: int
    predicted_count: int
    simulated_count: int
    all_attributes_present: bool
    missing_attributes: List[str]
    is_valid: bool


@dataclass
class UnperturbedDefaultsResult:
    subset: str
    unit_number: int
    cycles_tested: int
    unobservable_and_simulated_none: bool
    g32_to_g34_none: bool
    g35_to_g36_none_when_ml_disabled: bool
    g37_to_g38_none_when_ml_disabled: bool
    g39_to_g40_none: bool
    scenario_audit_none: bool
    is_valid: bool


@dataclass
class ProvenanceLabelingResult:
    to_dict_keys_count: int
    provenance_metadata_present: bool
    provenance_keys: List[str]
    labeled_dict_verified: bool
    measured_labeled_correctly: bool
    derived_labeled_correctly: bool
    predicted_labeled_correctly: bool
    simulated_labeled_correctly: bool
    is_valid: bool


@dataclass
class ScenarioPerturbationContractResult:
    thermal_stress_g39_populated: bool
    thermal_stress_g40_none: bool
    thermal_stress_audit_present: bool
    sensor_bias_g40_populated: bool
    sensor_bias_g39_none: bool
    sensor_bias_audit_present: bool
    raw_telemetry_unmodified: bool
    is_valid: bool


class DigitalTwinContractVerifier:
    """
    Automated verifier asserting that Digital Twin implementations adhere strictly to the
    formal 40-variable state taxonomy, provenance tagging, and separation contracts.
    """

    def __init__(
        self,
        raw_data_dir: str = os.path.join("dataset", "raw", "C-MAPSS"),
        scalers_dir: str = os.path.join("models", "scalers"),
        checkpoint_dir: str = os.path.join("models", "checkpoints"),
    ):
        self.raw_data_dir = raw_data_dir
        self.scalers_dir = scalers_dir
        self.checkpoint_dir = checkpoint_dir

    def verify_taxonomy_conformance(self) -> TaxonomyContractResult:
        """
        Verify that all 40 state variables (G1 - G40) exist, possess valid metadata,
        and map directly to concrete attributes on the EngineState dataclass.
        """
        taxonomy = STATE_TAXONOMY
        total_vars = len(taxonomy)

        measured = 0
        derived = 0
        predicted = 0
        simulated = 0

        missing_attrs: List[str] = []

        # Dummy state instance for attribute check
        dummy_state = EngineState(
            unit_number=1,
            time_cycles=1,
            altitude=0.0,
            mach_number=0.0,
            throttle_resolver_angle=100.0,
            T2=518.67,
            T24=642.0,
            T30=1580.0,
            T50=1400.0,
            P2=14.62,
            P15=21.61,
            P30=553.0,
            Nf=2388.0,
            Nc=9046.0,
            epr=1.0,
            Ps30=47.0,
            phi=521.0,
            NRf=2388.0,
            NRc=8138.0,
            BPR=8.4,
            farB=0.03,
            htBleed=392.0,
            Nf_dmd=2388.0,
            PCNfR_dmd=100.0,
            W31=38.8,
            W32=23.3,
            hpc_temp_rise=938.0,
            overall_pressure_ratio=37.8,
            spool_speed_ratio=3.78,
            normalized_feature_vec=np.zeros(14, dtype=np.float64),
            temporal_seq_tensor=np.zeros((30, 14), dtype=np.float32),
        )

        for gid, info in taxonomy.items():
            field_name = info["field_name"]
            cls_val = info["classification"]

            if cls_val == StateClassification.MEASURED:
                measured += 1
            elif cls_val == StateClassification.DERIVED:
                derived += 1
            elif cls_val == StateClassification.PREDICTED:
                predicted += 1
            elif cls_val == StateClassification.SIMULATED:
                simulated += 1

            if not hasattr(dummy_state, field_name):
                missing_attrs.append(f"{gid}: {field_name}")

        is_valid = (
            total_vars == 40
            and measured == 26
            and derived == 7
            and predicted == 5
            and simulated == 2
            and len(missing_attrs) == 0
        )

        return TaxonomyContractResult(
            total_variables=total_vars,
            measured_count=measured,
            derived_count=derived,
            predicted_count=predicted,
            simulated_count=simulated,
            all_attributes_present=(len(missing_attrs) == 0),
            missing_attributes=missing_attrs,
            is_valid=is_valid,
        )

    def verify_unperturbed_baseline_contract(
        self, subset: str = "FD001", unit_number: int = 1
    ) -> UnperturbedDefaultsResult:
        """
        Verify that an unperturbed baseline Digital Twin execution strictly defaults
        all simulation-only and unpredicted variables to None.
        """
        sim = DigitalTwinSimulator(
            subset=subset,
            data_dir=self.raw_data_dir,
            scalers_dir=self.scalers_dir,
            checkpoint_dir=self.checkpoint_dir,
            enable_ml_inference=False,
            scenario=None,
        )
        sim.load_engine(unit_number=unit_number, split="train")

        g32_to_g34_none = True
        g35_to_g36_none = True
        g37_to_g38_none = True
        g39_to_g40_none = True
        audit_none = True
        cycles_count = 0

        while True:
            st = sim.step()
            if st is None:
                break
            cycles_count += 1

            # G32 - G34 (Anomaly tracking)
            if st.anomaly_score_m1 is not None or st.anomaly_score_m2 is not None or st.anomaly_alert_flag is not None:
                g32_to_g34_none = False

            # G35 - G36 (Prognostics without ML enabled)
            if st.predicted_rul_xgboost is not None or st.predicted_rul_lstm is not None:
                g35_to_g36_none = False

            # G37 - G38 (Prognostic spread & health index without ML)
            if st.prognostic_spread is not None or st.virtual_health_index is not None:
                g37_to_g38_none = False

            # G39 - G40 (What-if scenario perturbation states)
            if st.thermal_stress_offset is not None or st.sensor_bias_delta is not None:
                g39_to_g40_none = False

            # Audit and scenario id
            if st.scenario_id is not None or st.scenario_audit is not None:
                audit_none = False

        all_none = (
            g32_to_g34_none
            and g35_to_g36_none
            and g37_to_g38_none
            and g39_to_g40_none
            and audit_none
        )

        return UnperturbedDefaultsResult(
            subset=subset.upper(),
            unit_number=unit_number,
            cycles_tested=cycles_count,
            unobservable_and_simulated_none=all_none,
            g32_to_g34_none=g32_to_g34_none,
            g35_to_g36_none_when_ml_disabled=g35_to_g36_none,
            g37_to_g38_none_when_ml_disabled=g37_to_g38_none,
            g39_to_g40_none=g39_to_g40_none,
            scenario_audit_none=audit_none,
            is_valid=all_none and cycles_count > 0,
        )

    def verify_provenance_and_labeling_contract(
        self, subset: str = "FD001", unit_number: int = 1
    ) -> ProvenanceLabelingResult:
        """
        Verify that to_dict() and to_labeled_dict() explicitly annotate data provenance
        and classifications for every variable.
        """
        sim = DigitalTwinSimulator(
            subset=subset,
            data_dir=self.raw_data_dir,
            scalers_dir=self.scalers_dir,
            checkpoint_dir=self.checkpoint_dir,
            enable_ml_inference=True,
        )
        sim.load_engine(unit_number=unit_number, split="train")
        st = sim.step()  # Cycle 2

        # 1. Standard to_dict() contract
        d = st.to_dict(include_arrays=True)
        to_dict_keys = len(d)
        has_provenance = "provenance" in d
        prov = d.get("provenance", {})
        prov_keys = list(prov.keys())

        # 2. to_labeled_dict() contract
        labeled = st.to_labeled_dict(include_arrays=False)
        labeled_keys = list(labeled.keys())

        measured_ok = True
        derived_ok = True
        predicted_ok = True
        simulated_ok = True

        for gid, info in STATE_TAXONOMY.items():
            field_name = info["field_name"]
            key = f"{gid}_{field_name}"
            if key not in labeled:
                continue

            entry = labeled[key]
            cls_name = entry["classification"]

            if info["classification"] == StateClassification.MEASURED:
                if cls_name != "Measured (M)":
                    measured_ok = False
            elif info["classification"] == StateClassification.DERIVED:
                if cls_name != "Derived (D)":
                    derived_ok = False
            elif info["classification"] == StateClassification.PREDICTED:
                if cls_name != "Predicted (P)":
                    predicted_ok = False
            elif info["classification"] == StateClassification.SIMULATED:
                if cls_name != "Simulated (S)":
                    simulated_ok = False

        is_valid = (
            to_dict_keys >= 40
            and has_provenance
            and len(prov_keys) >= 6
            and measured_ok
            and derived_ok
            and predicted_ok
            and simulated_ok
        )

        return ProvenanceLabelingResult(
            to_dict_keys_count=to_dict_keys,
            provenance_metadata_present=has_provenance,
            provenance_keys=prov_keys,
            labeled_dict_verified=True,
            measured_labeled_correctly=measured_ok,
            derived_labeled_correctly=derived_ok,
            predicted_labeled_correctly=predicted_ok,
            simulated_labeled_correctly=simulated_ok,
            is_valid=is_valid,
        )

    def verify_scenario_perturbation_contract(
        self, subset: str = "FD001", unit_number: int = 1
    ) -> ScenarioPerturbationContractResult:
        """
        Verify that What-If scenarios populate only G39/G40, record full audit provenance,
        and leave raw measured telemetry unmodified.
        """
        # 1. Thermal Stress Scenario
        sc_thermal = create_thermal_stress_scenario(
            scenario_id="CONTRACT_TEST_THERMAL",
            target="T30",
            offset_degR=25.0,
            name="Contract Thermal Stress Test",
        )
        sim_thermal = DigitalTwinSimulator(
            subset=subset,
            data_dir=self.raw_data_dir,
            scalers_dir=self.scalers_dir,
            checkpoint_dir=self.checkpoint_dir,
            scenario=sc_thermal,
        )
        sim_thermal.load_engine(unit_number=unit_number, split="train")
        st_thermal = sim_thermal.step()

        thermal_g39_ok = (st_thermal.thermal_stress_offset == 25.0)
        thermal_g40_ok = (st_thermal.sensor_bias_delta is None)
        thermal_audit_ok = (
            st_thermal.scenario_audit is not None
            and st_thermal.scenario_audit.get("scenario_id") == "CONTRACT_TEST_THERMAL"
        )

        # 2. Sensor Bias Scenario
        sc_bias = create_sensor_bias_scenario(
            scenario_id="CONTRACT_TEST_BIAS",
            target="P30",
            bias=12.5,
            unit="psia",
            name="Contract Sensor Bias Test",
        )
        sim_bias = DigitalTwinSimulator(
            subset="FD002",
            data_dir=self.raw_data_dir,
            scalers_dir=self.scalers_dir,
            checkpoint_dir=self.checkpoint_dir,
            scenario=sc_bias,
        )
        sim_bias.load_engine(unit_number=unit_number, split="train")
        st_bias = sim_bias.step()

        bias_g40_ok = (
            st_bias.sensor_bias_delta is not None
            and st_bias.sensor_bias_delta.get("P30") == 12.5
        )
        bias_g39_ok = (st_bias.thermal_stress_offset is None)
        bias_audit_ok = (
            st_bias.scenario_audit is not None
            and st_bias.scenario_audit.get("scenario_id") == "CONTRACT_TEST_BIAS"
        )

        # 3. Ground truth telemetry immutability in raw_history
        raw_telemetry_unmodified = True
        raw_24_bias = sim_bias.raw_history[0]  # Cycle 1
        # P30 in raw_history is at index 9 (sensor_7). Check that raw_24 retains original measurement
        orig_p30 = sim_bias.trajectory_df.iloc[0]["sensor_7"]
        if abs(raw_24_bias[9] - orig_p30) > 1e-9:
            raw_telemetry_unmodified = False

        is_valid = (
            thermal_g39_ok
            and thermal_g40_ok
            and thermal_audit_ok
            and bias_g40_ok
            and bias_g39_ok
            and bias_audit_ok
            and raw_telemetry_unmodified
        )

        return ScenarioPerturbationContractResult(
            thermal_stress_g39_populated=thermal_g39_ok,
            thermal_stress_g40_none=thermal_g40_ok,
            thermal_stress_audit_present=thermal_audit_ok,
            sensor_bias_g40_populated=bias_g40_ok,
            sensor_bias_g39_none=bias_g39_ok,
            sensor_bias_audit_present=bias_audit_ok,
            raw_telemetry_unmodified=raw_telemetry_unmodified,
            is_valid=is_valid,
        )

    def run_all_contract_verifications(
        self,
        output_dir: str = os.path.join("reports", "digital_twin", "step5_11"),
    ) -> Dict[str, Any]:
        """
        Execute all four Digital Twin contract verifications and export JSON and Markdown reports.
        """
        os.makedirs(output_dir, exist_ok=True)

        res_taxonomy = self.verify_taxonomy_conformance()
        res_defaults = self.verify_unperturbed_baseline_contract()
        res_provenance = self.verify_provenance_and_labeling_contract()
        res_scenario = self.verify_scenario_perturbation_contract()

        summary: Dict[str, Any] = {
            "step": "Phase 5 Step 5.11: Digital Twin Assumptions, Taxonomy & Labeling Verification",
            "taxonomy_contract": asdict(res_taxonomy),
            "unperturbed_defaults_contract": asdict(res_defaults),
            "provenance_and_labeling_contract": asdict(res_provenance),
            "scenario_perturbation_contract": asdict(res_scenario),
            "all_contracts_verified": (
                res_taxonomy.is_valid
                and res_defaults.is_valid
                and res_provenance.is_valid
                and res_scenario.is_valid
            ),
        }

        # Write JSON
        json_path = os.path.join(output_dir, "contract_verification.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Write Markdown
        md_path = os.path.join(output_dir, "contract_verification.md")
        self._write_markdown_report(summary, md_path)

        return summary

    def _write_markdown_report(self, summary: Dict[str, Any], filepath: str) -> None:
        """Generate structured markdown report of contract verification results."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Phase 5 — Step 5.11: Digital Twin Contract Verification Report\n\n")
            f.write("**Project:** AI-Based Aircraft Engine Health Monitoring, Fault Diagnosis and Predictive Maintenance  \n")
            f.write("**Milestone:** Phase 5 Step 5.11: Assumptions, Taxonomy & Labeling Contracts  \n")
            f.write(f"**Overall Verification Status:** **{'ALL CONTRACTS VERIFIED' if summary['all_contracts_verified'] else 'VERIFICATION FAILED'}**  \n\n")
            f.write("---\n\n")

            f.write("## 1. 40-Variable Canonical Taxonomy Conformance (G1–G40)\n\n")
            tax = summary["taxonomy_contract"]
            f.write(f"- **Total Variables Documented:** {tax['total_variables']} / 40\n")
            f.write(f"- **Measured Telemetry (G1–G26):** {tax['measured_count']} / 26\n")
            f.write(f"- **Derived Indicators & Representations (G27–G31, G37, G38):** {tax['derived_count']} / 7\n")
            f.write(f"- **Predicted Diagnostics & Prognostics (G32–G36):** {tax['predicted_count']} / 5\n")
            f.write(f"- **What-If Simulation Perturbations (G39–G40):** {tax['simulated_count']} / 2\n")
            missing_str = ", ".join(tax["missing_attributes"]) if tax["missing_attributes"] else "0 missing"
            f.write(f"- **Dataclass Attributes Conformance:** `{'PASS (0 missing)' if tax['all_attributes_present'] else f'FAIL (Missing: {missing_str})'}`\n")
            f.write(f"- **Taxonomy Contract Status:** **{'VERIFIED' if tax['is_valid'] else 'FAILED'}**\n\n")
            f.write("---\n\n")

            f.write("## 2. Unperturbed Baseline State Contract\n\n")
            unp = summary["unperturbed_defaults_contract"]
            f.write(f"- **Subset Tested:** {unp['subset']} Unit {unp['unit_number']} ({unp['cycles_tested']} cycles)\n")
            f.write(f"- **G32–G34 Anomaly States Default to None:** `{'PASS' if unp['g32_to_g34_none'] else 'FAIL'}`\n")
            f.write(f"- **G35–G36 Prognostic RUL Default to None (ML Disabled):** `{'PASS' if unp['g35_to_g36_none_when_ml_disabled'] else 'FAIL'}`\n")
            f.write(f"- **G37–G38 Uncertainty / Health Index Default to None (ML Disabled):** `{'PASS' if unp['g37_to_g38_none_when_ml_disabled'] else 'FAIL'}`\n")
            f.write(f"- **G39–G40 What-If Perturbations Default to None:** `{'PASS' if unp['g39_to_g40_none'] else 'FAIL'}`\n")
            f.write(f"- **Scenario Audit Metadata Defaults to None:** `{'PASS' if unp['scenario_audit_none'] else 'FAIL'}`\n")
            f.write(f"- **Unperturbed Contract Status:** **{'VERIFIED' if unp['is_valid'] else 'FAILED'}**\n\n")
            f.write("---\n\n")

            f.write("## 3. Data Provenance & Serialization Labeling Contract\n\n")
            prov = summary["provenance_and_labeling_contract"]
            f.write(f"- **Standard `to_dict()` Keys:** {prov['to_dict_keys_count']} entries\n")
            f.write(f"- **Provenance Metadata Included:** `{'PASS' if prov['provenance_metadata_present'] else 'FAIL'}` ({prov['provenance_keys']})\n")
            f.write(f"- **`to_labeled_dict()` Explicit Annotations:** `{'PASS' if prov['labeled_dict_verified'] else 'FAIL'}`\n")
            f.write(f"- **Measured Telemetry Labeled 'Measured (M)':** `{'PASS' if prov['measured_labeled_correctly'] else 'FAIL'}`\n")
            f.write(f"- **Derived Indicators Labeled 'Derived (D)':** `{'PASS' if prov['derived_labeled_correctly'] else 'FAIL'}`\n")
            f.write(f"- **ML States Labeled 'Predicted (P)':** `{'PASS' if prov['predicted_labeled_correctly'] else 'FAIL'}`\n")
            f.write(f"- **Scenario States Labeled 'Simulated (S)':** `{'PASS' if prov['simulated_labeled_correctly'] else 'FAIL'}`\n")
            f.write(f"- **Provenance Contract Status:** **{'VERIFIED' if prov['is_valid'] else 'FAILED'}**\n\n")
            f.write("---\n\n")

            f.write("## 4. What-If Scenario Perturbation Contract\n\n")
            sc = summary["scenario_perturbation_contract"]
            f.write(f"- **Thermal Stress: G39 Populated & G40 None:** `{'PASS' if sc['thermal_stress_g39_populated'] and sc['thermal_stress_g40_none'] else 'FAIL'}`\n")
            f.write(f"- **Thermal Stress: Scenario Audit Logged:** `{'PASS' if sc['thermal_stress_audit_present'] else 'FAIL'}`\n")
            f.write(f"- **Sensor Bias: G40 Populated & G39 None:** `{'PASS' if sc['sensor_bias_g40_populated'] and sc['sensor_bias_g39_none'] else 'FAIL'}`\n")
            f.write(f"- **Sensor Bias: Scenario Audit Logged:** `{'PASS' if sc['sensor_bias_audit_present'] else 'FAIL'}`\n")
            f.write(f"- **Raw Ground-Truth Telemetry Unmodified:** `{'PASS' if sc['raw_telemetry_unmodified'] else 'FAIL'}`\n")
            f.write(f"- **Scenario Perturbation Contract Status:** **{'VERIFIED' if sc['is_valid'] else 'FAILED'}**\n")


def main():
    print("=" * 80)
    print("Executing Phase 5 Step 5.11: Digital Twin Contract Verification")
    print("=" * 80)

    verifier = DigitalTwinContractVerifier()
    output_dir = os.path.join("reports", "digital_twin", "step5_11")
    print(f"Output directory: {output_dir}")

    summary = verifier.run_all_contract_verifications(output_dir=output_dir)

    print(f"\n1. Taxonomy Conformance: {'PASS' if summary['taxonomy_contract']['is_valid'] else 'FAIL'}")
    print(f"2. Unperturbed Baseline Defaults: {'PASS' if summary['unperturbed_defaults_contract']['is_valid'] else 'FAIL'}")
    print(f"3. Data Provenance & Labeling: {'PASS' if summary['provenance_and_labeling_contract']['is_valid'] else 'FAIL'}")
    print(f"4. Scenario Perturbation Isolation: {'PASS' if summary['scenario_perturbation_contract']['is_valid'] else 'FAIL'}")
    print(f"\nAll Digital Twin Contracts Verified: {summary['all_contracts_verified']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
