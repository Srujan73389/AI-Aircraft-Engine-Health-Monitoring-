"""
Unit and Integration Tests for Phase 5 Step 5.5: Physics-Based Component Feasibility & Implementation
Verifies thermodynamic equations, aerothermal speed similitude, boundary safeguards, numerical validation, and C-MAPSS integration.
"""

import hashlib
import math
import os
import unittest
import numpy as np

from digital_twin.physics_engine import (
    STANDARD_GAMMA_AIR,
    STANDARD_CP_AIR,
    STANDARD_T_STD_RANKINE,
    PhysicsIndicators,
    compute_compression_isentropic_efficiency,
    compute_compression_polytropic_efficiency,
    compute_compression_specific_work,
    compute_spool_speed_ratios,
    compute_aerothermal_speed_corrections,
    evaluate_engine_physics,
)
from digital_twin.twin_simulator import DigitalTwinSimulator


class TestDigitalTwinStep5_5(unittest.TestCase):
    """
    Test suite for Step 5.5 physics-based calculations and numerical validation.
    """

    # -------------------------------------------------------------------------
    # Core Analytical & Physics Tests (Preserved from initial implementation)
    # -------------------------------------------------------------------------
    def test_01_analytical_isentropic_efficiency(self):
        """Verify isentropic efficiency formula against exact hand-calculated analytical solution."""
        p2 = 14.696
        p30 = 554.36
        t2 = 518.67
        t30 = 1589.70

        gamma = 1.400
        exponent = (gamma - 1.0) / gamma  # 2/7
        expected_opr = p30 / p2
        expected_t_ratio = t30 / t2
        expected_eta_is = (math.pow(expected_opr, exponent) - 1.0) / (expected_t_ratio - 1.0)

        calc_eta_is = compute_compression_isentropic_efficiency(p2, p30, t2, t30, gamma=gamma)
        self.assertAlmostEqual(calc_eta_is, expected_eta_is, places=7)
        self.assertGreater(calc_eta_is, 0.85)
        self.assertLess(calc_eta_is, 0.92)

    def test_02_analytical_polytropic_efficiency(self):
        """Verify polytropic efficiency formula against exact analytical solution."""
        p2 = 14.696
        p30 = 554.36
        t2 = 518.67
        t30 = 1589.70

        gamma = 1.400
        exponent = (gamma - 1.0) / gamma
        expected_numerator = exponent * math.log(p30 / p2)
        expected_denominator = math.log(t30 / t2)
        expected_eta_poly = expected_numerator / expected_denominator

        calc_eta_poly = compute_compression_polytropic_efficiency(p2, p30, t2, t30, gamma=gamma)
        self.assertAlmostEqual(calc_eta_poly, expected_eta_poly, places=7)
        # Polytropic efficiency must be strictly greater than isentropic efficiency
        calc_eta_is = compute_compression_isentropic_efficiency(p2, p30, t2, t30, gamma=gamma)
        self.assertGreater(calc_eta_poly, calc_eta_is)

    def test_03_analytical_specific_work(self):
        """Verify specific work enthalpy formula against hand calculation."""
        t2 = 518.67
        t30 = 1589.70
        cp = 0.240
        expected_w = cp * (t30 - t2)

        calc_w = compute_compression_specific_work(t2, t30, cp=cp)
        self.assertAlmostEqual(calc_w, expected_w, places=7)
        self.assertAlmostEqual(calc_w, 257.0472, places=4)

    def test_04_analytical_spool_speed_ratios(self):
        """Verify spool speed ratio formulas."""
        nc = 9050.0
        nf = 2388.0
        nrc = 8125.0
        nrf = 2388.0

        ratios = compute_spool_speed_ratios(nc, nf, nrc, nrf)
        self.assertAlmostEqual(ratios["speed_ratio_phys"], 9050.0 / 2388.0, places=6)
        self.assertAlmostEqual(ratios["speed_ratio_corr"], 8125.0 / 2388.0, places=6)

    def test_05_aerothermal_speed_corrections(self):
        """Verify aerothermal speed correction formulas and residual bounds."""
        nf = 2388.0
        t2 = 518.67  # Standard day sea level -> theta_2 = 1.0 -> NRf_theory = Nf = 2388.0
        nc = 9050.0
        t24 = 642.50
        nrf = 2388.05
        nrc = 8125.0

        corrections = compute_aerothermal_speed_corrections(nf, t2, nc, t24, nrf, nrc)
        self.assertAlmostEqual(corrections["fan_speed_theoretical_corr"], 2388.0, places=5)
        self.assertAlmostEqual(corrections["fan_speed_corr_residual"], 0.05, places=5)

        theta_24 = t24 / 518.67
        expected_nrc_theory = nc / math.sqrt(theta_24)
        self.assertAlmostEqual(corrections["core_speed_theoretical_corr"], expected_nrc_theory, places=5)

    def test_06_boundary_and_unphysical_inputs_raise_errors(self):
        """Verify that unphysical thermodynamic conditions raise clear ValueErrors."""
        # P30 <= P2 (Expansion instead of compression)
        with self.assertRaises(ValueError):
            compute_compression_isentropic_efficiency(p2=50.0, p30=20.0, t2=500.0, t30=1000.0)

        # T30 <= T2
        with self.assertRaises(ValueError):
            compute_compression_isentropic_efficiency(p2=14.0, p30=100.0, t2=600.0, t30=500.0)

        # Negative pressure
        with self.assertRaises(ValueError):
            compute_compression_isentropic_efficiency(p2=-14.0, p30=100.0, t2=500.0, t30=1000.0)

        # Negative temperature
        with self.assertRaises(ValueError):
            compute_compression_isentropic_efficiency(p2=14.0, p30=100.0, t2=-500.0, t30=1000.0)

        # Negative speeds
        with self.assertRaises(ValueError):
            compute_spool_speed_ratios(nc=-100.0, nf=2000.0, nrc=8000.0, nrf=2000.0)

    def test_07_evaluate_engine_physics_on_cmapss_trajectory(self):
        """Verify physics indicators on real FD001 Unit 1 trajectory."""
        sim = DigitalTwinSimulator(subset="FD001")
        state_c1 = sim.load_engine(unit_number=1, split="train")

        physics = evaluate_engine_physics(state_c1)
        self.assertIsInstance(physics, PhysicsIndicators)
        self.assertEqual(len(physics.warnings), 0)

        # Compression efficiency should be solidly in nominal high-bypass turbofan range [0.85, 0.92]
        self.assertGreaterEqual(physics.comp_isentropic_eff, 0.85)
        self.assertLessEqual(physics.comp_isentropic_eff, 0.92)

        # Polytropic efficiency should be in [0.90, 0.95]
        self.assertGreaterEqual(physics.comp_polytropic_eff, 0.90)
        self.assertLessEqual(physics.comp_polytropic_eff, 0.95)

        # Fan speed residual should be very small (within sensor noise < 0.25 rpm)
        self.assertLess(abs(physics.fan_speed_corr_residual), 0.25)

        # Check dictionary export
        p_dict = physics.to_dict()
        self.assertIn("comp_isentropic_eff", p_dict)
        self.assertIn("comp_polytropic_eff", p_dict)
        self.assertIn("delta_t_hpc", p_dict)

    def test_08_compression_efficiency_degradation_trend(self):
        """Verify that isentropic compression efficiency decreases as engine degrades toward failure."""
        sim = DigitalTwinSimulator(subset="FD001")
        s1 = sim.load_engine(unit_number=1, split="train")
        p1 = evaluate_engine_physics(s1)

        # Run to end-of-life (cycle 192)
        states = sim.run_all()
        s_end = states[-1]
        self.assertEqual(s_end.time_cycles, 192)
        p_end = evaluate_engine_physics(s_end)

        # Start of life efficiency should be strictly higher than end of life efficiency
        self.assertGreater(
            p1.comp_isentropic_eff,
            p_end.comp_isentropic_eff,
            f"Expected Start of Life eta ({p1.comp_isentropic_eff:.4f}) > End of Life eta ({p_end.comp_isentropic_eff:.4f})",
        )
        eff_loss = p1.comp_isentropic_eff - p_end.comp_isentropic_eff
        self.assertGreater(eff_loss, 0.005)
        self.assertLess(eff_loss, 0.030)

    def test_09_multiregime_physics_evaluation_fd002(self):
        """Verify physics indicators across complex multi-regime FD002 flights."""
        sim = DigitalTwinSimulator(subset="FD002")
        state_c1 = sim.load_engine(unit_number=1, split="train")
        p1 = evaluate_engine_physics(state_c1)

        self.assertFalse(math.isnan(p1.comp_isentropic_eff))
        self.assertGreater(p1.comp_isentropic_eff, 0.82)
        self.assertLess(p1.comp_isentropic_eff, 0.93)
        self.assertGreater(p1.comp_polytropic_eff, 0.88)
        self.assertLess(p1.comp_polytropic_eff, 0.96)

    def test_10_dataset_hash_immutability(self):
        """Verify that train_FD001.txt is unmodified."""
        raw_path = os.path.join("dataset", "raw", "C-MAPSS", "train_FD001.txt")
        sha256 = hashlib.sha256()
        with open(raw_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        expected_hash = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8"
        self.assertEqual(sha256.hexdigest().lower(), expected_hash)

    # -------------------------------------------------------------------------
    # Expanded Numerical Validation Tests (Addressing Step 5.5 Review)
    # -------------------------------------------------------------------------
    def test_11_nan_pressure_input_rejected(self):
        """1. Verify that NaN pressure input raises ValueError."""
        with self.assertRaises(ValueError):
            compute_compression_isentropic_efficiency(p2=float("nan"), p30=554.0, t2=518.0, t30=1500.0)
        with self.assertRaises(ValueError):
            compute_compression_isentropic_efficiency(p2=14.0, p30=float("nan"), t2=518.0, t30=1500.0)
        with self.assertRaises(ValueError):
            compute_compression_polytropic_efficiency(p2=float("nan"), p30=554.0, t2=518.0, t30=1500.0)

    def test_12_infinite_temperature_input_rejected(self):
        """2. Verify that Infinite temperature input raises ValueError."""
        with self.assertRaises(ValueError):
            compute_compression_isentropic_efficiency(p2=14.0, p30=554.0, t2=float("inf"), t30=1500.0)
        with self.assertRaises(ValueError):
            compute_compression_isentropic_efficiency(p2=14.0, p30=554.0, t2=518.0, t30=float("inf"))
        with self.assertRaises(ValueError):
            compute_compression_polytropic_efficiency(p2=14.0, p30=554.0, t2=518.0, t30=float("inf"))
        with self.assertRaises(ValueError):
            compute_compression_specific_work(t2=518.0, t30=float("inf"))

    def test_13_invalid_gamma_in_polytropic_efficiency_rejected(self):
        """3. Verify that gamma <= 1.0, non-finite, or NaN in polytropic efficiency raises ValueError."""
        # gamma <= 1.0
        with self.assertRaises(ValueError):
            compute_compression_polytropic_efficiency(p2=14.0, p30=554.0, t2=518.0, t30=1500.0, gamma=1.0)
        with self.assertRaises(ValueError):
            compute_compression_polytropic_efficiency(p2=14.0, p30=554.0, t2=518.0, t30=1500.0, gamma=0.8)
        # NaN gamma
        with self.assertRaises(ValueError):
            compute_compression_polytropic_efficiency(p2=14.0, p30=554.0, t2=518.0, t30=1500.0, gamma=float("nan"))
        # Inf gamma
        with self.assertRaises(ValueError):
            compute_compression_polytropic_efficiency(p2=14.0, p30=554.0, t2=518.0, t30=1500.0, gamma=float("inf"))

    def test_14_zero_and_negative_specific_heat_rejected(self):
        """4. Verify that cp <= 0.0 raises ValueError in specific work calculation."""
        with self.assertRaises(ValueError):
            compute_compression_specific_work(t2=518.0, t30=1500.0, cp=0.0)
        with self.assertRaises(ValueError):
            compute_compression_specific_work(t2=518.0, t30=1500.0, cp=-0.240)

    def test_15_nan_specific_heat_rejected(self):
        """5. Verify that NaN specific heat raises ValueError."""
        with self.assertRaises(ValueError):
            compute_compression_specific_work(t2=518.0, t30=1500.0, cp=float("nan"))
        with self.assertRaises(ValueError):
            compute_compression_specific_work(t2=518.0, t30=1500.0, cp=float("inf"))

    def test_16_zero_and_negative_corrected_spool_speed_rejected(self):
        """6. Verify that zero or negative corrected spool speeds raise ValueError."""
        with self.assertRaises(ValueError):
            compute_spool_speed_ratios(nc=9000.0, nf=2000.0, nrc=0.0, nrf=2000.0)
        with self.assertRaises(ValueError):
            compute_spool_speed_ratios(nc=9000.0, nf=2000.0, nrc=8000.0, nrf=0.0)
        with self.assertRaises(ValueError):
            compute_spool_speed_ratios(nc=9000.0, nf=2000.0, nrc=-8000.0, nrf=2000.0)
        with self.assertRaises(ValueError):
            compute_aerothermal_speed_corrections(nf=2000.0, t2=518.0, nc=9000.0, t24=640.0, nrf=0.0, nrc=8000.0)

    def test_17_missing_telemetry_keys_raises_key_error(self):
        """7. Verify that evaluate_engine_physics raises KeyError on missing keys."""
        telemetry_incomplete = {
            "T2": 518.67,
            "T24": 642.50,
            # T30 missing!
            "P2": 14.696,
            "P15": 21.61,
            "P30": 554.36,
            "Nf": 2388.0,
            "Nc": 9050.0,
            "NRf": 2388.0,
            "NRc": 8125.0,
        }
        with self.assertRaises(KeyError):
            evaluate_engine_physics(telemetry_incomplete)

    def test_18_none_valued_telemetry_raises_value_error(self):
        """8. Verify that None-valued telemetry variables raise ValueError."""
        telemetry_with_none = {
            "T2": 518.67,
            "T24": 642.50,
            "T30": None,  # None value!
            "P2": 14.696,
            "P15": 21.61,
            "P30": 554.36,
            "Nf": 2388.0,
            "Nc": 9050.0,
            "NRf": 2388.0,
            "NRc": 8125.0,
        }
        with self.assertRaises(ValueError):
            evaluate_engine_physics(telemetry_with_none)

    def test_19_non_finite_telemetry_values_raise_value_error(self):
        """9. Verify that NaN or Inf telemetry variables raise ValueError."""
        telemetry_nan = {
            "T2": 518.67,
            "T24": 642.50,
            "T30": float("nan"),  # NaN value!
            "P2": 14.696,
            "P15": 21.61,
            "P30": 554.36,
            "Nf": 2388.0,
            "Nc": 9050.0,
            "NRf": 2388.0,
            "NRc": 8125.0,
        }
        with self.assertRaises(ValueError):
            evaluate_engine_physics(telemetry_nan)

        telemetry_inf = dict(telemetry_nan)
        telemetry_inf["T30"] = float("inf")
        with self.assertRaises(ValueError):
            evaluate_engine_physics(telemetry_inf)

    def test_20_unclipped_efficiency_and_diagnostic_warnings(self):
        """10. Verify efficiency results outside nominal physical range are NOT silently clipped."""
        # Case A: Low efficiency (eta_is < 0.75) due to severe thermal rise with poor pressure rise
        # P2 = 14.696, P30 = 200.0 (OPR = 13.609)
        # T2 = 518.67, T30 = 2000.0 (T_ratio = 3.856)
        # eta_is_math ≈ (13.609^(2/7) - 1) / (3.856 - 1) = (2.106 - 1) / 2.856 = 1.106 / 2.856 ≈ 0.387
        low_eta_telemetry = {
            "T2": 518.67,
            "T24": 800.0,
            "T30": 2000.0,
            "P2": 14.696,
            "P15": 21.61,
            "P30": 200.0,
            "Nf": 2388.0,
            "Nc": 9050.0,
            "NRf": 2388.0,
            "NRc": 8125.0,
        }
        res_low = evaluate_engine_physics(low_eta_telemetry)
        # Ensure it is NOT clipped to 0.75 or 0.0
        self.assertLess(res_low.comp_isentropic_eff, 0.75)
        self.assertGreater(res_low.comp_isentropic_eff, 0.0)
        self.assertAlmostEqual(res_low.comp_isentropic_eff, 0.3881, places=3)
        # Diagnostic warning must be triggered
        has_is_warning = any("Isentropic efficiency" in w and "outside nominal" in w for w in res_low.warnings)
        self.assertTrue(has_is_warning, f"Expected warning in {res_low.warnings}")

        # Case B: High efficiency (eta_is > 1.0) due to artificial thermodynamic anomaly
        # P2 = 14.696, P30 = 554.36 (OPR = 37.72)
        # T2 = 518.67, T30 = 1200.0 (T_ratio = 2.313)
        # Numerator: 37.72^(2/7) - 1 ≈ 1.8228
        # Denominator: 2.313 - 1 = 1.313
        # eta_is_math ≈ 1.8228 / 1.313 ≈ 1.388 (> 1.0)
        high_eta_telemetry = {
            "T2": 518.67,
            "T24": 642.50,
            "T30": 1200.0,
            "P2": 14.696,
            "P15": 21.61,
            "P30": 554.36,
            "Nf": 2388.0,
            "Nc": 9050.0,
            "NRf": 2388.0,
            "NRc": 8125.0,
        }
        res_high = evaluate_engine_physics(high_eta_telemetry)
        # Must NOT be clipped to 1.0
        self.assertGreater(res_high.comp_isentropic_eff, 1.0)
        self.assertAlmostEqual(res_high.comp_isentropic_eff, 1.3865, places=3)
        has_high_warning = any("Isentropic efficiency" in w and "outside nominal" in w for w in res_high.warnings)
        self.assertTrue(has_high_warning, f"Expected warning in {res_high.warnings}")

    def test_21_zero_and_negative_telemetry_raises_value_error(self):
        """11. Verify zero and negative physical measurements in telemetry raise ValueError."""
        base_valid_telemetry = {
            "T2": 518.67,
            "T24": 642.50,
            "T30": 1589.70,
            "P2": 14.696,
            "P15": 21.61,
            "P30": 554.36,
            "Nf": 2388.0,
            "Nc": 9050.0,
            "NRf": 2388.0,
            "NRc": 8125.0,
        }

        # Zero pressure
        telem = dict(base_valid_telemetry, P2=0.0)
        with self.assertRaises(ValueError):
            evaluate_engine_physics(telem)

        # Negative pressure
        telem = dict(base_valid_telemetry, P30=-100.0)
        with self.assertRaises(ValueError):
            evaluate_engine_physics(telem)

        # Zero temperature
        telem = dict(base_valid_telemetry, T2=0.0)
        with self.assertRaises(ValueError):
            evaluate_engine_physics(telem)

        # Negative temperature
        telem = dict(base_valid_telemetry, T30=-500.0)
        with self.assertRaises(ValueError):
            evaluate_engine_physics(telem)

        # Zero physical spool speed
        telem = dict(base_valid_telemetry, Nf=0.0)
        with self.assertRaises(ValueError):
            evaluate_engine_physics(telem)

        # Negative physical spool speed
        telem = dict(base_valid_telemetry, Nc=-9000.0)
        with self.assertRaises(ValueError):
            evaluate_engine_physics(telem)

        # Zero and negative corrected spool speeds
        telem = dict(base_valid_telemetry, NRf=0.0)
        with self.assertRaises(ValueError):
            evaluate_engine_physics(telem)

        telem = dict(base_valid_telemetry, NRc=-8000.0)
        with self.assertRaises(ValueError):
            evaluate_engine_physics(telem)


if __name__ == "__main__":
    unittest.main()
