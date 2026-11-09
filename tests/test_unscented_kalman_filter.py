"""
Unit Test Suite: Unscented Kalman Filter (UKF) State Estimation
==============================================================
Validates non-linear state estimation, unscented transform sigma points,
covariance positive definiteness, and turbofan state filtering for Stage 3 Digital Twin.
"""

import unittest
import numpy as np

from digital_twin.unscented_kalman_filter import (
    UnscentedKalmanFilter,
    TurbofanStateUKF,
)


class TestUnscentedKalmanFilter(unittest.TestCase):
    """Test suite for UKF and Turbofan State Estimator."""

    def setUp(self):
        np.random.seed(42)

    def test_01_weights_sum_to_one(self):
        """Verifies that unscented transform sigma weights sum to 1.0."""
        ukf = UnscentedKalmanFilter(dim_x=4, dim_z=2, alpha=1e-3, beta=2.0, kappa=0.0)
        # Weights for mean sum to 1.0
        self.assertAlmostEqual(float(np.sum(ukf.Wm)), 1.0, places=5)

    def test_02_sigma_points_generation(self):
        """Verifies that 2n+1 sigma points are correctly generated around mean."""
        dim_x = 3
        ukf = UnscentedKalmanFilter(dim_x=dim_x, dim_z=2)
        x = np.array([10.0, 20.0, 30.0])
        P = np.diag([1.0, 4.0, 9.0])

        sigmas = ukf.generate_sigma_points(x, P)
        # Total sigma points is 2*n + 1 = 7
        self.assertEqual(sigmas.shape, (7, 3))
        # Center sigma point is mean x
        np.testing.assert_allclose(sigmas[0], x)

    def test_03_linear_system_convergence(self):
        """Verifies UKF tracks and filters a 1D constant velocity noisy system."""
        # State: [position, velocity]
        def f_cv(x, u=None):
            dt = 1.0
            return np.array([x[0] + x[1] * dt, x[1]])

        def h_cv(x):
            return np.array([x[0]])

        ukf = UnscentedKalmanFilter(dim_x=2, dim_z=1, f_func=f_cv, h_func=h_cv)
        ukf.x = np.array([0.0, 5.0])  # Initial: pos=0, vel=5
        ukf.P = np.diag([10.0, 5.0])
        ukf.Q = np.diag([0.01, 0.01])
        ukf.R = np.array([[1.0]])

        true_pos = 0.0
        true_vel = 5.0

        for _ in range(20):
            true_pos += true_vel
            noisy_meas = np.array([true_pos + np.random.normal(0.0, 1.0)])
            ukf.predict()
            ukf.update(noisy_meas)

        # Estimated position and velocity should be close to true values
        self.assertAlmostEqual(ukf.x[0], true_pos, delta=3.0)
        self.assertAlmostEqual(ukf.x[1], true_vel, delta=1.5)

    def test_04_turbofan_state_ukf_step(self):
        """Verifies TurbofanStateUKF step execution and noise suppression."""
        estimator = TurbofanStateUKF()
        # Simulated noisy measurements: [T24, T30, T50, P30, Nf, Nc]
        noisy_telemetry = [643.0, 1588.0, 1402.0, 554.0, 2389.0, 9046.0]

        res = estimator.step(noisy_telemetry)
        self.assertIn("estimated_state", res)
        self.assertIn("filtered_t30", res)
        self.assertIn("filtered_p30", res)
        self.assertIn("hpc_efficiency_delta", res)
        self.assertIn("covariance_trace", res)

        est_state = res["estimated_state"]
        self.assertEqual(len(est_state), 8)
        self.assertGreater(est_state["T30"], 1500.0)
        self.assertLess(est_state["T30"], 1650.0)
        self.assertGreater(est_state["P30"], 500.0)
        self.assertLess(est_state["P30"], 600.0)

    def test_05_turbofan_ukf_measurement_validation(self):
        """Verifies error handling when wrong measurement dimension is passed."""
        estimator = TurbofanStateUKF()
        # Pass 3 measurements instead of required 6
        with self.assertRaises(ValueError):
            estimator.step([1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
