"""
Unscented Kalman Filter (UKF) for Digital Twin State Estimation
==============================================================
Architectural component for Stage 3 Digital Twin:
Provides non-linear state estimation, aerothermal health parameter tracking,
and sensor noise filtering over turbofan dynamic flight trajectories.

Mathematical Formulation:
- System Dynamics: x_k = f(x_{k-1}, u_{k-1}) + w_{k-1},  w ~ N(0, Q)
- Measurement:     y_k = h(x_k) + v_k,                   v ~ N(0, R)
- Unscented Transform (Van der Merwe scaled sigma points):
    lambda = alpha^2 * (n + kappa) - n
    gamma  = sqrt(n + lambda)
    chi_0  = x
    chi_i  = x + gamma * [sqrt(P)]_i,     i = 1 ... n
    chi_{i+n} = x - gamma * [sqrt(P)]_i, i = 1 ... n
- Weights:
    W_m^{(0)} = lambda / (n + lambda)
    W_c^{(0)} = lambda / (n + lambda) + (1 - alpha^2 + beta)
    W_m^{(i)} = W_c^{(i)} = 1 / (2 * (n + lambda)),  i = 1 ... 2n
- Predict Step:
    Propagate sigma points through f(.), compute x_pred and P_pred + Q.
- Update Step:
    Propagate sigma points through h(.), compute y_pred, P_yy + R, P_xy,
    Kalman gain K = P_xy * (P_yy)^{-1}, and correct state x and covariance P.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class UnscentedKalmanFilter:
    """
    General Non-Linear Unscented Kalman Filter (UKF).
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        f_func: Optional[Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]] = None,
        h_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.f_func = f_func or self._default_state_transition
        self.h_func = h_func or self._default_measurement_function

        # Scaling parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        self.num_sigmas = 2 * dim_x + 1
        self._compute_weights()

        # State and Covariances
        self.x = np.zeros(dim_x, dtype=np.float64)
        self.P = np.eye(dim_x, dtype=np.float64)
        self.Q = np.eye(dim_x, dtype=np.float64) * 0.01  # Process noise
        self.R = np.eye(dim_z, dtype=np.float64) * 0.1   # Measurement noise

        # Prior estimates
        self.x_prior = np.copy(self.x)
        self.P_prior = np.copy(self.P)
        self.sigmas_f = np.zeros((self.num_sigmas, dim_x), dtype=np.float64)
        self.sigmas_h = np.zeros((self.num_sigmas, dim_z), dtype=np.float64)

        # Innovation and Kalman Gain
        self.y = np.zeros(dim_z, dtype=np.float64)
        self.K = np.zeros((dim_x, dim_z), dtype=np.float64)
        self.S = np.zeros((dim_z, dim_z), dtype=np.float64)

    def _compute_weights(self) -> None:
        """Computes sigma point weights for mean and covariance."""
        n = self.dim_x
        lam = (self.alpha ** 2) * (n + self.kappa) - n
        self.gamma = np.sqrt(n + lam)
        self.lam = lam

        self.Wm = np.full(self.num_sigmas, 1.0 / (2.0 * (n + lam)), dtype=np.float64)
        self.Wc = np.full(self.num_sigmas, 1.0 / (2.0 * (n + lam)), dtype=np.float64)

        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = (lam / (n + lam)) + (1.0 - (self.alpha ** 2) + self.beta)

    def _default_state_transition(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Default identity state transition."""
        return np.copy(x)

    def _default_measurement_function(self, x: np.ndarray) -> np.ndarray:
        """Default linear measurement slicing."""
        return x[: self.dim_z]

    def _cholesky_stable(self, P: np.ndarray) -> np.ndarray:
        """Cholesky decomposition with jitter regularization for positive semi-definiteness."""
        P_sym = 0.5 * (P + P.T)
        try:
            return np.linalg.cholesky(P_sym)
        except np.linalg.LinAlgError:
            jitter = 1e-8 * np.eye(self.dim_x)
            return np.linalg.cholesky(P_sym + jitter)

    def generate_sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Generates 2n + 1 sigma points around x."""
        sigmas = np.zeros((self.num_sigmas, self.dim_x), dtype=np.float64)
        sigmas[0] = x

        chol = self._cholesky_stable(P)
        for i in range(self.dim_x):
            offset = self.gamma * chol[:, i]
            sigmas[i + 1] = x + offset
            sigmas[self.dim_x + i + 1] = x - offset
        return sigmas

    def predict(self, u: Optional[np.ndarray] = None) -> None:
        """
        Time Update: Propagates state and covariance forward by one time step.
        """
        sigmas = self.generate_sigma_points(self.x, self.P)

        # Propagate through non-linear dynamics
        for i in range(self.num_sigmas):
            self.sigmas_f[i] = self.f_func(sigmas[i], u)

        # Compute prior mean
        self.x_prior = np.dot(self.Wm, self.sigmas_f)

        # Compute prior covariance
        P_prior = np.zeros((self.dim_x, self.dim_x), dtype=np.float64)
        for i in range(self.num_sigmas):
            y_diff = self.sigmas_f[i] - self.x_prior
            P_prior += self.Wc[i] * np.outer(y_diff, y_diff)
        self.P_prior = P_prior + self.Q

        # Symmetrize
        self.P_prior = 0.5 * (self.P_prior + self.P_prior.T)
        self.x = np.copy(self.x_prior)
        self.P = np.copy(self.P_prior)

    def update(self, z: np.ndarray) -> None:
        """
        Measurement Update: Corrects prior estimate using observed sensor vector z.
        """
        z_arr = np.asarray(z, dtype=np.float64)

        # Generate sigma points from prior
        sig_prior = self.generate_sigma_points(self.x_prior, self.P_prior)

        # Transform sigma points through measurement function
        for i in range(self.num_sigmas):
            self.sigmas_h[i] = self.h_func(sig_prior[i])

        # Predicted measurement
        z_pred = np.dot(self.Wm, self.sigmas_h)

        # Innovation covariance S
        P_zz = np.zeros((self.dim_z, self.dim_z), dtype=np.float64)
        for i in range(self.num_sigmas):
            z_diff = self.sigmas_h[i] - z_pred
            P_zz += self.Wc[i] * np.outer(z_diff, z_diff)
        self.S = P_zz + self.R
        self.S = 0.5 * (self.S + self.S.T)

        # Cross-covariance P_xz
        P_xz = np.zeros((self.dim_x, self.dim_z), dtype=np.float64)
        for i in range(self.num_sigmas):
            x_diff = sig_prior[i] - self.x_prior
            z_diff = self.sigmas_h[i] - z_pred
            P_xz += self.Wc[i] * np.outer(x_diff, z_diff)

        # Kalman gain K = P_xz * S^{-1}
        self.K = np.linalg.solve(self.S.T, P_xz.T).T

        # Innovation residual
        self.y = z_arr - z_pred

        # Updated posterior state and covariance
        self.x = self.x_prior + np.dot(self.K, self.y)
        self.P = self.P_prior - np.dot(self.K, np.dot(self.S, self.K.T))
        self.P = 0.5 * (self.P + self.P.T)


class TurbofanStateUKF:
    """
    Specialized Unscented Kalman Filter for Turbofan Digital Twin State Estimation.
    Estimates 8 vital gas-path states from noisy C-MAPSS sensor measurements:
      x[0]: T24 (LPC exit temperature, °R)
      x[1]: T30 (HPC exit temperature, °R)
      x[2]: T50 (LPT exit temperature, °R)
      x[3]: P30 (HPC exit pressure, psia)
      x[4]: Nf  (Fan physical speed, rpm)
      x[5]: Nc  (Core physical speed, rpm)
      x[6]: delta_HPC_eff (HPC efficiency degradation delta, dimensionless)
      x[7]: delta_HPC_flow (HPC flow capacity degradation delta, dimensionless)
    """

    STATE_NAMES: List[str] = [
        "T24", "T30", "T50", "P30", "Nf", "Nc", "delta_HPC_eff", "delta_HPC_flow"
    ]

    MEASUREMENT_NAMES: List[str] = [
        "sensor_2_T24", "sensor_3_T30", "sensor_4_T50", "sensor_7_P30", "sensor_8_Nf", "sensor_9_Nc"
    ]

    def __init__(
        self,
        initial_state: Optional[np.ndarray] = None,
        process_noise_diag: Optional[List[float]] = None,
        measurement_noise_diag: Optional[List[float]] = None,
    ):
        dim_x = 8
        dim_z = 6

        # Nominal sea-level baseline
        if initial_state is None:
            init_x = np.array([642.35, 1585.0, 1400.0, 553.5, 2388.0, 9045.0, 0.0, 0.0], dtype=np.float64)
        else:
            init_x = np.asarray(initial_state, dtype=np.float64)

        self.ukf = UnscentedKalmanFilter(
            dim_x=dim_x,
            dim_z=dim_z,
            f_func=self._turbofan_dynamics,
            h_func=self._turbofan_measurement,
            alpha=0.1,
            beta=2.0,
            kappa=0.0,
        )

        self.ukf.x = init_x
        self.ukf.P = np.diag([2.0, 5.0, 4.0, 1.0, 5.0, 10.0, 0.005, 0.005])

        # Process noise: degradation states drift gradually
        q_diag = process_noise_diag or [0.1, 0.2, 0.15, 0.05, 0.1, 0.2, 1e-5, 1e-5]
        self.ukf.Q = np.diag(q_diag)

        # Measurement noise from C-MAPSS sensor tolerances
        r_diag = measurement_noise_diag or [0.5, 0.8, 0.6, 0.3, 1.0, 2.0]
        self.ukf.R = np.diag(r_diag)

    def _turbofan_dynamics(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Non-linear aerothermal state dynamics.
        Gas-path temperatures rise as HPC efficiency delta degrades.
        """
        x_next = np.copy(x)
        delta_eff = x[6]
        # Temperature increase proportional to degradation
        x_next[1] += abs(delta_eff) * 15.0  # T30 rises
        x_next[2] += abs(delta_eff) * 10.0  # T50 rises
        x_next[3] -= abs(delta_eff) * 5.0   # P30 slightly drops under loss of compression
        # Degradation state follows random walk / slow decay
        x_next[6] += -0.0001
        x_next[7] += -0.00008
        return x_next

    def _turbofan_measurement(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement function mapping 8-dim state to 6-dim sensor measurements.
        """
        # Directly measures [T24, T30, T50, P30, Nf, Nc]
        return x[:6]

    def step(self, sensor_measurements: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Executes one predict-update estimation cycle.

        Args:
            sensor_measurements: [s2_T24, s3_T30, s4_T50, s7_P30, s8_Nf, s9_Nc]

        Returns:
            Dict containing filtered state estimates, uncertainty variances, and innovation.
        """
        z = np.asarray(sensor_measurements, dtype=np.float64)
        if len(z) != 6:
            raise ValueError(f"Expected 6 sensor measurements, got {len(z)}")

        # 1. Predict
        self.ukf.predict()

        # 2. Update
        self.ukf.update(z)

        state_dict = {
            name: round(float(self.ukf.x[i]), 4)
            for i, name in enumerate(self.STATE_NAMES)
        }
        variance_dict = {
            f"{name}_var": round(float(self.ukf.P[i, i]), 6)
            for i, name in enumerate(self.STATE_NAMES)
        }

        return {
            "estimated_state": state_dict,
            "covariance_variances": variance_dict,
            "covariance_trace": round(float(np.trace(self.ukf.P)), 6),
            "innovation_residual": [round(float(v), 4) for v in self.ukf.y],
            "hpc_efficiency_delta": state_dict["delta_HPC_eff"],
            "filtered_t30": state_dict["T30"],
            "filtered_p30": state_dict["P30"],
        }
