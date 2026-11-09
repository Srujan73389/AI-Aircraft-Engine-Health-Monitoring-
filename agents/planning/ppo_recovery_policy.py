"""
Proximal Policy Optimization (PPO) Recovery Policy
==================================================
Architectural component for Recovery Planning Agent:
Implements an Actor-Critic reinforcement learning policy network capable of
continuous operational recovery control and parameter optimization under degraded states.

Mathematical Formulation:
- State Space S in R^8:
    [T30_norm, P30_norm, OPR_norm, Delta_T_HPC_norm, Virtual_HI, Predicted_RUL_norm, Current_Derate, Current_Bleed]
- Action Space A in R^3:
    * a_0: Delta Thrust Derate [%] in [-5.0, +5.0]
    * a_1: Delta Bleed Valve Trim [%] in [-2.0, +2.0]
    * a_2: Delta Thermal Offset [°R] in [-10.0, +10.0]
- Gaussian Policy:
    pi_theta(a|s) = N(mu_theta(s), diag(sigma^2))
- PPO Clipped Surrogate Loss:
    r_t(theta) = exp(log_pi_theta(a_t|s_t) - log_pi_old(a_t|s_t))
    L_CLIP(theta) = E_t [ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]
- Critic Value Function:
    V_phi(s) in R
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    """
    Continuous Gaussian Actor network mapping engine state to recovery control distribution.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (64, 64),
        action_scale: Tuple[float, ...] = (5.0, 2.0, 10.0),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared representation layers
        layers = []
        curr_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.Tanh())
            curr_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # Mean action head
        self.mu_head = nn.Linear(curr_dim, action_dim)
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Action scale register buffer
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mu: Mean actions scaled to operational bounds.
            std: Standard deviations.
        """
        feat = self.backbone(state)
        # Tanh activation scaled to action limits
        mu = torch.tanh(self.mu_head(feat)) * self.action_scale
        std = torch.exp(torch.clamp(self.log_std, min=-2.0, max=1.0))
        return mu, std

    def get_distribution(self, state: torch.Tensor) -> Normal:
        """Returns torch Normal distribution for action sampling."""
        mu, std = self.forward(state)
        return Normal(mu, std)


class CriticNetwork(nn.Module):
    """
    Critic Value network estimating state value V(s).
    """

    def __init__(self, state_dim: int = 8, hidden_dims: Tuple[int, ...] = (64, 64)):
        super().__init__()
        layers = []
        curr_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.Tanh())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimates state value V(s)."""
        return self.net(state)


class PPORecoveryPolicy:
    """
    High-level PPO Recovery Agent for engine mitigation policy deployment and evaluation.
    """

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 3,
        clip_epsilon: float = 0.2,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_epsilon = clip_epsilon
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim=state_dim).to(self.device)

        self._init_stable_weights()
        self.actor.eval()
        self.critic.eval()

    def _init_stable_weights(self) -> None:
        """Initializes weights with small orthogonal matrices for policy stability."""
        torch.manual_seed(42)
        for net in [self.actor, self.critic]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """
        Selects a continuous recovery control adjustment given the current state.

        Args:
            state: Array of shape (state_dim,) or (1, state_dim)
            deterministic: If True, uses action mean (no exploration noise).

        Returns:
            Dict containing action array, log_prob, state_value.
        """
        if isinstance(state, np.ndarray):
            st_tensor = torch.from_numpy(state).float()
            if st_tensor.dim() == 1:
                st_tensor = st_tensor.unsqueeze(0)
        elif isinstance(state, torch.Tensor):
            st_tensor = state.float()
            if st_tensor.dim() == 1:
                st_tensor = st_tensor.unsqueeze(0)
        else:
            raise TypeError(f"Expected numpy ndarray or torch Tensor, got {type(state).__name__}")

        if st_tensor.size(-1) != self.state_dim:
            raise ValueError(f"State dimension mismatch: expected {self.state_dim}, got {st_tensor.size(-1)}")

        st_tensor = st_tensor.to(self.device)

        with torch.no_grad():
            dist = self.actor.get_distribution(st_tensor)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(st_tensor).squeeze(-1)

        act_np = action.squeeze(0).cpu().numpy()
        return {
            "action": act_np,
            "delta_derate_pct": round(float(act_np[0]), 3),
            "delta_bleed_pct": round(float(act_np[1]), 3),
            "delta_thermal_offset_deg": round(float(act_np[2]), 3),
            "log_prob": float(log_prob.squeeze().item()),
            "state_value": float(value.squeeze().item()),
            "deterministic": deterministic,
        }

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates actions for PPO training step.

        Returns:
            log_probs: Log probabilities under current policy.
            values: State values.
            entropy: Distribution entropy.
        """
        dist = self.actor.get_distribution(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(states).squeeze(-1)
        return log_probs, values, entropy

    def compute_clipped_surrogate_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the standard PPO clipped surrogate objective loss:
            L_clip = -min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t).mean()
        """
        ratios = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        loss = -torch.min(surr1, surr2).mean()
        return loss
