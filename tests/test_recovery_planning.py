"""
Unit Test Suite: Recovery Planning (A* Search & PPO Policy)
===========================================================
Validates optimal trajectory search via A* planner and continuous control
via Proximal Policy Optimization (PPO) Actor-Critic policy for the Recovery Planning Agent.
"""

import unittest
import numpy as np
import torch

from agents.planning.a_star_planner import (
    AStarRecoveryPlanner,
    RecoveryNode,
    RecoveryPath,
    ACTION_PRIMITIVES,
)
from agents.planning.ppo_recovery_policy import (
    ActorNetwork,
    CriticNetwork,
    PPORecoveryPolicy,
)


class TestRecoveryPlanning(unittest.TestCase):
    """Test suite for A* Search and PPO Recovery Policy."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    # -------------------------------------------------------------------------
    # A* Search Planner Tests
    # -------------------------------------------------------------------------
    def test_01_a_star_already_safe_goal(self):
        """Verifies immediate return when engine is already in safe envelope."""
        planner = AStarRecoveryPlanner(t30_target_safe=1590.0)
        # Already safe: T30 = 1585.0 <= 1590.0
        path = planner.plan_recovery(current_t30=1585.0, current_p30=553.0, current_opr=37.5)
        self.assertTrue(path.success)
        self.assertEqual(len(path.actions), 0)
        self.assertEqual(path.total_cost, 0.0)

    def test_02_a_star_redline_initial_rejection(self):
        """Verifies immediate failure when starting state violates metallurgical redline."""
        planner = AStarRecoveryPlanner(t30_max_limit=1625.0)
        # Initial T30 = 1630.0 > 1625.0
        path = planner.plan_recovery(current_t30=1630.0)
        self.assertFalse(path.success)
        self.assertIn("redline", path.reason.lower())

    def test_03_a_star_successful_thermal_shedding(self):
        """Verifies A* planner finds optimal sequence to shed elevated thermal stress."""
        planner = AStarRecoveryPlanner(
            t30_max_limit=1625.0,
            t30_target_safe=1590.0,
            max_derate_pct=20.0,
        )
        # Elevated state: T30 = 1615.0 °R (excess of 25.0 °R above safe goal)
        path = planner.plan_recovery(current_t30=1615.0, current_p30=555.0, current_opr=38.0)
        self.assertTrue(path.success)
        self.assertGreater(len(path.actions), 0)
        self.assertGreater(path.total_cost, 0.0)

        # Final state must satisfy goal constraints
        final_state = path.states[-1]
        self.assertLessEqual(final_state.t30, 1590.0)
        self.assertGreaterEqual(final_state.p30, 500.0)
        self.assertGreaterEqual(final_state.opr, 35.0)
        self.assertLessEqual(final_state.thrust_derate_pct, 20.0)

    def test_04_a_star_admissible_heuristic(self):
        """Verifies heuristic is strictly admissible (under-estimates actual cost)."""
        planner = AStarRecoveryPlanner(t30_target_safe=1590.0)
        node = RecoveryNode(t30=1610.0, p30=550.0, opr=37.0)
        h_val = planner.heuristic(node)
        # 20 deg excess * 0.25 = 5.0
        self.assertAlmostEqual(h_val, 5.0, places=2)

    # -------------------------------------------------------------------------
    # PPO Recovery Policy Tests
    # -------------------------------------------------------------------------
    def test_05_ppo_actor_critic_forward_shapes(self):
        """Verifies Actor and Critic network forward output dimensions."""
        state_dim = 8
        action_dim = 3
        batch_size = 4

        actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim)
        critic = CriticNetwork(state_dim=state_dim)

        s = torch.randn(batch_size, state_dim)
        mu, std = actor(s)
        self.assertEqual(mu.shape, (batch_size, action_dim))
        self.assertEqual(std.shape, (action_dim,))

        val = critic(s)
        self.assertEqual(val.shape, (batch_size, 1))

    def test_06_ppo_policy_select_action_deterministic(self):
        """Verifies action selection in deterministic mode with bounds."""
        policy = PPORecoveryPolicy(state_dim=8, action_dim=3)
        state = np.array([0.5, 0.2, 0.1, 0.4, 0.9, 0.8, 0.0, 0.0], dtype=np.float32)

        res = policy.select_action(state, deterministic=True)
        self.assertTrue(res["deterministic"])
        self.assertIn("action", res)
        self.assertEqual(len(res["action"]), 3)
        self.assertIn("delta_derate_pct", res)
        self.assertIn("delta_bleed_pct", res)
        self.assertIn("delta_thermal_offset_deg", res)

        # Bounds checks
        self.assertGreaterEqual(res["delta_derate_pct"], -5.0)
        self.assertLessEqual(res["delta_derate_pct"], 5.0)
        self.assertGreaterEqual(res["delta_bleed_pct"], -2.0)
        self.assertLessEqual(res["delta_bleed_pct"], 2.0)

    def test_07_ppo_clipped_surrogate_loss(self):
        """Verifies clipped surrogate loss computation."""
        policy = PPORecoveryPolicy(state_dim=8, action_dim=3, clip_epsilon=0.2)
        batch = 8
        old_lp = torch.zeros(batch)
        new_lp = torch.tensor([0.05] * batch)
        adv = torch.ones(batch)

        loss = policy.compute_clipped_surrogate_loss(old_lp, new_lp, adv)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_08_ppo_state_dimension_validation(self):
        """Verifies error handling on mismatched state dimension."""
        policy = PPORecoveryPolicy(state_dim=8, action_dim=3)
        with self.assertRaises(ValueError):
            policy.select_action(np.zeros(4, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
