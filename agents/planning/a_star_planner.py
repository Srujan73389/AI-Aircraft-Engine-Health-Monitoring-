"""
A* Search Trajectory Recovery Planner
=====================================
Architectural component for Recovery Planning Agent:
Computes an optimal, constraint-adherent sequence of engine recovery actions
to navigate an anomalous or degraded turbofan from a high-stress operational point
back to the safe, nominal flight envelope with minimal mission/thrust penalty.

Graph Formulation:
- State Space S: Discretized engine state vector:
    (T30_bin, P30_bin, OPR_bin, thrust_derate_pct, bleed_adjust_pct)
- Action Primitives A:
    * DERATE_THRUST_MILD: Derate +2.0% (sheds ~8.0 °R, cost = 2.0)
    * DERATE_THRUST_MODERATE: Derate +5.0% (sheds ~20.0 °R, cost = 5.0)
    * ADJUST_BLEED_VALVE: Bleed open +2.0% (sheds ~6.0 °R, cost = 1.5)
    * TRIM_FUEL_RATIO: Fuel flow trim -1.0% (sheds ~10.0 °R, cost = 3.0)
    * STABILIZE_CRUISE: Hold current configuration (cost = 0.5)
- Transition Dynamics: T(s, a) -> s' based on aerothermal sensitivity derivatives.
- Hard Safety Constraints:
    * T30 <= 1625.0 °R (metallurgical redline boundary)
    * Max Derate <= 20.0% (flight safety margin)
    * Max Bleed <= 10.0%
- Target Goal Set G:
    * T30 <= 1590.0 °R (nominal operating zone)
    * P30 >= 500.0 psia
    * OPR >= 35.0
- Admissible Heuristic h(s):
    h(s) = max(0.0, s.T30 - 1590.0) * (min_cost_per_deg_reduction)
    Proved admissible: under-estimates the minimal cost to shed remaining thermal stress.
"""

from dataclasses import dataclass, field
import heapq
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass(frozen=True)
class RecoveryNode:
    """Discretized state node for A* search."""
    t30: float
    p30: float
    opr: float
    thrust_derate_pct: float = 0.0
    bleed_adjust_pct: float = 0.0

    def to_key(self, round_digits: int = 1) -> Tuple[float, float, float, float, float]:
        """Discretized tuple for closed set hash checks."""
        return (
            round(self.t30, round_digits),
            round(self.p30, round_digits),
            round(self.opr, round_digits),
            round(self.thrust_derate_pct, 1),
            round(self.bleed_adjust_pct, 1),
        )


@dataclass
class RecoveryAction:
    """Action primitive applied during recovery planning."""
    name: str
    delta_t30: float
    delta_p30: float
    delta_opr: float
    delta_derate: float
    delta_bleed: float
    base_cost: float


ACTION_PRIMITIVES: List[RecoveryAction] = [
    RecoveryAction(
        name="DERATE_THRUST_MILD",
        delta_t30=-8.0,
        delta_p30=-2.5,
        delta_opr=-0.15,
        delta_derate=2.0,
        delta_bleed=0.0,
        base_cost=2.0,
    ),
    RecoveryAction(
        name="DERATE_THRUST_MODERATE",
        delta_t30=-20.0,
        delta_p30=-6.0,
        delta_opr=-0.40,
        delta_derate=5.0,
        delta_bleed=0.0,
        base_cost=5.0,
    ),
    RecoveryAction(
        name="ADJUST_BLEED_VALVE",
        delta_t30=-6.0,
        delta_p30=-1.0,
        delta_opr=-0.08,
        delta_derate=0.0,
        delta_bleed=2.0,
        base_cost=1.5,
    ),
    RecoveryAction(
        name="TRIM_FUEL_RATIO",
        delta_t30=-10.0,
        delta_p30=-3.0,
        delta_opr=-0.20,
        delta_derate=1.0,
        delta_bleed=0.0,
        base_cost=3.0,
    ),
]


@dataclass
class RecoveryPath:
    """Result returned by the A* recovery planner."""
    success: bool
    total_cost: float
    actions: List[str]
    states: List[RecoveryNode]
    nodes_expanded: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "total_cost": round(self.total_cost, 2),
            "step_count": len(self.actions),
            "actions": self.actions,
            "final_state": {
                "T30": round(self.states[-1].t30, 2),
                "P30": round(self.states[-1].p30, 2),
                "OPR": round(self.states[-1].opr, 2),
                "thrust_derate_pct": round(self.states[-1].thrust_derate_pct, 2),
                "bleed_adjust_pct": round(self.states[-1].bleed_adjust_pct, 2),
            } if self.states else None,
            "nodes_expanded": self.nodes_expanded,
            "reason": self.reason,
        }


class AStarRecoveryPlanner:
    """
    A* Graph-Search Recovery Planner for Turbofan Engines.
    Guarantees optimal, minimal-derating recovery paths to safe operating envelopes.
    """

    def __init__(
        self,
        t30_max_limit: float = 1625.0,
        t30_target_safe: float = 1590.0,
        p30_min_safe: float = 500.0,
        opr_min_safe: float = 35.0,
        max_derate_pct: float = 20.0,
        max_bleed_pct: float = 10.0,
        max_search_depth: int = 15,
        max_nodes_expanded: int = 2000,
    ):
        self.t30_max_limit = t30_max_limit
        self.t30_target_safe = t30_target_safe
        self.p30_min_safe = p30_min_safe
        self.opr_min_safe = opr_min_safe
        self.max_derate_pct = max_derate_pct
        self.max_bleed_pct = max_bleed_pct
        self.max_search_depth = max_search_depth
        self.max_nodes_expanded = max_nodes_expanded

        # Admissible heuristic coefficient: min cost / max delta_T30
        # MODERATE derate sheds 20 °R for cost 5.0 -> 0.25 cost per °R
        self.heuristic_cost_per_deg = 0.25

    def is_goal_state(self, node: RecoveryNode) -> bool:
        """Evaluates whether the node has entered the safe operating envelope."""
        return (
            node.t30 <= self.t30_target_safe
            and node.p30 >= self.p30_min_safe
            and node.opr >= self.opr_min_safe
        )

    def is_valid_state(self, node: RecoveryNode) -> bool:
        """Evaluates physical and metallurgical constraints."""
        if node.t30 > self.t30_max_limit:
            return False  # Violates metallurgical redline
        if node.thrust_derate_pct > self.max_derate_pct:
            return False  # Exceeds maximum flight envelope derate
        if node.bleed_adjust_pct > self.max_bleed_pct:
            return False  # Exceeds maximum bleed schedule
        return True

    def heuristic(self, node: RecoveryNode) -> float:
        """
        Admissible heuristic under-estimating cost to reach safe goal set.
        """
        if self.is_goal_state(node):
            return 0.0
        thermal_excess = max(0.0, node.t30 - self.t30_target_safe)
        return thermal_excess * self.heuristic_cost_per_deg

    def plan_recovery(
        self,
        current_t30: float,
        current_p30: float = 553.9,
        current_opr: float = 37.7,
        initial_derate: float = 0.0,
        initial_bleed: float = 0.0,
    ) -> RecoveryPath:
        """
        Executes A* search to discover optimal recovery sequence.

        Args:
            current_t30: Current High Pressure Compressor exit temperature (°R)
            current_p30: Current HPC exit pressure (psia)
            current_opr: Current Overall Pressure Ratio
            initial_derate: Initial thrust derate percentage
            initial_bleed: Initial bleed adjustment percentage

        Returns:
            RecoveryPath object.
        """
        initial_state = RecoveryNode(
            t30=current_t30,
            p30=current_p30,
            opr=current_opr,
            thrust_derate_pct=initial_derate,
            bleed_adjust_pct=initial_bleed,
        )

        # Immediate goal check
        if self.is_goal_state(initial_state):
            return RecoveryPath(
                success=True,
                total_cost=0.0,
                actions=[],
                states=[initial_state],
                nodes_expanded=0,
                reason="Engine already operating within safe envelope.",
            )

        # Initial state hard limit check
        if not self.is_valid_state(initial_state):
            return RecoveryPath(
                success=False,
                total_cost=0.0,
                actions=[],
                states=[initial_state],
                nodes_expanded=0,
                reason=f"Initial state breaches metallurgical redline (T30={current_t30:.1f} °R > {self.t30_max_limit:.1f} °R).",
            )

        # Min-Heap Priority Queue: (f_score, g_score, counter, current_node, action_history, state_history)
        counter = 0
        h_init = self.heuristic(initial_state)
        open_set: List[Tuple[float, float, int, RecoveryNode, List[str], List[RecoveryNode]]] = []
        heapq.heappush(open_set, (h_init, 0.0, counter, initial_state, [], [initial_state]))

        # Closed set mapping state key to lowest g_score observed
        closed_set: Dict[Tuple[float, float, float, float, float], float] = {}
        nodes_expanded = 0

        while open_set and nodes_expanded < self.max_nodes_expanded:
            f, g, _, current_node, actions, states = heapq.heappop(open_set)
            nodes_expanded += 1

            # Check goal
            if self.is_goal_state(current_node):
                return RecoveryPath(
                    success=True,
                    total_cost=g,
                    actions=actions,
                    states=states,
                    nodes_expanded=nodes_expanded,
                    reason="A* search converged to safe operating envelope.",
                )

            state_key = current_node.to_key()
            if state_key in closed_set and closed_set[state_key] <= g:
                continue
            closed_set[state_key] = g

            if len(actions) >= self.max_search_depth:
                continue

            # Expand successors via action primitives
            for act in ACTION_PRIMITIVES:
                next_node = RecoveryNode(
                    t30=current_node.t30 + act.delta_t30,
                    p30=current_node.p30 + act.delta_p30,
                    opr=current_node.opr + act.delta_opr,
                    thrust_derate_pct=current_node.thrust_derate_pct + act.delta_derate,
                    bleed_adjust_pct=current_node.bleed_adjust_pct + act.delta_bleed,
                )

                if not self.is_valid_state(next_node):
                    continue

                next_g = g + act.base_cost
                next_key = next_node.to_key()

                if next_key in closed_set and closed_set[next_key] <= next_g:
                    continue

                next_h = self.heuristic(next_node)
                next_f = next_g + next_h
                counter += 1

                heapq.heappush(
                    open_set,
                    (
                        next_f,
                        next_g,
                        counter,
                        next_node,
                        actions + [act.name],
                        states + [next_node],
                    ),
                )

        return RecoveryPath(
            success=False,
            total_cost=0.0,
            actions=[],
            states=[initial_state],
            nodes_expanded=nodes_expanded,
            reason="Search budget exhausted without finding valid recovery path.",
        )
