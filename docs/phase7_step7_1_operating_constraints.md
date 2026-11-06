# Phase 7 — Step 7.1: Operating Constraints & Aerospace Assumptions

## 1. Executive Summary

In Phase 7, the digital twin and autonomous multi-agent health monitoring architecture are fortified with formal **Safety Verification and Simulated Recovery** capabilities. Step 7.1 establishes the mathematical boundaries, aerothermal invariants, and domain assumptions that govern realistic turbofan operations under nominal and degraded flight conditions.

All operating limits are explicitly cataloged in [`safety/constraints.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/constraints.py) with formal provenance tracing to one of four theoretical origins:
1. **`PHYSICAL_THERMODYNAMICS`**: Conservation laws, gas dynamic relationships, and aerodynamic stage compression physics.
2. **`OPERATIONAL_FLIGHT_ENVELOPE`**: Aircraft flight mechanics, civil aviation service ceilings, and engine airframe interface limits.
3. **`EMPIRICAL_BASELINE`**: Statistical distributions derived from NASA C-MAPSS nominal baseline fleet trajectories.
4. **`ACADEMIC_SIMULATION_ASSUMPTION`**: Simulation-specific modeling choices, normalization boundaries, and causal constraints.

---

## 2. Constraint Taxonomy & Severities

Each constraint enforces a designated severity level upon violation:

| Severity Level | Enforcement Policy | Operational Implication |
| :--- | :--- | :--- |
| **`CAUTION`** | Advisory Alert | Triggers automated diagnostic inspection; indicates incipient thermal or mechanical creep beyond nominal baseline without immediate structural hazard. |
| **`CRITICAL`** | Maintenance Intervention | Indicates significant operational exceedance requiring flight crew awareness, engine derating, or prioritized ground turnaround inspection. |
| **`HARD_LIMIT`** | Safety Firewall Abort / Rejection | Non-negotiable physical or metallurgical boundary (e.g., combustor casing burst pressure, shaft shear, mathematical invalidity). |

---

## 3. Comprehensive Operating Constraints Registry

The formal registry `OPERATING_CONSTRAINTS` contains 20 continuous range and relational invariant specifications:

### 3.1 Ambient Flight Envelope Constraints

| Constraint ID | Target Variable | Bounded Range | Units | Source | Severity | Aerothermal Assumption |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CONST_FLIGHT_ALTITUDE` | `altitude` | $[0.0, 45000.0]$ | $\text{ft}$ | `OPERATIONAL_FLIGHT_ENVELOPE` | `CRITICAL` | Subsonic civil transport certified ceiling (FL450) bounded by airframe pressurization and aerodynamic buffet margin. |
| `CONST_FLIGHT_MACH` | `mach_number` | $[0.0, 0.90]$ | dimensionless | `OPERATIONAL_FLIGHT_ENVELOPE` | `CRITICAL` | High-bypass turbofan nacelle drag divergence boundary $(M \le 0.90)$ preventing supersonic shock separation. |
| `CONST_FLIGHT_TRA` | `throttle_resolver_angle` | $[0.0, 100.0]$ | $\text{deg}$ | `OPERATIONAL_FLIGHT_ENVELOPE` | `CRITICAL` | Mechanical power lever resolver angle authority from flight idle ($0^\circ$) to max takeoff power ($100^\circ$). |

### 3.2 Core Gas-Path Temperatures

| Constraint ID | Target Variable | Bounded Range | Units | Source | Severity | Aerothermal Assumption |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CONST_TEMP_T2` | `T2` | $[400.0, 600.0]$ | $^\circ\text{R}$ | `PHYSICAL_THERMODYNAMICS` | `CRITICAL` | Fan inlet total stagnation temperature across ambient ISA high-altitude cruise ($400^\circ\text{R}$) to sea-level hot-day ram stagnation ($600^\circ\text{R}$). |
| `CONST_TEMP_T24` | `T24` | $[450.0, 750.0]$ | $^\circ\text{R}$ | `PHYSICAL_THERMODYNAMICS` | `CRITICAL` | Low-Pressure Compressor (LPC) exit stagnation temperature following isentropic work addition. |
| `CONST_TEMP_T30_CAUTION` | `T30` | $[1100.0, 1605.0]$ | $^\circ\text{R}$ | `EMPIRICAL_BASELINE` | `CAUTION` | High-Pressure Compressor (HPC) discharge thermal creep threshold where blade thermal degradation accelerates. |
| `CONST_TEMP_T30_MAX_WARNING` | `T30` | $[1100.0, 1625.0]$ | $^\circ\text{R}$ | `PHYSICAL_THERMODYNAMICS` | `HARD_LIMIT` | HPC exit structural metallurgical threshold preventing high-pressure compressor stage distortion. |
| `CONST_TEMP_T50` | `T50` | $[1000.0, 1600.0]$ | $^\circ\text{R}$ | `PHYSICAL_THERMODYNAMICS` | `CRITICAL` | Low-Pressure Turbine (LPT) exhaust gas temperature (EGT) survivability envelope balancing expansion work against creep life. |

### 3.3 Core Gas-Path Pressures & Derived Ratios

| Constraint ID | Target Variable | Bounded Range | Units | Source | Severity | Aerothermal Assumption |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CONST_PRESS_P2` | `P2` | $[2.5, 20.0]$ | $\text{psia}$ | `PHYSICAL_THERMODYNAMICS` | `CRITICAL` | Fan inlet total ram recovery pressure across cruise ambient to sea-level max dynamic head. |
| `CONST_PRESS_P15` | `P15` | $[5.0, 30.0]$ | $\text{psia}$ | `PHYSICAL_THERMODYNAMICS` | `CRITICAL` | Bypass duct total stagnation pressure after low-pressure fan pressure ratio (FPR) stage compression. |
| `CONST_PRESS_P30` | `P30` | $[100.0, 700.0]$ | $\text{psia}$ | `PHYSICAL_THERMODYNAMICS` | `HARD_LIMIT` | HPC discharge casing burst pressure safety containment limit under maximum core compression. |
| `CONST_PRESS_PS30` | `Ps30` | $[5.0, 65.0]$ | $\text{psia}$ | `PHYSICAL_THERMODYNAMICS` | `CRITICAL` | Combustor diffuser inlet static pressure feeding fuel-air mixing swirler nozzles. |
| `CONST_DERIVED_OPR` | `overall_pressure_ratio` | $[15.0, 55.0]$ | dimensionless | `PHYSICAL_THERMODYNAMICS` | `CRITICAL` | Overall compressor pressure ratio $(P_{30} / P_2)$ for stable aerothermal operation outside compressor surge/stall. |

### 3.4 Rotor Mechanical Speeds

| Constraint ID | Target Variable | Bounded Range | Units | Source | Severity | Aerothermal Assumption |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CONST_SPEED_NF` | `Nf` | $[1500.0, 3000.0]$ | $\text{rpm}$ | `PHYSICAL_THERMODYNAMICS` | `CRITICAL` | Fan spool mechanical speed bounded by fan blade tip transonic shock containment and shaft torsional resonance. |
| `CONST_SPEED_NC` | `Nc` | $[7000.0, 10500.0]$ | $\text{rpm}$ | `PHYSICAL_THERMODYNAMICS` | `CRITICAL` | Core HP spool rotational speed bounded by disk centrifugal hoop stresses avoiding plastic burst. |

### 3.5 Prognostics & Degradation Boundaries

| Constraint ID | Target Variable | Bounded Range | Units | Source | Severity | Aerothermal Assumption |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CONST_PROG_HEALTH_INDEX` | `virtual_health_index` | $[0.0, 1.0]$ | dimensionless | `ACADEMIC_SIMULATION_ASSUMPTION` | `HARD_LIMIT` | Normalized degradation state $H(t) \in [0.0, 1.0]$, where $1.0$ represents mint nominal health and $0.0$ marks operational retirement. |
| `CONST_PROG_RUL_NON_NEGATIVE` | `predicted_rul` | $[0.0, \infty)$ | $\text{cycles}$ | `ACADEMIC_SIMULATION_ASSUMPTION` | `HARD_LIMIT` | Physical causality requirement: engine time-to-failure cannot be negative. |

### 3.6 Aerothermal Ordering Invariants

| Constraint ID | Target Invariant | Relational Equation | Units | Source | Severity | Aerothermal Assumption |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CONST_INVARIANT_TEMP_ORDER` | Temperature Order | $T_{30} > T_{24} > T_2$ | $^\circ\text{R}$ | `PHYSICAL_THERMODYNAMICS` | `HARD_LIMIT` | Second Law of Thermodynamics: work addition across successive compressor stages monotonically increases stagnation temperature. |
| `CONST_INVARIANT_PRESS_ORDER` | Pressure Order | $P_{30} > P_{15} > P_2$ | $\text{psia}$ | `PHYSICAL_THERMODYNAMICS` | `HARD_LIMIT` | Monotonic compression pressure rise: forward gas flow requires continuous positive pressure gradient across stages. |
| `CONST_INVARIANT_SPOOL_ORDER` | Spool Hierarchy | $N_c > N_f$ | $\text{rpm}$ | `PHYSICAL_THERMODYNAMICS` | `HARD_LIMIT` | Rotor design geometry: smaller core spool operates at higher angular velocity than large low-pressure fan spool. |

---

## 4. Evaluation Mechanics & Noise Tolerances

The evaluator function [`evaluate_constraints(state_dict_or_engine_state)`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/constraints.py) handles input polymorphism and numerical precision:

1. **Input Normalization**:
   Accepts `EngineState` dataclass instances, dictionary representations (`state.to_dict()`), or raw mapping inputs with aliased column names (`sensor_3` $\rightarrow$ `T30`, `op_setting_1` $\rightarrow$ `altitude`).

2. **Unit Conversion**:
   Supports both flight altitude conventions ($40.0\text{ kft} \equiv 40,000\text{ ft}$).

3. **Sensor Noise Margins**:
   Accommodates small zero-point simulation noise in C-MAPSS ground-idle telemetry (e.g. altitude margin of $-50\text{ ft}$, Mach margin of $-0.01$) to prevent spurious false alarms on nominal trajectories while strictly detecting unphysical negative or excessive values.

4. **Structured Reporting**:
   Returns a [`ConstraintEvaluationReport`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/constraints.py) providing `total_evaluated`, `passed_count`, `violation_count`, `is_valid` flag, and detailed [`ConstraintViolation`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/safety/constraints.py) records.

---

## 5. Verification & Test Results

The dedicated test suite [`tests/test_safety_constraints_step7_1.py`](file:///c:/Users/LENOVO/Documents/Aircraft_Engine_AI/tests/test_safety_constraints_step7_1.py) validates all 10 unit test cases:

```
test_01_registry_completeness: Asserts all required flight, sensor, and model constraints exist. -> PASS
test_02_bounded_range_validity: Asserts min_value < max_value for all bounded numeric constraints. -> PASS
test_03_constraint_provenance: Asserts source and non-empty assumption strings for every constraint. -> PASS
test_04_temperature_constraint_violations: Asserts detection of out-of-range T30. -> PASS
test_05_pressure_constraint_violations: Asserts detection of out-of-range P30. -> PASS
test_06_flight_envelope_violations: Asserts detection of unphysical altitude/Mach. -> PASS
test_07_aerothermal_ordering_invariants: Asserts evaluation of thermodynamic order (T30 > T24 > T2, etc.). -> PASS
test_08_prognostics_bounds_violations: Asserts detection of negative RUL or invalid health index. -> PASS
test_09_nominal_state_evaluation: Evaluates a nominal EngineState; asserts zero violations. -> PASS
test_10_dataset_and_model_immutability: Cryptographic SHA-256 verification of 12 raw files, 4 scalers, 8 checkpoints. -> PASS
```

**Overall Test Suite Status**: **245 / 245 tests PASS** (235 prior tests + 10 Step 7.1 tests).
