"""
Fleet & Engine Trajectory API Router
Phase 8 - Step 8.2: Endpoints for subset discovery, engine inventory, and cycle state retrieval.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Path, Query

from backend.schemas.api_schemas import (
    FleetSubsetInfo,
    EngineSummary,
    EngineCycleState,
)
from digital_twin.trajectory_loader import TrajectoryLoader
from digital_twin.twin_simulator import DigitalTwinSimulator
from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER

router = APIRouter(tags=["Fleet & Engines"])

# Static dataset catalog metadata
FLEET_CATALOG = {
    "FD001": FleetSubsetInfo(
        subset="FD001",
        operating_conditions="Single (Sea Level)",
        fault_modes=["HPC Degradation"],
        training_engines_count=100,
        test_engines_count=100,
        description="Single sea-level flight regime with High-Pressure Compressor thermal degradation."
    ),
    "FD002": FleetSubsetInfo(
        subset="FD002",
        operating_conditions="Six Real-World Flight Regimes",
        fault_modes=["HPC Degradation"],
        training_engines_count=260,
        test_engines_count=259,
        description="Six complex operating flight envelopes (altitude/Mach combinations) with HPC degradation."
    ),
    "FD003": FleetSubsetInfo(
        subset="FD003",
        operating_conditions="Single (Sea Level)",
        fault_modes=["HPC Degradation", "Fan Degradation"],
        training_engines_count=100,
        test_engines_count=100,
        description="Single flight regime with compound High-Pressure Compressor and Fan fault modes."
    ),
    "FD004": FleetSubsetInfo(
        subset="FD004",
        operating_conditions="Six Real-World Flight Regimes",
        fault_modes=["HPC Degradation", "Fan Degradation"],
        training_engines_count=249,
        test_engines_count=248,
        description="Six operating regimes with compound HPC and Fan degradation modes."
    ),
}


@router.get("/fleet/subsets", response_model=List[FleetSubsetInfo])
def get_subsets() -> List[FleetSubsetInfo]:
    """Retrieve metadata catalog for all four NASA C-MAPSS subsets."""
    return list(FLEET_CATALOG.values())


@router.get("/fleet/{subset}/engines", response_model=List[int])
def get_subset_engines(
    subset: str = Path(..., description="C-MAPSS subset identifier ('FD001' to 'FD004')"),
    split: str = Query("train", pattern="^(train|test)$", description="Dataset partition ('train' or 'test')"),
) -> List[int]:
    """Retrieve list of available engine unit numbers for a given dataset subset and split."""
    sub_upper = subset.upper()
    split_str = split if isinstance(split, str) else "train"
    split_lower = split_str.lower()
    if sub_upper not in FLEET_CATALOG:
        raise HTTPException(
            status_code=404,
            detail=f"Subset '{subset}' not found. Supported subsets: {list(FLEET_CATALOG.keys())}"
        )
    try:
        loader = TrajectoryLoader()
        units = loader.get_available_units(subset=sub_upper, split=split_lower)
        return units
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load engine list for {sub_upper} ({split_lower}): {str(e)}")


@router.get("/engine/{subset}/{unit}/cycle/{cycle}", response_model=EngineCycleState)
def get_engine_cycle_state(
    subset: str = Path(..., description="C-MAPSS subset ('FD001' to 'FD004')"),
    unit: int = Path(..., description="Engine unit ID"),
    cycle: int = Path(..., description="Operational flight cycle"),
    split: str = Query("train", pattern="^(train|test)$", description="Dataset partition ('train' or 'test')"),
) -> EngineCycleState:
    """
    Constructs and returns the full 40-variable Digital Twin state vector (G1 to G40)
    for a given engine and operational cycle.
    """
    if cycle < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Requested cycle {cycle} must be a positive integer >= 1."
        )
    if unit < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Requested unit {unit} must be a positive integer >= 1."
        )

    sub_upper = subset.upper()
    split_str = split if isinstance(split, str) else "train"
    split_lower = split_str.lower()
    if sub_upper not in FLEET_CATALOG:
        raise HTTPException(
            status_code=404,
            detail=f"Subset '{subset}' not found. Supported subsets: {list(FLEET_CATALOG.keys())}"
        )

    try:
        loader = TrajectoryLoader()
        available_units = loader.get_available_units(subset=sub_upper, split=split_lower)
        if unit not in available_units:
            min_u = min(available_units)
            max_u = max(available_units)
            raise HTTPException(
                status_code=404,
                detail=f"Engine unit {unit} not found in subset '{sub_upper}' ({split_lower}). Valid engine ID range is {min_u} to {max_u} (total {len(available_units)} engines)."
            )

        sim = DigitalTwinSimulator(subset=sub_upper, enable_ml_inference=True)
        sim.load_engine(unit_number=unit, split=split_lower)
        max_cycle = sim.loader.load_engine_trajectory(sub_upper, unit, split=split_lower)["time_cycles"].max()
        if cycle > max_cycle:
            raise HTTPException(
                status_code=400,
                detail=f"Requested cycle {cycle} exceeds engine {unit} ({split_lower}) maximum cycle {max_cycle}."
            )

        sim.advance_to_cycle(cycle)

        raw_state = sim.current_state.to_dict(include_arrays=False)
        state_dict = dict(raw_state)
        for k, v in raw_state.items():
            if isinstance(k, str) and "_" in k and k.startswith("G"):
                prefix, rest = k.split("_", 1)
                if prefix[1:].isdigit() and rest not in state_dict:
                    state_dict[rest] = v

        return EngineCycleState(
            subset=sub_upper,
            unit_number=unit,
            cycle=cycle,
            split=split_lower,
            state_vector=state_dict,
            disclaimer=ACADEMIC_SIMULATION_DISCLAIMER,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error constructing state for {sub_upper} Unit {unit} ({split_lower}) Cycle {cycle}: {str(e)}"
        )
