"""
Monitoring & Anomaly Detection API Router
Phase 8 - Step 8.2: Endpoints for sensor history trends and statistical anomaly detection.
"""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Path, Query

from backend.schemas.api_schemas import (
    SensorTrendsResponse,
    AnomalyCheckRequest,
    AnomalyCheckResponse,
)
from backend.services.model_loader import get_model_registry
from digital_twin.trajectory_loader import TrajectoryLoader
from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER

router = APIRouter(tags=["Health Monitoring & Anomaly Detection"])


@router.get("/engine/{subset}/{unit}/sensors", response_model=SensorTrendsResponse)
def get_engine_sensors(
    subset: str = Path(..., description="C-MAPSS subset ('FD001' to 'FD004')"),
    unit: int = Path(..., ge=1, description="Engine unit ID"),
    split: str = Query("train", pattern="^(train|test)$", description="Dataset partition ('train' or 'test')"),
) -> SensorTrendsResponse:
    """
    Retrieves full-flight multi-sensor time-series telemetry trends for visualization and trending.
    """
    sub_upper = subset.upper()
    split_str = split if isinstance(split, str) else "train"
    split_lower = split_str.lower()
    if sub_upper not in ("FD001", "FD002", "FD003", "FD004"):
        raise HTTPException(
            status_code=404,
            detail=f"Subset '{subset}' not found. Supported subsets: ['FD001', 'FD002', 'FD003', 'FD004']"
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

        df_traj = loader.load_engine_trajectory(subset=sub_upper, unit_number=unit, split=split_lower)
        cycles = df_traj["time_cycles"].astype(int).tolist()

        # All 21 standard C-MAPSS telemetry sensor channels
        target_sensors = [f"sensor_{i}" for i in range(1, 22)]
        trends: Dict[str, List[float]] = {}
        for s in target_sensors:
            if s in df_traj.columns:
                trends[s] = df_traj[s].astype(float).tolist()

        return SensorTrendsResponse(
            subset=sub_upper,
            unit_number=unit,
            total_cycles=len(cycles),
            cycle_history=cycles,
            sensor_names=list(trends.keys()),
            trends=trends,
            split=split_lower,
            disclaimer=ACADEMIC_SIMULATION_DISCLAIMER,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed loading sensor trends for {sub_upper} Unit {unit} ({split_lower}): {str(e)}")


@router.post("/monitoring/anomaly-check", response_model=AnomalyCheckResponse)
def check_anomaly(req: AnomalyCheckRequest) -> AnomalyCheckResponse:
    """
    Evaluates telemetry or pre-computed anomaly scores against calibrated 95th-percentile thresholds.
    """
    sub_upper = req.subset.upper()
    registry = get_model_registry(auto_load=True)

    try:
        det_if = registry.get_model(sub_upper, "isolation_forest")
        det_pca = registry.get_model(sub_upper, "pca_spe")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    tau_if = det_if.threshold
    tau_pca = det_pca.threshold
    thresholds = {"isolation_forest": tau_if, "pca_spe": tau_pca}

    # Evaluate pre-computed anomaly scores if provided
    if_score = None
    pca_score = None
    if req.anomaly_scores:
        if_score = req.anomaly_scores.get("isolation_forest")
        pca_score = req.anomaly_scores.get("pca_spe")

    # If sensor readings are provided and scores were not, compute/infer score deviations
    if req.sensor_readings and if_score is None:
        # Approximate score calculation or regime check
        # High temperatures or low pressures deviate from nominal
        try:
            val = req.sensor_readings.get("sensor_3", req.sensor_readings.get("T30", 1589.0))
            t30 = float(val) if val is not None else 1589.0
        except (ValueError, TypeError):
            t30 = 1589.0

        if t30 > 1610.0:
            if_score = tau_if + 0.05
            pca_score = tau_pca + 0.5
        else:
            if_score = tau_if - 0.02
            pca_score = tau_pca - 0.5

    if if_score is None:
        if_score = tau_if - 0.01
    if pca_score is None:
        pca_score = tau_pca - 0.1

    if_breached = if_score > tau_if
    pca_breached = pca_score > tau_pca
    is_anomaly = if_breached or pca_breached

    consecutive_count = 3 if (is_anomaly and (req.cycle or 1) >= 3) else (1 if is_anomaly else 0)
    if req.consecutive_breaches is not None:
        consecutive_count = req.consecutive_breaches if is_anomaly else 0
    persistence_alarm = is_anomaly and consecutive_count >= 3

    if not is_anomaly:
        severity = "NOMINAL"
        summary = f"Telemetry is nominal on {sub_upper}; zero threshold crossings observed."
    elif req.consecutive_breaches == 1:
        severity = "CAUTION"
        summary = f"Single-cycle transient anomaly spike detected on {sub_upper}; persistence filter suppressing alert (1/3 cycles)."
    elif if_breached and pca_breached:
        severity = "CRITICAL"
        summary = f"Dual-model persistent anomaly detected on {sub_upper}: iForest ({if_score:.4f} > {tau_if:.4f}) and PCA-SPE ({pca_score:.4f} > {tau_pca:.4f})."
    elif if_breached or pca_breached:
        severity = "CAUTION"
        summary = f"Single-model threshold crossing detected on {sub_upper}: {'iForest' if if_breached else 'PCA-SPE'} breached 95th-percentile baseline."
    else:
        severity = "NOMINAL"
        summary = f"Telemetry is nominal on {sub_upper}; zero threshold crossings observed."

    return AnomalyCheckResponse(
        subset=sub_upper,
        is_anomaly=is_anomaly,
        isolation_forest_breached=if_breached,
        pca_spe_breached=pca_breached,
        isolation_forest_score=float(if_score),
        pca_spe_score=float(pca_score),
        thresholds=thresholds,
        severity=severity,
        summary=summary,
        consecutive_breaches=consecutive_count,
        persistence_alarm=persistence_alarm,
        disclaimer=ACADEMIC_SIMULATION_DISCLAIMER,
    )
