"""
Aircraft Engine AI - FastAPI Application Entrypoint
Phase 8 - Step 8.2: Backend API Serving Preprocessing Pipelines, Champion Models,
Fleet Discovery, Monitoring, RUL Forecasting, Simulation What-If, and Safety Governance.

Initializes FastAPI application, loads in-memory model registry during lifespan startup,
registers API routers under /api/v1 prefix, and exposes health monitoring with academic disclaimers.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.services.model_loader import ModelRegistry, get_model_registry
from backend.api.fleet import router as fleet_router
from backend.api.monitoring import router as monitoring_router
from backend.api.prognostics import router as prognostics_router
from backend.api.simulation import router as simulation_router
from backend.api.safety import router as safety_router
from safety.disclaimers import ACADEMIC_SIMULATION_DISCLAIMER


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager:
    Initializes and warms the ModelRegistry cache on server startup.
    """
    registry = get_model_registry(auto_load=True)
    app.state.model_registry = registry
    yield


app = FastAPI(
    title="Aircraft Engine AI Health Monitoring & Prognostics API",
    version="1.0.0",
    description=(
        "REST API serving NASA C-MAPSS Digital Twin preprocessing pipelines, "
        "champion machine learning prognostics (XGBoost & Bi-LSTM), interactive simulation, "
        "and aerothermal safety verification."
    ),
    lifespan=lifespan,
)

# CORS middleware for frontend communication (Phase 8 Step 8.3 & Dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_registry() -> ModelRegistry:
    """Dependency resolver for ModelRegistry instance."""
    return get_model_registry(auto_load=True)


# =============================================================================
# Defensive Exception Handlers (Phase 8 Step 8.10)
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for HTTP 422 validation errors.
    Returns structured APIErrorResponse with itemized field errors.
    """
    return JSONResponse(
        status_code=422,
        content={
            "status_code": 422,
            "error_code": "MISSING_INPUT",
            "message": "Request validation failed: one or more required fields are missing or malformed.",
            "detail": exc.errors(),
            "suggestion": "Review request payload parameters and verify required fields and types match API specifications.",
            "disclaimer": ACADEMIC_SIMULATION_DISCLAIMER,
        },
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_exception_handler(request: Request, exc: FileNotFoundError):
    """
    Custom handler for missing files / checkpoints returning HTTP 503 MODEL_UNAVAILABLE.
    """
    return JSONResponse(
        status_code=503,
        content={
            "status_code": 503,
            "error_code": "MODEL_UNAVAILABLE",
            "message": "Required champion model checkpoint or scaler pipeline file is unavailable.",
            "detail": f"MODEL_UNAVAILABLE: {str(exc)}",
            "suggestion": "Verify Phase 3 scalers (models/scalers/) and Phase 4 model checkpoints (models/checkpoints/) are present.",
            "disclaimer": ACADEMIC_SIMULATION_DISCLAIMER,
        },
    )


# =============================================================================
# Register API Routers under /api/v1 (and /api for compatibility)
# =============================================================================

app.include_router(fleet_router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")
app.include_router(prognostics_router, prefix="/api/v1")
app.include_router(simulation_router, prefix="/api/v1")
app.include_router(safety_router, prefix="/api/v1")

app.include_router(fleet_router, prefix="/api")
app.include_router(monitoring_router, prefix="/api")
app.include_router(prognostics_router, prefix="/api")
app.include_router(simulation_router, prefix="/api")
app.include_router(safety_router, prefix="/api")


# =============================================================================
# Root & System Health Endpoints
# =============================================================================

@app.get("/", tags=["Root"])
def root() -> Dict[str, Any]:
    """Root endpoint welcoming clients to the Aircraft Engine AI API."""
    return {
        "name": "Aircraft Engine AI Health Monitoring & Prognostics API",
        "version": "1.0.0",
        "phase": "Phase 8 - Backend API & Interactive Dashboard",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "models_status": "/api/models/status",
            "fleet_subsets": "/api/v1/fleet/subsets",
            "predict_rul": "/api/v1/predict/rul",
            "simulation_what_if": "/api/v1/simulation/what-if",
            "safety_verify": "/api/v1/safety/verify",
            "docs": "/docs",
        },
        "disclaimer": ACADEMIC_SIMULATION_DISCLAIMER,
    }


@app.get("/health", tags=["System Health"])
@app.get("/api/v1/health", tags=["System Health"])
def health(registry: ModelRegistry = Depends(get_registry)) -> Dict[str, Any]:
    """
    Health check endpoint:
    Returns API status, version, loaded model counts, and academic disclaimer.
    """
    return {
        "status": "healthy" if registry.is_ready else "degraded",
        "version": "1.0.0",
        "ready": registry.is_ready,
        "scalers_loaded": registry.scalers_count,
        "models_loaded": registry.models_count,
        "champion_checkpoints_loaded": registry.get_status()["champion_checkpoints_loaded"],
        "disclaimer": ACADEMIC_SIMULATION_DISCLAIMER,
    }


@app.get("/api/models/status", tags=["Model Registry"])
@app.get("/api/v1/models/status", tags=["Model Registry"])
def get_models_status(registry: ModelRegistry = Depends(get_registry)) -> Dict[str, Any]:
    """
    Detailed model registry status endpoint:
    Returns inventory verifying all 4 Phase 3 scalers and 8 Phase 4 champion models are loaded.
    """
    status_data = registry.get_status()
    return {
        "status": "success",
        "data": status_data,
        "disclaimer": ACADEMIC_SIMULATION_DISCLAIMER,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
