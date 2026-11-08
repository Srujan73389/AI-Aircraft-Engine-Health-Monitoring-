"""
Backend API Routes Package
"""

from backend.api.fleet import router as fleet_router
from backend.api.monitoring import router as monitoring_router
from backend.api.prognostics import router as prognostics_router
from backend.api.simulation import router as simulation_router
from backend.api.safety import router as safety_router

__all__ = [
    "fleet_router",
    "monitoring_router",
    "prognostics_router",
    "simulation_router",
    "safety_router",
]
