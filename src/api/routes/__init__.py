"""API route modules for CatanRL."""

from .feedback import router as feedback_router
from .health import router as health_router
from .predict import router as predict_router

__all__ = ["feedback_router", "health_router", "predict_router"]
