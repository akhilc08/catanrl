"""Strategy engine: explainability, planning, and natural language explanations."""

from .explainer import StrategyExplainer
from .planner import MonteCarloPlanner
from .templates import RESOURCE_NAMES, TEMPLATES, ExplanationGenerator

__all__ = [
    "StrategyExplainer",
    "MonteCarloPlanner",
    "ExplanationGenerator",
    "TEMPLATES",
    "RESOURCE_NAMES",
]
