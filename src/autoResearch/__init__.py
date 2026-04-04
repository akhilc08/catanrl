"""autoResearch — autonomous MAPPO hyperparameter search for CatanRL.

Mirrors Karpathy's autoresearch framework: proposes mutations, runs
time-bounded training trials, keeps improvements, discards regressions.
"""

from .evaluator import Evaluator, TrialResult
from .experiment import ExperimentRecord, ExperimentTracker
from .mutations import Mutation, MutationEngine
from .researcher import AutoResearcher, ResearchConfig

__all__ = [
    "AutoResearcher",
    "ResearchConfig",
    "Evaluator",
    "TrialResult",
    "ExperimentTracker",
    "ExperimentRecord",
    "Mutation",
    "MutationEngine",
]
