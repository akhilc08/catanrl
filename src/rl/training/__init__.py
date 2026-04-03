"""CatanRL training modules."""

from .curriculum import (
    CurriculumConfig,
    CurriculumPhase,
    CurriculumScheduler,
    default_curriculum,
)
from .mappo import MAPPOConfig, MAPPOTrainer
from .self_play import (
    OpponentEntry,
    OpponentPool,
    SelfPlayConfig,
    SelfPlayTrainer,
)

__all__ = [
    "CurriculumConfig",
    "CurriculumPhase",
    "CurriculumScheduler",
    "default_curriculum",
    "MAPPOConfig",
    "MAPPOTrainer",
    "OpponentEntry",
    "OpponentPool",
    "SelfPlayConfig",
    "SelfPlayTrainer",
]
