# Coding Conventions

**Analysis Date:** 2026-04-02

## Naming Patterns

**Files:**
- `snake_case.py` for all Python modules: `catan_env.py`, `gnn_encoder.py`, `drift_monitor.py`
- Test files prefixed with `test_`: `test_catan_env.py`, `test_policy.py`
- Config/data classes named with `PascalCase` suffix matching purpose: `CurriculumConfig`, `CurriculumPhase`

**Functions:**
- `snake_case` for all functions and methods: `get_action_mask()`, `set_reference_from_data()`, `encode_action()`
- Private helpers prefixed with single underscore: `_generate_board_topology()`, `_orthogonal_init()`, `_make_data()` (in tests), `_bank_trade_index()`
- Boolean-returning methods named with verb phrases: `is_final_phase`, `check_drift()`

**Classes:**
- `PascalCase` throughout: `CatanEnv`, `CatanPolicy`, `DriftMonitor`, `ActionSpace`, `ModelManager`
- Domain prefix on classes within the same module: `CatanGNNEncoder`, `CatanPolicy`

**Variables:**
- `snake_case` for all local and instance variables: `num_players`, `reference_histograms`, `current_phase_idx`
- `SCREAMING_SNAKE_CASE` for module-level constants: `MAX_TURNS`, `WIN_VP`, `NUM_HEX`, `WOOD`, `BRICK`

**Type Parameters:**
- Standard `Any` from `typing` when generic types are needed
- Union types written with `|` syntax (Python 3.10+): `str | None`, `reference_path: str | None = None`

## Code Style

**Formatting:**
- Tool: `ruff` (configured in `pyproject.toml`)
- `target-version = "py310"`, line length: 100
- `src = ["src", "tests"]` in ruff config

**Linting:**
- Ruff rules enabled: `E` (pycodestyle errors), `F` (pyflakes), `I` (isort), `N` (pep8-naming), `W` (pycodestyle warnings), `UP` (pyupgrade)
- `mypy` with `ignore_missing_imports = true`, `disallow_untyped_defs = false`
- Inline suppression via `# type: ignore[...]` with specific error codes (not bare `# type: ignore`)
- `# noqa: E402` used when import order is constrained (e.g., after conditional imports)

## Import Organization

**Order (enforced by ruff/isort):**
1. `from __future__ import annotations` — present in every source file
2. Standard library imports
3. Third-party imports
4. Local/project imports (using `from src.module import ...` pattern)

**Path Style:**
- Absolute imports from `src.*` root: `from src.rl.env.catan_env import CatanEnv`
- Relative imports within a package: `from .action_space import ActionSpace`
- No `__init__.py` re-exports of significance; direct module paths are used

## Module Docstrings

Every source file opens with a triple-quoted module-level docstring describing the module's purpose, key design decisions, and any important constants or encodings. Example from `action_space.py`:

```python
"""Action space encoding and masking for the Catan RL environment.

Flat integer action encoding with boolean masking for legal actions.

Action Categories and Offsets:
    ROLL_DICE:  1 action  (offset 0)
    ...
"""
```

## Class and Function Documentation

**NumPy/Google-style docstrings with Parameters section:**
```python
class DriftMonitor:
    """Monitors feature distribution drift using Jensen-Shannon divergence.

    Parameters
    ----------
    reference_path : str, optional
        Path to a JSON file containing reference histograms.
    threshold : float
        JS divergence threshold for triggering drift detection.
    """
```

Single-line docstrings used for simple methods: `"""Return a fresh CatanEnv reset with a fixed seed."""`

## Type Annotations

- `from __future__ import annotations` in every file to enable PEP 563 deferred evaluation
- Return types annotated on all public methods: `def get_action_mask(self) -> np.ndarray:`
- `@dataclass` used for configuration/data-transfer objects in `src/rl/training/curriculum.py`, `src/rl/training/mappo.py`, `src/rl/training/self_play.py`
- Pydantic `BaseModel` used for API request/response schemas in `src/api/schemas.py`
- `collections.abc.Mapping` preferred over `typing.Mapping`

## Error Handling

**Patterns:**
- `ValueError` raised for invalid input at domain boundaries: `raise ValueError(f"Unknown action type: {action_type}")`
- `RuntimeError` raised for invalid operational state: `raise RuntimeError("Model not loaded. Call load() first.")`
- Broad `except Exception as e` used in infrastructure/IO code (CloudWatch, S3 download, drift monitor load) with structured log of the error
- Specific exception types used when the type is known: `except urllib.error.HTTPError as e:`
- No bare `except:` clauses

## Logging

**Framework:** `structlog` in API and monitoring modules; stdlib `logging` in training modules

**Module-level logger pattern:**
```python
# API/monitoring modules
import structlog
logger = structlog.get_logger()

# Training modules
import logging
logger = logging.getLogger(__name__)
```

**Event style:** `structlog` calls use keyword arguments for structured context:
```python
logger.info("model_loaded", version=self.version)
logger.warning("s3_download_failed", error=str(exc))
logger.info("request_complete", request_id=request_id, path=request.url.path, latency_ms=round(latency, 1))
```

## Constants

- Module-level constants defined at the top of the file after imports, grouped under a `# --- Constants ---` separator comment
- Integer constants for game indices use tuple unpacking: `WOOD, BRICK, SHEEP, WHEAT, ORE, DESERT = 0, 1, 2, 3, 4, 5`
- Class-level constants (offsets, sizes) are `SCREAMING_SNAKE_CASE` class attributes, not instance attributes: `ActionSpace.BUILD_ROAD_OFFSET = 2`

## Function Design

**Size:** Large classes are accepted for complex domain logic (`catan_env.py` at 1076 lines), but helper logic is factored into private module-level functions.

**Parameters:** Configuration objects passed as dataclasses (`MAPPOConfig`, `CurriculumConfig`) rather than long keyword argument lists.

**Return Values:** Multi-value returns use tuples; public APIs annotate them explicitly: `-> tuple[dict, float, bool, bool, dict]`

## Module Design

**Exports:** No explicit `__all__` declarations; consumers import directly from submodules.

**Barrel Files:** `__init__.py` files are present but empty (used only for package recognition), not for re-exporting.

---

*Convention analysis: 2026-04-02*
