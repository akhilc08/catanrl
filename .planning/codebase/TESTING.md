# Testing Patterns

**Analysis Date:** 2026-04-02

## Test Framework

**Runner:**
- `pytest >= 7.4.0`
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`
  - `testpaths = ["tests"]`
  - `pythonpath = ["."]` (enables `from src.x import y` without install)

**Assertion Library:**
- pytest built-in assertions
- `numpy.testing.assert_array_almost_equal` for floating-point array comparisons
- `pydantic.ValidationError` raised and caught with `pytest.raises`

**Additional plugins:**
- `pytest-cov >= 4.1.0` — coverage reporting
- `pytest-asyncio >= 0.23.0` — async test support for FastAPI routes
- `httpx >= 0.25.0` — async HTTP client for API integration tests

**Run Commands:**
```bash
pytest tests/ -x --cov=src --cov-report=term-missing   # Full suite with coverage
pytest tests/unit/                                       # Unit tests only
pytest tests/integration/                               # Integration tests only
make test                                               # Alias for the full command above
```

## Test File Organization

**Location:** Separate `tests/` directory at project root, mirroring the source structure.

**Structure:**
```
tests/
├── conftest.py                         # Shared fixtures
├── __init__.py
├── unit/
│   ├── __init__.py
│   ├── test_action_space.py           # src/rl/env/action_space.py
│   ├── test_catan_env.py              # src/rl/env/catan_env.py
│   ├── test_gnn_encoder.py            # src/rl/models/gnn_encoder.py
│   ├── test_policy.py                 # src/rl/models/policy.py
│   ├── test_curriculum.py             # src/rl/training/curriculum.py
│   ├── test_drift_monitor.py          # src/monitoring/drift_monitor.py
│   └── test_api_schemas.py            # src/api/schemas.py
└── integration/
    ├── __init__.py
    ├── test_full_pipeline.py          # Env + model end-to-end
    └── test_predict_endpoint.py       # FastAPI routes via httpx
```

**Naming:** `test_{module_name}.py`, one file per source module.

## Test Structure

**Suite Organization — class-per-feature-group:**
```python
class TestReset:
    """Tests for env.reset()."""

    def test_returns_obs_and_info(self, env: CatanEnv):
        obs, info = env.reset(seed=123)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_observation_keys(self, env: CatanEnv):
        obs, _ = env.reset(seed=123)
        expected_keys = {"hex_features", "vertex_features", ...}
        assert set(obs.keys()) == expected_keys
```

Classes group related tests by the method or behavior under test (e.g. `TestReset`, `TestStep`, `TestBuildingCosts`). Each test method covers one specific assertion or scenario.

**Docstrings on non-obvious tests:** Short docstrings explain the intent of tests that require context:
```python
def test_masked_actions_have_zero_probability(self, policy: CatanPolicy):
    """Actions that are masked out should have ~0 probability."""
```

## Fixtures

**Shared fixtures in `tests/conftest.py`:**
```python
@pytest.fixture
def env() -> CatanEnv:
    """Return a fresh CatanEnv reset with a fixed seed."""
    e = CatanEnv(num_players=4)
    e.reset(seed=42)
    return e

@pytest.fixture
def action_space() -> ActionSpace:
    """Return an ActionSpace instance."""
    return ActionSpace()
```

**Local fixtures in test files for module-specific setup:**
```python
# tests/unit/test_policy.py
@pytest.fixture
def policy() -> CatanPolicy:
    """Create a small policy for testing."""
    encoder = CatanGNNEncoder.from_env_defaults(
        hidden_dim=32, num_heads=2, num_layers=1, output_dim=64
    )
    return CatanPolicy(gnn_encoder=encoder, action_dim=261, hidden_dim=64)

# tests/unit/test_drift_monitor.py
@pytest.fixture
def monitor() -> DriftMonitor:
    """Create a fresh DriftMonitor."""
    return DriftMonitor(threshold=0.15, num_bins=50)

# tests/integration/test_predict_endpoint.py
@pytest_asyncio.fixture
async def client():
    """Create an async test client with app state initialized."""
    model_manager = ModelManager()
    model_manager.load()
    app.state.model = model_manager
    app.state.start_time = time.time()
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
```

**Helper functions in conftest (not fixtures):**
```python
def step_through_setup(env: CatanEnv) -> None:
    """Helper: play through the entire setup phase with random valid actions."""
    while env.game_phase in (env.PHASE_SETUP_FIRST, env.PHASE_SETUP_SECOND):
        ...

def take_random_valid_action(env: CatanEnv) -> tuple[dict, float, bool, bool, dict]:
    """Take a single random valid action and return step result."""
    ...
```

## Mocking

**No mocking framework used.** The codebase does not use `unittest.mock`, `pytest-mock`, or any patching. Tests rely on:
- Real object instantiation with controlled inputs
- Seeded random number generators (`np.random.seed(42)`, `env.reset(seed=42)`)
- Direct state manipulation to set up test scenarios (e.g. `env.player_resources[0] = np.array([10, 10, 10, 10, 10])`)
- `pytest.importorskip("torch")` to skip torch-dependent tests gracefully when PyTorch is unavailable

## Test Data and Factories

**Private helper functions (not fixtures) generate reusable test data:**
```python
# tests/unit/test_policy.py
def _make_batch_inputs(batch_size: int = 1):
    """Generate random batched tensor inputs."""
    hex_feat = torch.randn(batch_size, 19, 9)
    ...
    return hex_feat, vert_feat, edge_feat, player_feat, current_player, action_mask

def _make_obs_dict():
    """Generate a numpy observation dict like CatanEnv produces."""
    return {
        "hex_features": np.random.randn(19, 9).astype(np.float32),
        ...
    }

# tests/unit/test_api_schemas.py
def _make_board_state() -> dict:
    """Create a minimal valid board state dict."""
    hexes = [{"type": "wood", "number": 5, "position": i} for i in range(19)]
    ...

# tests/unit/test_drift_monitor.py
def _make_data(n: int = 100, mean: float = 0.5, std: float = 0.1, seed: int = 42):
    """Generate synthetic feature data."""
    rng = np.random.default_rng(seed)
    ...
```

Convention: module-level private helpers are named `_make_*` and accept parameters for customization.

## Parametrize

Used for exhaustive encode/decode roundtrip testing across all action types:
```python
@pytest.mark.parametrize(
    "action_type,param",
    [
        ("ROLL_DICE", 0),
        ("END_TURN", 0),
        ("BUILD_ROAD", 0),
        ("BUILD_ROAD", 71),
        ...
    ],
)
def test_roundtrip(self, action_type: str, param: int):
    encoded = ActionSpace.encode_action(action_type, param)
    decoded_type, decoded_param = ActionSpace.decode_action(encoded)
    assert decoded_type == action_type
    assert decoded_param == param
```

## Async Testing

All FastAPI endpoint tests use `pytest-asyncio` with `@pytest.mark.asyncio` and `httpx.AsyncClient` via `ASGITransport` (no real network calls):

```python
@pytest.mark.asyncio
async def test_predict_valid_request(client: httpx.AsyncClient):
    body = _make_valid_predict_body()
    response = await client.post("/predict", json=body)
    assert response.status_code == 200
    data = response.json()
    assert "moves" in data
```

The `@pytest_asyncio.fixture` decorator (not `@pytest.fixture`) is required for async fixtures.

## Error Testing

`pytest.raises` used as a context manager for both Pydantic validation errors and domain errors:

```python
def test_invalid_player_index_too_high(self):
    with pytest.raises(ValidationError):
        PredictRequest(..., player_index=5, ...)

def test_empty_phases_raises(self):
    with pytest.raises(ValueError, match="at least one phase"):
        CurriculumScheduler(CurriculumConfig(phases=[]))
```

Use `match=` to assert the error message when the message is part of the contract.

## Test Types

**Unit Tests (`tests/unit/`):**
- Scope: Single class or module in isolation
- No external services, no network
- State manipulation via direct attribute assignment to reach specific scenarios
- Seeded RNG for deterministic randomness

**Integration Tests (`tests/integration/`):**
- `test_full_pipeline.py`: Tests the full env → GNN encoder → policy pipeline together using real objects
- `test_predict_endpoint.py`: Tests FastAPI routes end-to-end using `httpx.AsyncClient` and `ASGITransport`
- Integration tests instantiate `ModelManager` with a real (untrained fallback) model — no mocks

## Coverage

**Requirements:** No enforced minimum; coverage reported to terminal as part of `make test`.

**View Coverage:**
```bash
pytest tests/ -x --cov=src --cov-report=term-missing
pytest tests/ --cov=src --cov-report=html   # HTML report in htmlcov/
```

**Known coverage:** 61% (per commit history) across 140 unit + integration tests.

## Seeding and Determinism

All environment-based tests seed both numpy and the environment for determinism:
```python
np.random.seed(42)
env.reset(seed=42)
```

New tests should follow this pattern to prevent flakiness.

---

*Testing analysis: 2026-04-02*
