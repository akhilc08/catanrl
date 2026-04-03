# Codebase Concerns

**Analysis Date:** 2026-04-02

## Tech Debt

**autoResearch module is a stub:**
- Issue: `src/autoResearch/__init__.py` is the only file in the module — it is empty (1 line). The commit history advertises "AutoResearch orchestrator, Modal dispatcher, research directions" but no implementation exists.
- Files: `src/autoResearch/__init__.py`
- Impact: Any code that imports from `src.autoResearch` gets nothing. Feature is completely unimplemented.
- Fix approach: Implement the orchestrator and Modal dispatcher, or remove the package and revert commit claims.

**Curriculum environment flags are declared but not enforced:**
- Issue: `CurriculumPhase` defines `enable_trading`, `enable_dev_cards`, and `enable_robber` flags, but `CatanEnv` has no corresponding constructor parameters to honour them. The scheduler passes a phase to callers but there is no mechanism in the env to restrict mechanics based on those flags.
- Files: `src/rl/training/curriculum.py` (lines 46-55), `src/rl/env/catan_env.py`
- Impact: Curriculum phases 1-N all run full Catan regardless of declared restrictions, making early curriculum stages equivalent to the final stage. Training signal in early phases is not simplified as intended.
- Fix approach: Add `enable_trading`, `enable_dev_cards`, `enable_robber` parameters to `CatanEnv.__init__` and gate the relevant action mask entries and reward handlers behind them.

**Duplicate `get_current_phase` / `current_phase` accessors:**
- Issue: `CurriculumScheduler` exposes both a `@property current_phase` and a method `get_current_phase()` that does nothing but call the property.
- Files: `src/rl/training/curriculum.py` (lines 112-118)
- Impact: Minor dead code / API confusion.
- Fix approach: Remove `get_current_phase()`.

**MAPPO training logs to stdout via `print()` instead of structured logging:**
- Issue: All progress output in `MAPPOTrainer.train()` and `SelfPlayTrainer.train()` uses bare `print()` calls rather than the `structlog` logger used everywhere else.
- Files: `src/rl/training/mappo.py` (lines 564-587), `src/rl/training/self_play.py` (lines 492-526)
- Impact: Training output cannot be captured by the structured log pipeline or routed to CloudWatch.
- Fix approach: Replace `print()` calls with `logger.info(...)` using structured key-value pairs.

**`torch.load` uses `weights_only=False`:**
- Issue: `ModelManager._load_checkpoint()` calls `torch.load(path, weights_only=False)`. This allows arbitrary code execution via pickle if the checkpoint file is replaced.
- Files: `src/api/model_loader.py` (line 70)
- Impact: Supply-chain or S3-write attack on the model artifact can execute arbitrary code on the inference server.
- Fix approach: Use `weights_only=True` and migrate checkpoints to the safe serialization format, or explicitly validate checksum before loading.

**`mypy` configured with `disallow_untyped_defs = false`:**
- Issue: `pyproject.toml` sets `disallow_untyped_defs = false` and `warn_return_any = false`, severely limiting the type safety guarantees that mypy would otherwise provide.
- Files: `pyproject.toml` (lines 17-20)
- Impact: Many type errors will pass CI silently, especially in `src/strategy/explainer.py` where `policy` and `env` parameters are untyped.
- Fix approach: Gradually enable `disallow_untyped_defs = true` per module using per-module mypy overrides.

## Known Bugs

**`_roller_player` returns wrong player during discard phase:**
- Symptoms: When a 7 is rolled and multiple players must discard, control passes to each discarding player. The `_roller_player` property computes `(turn_counter - 1) % num_players`, which is correct only if no mid-turn player switching has happened. After discard sub-phases cycle through multiple players, `current_player` is restored using `_roller_player`, but that value may be stale if `turn_counter` was advanced.
- Files: `src/rl/env/catan_env.py` (lines 631-633, 768-774)
- Trigger: Roll a 7 when 2+ players have >7 resources; the active player after discard ends will sometimes be the wrong player.
- Workaround: None — games may proceed with an incorrect current player.

**Hex feature indexing redundancy / off-by-one in `_get_observation`:**
- Symptoms: `hex_feat[hi, 5]` is set when `ht == DESERT` (DESERT = 5), but column 5 of the one-hot encoding is already the DESERT class slot (indices 0-5 for WOOD/BRICK/SHEEP/WHEAT/ORE/DESERT). Column 6 is then set again as an explicit desert flag, duplicating information and creating an observation shape mismatch comment ("6 type one-hot + desert flag + number + robber = 9") vs actual usage.
- Files: `src/rl/env/catan_env.py` (lines 940-951)
- Trigger: Always — every observation produced by the environment carries this inconsistency.
- Workaround: Model still trains, but the observation encoding wastes a feature dimension and may confuse the encoder.

**`_find_next_discard_player` iterates from `after = -1` at start:**
- Symptoms: Called with `after = -1` on first discard; `(−1 + 1) % 4 = 0`, which works by accident. However the function will skip player 0 if they need to discard and the roller is also player 0, because `after` is not the roller index — it is initialized as `-1` unconditionally.
- Files: `src/rl/env/catan_env.py` (line 572)
- Trigger: Player 0 rolls a 7 and also has >7 cards.

**Bank trade allows self-trade (give and receive same resource):**
- Symptoms: `BANK_TRADE` encodes 20 actions as `give_res * 4 + get_res`. With 5 resources and 4 choices this actually encodes 20 pairs, but the encoding does not exclude `give_res == get_res` for all combinations. The action mask does not validate that `give_res != get_res`.
- Files: `src/rl/env/action_space.py` (BANK_TRADE section), `src/rl/env/catan_env.py` `_handle_bank_trade`
- Trigger: Agent selects the degenerate trade action; resources are deducted and re-added at a loss (ratio 4).

## Security Considerations

**Unauthenticated inference and feedback endpoints:**
- Risk: `/predict` and `/feedback` routes have no authentication or rate limiting. Any client can call the inference API or write to the feedback database.
- Files: `src/api/main.py`, `src/api/routes/predict.py`, `src/api/routes/feedback.py`
- Current mitigation: None.
- Recommendations: Add API key middleware or OAuth2 dependency; add rate limiting via `slowapi` or a reverse proxy; restrict `/feedback` writes.

**SQLite feedback database at computed absolute path:**
- Risk: `_DB_PATH` is resolved via `Path(__file__).resolve().parent.parent.parent.parent / "data" / "feedback.db"`. This path escapes the `src/` subtree and writes to the project root. In a container or read-only filesystem this will fail silently. In a shared host it may conflict with other services.
- Files: `src/api/routes/feedback.py` (line 18)
- Current mitigation: None.
- Recommendations: Make the database path configurable via an environment variable (e.g. `FEEDBACK_DB_PATH`); validate writability at startup.

**No input size validation on `PredictRequest`:**
- Risk: `BoardState.hexes`, `vertices`, `edges` are `list[...]` with no `max_items` constraint. A malicious client can send lists with thousands of items, causing out-of-memory or excessively slow numpy stack operations.
- Files: `src/api/schemas.py`, `src/api/routes/predict.py`
- Current mitigation: None.
- Recommendations: Add `max_length` constraints to all list fields in `BoardState` (hexes=19, vertices=54, edges=72).

**S3 model download to `/tmp` with no integrity check:**
- Risk: After downloading `policy.pt` from S3, the file is loaded without any checksum verification. A compromised S3 bucket can deliver a malicious pickle.
- Files: `src/api/model_loader.py` (lines 46-57)
- Current mitigation: None.
- Recommendations: Store and verify a SHA-256 digest alongside the model artifact in S3 before loading.

**GitHub token scope not limited:**
- Risk: `RetrainingTrigger` reads `GITHUB_TOKEN` from the environment and uses it to dispatch workflow events. If `GITHUB_TOKEN` has broad repository scope it could be used to trigger arbitrary workflows.
- Files: `src/monitoring/retraining_trigger.py` (line 31)
- Current mitigation: Token absence causes a logged error and no action.
- Recommendations: Document that only `actions:write` scope is needed; validate the token is not a broad PAT.

## Performance Bottlenecks

**`_calc_longest_road` runs DFS from every player edge on every road or settlement build:**
- Problem: `_update_longest_road` calls `_calc_longest_road` for every player, which in turn runs a DFS starting from every edge in that player's road set. In the worst case (15 roads × 4 players) this is O(60 × E) per build action.
- Files: `src/rl/env/catan_env.py` (lines 855-913)
- Cause: No incremental update or early-exit; DFS re-runs from scratch after every road or settlement placement.
- Improvement path: Cache the longest road length per player and only recompute for players whose road network changed.

**Board topology generated on every `CatanGNNEncoder` device move:**
- Problem: `_get_edge_indices()` caches topology per device, but the cache is per-instance, not process-global. When multiple encoder instances exist (e.g. during self-play evaluation with freshly instantiated opponent policies), each new instance recomputes topology and moves tensors to the device.
- Files: `src/rl/models/gnn_encoder.py` (lines 225-249)
- Cause: `_TOPOLOGY_CACHE` is process-global but `_edge_indices` is instance-level.
- Improvement path: Move the device-mapped cache to the module level, keyed by device string.

**Sequential environment stepping during rollout collection:**
- Problem: `MAPPOTrainer.collect_rollout()` steps each of the `num_envs` environments in a Python `for` loop. There is no parallelism; environments are CPU-bound Python objects.
- Files: `src/rl/training/mappo.py` (lines 282-307)
- Cause: `CatanEnv` is a plain Python class with no multiprocessing wrapper.
- Improvement path: Wrap environments with `gymnasium.vector.AsyncVectorEnv` or `SyncVectorEnv` to enable batched stepping.

**`_request_to_obs` iterates Python lists for every inference call:**
- Problem: The predict route reconstructs numpy feature arrays by iterating Pydantic model lists (hexes, vertices, edges) in Python loops on every request.
- Files: `src/api/routes/predict.py` (lines 83-130)
- Cause: No caching of the board layout; full reconstruction per request.
- Improvement path: Accept pre-encoded numpy arrays directly, or vectorize the construction using numpy indexing.

## Fragile Areas

**`CatanEnv` game state is entirely mutable public attributes:**
- Files: `src/rl/env/catan_env.py`
- Why fragile: All game state (`vertex_building`, `edge_road`, `player_resources`, etc.) is exposed as public numpy arrays with no encapsulation. Any code with an env reference can corrupt state silently. `MonteCarloPlanner` depends on `copy.deepcopy` of environment state dicts for rollout isolation, which will silently break if new attributes are added without updating `_restore_env`.
- Safe modification: Treat `catan_env.py` as a state machine — only mutate state through the `step()` path. When adding attributes, update `MonteCarloPlanner._restore_env` in `src/strategy/planner.py`.
- Test coverage: `tests/unit/test_catan_env.py` covers main game paths but not all discard/robber edge cases.

**`OpponentPool._find_entry` uses identity comparison (`is`) on state dicts:**
- Files: `src/rl/training/self_play.py` (lines 310-315)
- Why fragile: Pool entries are deep-copied on insertion (line 154), but `_find_entry` searches by object identity. After any serialization/deserialization round-trip or copy, the identity will not match and `opp_entry` will always be `None`, silently skipping stat updates.
- Safe modification: Key entries by a deterministic hash of the state dict or an explicit integer ID.
- Test coverage: No tests for `_find_entry` identity matching.

**`_DB_INITIALIZED` is a module-level boolean:**
- Files: `src/api/routes/feedback.py` (line 19)
- Why fragile: Under any multi-worker ASGI deployment (e.g. `uvicorn --workers 4`) each process has its own `_DB_INITIALIZED = False`. The `CREATE TABLE IF NOT EXISTS` is idempotent so this is safe in SQLite, but the flag provides a false sense of "initialized once" semantics. Concurrent writes from multiple workers to a single SQLite file will cause lock contention.
- Safe modification: Replace SQLite with a proper database (PostgreSQL) or use a connection pool with WAL mode enabled.
- Test coverage: Not tested under concurrent access.

**`action_mask` cast to `np.int8` in observation, but consumed as `np.bool_` in policy:**
- Files: `src/rl/env/catan_env.py` (line 995), `src/api/model_loader.py` (line 111)
- Why fragile: The env returns `action_mask.astype(np.int8)`, but `ModelManager.predict` calls `torch.from_numpy(action_mask).float()` which works for int8. However `_request_to_obs` in predict route creates the mask as `np.bool_` (line 135), while the training rollout code uses `.float()` on the mask tensor. These different dtypes produce valid results but the inconsistency is a source of subtle bugs if dtype assumptions change.
- Safe modification: Standardize on a single dtype (recommend `np.bool_`) in the env and document the expected input dtype in `ModelManager.predict`.

## Scaling Limits

**SQLite for feedback storage:**
- Current capacity: Single-file SQLite, handles low single-digit concurrent writes.
- Limit: SQLite write lock contention degrades rapidly above ~10 concurrent writes/second. Unsuitable if the inference API is horizontally scaled.
- Scaling path: Migrate to PostgreSQL (or DynamoDB for AWS-native deployment); update `feedback.py` to use an async driver (e.g. `asyncpg`).

**In-memory opponent pool:**
- Current capacity: `pool_size` defaults to 10 checkpoints held in process memory. Each CatanPolicy checkpoint is ~10MB+ depending on hidden dim.
- Limit: A pool of 10 checkpoints × ~10MB = ~100MB baseline, growing with model size. Not shared across training processes.
- Scaling path: Persist pool state to S3/disk and load on demand; use a shared checkpoint registry for distributed training.

**Drift monitor `recent_data` list grows unbounded:**
- Current capacity: `DriftMonitor.recent_data` is a plain Python list with no max length.
- Limit: In a long-running process, every inference call appended via `add_observation()` accumulates forever, causing unbounded memory growth.
- Scaling path: Replace with a fixed-size circular buffer (`collections.deque(maxlen=N)`).
- Files: `src/monitoring/drift_monitor.py` (line 113)

## Dependencies at Risk

**`torch-geometric` (PyG) version pinning is loose:**
- Risk: `requirements.txt` specifies `torch-geometric>=2.4.0` with no upper bound. PyG frequently changes its `HeteroConv` and `GATConv` APIs between minor versions. The GNN encoder uses internal PyG APIs directly.
- Impact: A `pip install --upgrade` can silently break `CatanGNNEncoder.forward()`.
- Files: `requirements.txt`, `src/rl/models/gnn_encoder.py`
- Migration plan: Pin to an exact version or a tight range (`>=2.4.0,<2.6.0`); add a compatibility test in CI.

**`mlflow` is an optional but untested soft dependency:**
- Risk: `requirements.txt` includes `mlflow>=2.9.0` but the training code catches `ImportError` and silently disables logging. If mlflow is present but at the wrong version, calls like `mlflow.log_params` may fail with opaque errors mid-training.
- Files: `src/rl/training/mappo.py` (lines 492-517), `requirements.txt`
- Migration plan: Move mlflow to `requirements-dev.txt` or add explicit version validation on import.

## Missing Critical Features

**No authentication on the inference API:**
- Problem: The FastAPI app has no auth middleware, API key validation, or rate limiting.
- Blocks: Cannot safely expose the service publicly without risk of abuse or resource exhaustion.

**No model versioning or rollback mechanism:**
- Problem: `ModelManager` loads a single champion checkpoint. There is no registry of deployed versions, no A/B capability, and no rollback if a newly promoted model regresses.
- Blocks: Safe continuous deployment of retrained models.

**`MonteCarloPlanner` cannot restore arbitrary env state:**
- Problem: `_restore_env` in `src/strategy/planner.py` reconstructs `CatanEnv` from a state dict, but `CatanEnv` has no `get_state()` / `set_state()` API. The planner must manually enumerate every attribute — any attribute added to the env silently breaks rollout isolation.
- Files: `src/strategy/planner.py`
- Blocks: Reliable lookahead planning.

## Test Coverage Gaps

**Self-play trainer has no unit tests:**
- What's not tested: `SelfPlayTrainer.train()`, `OpponentPool.evaluate_against_pool()`, `_play_one_game()`, and `_select_action()`.
- Files: `src/rl/training/self_play.py`
- Risk: Bugs in pool eviction logic, win-rate threshold gating, or game simulation run undetected.
- Priority: High

**`MonteCarloPlanner` and `StrategyExplainer` have no tests:**
- What's not tested: Any of `src/strategy/planner.py` or `src/strategy/explainer.py`.
- Files: `src/strategy/planner.py`, `src/strategy/explainer.py`
- Risk: Strategy explanation and planning features may silently produce incorrect results without failing.
- Priority: High

**`RetrainingTrigger` and `CloudWatchEmitter` have no tests:**
- What's not tested: `src/monitoring/retraining_trigger.py`, `src/monitoring/cloudwatch.py`
- Risk: Broken monitoring pipeline is not caught in CI; drift-triggered retraining could silently fail.
- Priority: Medium

**MAPPO training loop has no tests:**
- What's not tested: `MAPPOTrainer.collect_rollout()`, `MAPPOTrainer.update()`, `MAPPOTrainer.train()`.
- Files: `src/rl/training/mappo.py`
- Risk: Gradient computation errors, GAE bugs, or checkpoint-saving failures go undetected.
- Priority: High

**No integration test for the feedback endpoint persistence:**
- What's not tested: Whether submitted feedback is actually persisted and readable from the SQLite database.
- Files: `tests/integration/test_predict_endpoint.py` (covers predict only), `src/api/routes/feedback.py`
- Risk: Feedback data could be silently lost (e.g. path resolution failure) without a CI failure.
- Priority: Medium

---

*Concerns audit: 2026-04-02*
