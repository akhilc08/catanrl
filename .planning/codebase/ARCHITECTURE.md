# Architecture

**Analysis Date:** 2026-04-02

## Pattern Overview

**Overall:** Multi-tier ML system — Gymnasium-based RL training pipeline feeding a FastAPI inference service, with an autonomous monitoring and retraining loop.

**Key Characteristics:**
- Strict separation between training (offline) and inference (online serving)
- GNN-based observation encoding shared between actor and critic heads (parameter sharing)
- Flat integer action space (261 actions) with boolean masking for legal moves
- Champion/challenger promotion pattern for model lifecycle
- Drift-triggered automated retraining via GitHub Actions repository_dispatch

## Layers

**Environment Layer:**
- Purpose: Full Gymnasium-compatible Catan game simulation
- Location: `src/rl/env/`
- Contains: `CatanEnv` (game engine), `ActionSpace` (flat 261-action encoding with masking)
- Depends on: numpy, gymnasium
- Used by: training loop, evaluation scripts, inference server (action mask decoding)

**Model Layer:**
- Purpose: Neural network components for policy and value estimation
- Location: `src/rl/models/`
- Contains: `CatanGNNEncoder` (3-layer heterogeneous GAT), `CatanPolicy` (actor-critic), `TradeModule`
- Depends on: torch, torch_geometric
- Used by: training, inference API, strategy explainer, Monte Carlo planner

**Training Layer:**
- Purpose: MAPPO training loop with self-play and curriculum scheduling
- Location: `src/rl/training/`
- Contains: `MAPPOTrainer` + `MAPPOConfig` + `RolloutBuffer` (mappo.py), `SelfPlayManager` (self_play.py), `CurriculumScheduler` (curriculum.py)
- Depends on: model layer, environment layer, MLflow, torch
- Used by: `scripts/dispatch_training.py`, GitHub Actions retrain workflow

**API Layer:**
- Purpose: FastAPI inference service with model lifecycle management
- Location: `src/api/`
- Contains: `main.py` (app + middleware), `ModelManager` (model_loader.py), Pydantic schemas, route handlers
- Depends on: model layer, environment layer (action decoding), boto3 (S3 model loading)
- Used by: ECS deployment, external callers

**Strategy Layer:**
- Purpose: Post-hoc explainability and multi-step planning on top of the policy
- Location: `src/strategy/`
- Contains: `StrategyExplainer` (SHAP/attention extraction), `MonteCarloPlanner` (rollout-based evaluation), `ExplanationGenerator` + NL templates (templates.py)
- Depends on: model layer, environment layer
- Used by: API (optional enrichment), external callers

**Monitoring Layer:**
- Purpose: Runtime observability and drift detection feeding back into retraining
- Location: `src/monitoring/`
- Contains: `DriftMonitor` (JS-divergence on feature distributions), `CloudWatchEmitter` (AWS CloudWatch metrics), `RetrainingTrigger` (GitHub Actions repository_dispatch)
- Depends on: scipy, boto3, structlog
- Used by: API (can be wired per-request), retraining workflow

## Data Flow

**Inference Request:**
1. HTTP POST `/predict` arrives at `src/api/routes/predict.py`
2. `PredictRequest` (Pydantic) validated from JSON body
3. `_request_to_obs()` converts board state into numpy observation dict with shapes `hex_features(19,9)`, `vertex_features(54,7)`, `edge_features(72,5)`, `player_features(4,14)`
4. Phase-based action mask (`np.bool_[261]`) computed from `game_phase`
5. `ModelManager.predict()` passes obs + mask through `CatanPolicy._encode()` (GNN) → actor head (masked softmax) + critic head (sigmoid → win prob)
6. Top-3 action IDs decoded via `ActionSpace.decode_action()`, human explanations generated
7. `PredictResponse` returned with moves, strategy summary, win probability, model version, latency

**Training Pipeline:**
1. `scripts/dispatch_training.py` instantiates `MAPPOConfig`, `CatanGNNEncoder`, `CatanPolicy`, `MAPPOTrainer`
2. `MAPPOTrainer` runs parallel environments (`num_envs=8`), collects rollouts into `RolloutBuffer`
3. GAE computed; PPO clipped updates applied over `update_epochs` minibatch epochs
4. `SelfPlayManager` periodically evaluates and manages opponent pool checkpoints
5. `CurriculumScheduler` advances phases (random → heuristic → self-play opponents)
6. Checkpoints saved to `models/checkpoints/`

**Retraining Loop:**
1. `DriftMonitor` detects JS-divergence above threshold in feature distributions
2. `RetrainingTrigger` POSTs `repository_dispatch` event `drift-detected` to GitHub API
3. `.github/workflows/retrain.yml` triggers: dispatch training → evaluate challenger vs champion → promote if win rate delta ≥ 0.02 → upload champion to S3 → rolling ECS deploy

**Model Loading (API startup):**
1. `ModelManager.load()` resolves model in priority order: explicit path → `MODEL_PATH` env var → S3 (`S3_MODEL_BUCKET`/`S3_MODEL_KEY`) → untrained fallback
2. Loaded policy stored in `app.state.model` (FastAPI lifespan context)

## Key Abstractions

**CatanEnv:**
- Purpose: Full 4-player Catan game as a Gymnasium environment
- Examples: `src/rl/env/catan_env.py`
- Pattern: Standard `reset(seed) → obs_dict`, `step(action) → (obs, reward, terminated, truncated, info)`, `get_action_mask() → np.bool_[261]`

**ActionSpace:**
- Purpose: Flat integer encoding of all 261 Catan actions with encode/decode/masking
- Examples: `src/rl/env/action_space.py`
- Pattern: Class-level offset constants (e.g., `BUILD_ROAD_OFFSET = 2`); `decode_action(id) → (type_str, param_int)`

**CatanGNNEncoder:**
- Purpose: Heterogeneous GAT over board graph (hex/vertex/edge node types, 5 relation types)
- Examples: `src/rl/models/gnn_encoder.py`
- Pattern: `from_env_defaults()` factory; outputs fixed-size embedding; exposes `get_attention_weights()` for explainability

**CatanPolicy:**
- Purpose: Shared-parameter actor-critic; GNN encoder + actor MLP (261 logits) + critic MLP (scalar value)
- Examples: `src/rl/models/policy.py`
- Pattern: `_encode(obs_dict) → embedding`; `get_action_and_value(obs, mask) → (action, log_prob, entropy, value)`

**ModelManager:**
- Purpose: Model lifecycle: load (local/S3/fallback), inference, version tracking
- Examples: `src/api/model_loader.py`
- Pattern: `load()` then `predict(obs_dict, action_mask) → ([(action_id, prob)], win_prob)`

**MAPPOConfig / MAPPOTrainer:**
- Purpose: Hyperparameter container and MAPPO training loop with GAE and clipped PPO
- Examples: `src/rl/training/mappo.py`
- Pattern: `MAPPOTrainer(config, policy).train() → metrics_dict`

## Entry Points

**API Server:**
- Location: `src/api/main.py`
- Triggers: `uvicorn src.api.main:app` (Docker/ECS)
- Responsibilities: Loads model on startup, attaches middleware (request ID, latency logging), mounts routers for `/predict`, `/health`, `/feedback`

**Training Script:**
- Location: `scripts/dispatch_training.py`
- Triggers: CLI (`python scripts/dispatch_training.py`), GitHub Actions retrain workflow
- Responsibilities: Dispatch training to Modal GPU or run locally; creates `MAPPOTrainer` and calls `.train()`

**Evaluation/Promotion Script:**
- Location: `scripts/evaluate_and_promote.py`
- Triggers: CLI, GitHub Actions retrain workflow (after dispatch)
- Responsibilities: Head-to-head evaluation of challenger vs champion over N games; promotes if win rate delta ≥ threshold; archives previous champion

**Retraining Workflow:**
- Location: `.github/workflows/retrain.yml`
- Triggers: Weekly cron (Sunday 02:00 UTC), `workflow_dispatch`, `repository_dispatch` event `drift-detected`
- Responsibilities: Full retrain + evaluate + promote + S3 upload + ECS rolling deploy

**Deploy Workflow:**
- Location: `.github/workflows/deploy.yml`
- Triggers: Push to `main`
- Responsibilities: Build Docker image → push to ECR → update ECS service

## Error Handling

**Strategy:** Graceful degradation at every integration boundary.

**Patterns:**
- `ModelManager.load()` falls back through local → S3 → untrained dev policy rather than failing at startup
- `CloudWatchEmitter.__init__` catches ImportError and boto3 exceptions; sets `self.enabled = False` — callers check flag before emitting
- `RetrainingTrigger.trigger()` returns `bool` rather than raising; logs errors via structlog
- Unknown game phases in `_request_to_obs()` enable full action mask as safe fallback
- API middleware catches unhandled exceptions implicitly via FastAPI; request ID injected via middleware for traceability

## Cross-Cutting Concerns

**Logging:** structlog throughout API and monitoring layers (`structlog.get_logger()`); standard `logging` in training layer
**Validation:** Pydantic v2 schemas at API boundary (`src/api/schemas.py`); gymnasium spaces enforce observation/action shapes at env boundary
**Authentication:** None — API is unauthenticated; AWS access uses GitHub OIDC role assumption (no long-lived credentials)

---

*Architecture analysis: 2026-04-02*
