# CatanRL

Multi-agent reinforcement learning system for Settlers of Catan. A full production ML pipeline — from a custom Gymnasium environment and heterogeneous GNN policy to a FastAPI inference service, automated drift detection, and continuous retraining on AWS.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Neural Network Design](#neural-network-design)
- [Training Pipeline](#training-pipeline)
- [Strategy Layer](#strategy-layer)
- [Inference API](#inference-api)
- [Monitoring and Drift Detection](#monitoring-and-drift-detection)
- [Infrastructure](#infrastructure)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Testing](#testing)
- [Configuration](#configuration)

---

## Overview

CatanRL trains a 4-player Catan AI using Multi-Agent Proximal Policy Optimization (MAPPO) with self-play and curriculum scheduling. The trained policy is served via a FastAPI REST API deployed on AWS ECS Fargate. A background monitoring loop detects distribution drift in inference data and automatically triggers retraining via GitHub Actions.

**74% win rate vs random · 52% vs rule-based agents (25% baseline) · autonomous UCB1 hyperparameter search · drift-triggered AWS ECS retraining**

**Key design decisions:**

- Heterogeneous Graph Attention Network encodes the board as a typed graph (hexes, vertices, edges) rather than a flat feature vector
- Flat integer action space (261 actions) with boolean masking enforces legal moves without any special-casing in the policy
- Champion/challenger promotion pattern gates every model version before it reaches production
- Drift-triggered retraining closes the loop between production observations and training data
- Autonomous research agent uses UCB1 bandit selection to propose hyperparameter mutations, run self-play trials, and promote configs that improve win rate — continuously discovering better training configurations without manual tuning

---

## Architecture

The system is composed of six layers with strict boundaries:

```
┌─────────────────────────────────────────────────────────────────┐
│  Inference API  (src/api/)                                      │
│  FastAPI · Pydantic · ModelManager · champion.pt                │
├─────────────────────────────────────────────────────────────────┤
│  Strategy Layer  (src/strategy/)                                │
│  SHAP/Attention Explainer · Monte Carlo Planner · NL Templates  │
├─────────────────────────────────────────────────────────────────┤
│  Model Layer  (src/rl/models/)                                  │
│  CatanGNNEncoder · CatanPolicy · TradeModule                    │
├─────────────────────────────────────────────────────────────────┤
│  Training Layer  (src/rl/training/)                             │
│  MAPPOTrainer · SelfPlayManager · CurriculumScheduler           │
├─────────────────────────────────────────────────────────────────┤
│  Environment Layer  (src/rl/env/)                               │
│  CatanEnv (full Catan engine) · ActionSpace (261 actions)       │
├─────────────────────────────────────────────────────────────────┤
│  Monitoring Layer  (src/monitoring/)                            │
│  DriftMonitor · CloudWatchEmitter · RetrainingTrigger           │
└─────────────────────────────────────────────────────────────────┘
```

### Inference Request Flow

```
POST /predict
  → Pydantic validation (PredictRequest)
  → _request_to_obs(): board → numpy dict
      hex_features    (19, 9)   — 19 tiles, 6 type one-hot + desert flag + number token + robber
      vertex_features (54, 7)   — 54 intersections, building type + owner + port flags
      edge_features   (72, 5)   — 72 roads, road presence + owner
      player_features  (4, 14)  — 4 players, resource counts + VP + dev cards
  → phase-based action mask (bool[261])
  → ModelManager.predict()
      → CatanPolicy._encode(): GNN → 256-dim embedding
      → actor head: masked softmax over 261 logits
      → critic head: sigmoid → win probability
  → top-3 actions decoded + NL explanations generated
  → PredictResponse (moves, strategy_summary, win_prob, latency_ms, model_version)
```

### Automated Retraining Loop

```
DriftMonitor (every 6h via EventBridge)
  → JS-divergence on feature histograms > 0.15 threshold
  → RetrainingTrigger: POST repository_dispatch "drift-detected" to GitHub API
  → retrain.yml workflow:
      1. dispatch_training.py → Modal A10G GPU
      2. evaluate_and_promote.py → challenger vs champion (N games)
         win rate delta ≥ 0.02 → promote
      3. aws s3 cp models/champion.pt s3://catanrl-models/champion.pt
      4. ECS rolling deploy
```

---

## Neural Network Design

### Board Representation: Heterogeneous GNN

The Catan board is a typed graph with three node types and five relation types:

| Node Type | Count | Features |
|-----------|-------|---------|
| Hex | 19 | terrain type (6 one-hot), number token, robber flag |
| Vertex | 54 | building type, owner index, port flags |
| Edge | 72 | road presence, owner index |

| Relation Type | Direction |
|---------------|----------|
| hex → vertex | adjacency |
| vertex → hex | adjacency |
| vertex → vertex | neighbor |
| edge → vertex | endpoint |
| vertex → edge | endpoint |

**`CatanGNNEncoder`** (`src/rl/models/gnn_encoder.py`) applies three layers of `HeteroConv` using `GATConv` (Graph Attention Convolution) per relation type. Multi-head attention allows the encoder to learn which neighboring nodes are most relevant for each node type. The final embedding is a fixed-size vector regardless of board state.

```
Board graph → HeteroConv (layer 1) → HeteroConv (layer 2) → HeteroConv (layer 3) → pool → 256-dim embedding
```

Attention weights are exposed via `get_attention_weights()` for downstream explainability.

The encoder is constructed with `CatanGNNEncoder.from_env_defaults()`, which reads graph topology directly from `CatanEnv` arrays, keeping the encoder and environment graph definitions in sync.

### Policy Network: Shared Actor-Critic

**`CatanPolicy`** (`src/rl/models/policy.py`) uses parameter sharing — the GNN encoder feeds both the actor and critic:

```
obs_dict → CatanGNNEncoder → 256-dim embedding
                                ├── Actor MLP → 261 logits → masked softmax → action distribution
                                └── Critic MLP → scalar → sigmoid → win probability
```

The action mask is applied before softmax: masked-out logits are set to `-1e9`. This guarantees the policy assigns exactly zero probability to illegal actions without any policy gradient through the mask.

`get_action_and_value(obs, mask)` returns `(action, log_prob, entropy, value)` — the full tuple needed by PPO.

### Action Space: Flat Integer Encoding

**`ActionSpace`** (`src/rl/env/action_space.py`) encodes all 261 Catan actions as a flat integer range using class-level offset constants:

| Category | Count | Offset |
|----------|-------|--------|
| ROLL_DICE | 1 | 0 |
| END_TURN | 1 | 1 |
| BUILD_ROAD | 72 | 2 |
| BUILD_SETTLEMENT | 54 | 74 |
| BUILD_CITY | 54 | 128 |
| BUY_DEV_CARD | 1 | 182 |
| PLAY_DEV_CARD | 5 | 183 |
| BANK_TRADE | 20 | 188 |
| MARITIME_TRADE | 10 | 208 |
| MOVE_ROBBER | 19 | 218 |
| DISCARD | 24 | 237 |

`decode_action(id) → (type_str, param_int)` is used at inference time to convert model outputs into human-readable moves.

### Trade Module

**`TradeModule`** (`src/rl/models/trade_module.py`) is a specialized policy head for trade actions. It operates over the trade sub-space of the action distribution and can be composed with `CatanPolicy` for trade-specific reasoning.

---

## Training Pipeline

### MAPPO: Multi-Agent PPO

**`MAPPOTrainer`** (`src/rl/training/mappo.py`) implements the MAPPO algorithm:

1. **Rollout Collection**: `num_envs=8` environments stepped in parallel, collecting `(obs, action, reward, value, log_prob, done)` tuples into `RolloutBuffer`
2. **GAE**: Generalized Advantage Estimation computed over the collected rollout with configurable `gamma` and `lambda`
3. **PPO Update**: Clipped surrogate objective over `update_epochs` epochs with minibatch shuffling; entropy bonus encourages exploration
4. **Checkpointing**: Policy state dicts saved to `models/checkpoints/policy_update_N.pt` on a configurable interval
5. **MLflow Logging**: Metrics (`policy_loss`, `value_loss`, `entropy`, `approx_kl`) logged per update; graceful `ImportError` fallback if MLflow is unavailable

All hyperparameters live in the `MAPPOConfig` dataclass — no magic numbers in the training loop.

### Self-Play

**`SelfPlayManager`** (`src/rl/training/self_play.py`) maintains an opponent pool of past policy checkpoints:

- Pool capacity: configurable `pool_size` (default 10)
- Eviction: oldest checkpoint removed when pool is full
- Evaluation: current policy plays against all pool members; win rate tracked per opponent
- Promotion threshold: configurable win rate delta before advancing the pool

Self-play prevents strategy collapse by forcing the policy to generalize against a distribution of opponents rather than a fixed partner.

### Curriculum Scheduling

**`CurriculumScheduler`** (`src/rl/training/curriculum.py`) defines a sequence of `CurriculumPhase` objects, each specifying:

- `enable_trading` — whether bank/maritime trade actions are included
- `enable_dev_cards` — whether development cards are active
- `enable_robber` — whether the robber mechanic is active
- Advancement condition based on rolling average reward or win rate

Phases progress from simplified Catan (no trading, no robber) toward full rules, giving the policy a tractable learning signal before facing full game complexity.

---

## Strategy Layer

### Explainability

**`StrategyExplainer`** (`src/strategy/explainer.py`) provides post-hoc explanations of policy decisions using two complementary methods:

- **Attention Extraction**: Reads GATConv attention weights from `CatanGNNEncoder.get_attention_weights()` to identify which board tiles and settlements the policy is attending to
- **SHAP-based Attribution**: Computes gradient-based feature importance scores over the observation dict, attributing action probabilities back to input features

Explanations are structured as feature importance scores over the `hex_features`, `vertex_features`, `edge_features`, and `player_features` tensors.

### Monte Carlo Planner

**`MonteCarloPlanner`** (`src/strategy/planner.py`) runs lookahead rollouts from the current game state:

1. Deep-copy current `CatanEnv` state
2. Step the environment forward using the policy's action distribution for N steps
3. Average value estimates across M rollouts to rank candidate actions
4. Restore original environment state via `_restore_env`

This provides a lookahead capability beyond the policy's single-step action distribution, at the cost of additional inference calls.

### Natural Language Templates

**`ExplanationGenerator`** (`src/strategy/templates.py`) maps decoded action types and SHAP scores to human-readable strategy summaries using a template library. Used by the API to populate the `strategy_summary` field in `PredictResponse`.

---

## Inference API

Built with FastAPI, served via Uvicorn, containerized with Docker.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Main inference endpoint |
| `GET` | `/health` | Liveness/readiness check |
| `POST` | `/feedback` | Submit game outcome feedback |

### Request / Response

```python
# POST /predict
{
  "board_state": {
    "hexes":    [{"type": "wood", "number": 5, "position": 0}, ...],  # 19 hexes
    "vertices": [{"building": null, "owner": null, "port": null}, ...],  # 54 vertices
    "edges":    [{"road": false, "owner": null}, ...],  # 72 edges
  },
  "player_resources": [[3, 2, 1, 0, 1], ...],  # 4 players × 5 resources
  "player_index": 0,
  "game_phase": "main"
}

# Response
{
  "moves": [
    {"action_id": 74, "action_type": "BUILD_SETTLEMENT", "param": 0, "probability": 0.42},
    ...
  ],
  "strategy_summary": "Prioritize settlement at vertex 0 to secure wood and wheat income.",
  "win_probability": 0.31,
  "model_version": "champion-v3",
  "latency_ms": 14.2
}
```

### Model Loading

`ModelManager` (`src/api/model_loader.py`) resolves the model in priority order at startup:

1. Explicit path argument
2. `MODEL_PATH` environment variable
3. S3 download (`S3_MODEL_BUCKET` / `S3_MODEL_KEY`)
4. Untrained fallback (for development)

The loaded policy is stored in `app.state.model` via FastAPI's lifespan context and reused across all requests.

---

## Monitoring and Drift Detection

### Drift Monitor

**`DriftMonitor`** (`src/monitoring/drift_monitor.py`) tracks statistical drift in inference-time feature distributions:

- Maintains reference histograms from the training data distribution
- On each check, computes Jensen-Shannon divergence between reference and recent histograms using `scipy.spatial.distance.jensenshannon`
- Threshold: `0.15` JS divergence (configurable)
- Schedule: Every 6 hours via AWS EventBridge → Lambda

JS divergence is symmetric and bounded in `[0, 1]`, making it more stable than KL divergence for comparing empirical distributions.

### Retraining Trigger

When drift exceeds threshold, **`RetrainingTrigger`** (`src/monitoring/retraining_trigger.py`) fires a `repository_dispatch` event to GitHub using `urllib.request` (no `requests` dependency):

```
POST https://api.github.com/repos/{owner}/catanrl/dispatches
Authorization: Bearer {GITHUB_TOKEN}
{"event_type": "drift-detected"}
```

This triggers `.github/workflows/retrain.yml`.

### CloudWatch Metrics

**`CloudWatchEmitter`** (`src/monitoring/cloudwatch.py`) publishes to namespace `CatanRL`:

| Metric | Dimension | Description |
|--------|-----------|-------------|
| `InferenceLatencyMs` | `ModelVersion` | Per-request wall-clock latency |
| `InferenceConfidence` | `ModelVersion` | Policy confidence on selected action |
| `RequestCount` | — | Total inference requests |
| `JSDivergence` | — | Drift monitor score |

The emitter degrades gracefully: if `boto3` is unavailable or credentials are missing, it sets `self.enabled = False` and all emit calls become no-ops.

---

## Infrastructure

All infrastructure runs on AWS in `us-east-1`.

### Compute

- **ECS Fargate** — inference API container (cluster: `catanrl`, service: `catanrl-api`)
- **AWS Lambda** — drift monitor execution (triggered by EventBridge every 6 hours)
- **Modal** — GPU training (A10G via `scripts/dispatch_training.py`)

### Storage

- **ECR** — Docker image registry; images tagged with `github.sha`
- **S3** (`catanrl-models`) — model artifacts (`champion.pt`, `challenger.pt`)
- **MLflow** — experiment tracking during training (local `./mlruns` in dev, self-hosted in prod)

### CI/CD: Three Workflows

**`ci.yml`** — runs on PR and push to `main`:
```
ruff lint → mypy type check → pytest unit → pytest integration
```

**`deploy.yml`** — runs on push to `main`:
```
AWS OIDC auth → ECR login → Docker build (with GHA cache) → ECR push → ECS rolling update
```

**`retrain.yml`** — runs weekly (Sunday 02:00 UTC), on `workflow_dispatch`, or on `drift-detected`:
```
install deps → AWS OIDC auth → dispatch training (Modal GPU) → evaluate challenger vs champion
  → if win_rate_delta ≥ 0.02: promote challenger → s3 cp champion.pt → ECS rolling update
```

### Authentication

All GitHub Actions → AWS authentication uses **GitHub OIDC** (no long-lived credentials):

```json
// infra/iam-github-oidc.json
{
  "Principal": {"Federated": "token.actions.githubusercontent.com"},
  "Condition": {"StringLike": {"token.actions.githubusercontent.com:sub": "repo:*/catanrl:*"}}
}
```

The IAM role `catanrl-github-actions` is assumed by workflows; no `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` secrets are stored in GitHub.

---

## Project Structure

```
catanrl/
├── src/
│   ├── api/                    # FastAPI inference service
│   │   ├── main.py             # App factory, lifespan, middleware
│   │   ├── model_loader.py     # ModelManager: load → S3/local/fallback, predict
│   │   ├── schemas.py          # Pydantic v2 request/response models
│   │   └── routes/
│   │       ├── predict.py      # POST /predict
│   │       ├── health.py       # GET /health
│   │       └── feedback.py     # POST /feedback (SQLite persistence)
│   ├── rl/
│   │   ├── env/
│   │   │   ├── catan_env.py    # Full 4-player Catan game engine
│   │   │   └── action_space.py # Flat 261-action encoding + masking
│   │   ├── models/
│   │   │   ├── gnn_encoder.py  # Heterogeneous GAT board encoder
│   │   │   ├── policy.py       # Shared actor-critic (CatanPolicy)
│   │   │   └── trade_module.py # Trade-specific policy head
│   │   └── training/
│   │       ├── mappo.py        # MAPPOTrainer + MAPPOConfig + RolloutBuffer
│   │       ├── self_play.py    # OpponentPool + SelfPlayManager
│   │       └── curriculum.py   # CurriculumScheduler + phase definitions
│   ├── strategy/
│   │   ├── explainer.py        # SHAP/attention explainability
│   │   ├── planner.py          # Monte Carlo rollout planner
│   │   └── templates.py        # NL explanation templates
│   └── monitoring/
│       ├── cloudwatch.py       # AWS CloudWatch metrics emitter
│       ├── drift_monitor.py    # JS-divergence drift detection
│       └── retraining_trigger.py  # GitHub Actions repository_dispatch
├── tests/
│   ├── conftest.py             # Shared fixtures (env, action_space, helpers)
│   ├── unit/                   # 7 unit test modules (one per source module)
│   └── integration/            # Full pipeline + API endpoint tests
├── scripts/
│   ├── dispatch_training.py    # Launch training on Modal or locally
│   ├── evaluate_and_promote.py # Champion/challenger head-to-head evaluation
│   └── bootstrap_aws.sh        # One-time AWS infrastructure setup
├── infra/
│   ├── ecs-task-definition.json
│   ├── eventbridge-rule.json
│   └── iam-github-oidc.json
├── .github/workflows/
│   ├── ci.yml
│   ├── deploy.yml
│   └── retrain.yml
├── Dockerfile                  # Multi-stage build → python:3.10-slim
├── docker-compose.yml          # api + mlflow (port 5000) + drift_monitor
├── pyproject.toml              # Project metadata, ruff, mypy, pytest config
├── requirements.txt
└── requirements-dev.txt
```

---

## Setup

### Local (Python)

```bash
# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run the API
uvicorn src.api.main:app --reload

# Train locally (no GPU)
python scripts/dispatch_training.py --local

# Train on Modal GPU
python scripts/dispatch_training.py --gpu A10G

# Evaluate and promote challenger
python scripts/evaluate_and_promote.py
```

### Local (Docker Compose)

```bash
docker compose up
# API:    http://localhost:8000
# MLflow: http://localhost:5000
```

### AWS Bootstrap

```bash
# One-time infrastructure setup
bash scripts/bootstrap_aws.sh
```

---

## Testing

```bash
# Full test suite with coverage
pytest tests/ -x --cov=src --cov-report=term-missing

# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Via Makefile
make test
```

**140 tests, 61% coverage** across unit and integration suites.

Tests do not use mocks. All tests use real object instantiation with seeded randomness (`env.reset(seed=42)`, `np.random.seed(42)`). Integration tests wire the real `ModelManager` (with untrained fallback policy) against the live FastAPI app via `httpx.AsyncClient` + `ASGITransport` — no network required.

---

## Configuration

All runtime configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/app/models/champion.pt` | Path to champion model file |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region for CloudWatch and S3 |
| `GITHUB_REPO` | `owner/catanrl` | `owner/repo` for retraining dispatch |
| `GITHUB_TOKEN` | — | GitHub PAT (`actions:write` scope) for retraining dispatch |
| `S3_MODEL_BUCKET` | — | S3 bucket name for model artifact download |
| `S3_MODEL_KEY` | — | S3 key for champion model |

GitHub Actions secrets required: `AWS_ACCOUNT_ID`, `ECR_REGISTRY`.

Training hyperparameters are configured via the `MAPPOConfig` dataclass in `src/rl/training/mappo.py`. Curriculum phases are defined in `src/rl/training/curriculum.py`.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.1.0 | Neural network training and inference |
| `torch-geometric` | ≥2.4.0 | Heterogeneous GNN (`HeteroConv`, `GATConv`) |
| `gymnasium` | ≥0.29.0 | RL environment interface |
| `fastapi` | ≥0.104.0 | Inference REST API |
| `uvicorn[standard]` | ≥0.24.0 | ASGI server |
| `pydantic` | ≥2.5.0 | API schema validation |
| `scipy` | ≥1.11.0 | Jensen-Shannon divergence for drift detection |
| `mlflow` | ≥2.9.0 | Experiment tracking (optional) |
| `boto3` | ≥1.29.0 | AWS S3 and CloudWatch (optional) |
| `structlog` | ≥23.2.0 | Structured JSON logging |
| `numpy` | ≥1.24.0 | Array operations throughout |
