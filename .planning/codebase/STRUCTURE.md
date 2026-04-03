# Codebase Structure

**Analysis Date:** 2026-04-02

## Directory Layout

```
catanrl/
├── src/                        # All application source code
│   ├── api/                    # FastAPI inference service
│   │   ├── main.py             # App factory, lifespan, middleware
│   │   ├── model_loader.py     # ModelManager: load/predict lifecycle
│   │   ├── schemas.py          # Pydantic request/response models
│   │   └── routes/             # FastAPI routers
│   │       ├── predict.py      # POST /predict (main inference endpoint)
│   │       ├── health.py       # GET /health
│   │       └── feedback.py     # POST /feedback
│   ├── rl/                     # Reinforcement learning core
│   │   ├── env/                # Gymnasium environment
│   │   │   ├── catan_env.py    # Full 4-player Catan game engine (~42KB)
│   │   │   └── action_space.py # Flat 261-action encoding + masking
│   │   ├── models/             # Neural network components
│   │   │   ├── gnn_encoder.py  # Heterogeneous GAT board encoder
│   │   │   ├── policy.py       # Actor-critic policy (CatanPolicy)
│   │   │   └── trade_module.py # Trade-specific policy module
│   │   └── training/           # Training algorithms
│   │       ├── mappo.py        # MAPPO trainer + config + rollout buffer
│   │       ├── self_play.py    # Opponent pool management
│   │       └── curriculum.py   # Progressive curriculum scheduler
│   ├── strategy/               # Explainability and planning
│   │   ├── explainer.py        # SHAP/attention-based explanations
│   │   ├── planner.py          # Monte Carlo rollout planner
│   │   └── templates.py        # NL explanation templates + generator
│   ├── monitoring/             # Runtime observability
│   │   ├── cloudwatch.py       # AWS CloudWatch metrics emitter
│   │   ├── drift_monitor.py    # JS-divergence feature drift detection
│   │   └── retraining_trigger.py # GitHub Actions repository_dispatch
│   └── autoResearch/           # AutoResearch orchestrator (stub)
│       └── __init__.py
├── tests/                      # All tests
│   ├── conftest.py             # Shared fixtures (env, action_space, helpers)
│   ├── unit/                   # Unit tests (7 modules)
│   │   ├── test_catan_env.py
│   │   ├── test_action_space.py
│   │   ├── test_gnn_encoder.py
│   │   ├── test_policy.py
│   │   ├── test_curriculum.py
│   │   ├── test_drift_monitor.py
│   │   └── test_api_schemas.py
│   └── integration/            # Integration tests (2 modules)
│       ├── test_full_pipeline.py
│       └── test_predict_endpoint.py
├── scripts/                    # Operational scripts (executable)
│   ├── dispatch_training.py    # Launch training (Modal or local)
│   ├── evaluate_and_promote.py # Champion/challenger evaluation
│   └── bootstrap_aws.sh        # AWS infrastructure bootstrap
├── infra/                      # AWS infrastructure definitions
│   ├── ecs-task-definition.json
│   ├── eventbridge-rule.json
│   └── iam-github-oidc.json
├── .github/
│   └── workflows/
│       ├── ci.yml              # CI: lint + test on PR
│       ├── deploy.yml          # CD: push to ECR + ECS on main push
│       └── retrain.yml         # Retrain: weekly/drift-triggered pipeline
├── Dockerfile                  # Container image for API service
├── docker-compose.yml          # Local dev stack
├── pyproject.toml              # Project metadata + tool config
├── requirements.txt            # Runtime dependencies
└── requirements-dev.txt        # Dev/test dependencies
```

## Directory Purposes

**`src/api/`:**
- Purpose: FastAPI HTTP inference service
- Contains: App entry point, model lifecycle, Pydantic schemas, route handlers
- Key files: `src/api/main.py`, `src/api/model_loader.py`, `src/api/schemas.py`

**`src/rl/env/`:**
- Purpose: Game simulation — the environment all agents interact with
- Contains: Full Catan rules engine, action encoding
- Key files: `src/rl/env/catan_env.py` (authoritative game state), `src/rl/env/action_space.py` (action constants/decode)

**`src/rl/models/`:**
- Purpose: Neural network architecture
- Contains: GNN encoder, policy network, trade module
- Key files: `src/rl/models/gnn_encoder.py`, `src/rl/models/policy.py`

**`src/rl/training/`:**
- Purpose: Training algorithms and scheduling
- Contains: MAPPO loop, self-play opponent pool, curriculum phases
- Key files: `src/rl/training/mappo.py` (primary), `src/rl/training/self_play.py`, `src/rl/training/curriculum.py`

**`src/strategy/`:**
- Purpose: Post-hoc analysis tools layered on top of the trained policy
- Contains: Attention/gradient explainability, Monte Carlo lookahead, NL templates
- Key files: `src/strategy/explainer.py`, `src/strategy/planner.py`, `src/strategy/templates.py`

**`src/monitoring/`:**
- Purpose: Production observability — metrics, drift, retraining triggers
- Contains: CloudWatch emitter, JS-divergence drift detector, GitHub dispatch trigger
- Key files: `src/monitoring/drift_monitor.py`, `src/monitoring/retraining_trigger.py`, `src/monitoring/cloudwatch.py`

**`src/autoResearch/`:**
- Purpose: AutoResearch orchestrator (currently a stub — `__init__.py` only)
- Contains: Placeholder for future research automation

**`tests/unit/`:**
- Purpose: Isolated unit tests per module
- Contains: Tests for env, action space, GNN encoder, policy, curriculum, drift monitor, API schemas

**`tests/integration/`:**
- Purpose: End-to-end pipeline and API endpoint tests
- Contains: Full training pipeline test, HTTP predict endpoint test

**`scripts/`:**
- Purpose: Operational CLI tools run by humans or CI
- Contains: Training dispatch, model evaluation/promotion, AWS bootstrap

**`infra/`:**
- Purpose: AWS infrastructure-as-config (not Terraform — raw JSON definitions)
- Contains: ECS task definition, EventBridge rule, IAM OIDC role policy

## Key File Locations

**Entry Points:**
- `src/api/main.py`: FastAPI app factory and lifespan — import target for uvicorn
- `scripts/dispatch_training.py`: Training entry point (CLI + CI)
- `scripts/evaluate_and_promote.py`: Model promotion entry point (CLI + CI)

**Configuration:**
- `pyproject.toml`: Project metadata, pytest config, linting config
- `src/rl/training/mappo.py`: `MAPPOConfig` dataclass — all training hyperparameters
- `src/rl/training/self_play.py`: `SelfPlayConfig` dataclass
- `src/rl/training/curriculum.py`: `CurriculumConfig` and phase definitions

**Core Logic:**
- `src/rl/env/catan_env.py`: Game rules, board topology, reward shaping
- `src/rl/env/action_space.py`: Action offsets, encode/decode — referenced everywhere
- `src/rl/models/gnn_encoder.py`: Board → embedding; topology built from `catan_env` arrays
- `src/rl/models/policy.py`: `CatanPolicy` — actor + critic heads; `_encode()` used by API and training
- `src/api/model_loader.py`: `ModelManager` — S3/local model loading and inference

**Testing:**
- `tests/conftest.py`: Shared fixtures (`env`, `action_space`, `step_through_setup`, `take_random_valid_action`)
- `tests/unit/`: One file per major module
- `tests/integration/`: `test_full_pipeline.py`, `test_predict_endpoint.py`

## Naming Conventions

**Files:**
- `snake_case.py` throughout — e.g., `catan_env.py`, `gnn_encoder.py`, `drift_monitor.py`
- Test files prefixed `test_` — e.g., `test_catan_env.py`
- Config JSON uses kebab-case — e.g., `ecs-task-definition.json`

**Directories:**
- `snake_case` for Python packages — e.g., `autoResearch`, `rl`, `api`
- Flat package structure within each module (no deep nesting beyond 3 levels)

**Classes:**
- PascalCase — e.g., `CatanEnv`, `CatanGNNEncoder`, `CatanPolicy`, `MAPPOTrainer`, `DriftMonitor`
- Dataclasses for config — e.g., `MAPPOConfig`, `SelfPlayConfig`, `CurriculumPhase`

**Functions:**
- `snake_case` — e.g., `get_action_mask`, `decode_action`, `emit_inference_metrics`
- Private helpers prefixed `_` — e.g., `_request_to_obs`, `_explain_action`, `_build_catan_topology`

## Where to Add New Code

**New API endpoint:**
- Route handler: `src/api/routes/<name>.py` (create new router module)
- Pydantic schemas: add to `src/api/schemas.py`
- Register router in: `src/api/main.py` via `app.include_router()`
- Tests: `tests/integration/test_<name>_endpoint.py`

**New RL model component:**
- Implementation: `src/rl/models/<component>.py`
- Unit tests: `tests/unit/test_<component>.py`

**New training feature:**
- Implementation: `src/rl/training/<feature>.py` or extend existing files in `src/rl/training/`
- Config: Add to relevant `*Config` dataclass

**New environment mechanic:**
- Implementation: `src/rl/env/catan_env.py` (extend existing game engine)
- If new actions required: update offsets and `TOTAL_ACTIONS` in `src/rl/env/action_space.py`
- Tests: `tests/unit/test_catan_env.py`

**New monitoring concern:**
- Implementation: `src/monitoring/<concern>.py`
- Wire into API: `src/api/routes/predict.py` or `src/api/main.py` lifespan

**New utility script:**
- Location: `scripts/<verb>_<noun>.py` (executable, argparse-based)

## Special Directories

**`.planning/`:**
- Purpose: GSD planning documents
- Generated: No (human/AI authored)
- Committed: Yes

**`.github/workflows/`:**
- Purpose: CI/CD, retraining, and deploy automation
- Generated: No
- Committed: Yes

**`infra/`:**
- Purpose: AWS resource definitions (ECS, EventBridge, IAM)
- Generated: No
- Committed: Yes

---

*Structure analysis: 2026-04-02*
