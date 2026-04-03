# Technology Stack

**Analysis Date:** 2026-04-02

## Languages

**Primary:**
- Python 3.10 - All source code, training, inference, monitoring, scripts

## Runtime

**Environment:**
- Python 3.10 (CPython), constrained by `requires-python = ">=3.10"` in `pyproject.toml`

**Package Manager:**
- pip (no Poetry or uv detected)
- Lockfile: None (only `requirements.txt` and `requirements-dev.txt` — no pinned lockfile)

## Frameworks

**Core:**
- FastAPI >=0.104.0 - REST inference API (`src/api/main.py`)
- Uvicorn[standard] >=0.24.0 - ASGI server, entry point `src/api/main:app`
- Pydantic >=2.5.0 - Request/response schema validation (`src/api/schemas.py`)
- Gymnasium >=0.29.0 - RL environment interface (`src/rl/env/catan_env.py`)

**Machine Learning:**
- PyTorch >=2.1.0 - All neural network models, training, inference
- PyTorch Geometric (torch-geometric) >=2.4.0 - Heterogeneous GNN (`src/rl/models/gnn_encoder.py`); uses `GATConv`, `HeteroConv`
- NumPy >=1.24.0 - Array operations throughout
- SciPy >=1.11.0 - Jensen-Shannon divergence for drift detection (`src/monitoring/drift_monitor.py`)

**Experiment Tracking:**
- MLflow >=2.9.0 - Training run tracking, metric/param logging (`src/rl/training/mappo.py`)

**Logging:**
- structlog >=23.2.0 - Structured JSON logging across all modules

**Testing:**
- pytest >=7.4.0 - Test runner
- pytest-cov >=4.1.0 - Coverage reporting
- pytest-asyncio >=0.23.0 - Async test support
- httpx >=0.25.0 - HTTP client for API integration tests

**Dev / Linting:**
- ruff >=0.1.0 - Linter and formatter (`pyproject.toml`: line-length=100, targets E/F/I/N/W/UP)
- mypy >=1.7.0 - Static type checking (configured in `pyproject.toml`)
- cleanrl >=0.1.0 - Dev dependency (RL utility reference)

**Build/Dev:**
- Docker (multi-stage build) - `Dockerfile` produces a `python:3.10-slim` runtime image
- Docker Compose - Local multi-service dev stack (`docker-compose.yml`)
- GNU Make - `Makefile` for common dev commands

## Key Dependencies

**Critical:**
- `torch>=2.1.0` - Core ML framework; model training, inference, checkpointing (`.pt` files)
- `torch-geometric>=2.4.0` - GNN encoder backbone (`CatanGNNEncoder` in `src/rl/models/gnn_encoder.py`); heterogeneous GAT with 3 node types and 5 relation types
- `fastapi>=0.104.0` - Inference REST API with lifespan model loading
- `gymnasium>=0.29.0` - Catan environment step/reset interface (`src/rl/env/catan_env.py`)
- `mlflow>=2.9.0` - Experiment tracking; used inside MAPPO training loop with graceful ImportError fallback
- `boto3>=1.29.0` - AWS SDK; used by `CloudWatchEmitter` (`src/monitoring/cloudwatch.py`) with graceful ImportError fallback

**Infrastructure:**
- `scipy>=1.11.0` - `jensenshannon` from `scipy.spatial.distance` used by `DriftMonitor`
- `structlog>=23.2.0` - All logging; structured key=value pairs suitable for CloudWatch log ingestion
- `pydantic>=2.5.0` - API schema validation

## Configuration

**Environment:**
- Configuration via OS environment variables; no dotenv library detected
- Key env vars: `MODEL_PATH` (default `/app/models/champion.pt`), `LOG_LEVEL`, `AWS_DEFAULT_REGION`, `GITHUB_REPO`, `GITHUB_TOKEN`
- `.env` file presence: Not detected in repository

**Build:**
- `pyproject.toml` - Project metadata, ruff, mypy, pytest configuration
- `Dockerfile` - Two-stage build (builder + runtime); exposes port 8000
- `docker-compose.yml` - Three services: `api`, `mlflow` (port 5000), `drift_monitor`

## Model Artifacts

**Format:** PyTorch `.pt` state dicts
**Locations:**
- `models/champion.pt` - Production model loaded by API on startup
- `models/challenger.pt` - Candidate model for evaluation
- `models/checkpoints/policy_update_N.pt` - Training checkpoints
- `models/` directory is volume-mounted in `docker-compose.yml`

## Platform Requirements

**Development:**
- Python 3.10+
- Docker + Docker Compose for full local stack
- GPU optional (PyTorch CPU fallback supported; Modal used for remote GPU training)

**Production:**
- AWS ECS Fargate (us-east-1) — `infra/ecs-task-definition.json`
- AWS ECR - Container image registry
- AWS S3 - Model artifact storage (`s3://catanrl-models/champion.pt`)
- AWS CloudWatch - Metrics and logs (`/catanrl/api` log group)
- AWS Lambda - Drift monitor scheduled execution
- AWS EventBridge - 6-hour cron trigger for drift monitor (`infra/eventbridge-rule.json`)

---

*Stack analysis: 2026-04-02*
