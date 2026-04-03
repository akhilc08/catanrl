# External Integrations

**Analysis Date:** 2026-04-02

## APIs & External Services

**Serverless GPU Training:**
- Modal - Remote GPU training dispatch
  - SDK/Client: `modal` (optional import; falls back to local if not installed)
  - GPU target: A10G (configurable via `--gpu` flag)
  - Entry point: `scripts/dispatch_training.py`
  - Auth: Modal account credentials (implicit from Modal CLI)

**CI/CD Automation:**
- GitHub Actions REST API - Repository dispatch for retraining triggers
  - Endpoint: `POST https://api.github.com/repos/{owner}/catanrl/dispatches`
  - Event type: `drift-detected`
  - Auth: `GITHUB_TOKEN` env var (Personal Access Token with repo scope)
  - Implementation: `src/monitoring/retraining_trigger.py` using `urllib.request` (no `requests` dependency)

## Data Storage

**Databases:**
- None detected — no SQL/NoSQL database client libraries present

**File Storage:**
- AWS S3 - Model artifact storage
  - Bucket: `catanrl-models`
  - Key paths: `champion.pt`, `challenger.pt`
  - Client: `boto3>=1.29.0`
  - Auth: IAM role via GitHub OIDC (`infra/iam-github-oidc.json`)
  - Used in: `.github/workflows/retrain.yml` (`aws s3 cp models/champion.pt s3://catanrl-models/champion.pt`)
- Local filesystem - Model checkpoints during training
  - Path: `models/checkpoints/`
  - Docker volume mount: `./models:/app/models`

**Caching:**
- None (no Redis, Memcached, or similar)

**Experiment Tracking Storage:**
- MLflow - Local `./mlruns` directory (Docker Compose) or self-hosted MLflow server
  - Service: `mlflow` container in `docker-compose.yml` (port 5000)
  - Client: `mlflow>=2.9.0`
  - Experiment name: `catanrl`

## Authentication & Identity

**Auth Provider:**
- GitHub OIDC - Keyless AWS authentication for GitHub Actions
  - IAM role: `catanrl-github-actions`
  - Trust policy: `infra/iam-github-oidc.json`
  - OIDC provider: `token.actions.githubusercontent.com`
  - Used in: `.github/workflows/deploy.yml` and `.github/workflows/retrain.yml`
  - Secret: `AWS_ACCOUNT_ID`, `ECR_REGISTRY` (GitHub Actions secrets)

**API Authentication:**
- FastAPI inference API — No authentication detected on endpoints
- GitHub API calls use Bearer token (`GITHUB_TOKEN`)

## Monitoring & Observability

**Metrics:**
- AWS CloudWatch - Custom metrics namespace `CatanRL`
  - Client: `boto3` CloudWatch client (`src/monitoring/cloudwatch.py`)
  - Region: `us-east-1` (configurable)
  - Metrics emitted:
    - `InferenceLatencyMs` — per-request latency with `ModelVersion` dimension
    - `InferenceConfidence` — model confidence score with `ModelVersion` dimension
    - `RequestCount` — request count
    - `JSDivergence` — drift detection score
  - Graceful degradation: disables itself if `boto3` unavailable or client init fails

**Logs:**
- structlog structured logging throughout all modules
- AWS CloudWatch Logs in production
  - Log group: `/catanrl/api`
  - Log driver: `awslogs` (configured in `infra/ecs-task-definition.json`)
  - Stream prefix: `ecs`

**Drift Detection:**
- Custom `DriftMonitor` (`src/monitoring/drift_monitor.py`)
  - Algorithm: Jensen-Shannon divergence (via `scipy.spatial.distance.jensenshannon`)
  - Threshold: 0.15 JS divergence (configurable)
  - Schedule: Every 6 hours via AWS EventBridge + Lambda (`infra/eventbridge-rule.json`)
  - On drift: calls `RetrainingTrigger` → fires GitHub Actions `repository_dispatch`

**Error Tracking:**
- None detected (no Sentry, Datadog, Rollbar)

## CI/CD & Deployment

**Hosting:**
- AWS ECS Fargate (cluster: `catanrl`, service: `catanrl-api`, region: `us-east-1`)
- Container registry: AWS ECR

**CI Pipeline:**
- GitHub Actions — three workflows:
  - `.github/workflows/ci.yml` - Runs on PR and push to `main`
    - Steps: ruff lint → mypy type check → pytest unit → pytest integration
    - Python: 3.10
  - `.github/workflows/deploy.yml` - Runs on push to `main`
    - Steps: AWS OIDC auth → ECR login → Docker build+push → ECS service update
    - Image tagged with `github.sha`
    - Build cache: GitHub Actions cache (`type=gha`)
  - `.github/workflows/retrain.yml` - Runs weekly (Sunday 02:00 UTC), on `workflow_dispatch`, and on `repository_dispatch` with type `drift-detected`
    - Steps: install deps → AWS OIDC auth → dispatch training (Modal) → evaluate and promote → upload champion to S3 → rolling ECS deploy

## Webhooks & Callbacks

**Incoming:**
- `POST /feedback` - Feedback route on FastAPI API (`src/api/routes/feedback.py`)
- `repository_dispatch` from GitHub (triggers retrain workflow on `drift-detected` event)

**Outgoing:**
- `POST https://api.github.com/repos/{owner}/catanrl/dispatches` - Fired by `RetrainingTrigger` when drift exceeds threshold

## Environment Configuration

**Required env vars (production):**
- `MODEL_PATH` - Path to champion model file (default: `/app/models/champion.pt`)
- `LOG_LEVEL` - Logging verbosity (default: `INFO`)
- `AWS_DEFAULT_REGION` - AWS region (default: `us-east-1`)
- `GITHUB_REPO` - Repo in `owner/repo` format for retraining dispatch (default: `owner/catanrl`)
- `GITHUB_TOKEN` - GitHub PAT with repo scope for retraining dispatch

**GitHub Actions secrets required:**
- `AWS_ACCOUNT_ID` - AWS account ID for OIDC role ARN
- `ECR_REGISTRY` - ECR registry URL for image tagging

**Secrets location:**
- Runtime secrets: OS environment variables injected via ECS task definition or GitHub Actions secrets
- No secrets stored in code or committed files

---

*Integration audit: 2026-04-02*
