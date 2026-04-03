#!/usr/bin/env bash
set -euo pipefail

# CatanRL AWS Bootstrap Script
# Run once to create all required AWS resources

REGION="us-east-1"
PROJECT="catanrl"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "=== CatanRL AWS Bootstrap ==="
echo "Account: $ACCOUNT_ID"
echo "Region: $REGION"
echo ""

# 1. ECR Repository
echo "[1/6] Creating ECR repository..."
aws ecr create-repository \
    --repository-name $PROJECT \
    --region $REGION \
    --image-scanning-configuration scanOnPush=true \
    2>/dev/null || echo "  ECR repo already exists"

# 2. S3 Bucket for models + artifacts
echo "[2/6] Creating S3 bucket..."
aws s3 mb "s3://${PROJECT}-models" --region $REGION 2>/dev/null || echo "  S3 bucket already exists"
aws s3api put-bucket-versioning \
    --bucket "${PROJECT}-models" \
    --versioning-configuration Status=Enabled

# 3. ECS Cluster
echo "[3/6] Creating ECS cluster..."
aws ecs create-cluster \
    --cluster-name $PROJECT \
    --region $REGION \
    --capacity-providers FARGATE \
    --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
    2>/dev/null || echo "  ECS cluster already exists"

# 4. CloudWatch Log Group
echo "[4/6] Creating CloudWatch log group..."
aws logs create-log-group \
    --log-group-name "/${PROJECT}/api" \
    --region $REGION \
    2>/dev/null || echo "  Log group already exists"

# 5. IAM Role for GitHub Actions OIDC
echo "[5/6] Creating IAM role for GitHub Actions..."
# Create OIDC provider (idempotent)
aws iam create-open-id-connect-provider \
    --url "https://token.actions.githubusercontent.com" \
    --client-id-list "sts.amazonaws.com" \
    --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1" \
    2>/dev/null || echo "  OIDC provider already exists"

# Create IAM role with trust policy
TRUST_POLICY=$(cat infra/iam-github-oidc.json | sed "s/ACCOUNT_ID/$ACCOUNT_ID/g")
aws iam create-role \
    --role-name "${PROJECT}-github-actions" \
    --assume-role-policy-document "$TRUST_POLICY" \
    2>/dev/null || echo "  IAM role already exists"

# Attach required policies
for policy in AmazonECS_FullAccess AmazonEC2ContainerRegistryPowerUser AmazonS3FullAccess CloudWatchLogsFullAccess; do
    aws iam attach-role-policy \
        --role-name "${PROJECT}-github-actions" \
        --policy-arn "arn:aws:iam::aws:policy/$policy" \
        2>/dev/null || true
done

# 6. ECS Task Execution Role
echo "[6/6] Creating ECS task execution role..."
aws iam create-role \
    --role-name "${PROJECT}-ecs-task-execution" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    2>/dev/null || echo "  ECS task execution role already exists"

aws iam attach-role-policy \
    --role-name "${PROJECT}-ecs-task-execution" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy" \
    2>/dev/null || true

echo ""
echo "=== Bootstrap complete ==="
echo "Next steps:"
echo "  1. Set OWNER/catanrl in infra/iam-github-oidc.json"
echo "  2. Add GitHub secrets: AWS_ACCOUNT_ID, ECR_REGISTRY"
echo "  3. Build and push first image: docker build -t $PROJECT . && docker tag ..."
echo "  4. Register ECS task definition: aws ecs register-task-definition --cli-input-json file://infra/ecs-task-definition.json"
echo "  5. Create ECS service: aws ecs create-service ..."
