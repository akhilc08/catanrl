"""CloudWatch metrics emitter for CatanRL inference monitoring."""

from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger()


class CloudWatchEmitter:
    """Emits custom metrics to AWS CloudWatch.

    Gracefully degrades if boto3 is unavailable or client creation fails,
    logging warnings instead of raising exceptions.

    Parameters
    ----------
    namespace : str
        CloudWatch namespace for all metrics.
    region : str
        AWS region for the CloudWatch client.
    """

    def __init__(self, namespace: str = "CatanRL", region: str = "us-east-1"):
        self.namespace = namespace
        self.enabled = False
        self.client: Any = None

        try:
            import boto3

            self.client = boto3.client("cloudwatch", region_name=region)
            self.enabled = True
            logger.info("cloudwatch_enabled", namespace=namespace, region=region)
        except ImportError:
            logger.warning("cloudwatch_disabled", reason="boto3 not installed")
        except Exception as e:
            logger.warning("cloudwatch_disabled", reason=str(e))

    def emit_inference_metrics(
        self, latency_ms: float, confidence: float, model_version: str
    ) -> None:
        """Emit per-request inference metrics.

        Parameters
        ----------
        latency_ms : float
            Inference latency in milliseconds.
        confidence : float
            Model confidence score (0-1).
        model_version : str
            Model version identifier.
        """
        metrics = [
            {
                "MetricName": "InferenceLatencyMs",
                "Value": latency_ms,
                "Unit": "Milliseconds",
                "Dimensions": [
                    {"Name": "ModelVersion", "Value": model_version},
                ],
            },
            {
                "MetricName": "InferenceConfidence",
                "Value": confidence,
                "Unit": "None",
                "Dimensions": [
                    {"Name": "ModelVersion", "Value": model_version},
                ],
            },
        ]
        self._put_metrics(metrics)

    def emit_drift_score(self, js_divergence: float) -> None:
        """Emit drift detection score.

        Parameters
        ----------
        js_divergence : float
            Jensen-Shannon divergence between reference and recent distributions.
        """
        metrics = [
            {
                "MetricName": "JSDivergence",
                "Value": js_divergence,
                "Unit": "None",
            },
        ]
        self._put_metrics(metrics)

    def emit_request_count(self, count: int = 1) -> None:
        """Emit request count metric.

        Parameters
        ----------
        count : int
            Number of requests to record.
        """
        metrics = [
            {
                "MetricName": "RequestCount",
                "Value": float(count),
                "Unit": "Count",
            },
        ]
        self._put_metrics(metrics)

    def _put_metrics(self, metrics: list[dict]) -> None:
        """Batch put metrics to CloudWatch.

        Parameters
        ----------
        metrics : list[dict]
            List of metric data dicts in CloudWatch PutMetricData format.
        """
        if not self.enabled or self.client is None:
            logger.debug("cloudwatch_skip", reason="disabled", num_metrics=len(metrics))
            return

        try:
            # Add timestamp to all metrics that don't have one
            import datetime

            now = datetime.datetime.now(datetime.timezone.utc)
            for m in metrics:
                if "Timestamp" not in m:
                    m["Timestamp"] = now

            self.client.put_metric_data(
                Namespace=self.namespace,
                MetricData=metrics,
            )
            logger.debug("cloudwatch_put", num_metrics=len(metrics))
        except Exception as e:
            logger.error("cloudwatch_put_failed", error=str(e), num_metrics=len(metrics))
