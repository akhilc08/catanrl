"""Retraining trigger via GitHub Actions repository_dispatch."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import structlog

logger = structlog.get_logger()


class RetrainingTrigger:
    """Triggers model retraining via GitHub Actions repository_dispatch API.

    Uses urllib.request to avoid adding a requests dependency.

    Parameters
    ----------
    repo : str, optional
        GitHub repository in "owner/repo" format. Falls back to
        GITHUB_REPO environment variable.
    token : str, optional
        GitHub personal access token with repo scope. Falls back to
        GITHUB_TOKEN environment variable.
    """

    def __init__(self, repo: str | None = None, token: str | None = None):
        self.repo = repo or os.environ.get("GITHUB_REPO", "owner/catanrl")
        self.token = token or os.environ.get("GITHUB_TOKEN")

    def trigger(self, reason: str, metadata: dict | None = None) -> bool:
        """POST to GitHub Actions repository_dispatch API.

        Parameters
        ----------
        reason : str
            Human-readable reason for triggering retraining.
        metadata : dict, optional
            Additional metadata to include in the dispatch payload.

        Returns
        -------
        bool
            True if the dispatch was successful, False otherwise.
        """
        if not self.token:
            logger.error("retraining_trigger_no_token", repo=self.repo)
            return False

        url = f"https://api.github.com/repos/{self.repo}/dispatches"

        payload = {
            "event_type": "drift-detected",
            "client_payload": {
                "reason": reason,
                **(metadata or {}),
            },
        }

        body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "CatanRL-DriftMonitor",
            },
        )

        try:
            with urllib.request.urlopen(req) as response:
                status = response.status
                logger.info(
                    "retraining_triggered",
                    repo=self.repo,
                    status=status,
                    reason=reason,
                )
                return 200 <= status < 300
        except urllib.error.HTTPError as e:
            logger.error(
                "retraining_trigger_http_error",
                repo=self.repo,
                status=e.code,
                reason=e.reason,
                body=e.read().decode("utf-8", errors="replace")[:500],
            )
            return False
        except urllib.error.URLError as e:
            logger.error(
                "retraining_trigger_url_error",
                repo=self.repo,
                error=str(e.reason),
            )
            return False
        except Exception as e:
            logger.error(
                "retraining_trigger_error",
                repo=self.repo,
                error=str(e),
            )
            return False

    def trigger_if_needed(self, drift_result: dict) -> bool:
        """Trigger retraining if drift was detected.

        Parameters
        ----------
        drift_result : dict
            Result from DriftMonitor.check_drift().

        Returns
        -------
        bool
            True if retraining was triggered, False otherwise.
        """
        if not drift_result.get("drift_detected", False):
            logger.debug(
                "retraining_not_needed",
                js_divergence=drift_result.get("js_divergence", 0.0),
            )
            return False

        features_drifted = drift_result.get("features_drifted", [])
        js_divergence = drift_result.get("js_divergence", 0.0)

        reason = (
            f"Feature drift detected: JS divergence={js_divergence:.4f}, "
            f"{len(features_drifted)} features drifted"
        )

        metadata = {
            "js_divergence": js_divergence,
            "features_drifted": features_drifted[:20],  # limit size
            "num_features_drifted": len(features_drifted),
            "num_recent_samples": drift_result.get("num_recent_samples", 0),
        }

        return self.trigger(reason, metadata)
