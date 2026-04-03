"""Feature distribution drift monitoring using Jensen-Shannon divergence."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import structlog
from scipy.spatial.distance import jensenshannon

logger = structlog.get_logger()


class DriftMonitor:
    """Monitors feature distribution drift using Jensen-Shannon divergence.

    Compares recent observation feature distributions against a reference
    distribution (typically computed from training data). When JS divergence
    exceeds a threshold, drift is detected.

    Parameters
    ----------
    reference_path : str, optional
        Path to a JSON file containing reference histograms.
    threshold : float
        JS divergence threshold for triggering drift detection.
    num_bins : int
        Number of histogram bins for discretizing continuous features.
    """

    def __init__(
        self,
        reference_path: str | None = None,
        threshold: float = 0.15,
        num_bins: int = 50,
    ):
        self.threshold = threshold
        self.num_bins = num_bins
        self.reference_histograms: dict[str, np.ndarray] = {}
        self.recent_data: list[dict[str, float]] = []

        if reference_path and os.path.exists(reference_path):
            self.load_reference(reference_path)

    def load_reference(self, path: str) -> None:
        """Load reference distribution histograms from a JSON file.

        Parameters
        ----------
        path : str
            Path to the JSON file.
        """
        try:
            with open(path) as f:
                data = json.load(f)
            self.reference_histograms = {
                k: np.array(v, dtype=np.float64) for k, v in data.items()
            }
            logger.info("reference_loaded", path=path, num_features=len(self.reference_histograms))
        except Exception as e:
            logger.error("reference_load_failed", path=path, error=str(e))

    def save_reference(self, path: str) -> None:
        """Save current reference distribution histograms to a JSON file.

        Parameters
        ----------
        path : str
            Path to write the JSON file.
        """
        data = {k: v.tolist() for k, v in self.reference_histograms.items()}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("reference_saved", path=path, num_features=len(data))

    def set_reference_from_data(self, data: list[dict[str, float]]) -> None:
        """Compute reference histograms from a list of feature observations.

        Parameters
        ----------
        data : list[dict]
            List of feature dictionaries, where each dict maps feature names
            to scalar values.
        """
        if not data:
            logger.warning("set_reference_empty_data")
            return

        # Collect all feature names
        feature_names = sorted(data[0].keys())

        for feat in feature_names:
            values = np.array([d[feat] for d in data if feat in d], dtype=np.float64)
            self.reference_histograms[feat] = self._compute_histogram(values)

        logger.info(
            "reference_set_from_data",
            num_features=len(feature_names),
            num_samples=len(data),
        )

    def add_observation(self, features: dict[str, float]) -> None:
        """Add a single observation to the recent data buffer.

        Parameters
        ----------
        features : dict
            Feature name to scalar value mapping.
        """
        self.recent_data.append(features)

    def check_drift(self) -> dict[str, Any]:
        """Compute JS divergence between reference and recent distributions.

        Returns
        -------
        dict
            {
                "js_divergence": float (mean across features),
                "drift_detected": bool,
                "features_drifted": list[str],
                "per_feature_js": dict[str, float],
                "num_recent_samples": int,
            }
        """
        if not self.reference_histograms:
            logger.warning("drift_check_no_reference")
            return {
                "js_divergence": 0.0,
                "drift_detected": False,
                "features_drifted": [],
                "per_feature_js": {},
                "num_recent_samples": len(self.recent_data),
            }

        if len(self.recent_data) < 10:
            logger.warning("drift_check_insufficient_data", count=len(self.recent_data))
            return {
                "js_divergence": 0.0,
                "drift_detected": False,
                "features_drifted": [],
                "per_feature_js": {},
                "num_recent_samples": len(self.recent_data),
            }

        per_feature_js: dict[str, float] = {}
        features_drifted: list[str] = []

        for feat, ref_hist in self.reference_histograms.items():
            values = np.array(
                [d[feat] for d in self.recent_data if feat in d],
                dtype=np.float64,
            )
            if len(values) == 0:
                continue

            recent_hist = self._compute_histogram(values)

            # Ensure histograms have the same length
            min_len = min(len(ref_hist), len(recent_hist))
            ref_h = ref_hist[:min_len]
            rec_h = recent_hist[:min_len]

            # Add small epsilon to avoid zero bins
            eps = 1e-10
            ref_h = ref_h + eps
            rec_h = rec_h + eps

            # Normalize
            ref_h = ref_h / ref_h.sum()
            rec_h = rec_h / rec_h.sum()

            js = float(jensenshannon(ref_h, rec_h))
            per_feature_js[feat] = js

            if js > self.threshold:
                features_drifted.append(feat)

        if per_feature_js:
            mean_js = float(np.mean(list(per_feature_js.values())))
        else:
            mean_js = 0.0

        drift_detected = len(features_drifted) > 0

        if drift_detected:
            logger.warning(
                "drift_detected",
                js_divergence=mean_js,
                num_features_drifted=len(features_drifted),
                features=features_drifted[:5],
            )
        else:
            logger.info("drift_check_ok", js_divergence=mean_js)

        return {
            "js_divergence": mean_js,
            "drift_detected": drift_detected,
            "features_drifted": features_drifted,
            "per_feature_js": per_feature_js,
            "num_recent_samples": len(self.recent_data),
        }

    def _compute_histogram(self, values: np.ndarray) -> np.ndarray:
        """Compute normalized histogram for a feature.

        Parameters
        ----------
        values : np.ndarray
            1-D array of feature values.

        Returns
        -------
        np.ndarray
            Normalized histogram (sums to 1).
        """
        if len(values) == 0:
            return np.zeros(self.num_bins, dtype=np.float64)

        # Use fixed range [0, 1] since most features are normalized
        # Fall back to data range if values exceed [0, 1]
        v_min = float(values.min())
        v_max = float(values.max())

        if v_min >= 0.0 and v_max <= 1.0:
            bin_range = (0.0, 1.0)
        else:
            margin = max(abs(v_max - v_min) * 0.01, 1e-6)
            bin_range = (v_min - margin, v_max + margin)

        hist, _ = np.histogram(values, bins=self.num_bins, range=bin_range)
        hist = hist.astype(np.float64)

        total = hist.sum()
        if total > 0:
            hist /= total

        return hist


def trigger_retraining(drift_result: dict) -> None:
    """Convenience function to trigger retraining from drift result."""
    from .retraining_trigger import RetrainingTrigger

    trigger = RetrainingTrigger()
    trigger.trigger_if_needed(drift_result)


def main() -> None:
    """Run drift check. CLI entry point for Lambda / scheduled execution."""
    import argparse

    parser = argparse.ArgumentParser(description="CatanRL Drift Monitor")
    parser.add_argument(
        "--interval",
        type=int,
        default=21600,
        help="Check interval in seconds (used by scheduler, not this script)",
    )
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument(
        "--reference",
        type=str,
        default="data/reference_distribution.json",
    )
    parser.add_argument(
        "--recent-data",
        type=str,
        default=None,
        help="Path to JSON file with recent feature observations",
    )
    args = parser.parse_args()

    monitor = DriftMonitor(reference_path=args.reference, threshold=args.threshold)

    # Load recent data if provided
    if args.recent_data and os.path.exists(args.recent_data):
        with open(args.recent_data) as f:
            recent = json.load(f)
        for obs in recent:
            monitor.add_observation(obs)

    result = monitor.check_drift()
    print(json.dumps(result, indent=2, default=str))

    if result["drift_detected"]:
        trigger_retraining(result)


if __name__ == "__main__":
    main()
