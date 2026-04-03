"""Tests for the DriftMonitor."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from src.monitoring.drift_monitor import DriftMonitor


@pytest.fixture
def monitor() -> DriftMonitor:
    """Create a fresh DriftMonitor."""
    return DriftMonitor(threshold=0.15, num_bins=50)


def _make_data(n: int = 100, mean: float = 0.5, std: float = 0.1, seed: int = 42):
    """Generate synthetic feature data."""
    rng = np.random.default_rng(seed)
    return [
        {"feature_a": float(rng.normal(mean, std)), "feature_b": float(rng.uniform(0, 1))}
        for _ in range(n)
    ]


class TestDriftMonitorCreation:

    def test_creates_without_error(self):
        dm = DriftMonitor()
        assert dm is not None
        assert dm.threshold == 0.15

    def test_custom_threshold(self):
        dm = DriftMonitor(threshold=0.3, num_bins=25)
        assert dm.threshold == 0.3
        assert dm.num_bins == 25

    def test_loads_reference_from_nonexistent_path(self):
        dm = DriftMonitor(reference_path="/nonexistent/path.json")
        assert len(dm.reference_histograms) == 0


class TestSetReference:

    def test_set_reference_from_data_stores_histograms(self, monitor: DriftMonitor):
        data = _make_data(100)
        monitor.set_reference_from_data(data)
        assert "feature_a" in monitor.reference_histograms
        assert "feature_b" in monitor.reference_histograms
        assert len(monitor.reference_histograms["feature_a"]) == 50

    def test_set_reference_empty_data(self, monitor: DriftMonitor):
        monitor.set_reference_from_data([])
        assert len(monitor.reference_histograms) == 0


class TestCheckDrift:

    def test_no_drift_with_identical_data(self, monitor: DriftMonitor):
        data = _make_data(100, seed=42)
        monitor.set_reference_from_data(data)

        # Add the same data as recent observations
        same_data = _make_data(100, seed=42)
        for obs in same_data:
            monitor.add_observation(obs)

        result = monitor.check_drift()
        assert result["drift_detected"] is False
        assert result["js_divergence"] < 0.1
        assert len(result["features_drifted"]) == 0

    def test_drift_with_very_different_data(self, monitor: DriftMonitor):
        # Reference: centered at 0.5
        ref_data = _make_data(200, mean=0.5, std=0.05, seed=42)
        monitor.set_reference_from_data(ref_data)

        # Recent data: centered at 5.0 (very different)
        recent_data = _make_data(200, mean=5.0, std=0.05, seed=99)
        for obs in recent_data:
            monitor.add_observation(obs)

        result = monitor.check_drift()
        assert result["drift_detected"] is True
        assert result["js_divergence"] > 0.1
        assert len(result["features_drifted"]) > 0

    def test_no_reference_returns_no_drift(self, monitor: DriftMonitor):
        for obs in _make_data(20):
            monitor.add_observation(obs)
        result = monitor.check_drift()
        assert result["drift_detected"] is False

    def test_insufficient_data_returns_no_drift(self, monitor: DriftMonitor):
        monitor.set_reference_from_data(_make_data(100))
        # Only add 5 observations (below the 10 minimum)
        for obs in _make_data(5):
            monitor.add_observation(obs)
        result = monitor.check_drift()
        assert result["drift_detected"] is False
        assert result["num_recent_samples"] == 5


class TestSaveLoadReference:

    def test_save_load_roundtrip(self, monitor: DriftMonitor):
        data = _make_data(100)
        monitor.set_reference_from_data(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "reference.json")
            monitor.save_reference(path)

            assert os.path.exists(path)

            # Load into a new monitor
            monitor2 = DriftMonitor()
            monitor2.load_reference(path)

            assert set(monitor2.reference_histograms.keys()) == set(
                monitor.reference_histograms.keys()
            )
            for key in monitor.reference_histograms:
                np.testing.assert_array_almost_equal(
                    monitor.reference_histograms[key],
                    monitor2.reference_histograms[key],
                )


class TestAddObservation:

    def test_buffers_data(self, monitor: DriftMonitor):
        assert len(monitor.recent_data) == 0
        monitor.add_observation({"feature_a": 0.5, "feature_b": 0.3})
        assert len(monitor.recent_data) == 1
        monitor.add_observation({"feature_a": 0.6, "feature_b": 0.4})
        assert len(monitor.recent_data) == 2

    def test_buffered_data_matches_input(self, monitor: DriftMonitor):
        obs = {"feature_a": 0.123, "feature_b": 0.456}
        monitor.add_observation(obs)
        assert monitor.recent_data[0] == obs
