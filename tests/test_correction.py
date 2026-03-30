"""Tests for ANS calibration helper functions."""
import pytest
from ans_sampler import (
    _compute_correction,
    _compute_binned_corrections,
    _get_phase_correction,
)


class TestComputeCorrection:
    def test_empty_samples_returns_one(self):
        corr, median = _compute_correction([], 0.5, 0.8, 1.15)
        assert corr == 1.0
        assert median == 0.0

    def test_excess_above_one_dampens(self):
        corr, median = _compute_correction([2.0, 2.0, 2.0], 0.5, 0.8, 1.15)
        assert abs(corr - 0.8) < 1e-6
        assert abs(median - 2.0) < 1e-6

    def test_excess_below_one_boosts(self):
        corr, median = _compute_correction([0.5], 0.5, 0.8, 1.15)
        assert abs(corr - 1.15) < 1e-6

    def test_unclamped_correction(self):
        corr, median = _compute_correction([1.0], 0.5, 0.8, 1.15)
        assert abs(corr - 1.0) < 1e-6

    def test_floor_clamps_minimum(self):
        corr, _ = _compute_correction([100.0], 0.5, 0.8, 1.15)
        assert corr == 0.8

    def test_ceiling_clamps_maximum(self):
        corr, _ = _compute_correction([0.001], 0.5, 0.8, 1.15)
        assert corr == 1.15

    def test_median_is_middle_element(self):
        corr, median = _compute_correction([3.0, 1.0, 2.0], 1.0, 0.0, 10.0)
        assert abs(median - 2.0) < 1e-6
        assert abs(corr - 0.5) < 1e-6


class TestComputeBinnedCorrections:
    def test_bins_with_enough_samples_get_own_correction(self):
        bins = {
            "structural": [2.0, 2.0, 2.0],
            "texture": [0.5, 0.5, 0.5],
            "cleanup": [1.0],
        }
        result = _compute_binned_corrections(bins, 0.9, 0.5, 0.8, 1.15, min_samples=3)
        assert abs(result["structural"] - 0.8) < 1e-6
        assert abs(result["texture"] - 1.15) < 1e-6
        assert abs(result["cleanup"] - 0.9) < 1e-6

    def test_all_bins_below_min_samples_use_global(self):
        bins = {"structural": [2.0], "texture": [0.5], "cleanup": []}
        result = _compute_binned_corrections(bins, 0.95, 0.5, 0.8, 1.15, min_samples=3)
        assert all(abs(v - 0.95) < 1e-6 for v in result.values())


class TestGetPhaseCorrection:
    def test_structural_phase(self):
        bins = {"structural": 0.85, "texture": 0.90, "cleanup": 0.95}
        assert _get_phase_correction(6.0, bins, 1.0) == 0.85

    def test_texture_phase(self):
        bins = {"structural": 0.85, "texture": 0.90, "cleanup": 0.95}
        assert _get_phase_correction(1.0, bins, 1.0) == 0.90

    def test_cleanup_phase(self):
        bins = {"structural": 0.85, "texture": 0.90, "cleanup": 0.95}
        assert _get_phase_correction(0.3, bins, 1.0) == 0.95

    def test_none_bins_returns_global(self):
        assert _get_phase_correction(1.0, None, 0.88) == 0.88

    def test_boundary_sigma_5_is_texture(self):
        bins = {"structural": 0.85, "texture": 0.90, "cleanup": 0.95}
        assert _get_phase_correction(5.0, bins, 1.0) == 0.90

    def test_boundary_sigma_0_5_is_cleanup(self):
        bins = {"structural": 0.85, "texture": 0.90, "cleanup": 0.95}
        assert _get_phase_correction(0.5, bins, 1.0) == 0.95
