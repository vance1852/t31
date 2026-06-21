from __future__ import annotations

import numpy as np
import pytest

from nano_dispersion.algorithms.batch_stats import (
    aggregate_trajectory_results,
    flag_worst_trajectories,
)


def test_short_trajectories_excluded_from_distribution():
    rng = np.random.default_rng(42)
    results = []

    for i in range(5):
        D_val = float(rng.normal(2.0, 0.3))
        results.append({
            "trajectory_id": i,
            "particle_id": i,
            "channel_id": 0,
            "num_frames": int(rng.integers(5, 18)),
            "diffusion_D_um2_s": D_val,
            "alpha_exponent": 1.0,
            "fit_r2": 0.95,
            "hydro_radius_nm": 50.0,
            "excluded_from_distribution": True,
            "exclude_reason": "short_trajectory_10_frames_less_than_20",
        })

    for i in range(5, 15):
        D_val = float(rng.normal(2.0, 0.3))
        results.append({
            "trajectory_id": i,
            "particle_id": i,
            "channel_id": 0,
            "num_frames": int(rng.integers(30, 100)),
            "diffusion_D_um2_s": D_val,
            "alpha_exponent": 1.0,
            "fit_r2": 0.95,
            "hydro_radius_nm": 50.0,
            "excluded_from_distribution": False,
            "exclude_reason": None,
        })

    stats = aggregate_trajectory_results(results, exclude_short=True)

    assert stats["n_total"] == 15
    assert stats["n_valid"] == 10
    assert stats["n_excluded"] == 5


def test_per_channel_differences():
    rng = np.random.default_rng(555)
    results = []

    for ch in ["channel_A", "channel_B"]:
        mean_D = 1.0 if ch == "channel_A" else 2.0
        for i in range(10):
            D_val = float(rng.normal(mean_D, 0.1))
            results.append({
                "trajectory_id": i if ch == "channel_A" else i + 10,
                "particle_id": i if ch == "channel_A" else i + 10,
                "channel_id": ch,
                "num_frames": 50,
                "diffusion_D_um2_s": D_val,
                "alpha_exponent": 1.0,
                "fit_r2": 0.95,
                "hydro_radius_nm": 50.0,
                "excluded_from_distribution": False,
            })

    stats = aggregate_trajectory_results(results, exclude_short=True)

    per_channel = stats["per_channel_stats"]
    assert "channel_A" in per_channel
    assert "channel_B" in per_channel

    ch0_mean = per_channel["channel_A"]["mean_D"]
    ch1_mean = per_channel["channel_B"]["mean_D"]
    assert ch1_mean > ch0_mean


def test_worst_scoring_ranks_correctly():
    results = [
        {
            "trajectory_id": 0,
            "particle_id": 0,
            "channel_id": 0,
            "num_frames": 150,
            "diffusion_D_um2_s": 2.0,
            "alpha_exponent": 1.0,
            "fit_r2": 0.98,
            "hydro_radius_nm": 50.0,
            "excluded_from_distribution": False,
        },
        {
            "trajectory_id": 1,
            "particle_id": 1,
            "channel_id": 0,
            "num_frames": 10,
            "diffusion_D_um2_s": 2.0,
            "alpha_exponent": 1.0,
            "fit_r2": 0.98,
            "hydro_radius_nm": 50.0,
            "excluded_from_distribution": True,
            "exclude_reason": "short",
        },
        {
            "trajectory_id": 2,
            "particle_id": 2,
            "channel_id": 0,
            "num_frames": 150,
            "diffusion_D_um2_s": 2.0,
            "alpha_exponent": 0.3,
            "fit_r2": 0.5,
            "hydro_radius_nm": 50.0,
            "excluded_from_distribution": False,
        },
    ]

    worst = flag_worst_trajectories(results, n=3)

    assert len(worst) == 3
    scores = [w["total_score"] for w in worst]
    assert scores[0] >= scores[-1]
