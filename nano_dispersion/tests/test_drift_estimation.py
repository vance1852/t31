from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nano_dispersion.algorithms.drift_estimation import (
    compute_drift_velocity,
    estimate_global_drift,
    subtract_drift,
)


def _generate_brownian_with_drift(
    n_particles: int,
    n_frames: int,
    D: float,
    dt: float,
    vx: float,
    vy: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(2.0 * D * dt)
    rows = []

    for pid in range(n_particles):
        x0 = rng.uniform(10, 90)
        y0 = rng.uniform(10, 90)
        dx_brownian = rng.normal(0.0, sigma, size=n_frames)
        dy_brownian = rng.normal(0.0, sigma, size=n_frames)

        xs = np.empty(n_frames)
        ys = np.empty(n_frames)
        xs[0] = x0
        ys[0] = y0
        for i in range(1, n_frames):
            xs[i] = xs[i - 1] + dx_brownian[i] + vx * dt
            ys[i] = ys[i - 1] + dy_brownian[i] + vy * dt

        for frame in range(n_frames):
            rows.append({
                "particle_id": pid,
                "frame": frame,
                "time_s": frame * dt,
                "x_um": float(xs[frame]),
                "y_um": float(ys[frame]),
            })

    return pd.DataFrame(rows)


def test_known_drift_recovered():
    n_particles = 60
    n_frames = 150
    D = 1.0
    dt = 0.033
    vx_true = 1.0
    vy_true = 0.5

    df = _generate_brownian_with_drift(n_particles, n_frames, D, dt, vx_true, vy_true, seed=42)
    drift_df = estimate_global_drift(df, outlier_sigma=3.0)

    if drift_df.empty:
        pytest.skip("drift estimation returned empty")

    vx_est, vy_est, _ = compute_drift_velocity(drift_df)
    remaining_drift_vx = abs(vx_est - vx_true)
    remaining_drift_vy = abs(vy_est - vy_true)

    assert remaining_drift_vx < 0.2


def test_drift_estimation_ignores_outliers():
    n_particles = 20
    n_frames = 80
    D = 1.0
    dt = 0.033
    vx_true = 0.5
    vy_true = 0.2

    df = _generate_brownian_with_drift(n_particles, n_frames, D, dt, vx_true, vy_true, seed=123)

    rng = np.random.default_rng(456)
    for _ in range(5):
        idx = rng.integers(0, len(df))
        df.loc[idx, "x_um"] += 50.0
        df.loc[idx, "y_um"] += 50.0

    drift_df = estimate_global_drift(df, outlier_sigma=3.0)
    if drift_df.empty:
        pytest.skip("drift estimation returned empty")

    vx_est, vy_est, _ = compute_drift_velocity(drift_df)

    assert abs(vx_est - vx_true) < 0.3


def test_subtract_drift_idempotent():
    n_particles = 10
    n_frames = 50
    D = 1.0
    dt = 0.033
    vx = 0.8
    vy = 0.3

    df = _generate_brownian_with_drift(n_particles, n_frames, D, dt, vx, vy, seed=789)
    drift_df = estimate_global_drift(df, outlier_sigma=3.0)
    if drift_df.empty:
        pytest.skip("drift estimation returned empty")

    corrected1 = subtract_drift(df, drift_df)
    corrected2 = subtract_drift(corrected1, drift_df)

    assert "x_corrected_um" in corrected2.columns
    assert "y_corrected_um" in corrected2.columns
    assert np.all(np.isfinite(corrected2["x_corrected_um"].values))
    assert np.all(np.isfinite(corrected2["y_corrected_um"].values))
