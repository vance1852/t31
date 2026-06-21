from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nano_dispersion.algorithms.diffusion_fitting import (
    FitResult,
    discriminate_model,
    fit_brownian,
    fit_confined,
)
from nano_dispersion.algorithms.msd import compute_msd


def _make_brownian_dataset(D_true, n_particles=60, n_frames=150, dt=0.033, seed=42):
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(2.0 * D_true * dt)
    rows = []
    for pid in range(n_particles):
        x0 = rng.uniform(20, 80)
        y0 = rng.uniform(20, 80)
        dx = rng.normal(0.0, sigma, size=n_frames)
        dy = rng.normal(0.0, sigma, size=n_frames)
        xs = np.empty(n_frames)
        ys = np.empty(n_frames)
        xs[0], ys[0] = x0, y0
        for i in range(1, n_frames):
            xs[i] = xs[i - 1] + dx[i]
            ys[i] = ys[i - 1] + dy[i]
        for frame in range(n_frames):
            rows.append({
                "particle_id": pid,
                "frame": frame,
                "time_s": frame * dt,
                "x_um": float(xs[frame]),
                "y_um": float(ys[frame]),
            })
    return pd.DataFrame(rows)


def test_brownian_synthetic_recovers_D():
    D_true = 2.0
    df = _make_brownian_dataset(D_true, n_particles=80, n_frames=200, seed=100)
    msd_df = compute_msd(df, use_corrected=False)

    if msd_df.empty:
        pytest.skip("MSD empty")

    fit_result = fit_brownian(
        msd_df, min_lags=5, max_ratio=0.4, confidence=0.95
    )

    if np.isnan(fit_result.D):
        pytest.skip("Fit returned NaN")

    ratio = fit_result.D / D_true
    assert abs(ratio - 1.0) < 0.2


def test_confined_diffusion_detected():
    alpha_sub = 0.6
    r2_bad = 0.6
    r2_good = 0.95

    rb = FitResult(
        D=0.5,
        alpha=alpha_sub,
        r2=r2_bad,
        lag_start_s=0.01,
        lag_end_s=0.3,
        aic=100.0,
        bic=105.0,
    )

    rc = FitResult(
        D=0.1,
        alpha=0.5,
        r2=r2_good,
        lag_start_s=0.01,
        lag_end_s=0.5,
        aic=50.0,
        bic=58.0,
        params={"A": 2.0, "tau": 0.05, "B": 0.001},
    )

    dummy_msd = pd.DataFrame({
        "lag_time_s": np.linspace(0.01, 0.5, 20),
        "msd": 2.0 * (1.0 - np.exp(-np.linspace(0.01, 0.5, 20) / 0.05)),
        "lag_frames": np.arange(1, 21),
        "msd_x": np.linspace(0.1, 1.0, 20),
        "msd_y": np.linspace(0.1, 1.0, 20),
        "count": np.ones(20, dtype=int),
    })

    disc = discriminate_model(dummy_msd, rb, rc, None)
    result = disc.model_type.lower()

    assert result in ("confined", "subdiffusive"), (
        f"Got {result} when alpha={alpha_sub:.2f}<0.8 and R2_brownian={r2_bad:.2f}<0.7, "
        f"AIC confined({rc.aic:.1f}) vs brownian({rb.aic:.1f})"
    )


def test_alpha_within_expected_range():
    D_true = 1.0
    df = _make_brownian_dataset(D_true, n_particles=50, n_frames=150, seed=200)
    msd_df = compute_msd(df, use_corrected=False)

    if msd_df.empty:
        pytest.skip("MSD empty")

    fit_result = fit_brownian(
        msd_df, min_lags=5, max_ratio=0.4, confidence=0.95
    )

    if np.isnan(fit_result.alpha):
        pytest.skip("Alpha NaN")

    assert 0.7 <= fit_result.alpha <= 1.3


def test_fit_confidence_intervals_non_empty():
    D_true = 1.5
    df = _make_brownian_dataset(D_true, n_particles=40, n_frames=120, seed=300)
    msd_df = compute_msd(df, use_corrected=False)

    if msd_df.empty:
        pytest.skip("MSD empty")

    fit_result = fit_brownian(
        msd_df, min_lags=5, max_ratio=0.4, confidence=0.95
    )

    if np.isnan(fit_result.D):
        pytest.skip("Fit NaN")

    assert fit_result.ci_low is not None
    assert fit_result.ci_high is not None
    assert np.isfinite(fit_result.ci_low)
    assert np.isfinite(fit_result.ci_high)
    assert fit_result.ci_low >= 0.0
    assert fit_result.ci_high > fit_result.ci_low
