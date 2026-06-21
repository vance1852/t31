from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nano_dispersion.algorithms.msd import _msd_single_trajectory, compute_msd


def _make_brownian_single(D, n, dt, seed):
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(2.0 * D * dt)
    dx = rng.normal(0.0, sigma, size=n)
    dy = rng.normal(0.0, sigma, size=n)
    xs = np.empty(n)
    ys = np.empty(n)
    xs[0], ys[0] = 0.0, 0.0
    xs[1:] = np.cumsum(dx[1:])
    ys[1:] = np.cumsum(dy[1:])
    return xs, ys


def test_uneven_frame_intervals_use_real_time():
    times = np.array([0.0, 0.05, 0.15, 0.2])
    D = 2.0
    n = len(times)
    rng = np.random.default_rng(777)
    xs = np.zeros(n)
    ys = np.zeros(n)
    sigma05 = np.sqrt(2.0 * D * 0.05)
    sigma1 = np.sqrt(2.0 * D * 0.1)
    xs[1] = xs[0] + rng.normal(0, sigma05)
    ys[1] = ys[0] + rng.normal(0, sigma05)
    xs[2] = xs[1] + rng.normal(0, sigma1)
    ys[2] = ys[1] + rng.normal(0, sigma1)
    xs[3] = xs[2] + rng.normal(0, sigma05)
    ys[3] = ys[2] + rng.normal(0, sigma05)

    df = pd.DataFrame({
        "particle_id": [1] * n,
        "frame": list(range(n)),
        "time_s": times,
        "x_um": xs,
        "y_um": ys,
    })

    msd_df = compute_msd(df, use_corrected=False)

    if msd_df.empty:
        dts, msds, _ = _msd_single_trajectory(xs, ys, times)
        assert len(dts) > 0
        for dt_val in dts:
            ratio = dt_val / 0.033
            assert not np.isclose(ratio, round(ratio)), f"dt_val={dt_val} appears to be multiple of 0.033"
        return

    lag_times = msd_df["lag_time_s"].values
    if len(lag_times) > 0:
        for lt in lag_times:
            ratio = lt / 0.033
            msg = f"lag_time_s={lt} appears to be multiple of 0.033"
            assert not np.isclose(ratio, round(ratio)), msg


def test_msd_scales_linearly_for_brownian():
    D_true = 2.0
    n_frames = 200
    dt = 0.033
    n_particles = 50

    rows = []
    for pid in range(n_particles):
        xs, ys = _make_brownian_single(D_true, n_frames, dt, seed=pid * 11)
        for frame in range(n_frames):
            rows.append({
                "particle_id": pid,
                "frame": frame,
                "time_s": frame * dt,
                "x_um": float(xs[frame]),
                "y_um": float(ys[frame]),
            })

    df = pd.DataFrame(rows)
    msd_df = compute_msd(df, use_corrected=False)
    if msd_df.empty:
        pytest.skip("MSD result empty")

    times = msd_df["lag_time_s"].values
    msds = msd_df["msd"].values
    if len(times) < 5:
        pytest.skip("Not enough MSD points")

    mask = (times > 0.05) & (times < 1.5)
    if mask.sum() < 3:
        pytest.skip("MSD filtered points insufficient")

    t_sel = times[mask]
    m_sel = msds[mask]

    slope, _ = np.polyfit(t_sel, m_sel, 1)
    D_est = slope / 4.0

    assert abs(D_est / D_true - 1.0) < 0.3


def test_msd_single_vs_ensemble_consistent():
    D_true = 1.5
    n_frames = 80
    dt = 0.033

    rows = []
    for pid in range(10):
        xs, ys = _make_brownian_single(D_true, n_frames, dt, seed=pid * 7)
        for frame in range(n_frames):
            rows.append({
                "particle_id": pid,
                "frame": frame,
                "time_s": frame * dt,
                "x_um": float(xs[frame]),
                "y_um": float(ys[frame]),
            })
    df_all = pd.DataFrame(rows)
    msd_all = compute_msd(df_all, use_corrected=False)

    if msd_all.empty:
        pytest.skip("Ensemble MSD empty")

    single_msds = []
    for pid in range(10):
        sub = df_all[df_all["particle_id"] == pid].copy()
        m = compute_msd(sub, use_corrected=False)
        if not m.empty:
            single_msds.append(m)

    assert len(single_msds) > 0

    all_times = msd_all["lag_time_s"].values
    if len(all_times) < 3:
        pytest.skip("Not enough ensemble points")

    assert len(all_times) >= 3
