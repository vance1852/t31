from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nano_dispersion.algorithms.trajectory_qc import (
    detect_cross_channel,
    detect_outliers,
    detect_short_trajectories,
    reconnect_broken_trajectories,
    sort_and_validate,
)


def test_sort_and_validate():
    rows = [
        {"particle_id": 1, "frame": 5, "time_s": 0.165, "x_um": 1.0, "y_um": 2.0},
        {"particle_id": 1, "frame": 3, "time_s": 0.099, "x_um": 0.8, "y_um": 1.8},
        {"particle_id": 1, "frame": 4, "time_s": 0.132, "x_um": 0.9, "y_um": 1.9},
        {"particle_id": 1, "frame": 8, "time_s": 0.264, "x_um": 1.3, "y_um": 2.3},
        {"particle_id": 2, "frame": 0, "time_s": 0.0, "x_um": 5.0, "y_um": 5.0},
        {"particle_id": 2, "frame": 1, "time_s": 0.033, "x_um": 5.1, "y_um": 5.1},
    ]
    df = pd.DataFrame(rows)
    sorted_df, issues = sort_and_validate(df)

    assert list(sorted_df.columns)[:5] == ["particle_id", "frame", "time_s", "x_um", "y_um"]
    frames_p1 = sorted_df[sorted_df["particle_id"] == 1]["frame"].values
    assert np.all(np.diff(frames_p1) >= 1)
    assert len(issues) >= 1
    assert any("缺失" in issue for issue in issues)
    assert "is_outlier" in sorted_df.columns
    assert "anomaly_tag" in sorted_df.columns


def test_detect_short_trajectories():
    long_df = pd.DataFrame({
        "frame": list(range(50)),
        "time_s": [i * 0.033 for i in range(50)],
        "x_um": [0.0] * 50,
        "y_um": [0.0] * 50,
    })
    short_df = pd.DataFrame({
        "frame": list(range(10)),
        "time_s": [i * 0.033 for i in range(10)],
        "x_um": [0.0] * 10,
        "y_um": [0.0] * 10,
    })
    trajectories = {1: long_df, 2: short_df, 3: long_df}
    short_ids = detect_short_trajectories(trajectories, min_frames=20)

    assert 2 in short_ids
    assert 1 not in short_ids
    assert 3 not in short_ids


def test_reconnect_broken_trajectories_no_false_connections():
    dt = 0.033

    traj_a_end = pd.DataFrame({
        "frame": [0, 1, 2, 3, 4],
        "time_s": [0, dt, 2 * dt, 3 * dt, 4 * dt],
        "x_um": [10.0, 10.1, 10.2, 10.3, 10.4],
        "y_um": [20.0, 20.05, 20.1, 20.15, 20.2],
    })

    traj_b_start = pd.DataFrame({
        "frame": [7, 8, 9, 10, 11],
        "time_s": [7 * dt, 8 * dt, 9 * dt, 10 * dt, 11 * dt],
        "x_um": [15.5, 15.6, 15.7, 15.8, 15.9],
        "y_um": [25.5, 25.55, 25.6, 25.65, 25.7],
    })

    traj_c_end = pd.DataFrame({
        "frame": [0, 1, 2, 3, 4],
        "time_s": [0, dt, 2 * dt, 3 * dt, 4 * dt],
        "x_um": [30.0, 30.1, 30.2, 30.3, 30.4],
        "y_um": [40.0, 40.05, 40.1, 40.15, 40.2],
    })

    traj_d_start = pd.DataFrame({
        "frame": [6, 7, 8, 9, 10],
        "time_s": [6 * dt, 7 * dt, 8 * dt, 9 * dt, 10 * dt],
        "x_um": [30.5, 30.6, 30.7, 30.8, 30.9],
        "y_um": [40.22, 40.27, 40.32, 40.37, 40.42],
    })

    trajectories_far = {1: traj_a_end, 2: traj_b_start}
    mapping_far, log_far = reconnect_broken_trajectories(
        trajectories_far,
        max_gap_frames=5,
        max_distance_um=3.0,
        max_velocity_um_per_s=50.0,
    )
    assert len(mapping_far) == 0
    assert len(log_far) == 0

    trajectories_close = {3: traj_c_end, 4: traj_d_start}
    mapping_close, log_close = reconnect_broken_trajectories(
        trajectories_close,
        max_gap_frames=5,
        max_distance_um=3.0,
        max_velocity_um_per_s=50.0,
    )
    assert len(mapping_close) == 1
    assert 4 in mapping_close
    assert mapping_close[4] == 3


def test_detect_outliers_marks_only():
    rng = np.random.default_rng(99)
    n = 50
    base_x = np.cumsum(rng.normal(0, 0.1, n))
    base_y = np.cumsum(rng.normal(0, 0.1, n))
    base_x[25] += 100.0
    base_y[25] += 100.0

    df = pd.DataFrame({
        "particle_id": [1] * n,
        "frame": list(range(n)),
        "time_s": [i * 0.033 for i in range(n)],
        "x_um": base_x,
        "y_um": base_y,
    })

    result = detect_outliers(df, sigma=3.0)

    assert len(result) == len(df)
    assert "is_outlier" in result.columns
    assert True in result["is_outlier"].values
    assert result["is_outlier"].iloc[25] == True


def test_detect_cross_channel_triggers_blocking():
    channel_bounds = {
        0: (0.0, 200.0, 0.0, 50.0),
        1: (200.0, 400.0, 0.0, 50.0),
    }

    inside_df = pd.DataFrame({
        "particle_id": [1] * 10,
        "frame": list(range(10)),
        "time_s": [i * 0.033 for i in range(10)],
        "x_um": [50.0 + i * 0.5 for i in range(10)],
        "y_um": [25.0] * 10,
    })

    cross_df = pd.DataFrame({
        "particle_id": [2] * 10,
        "frame": list(range(10)),
        "time_s": [i * 0.033 for i in range(10)],
        "x_um": [150.0 + i * 8.0 for i in range(10)],
        "y_um": [25.0] * 10,
    })

    combined = pd.concat([inside_df, cross_df], ignore_index=True)
    cross_pids = detect_cross_channel(combined, channel_bounds)

    assert 2 in cross_pids
    assert 1 not in cross_pids
