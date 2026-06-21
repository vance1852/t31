from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import theilslopes


def _robust_median_displacements(
    points_by_particle: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    all_dx: list[float] = []
    all_dy: list[float] = []
    all_frames: list[int] = []
    all_times: list[float] = []

    for pid, traj in points_by_particle.items():
        traj_sorted = traj.sort_values("frame")
        if len(traj_sorted) < 2:
            continue

        frames = traj_sorted["frame"].values
        times = traj_sorted["time_s"].values
        x = traj_sorted["x_um"].values
        y = traj_sorted["y_um"].values

        for i in range(1, len(traj_sorted)):
            dt = times[i] - times[i - 1]
            if dt <= 0:
                continue
            df = int(frames[i])
            all_frames.append(df)
            all_times.append(float(times[i]))
            all_dx.append(float(x[i] - x[i - 1]))
            all_dy.append(float(y[i] - y[i - 1]))

    if not all_frames:
        return pd.DataFrame(
            columns=["frame", "time_s", "dx_um", "dy_um", "n_particles"]
        )

    raw = pd.DataFrame(
        {
            "frame": all_frames,
            "time_s": all_times,
            "dx_um": all_dx,
            "dy_um": all_dy,
        }
    )

    grouped = raw.groupby("frame")
    median_df = grouped.agg(
        dx_um=("dx_um", "median"),
        dy_um=("dy_um", "median"),
        n_particles=("dx_um", "count"),
    ).reset_index()

    time_df = raw.groupby("frame")["time_s"].median().reset_index()
    median_df = median_df.merge(time_df, on="frame", how="left")

    median_df = median_df.sort_values("frame").reset_index(drop=True)
    return median_df[["frame", "time_s", "dx_um", "dy_um", "n_particles"]]


def estimate_global_drift(
    all_points_df: pd.DataFrame, outlier_sigma: float = 3.0
) -> pd.DataFrame:
    points_by_particle: dict[int, pd.DataFrame] = {}
    for pid, grp in all_points_df.groupby("particle_id"):
        points_by_particle[pid] = grp

    median_disp = _robust_median_displacements(points_by_particle)

    if len(median_disp) < 3:
        empty = pd.DataFrame(
            columns=["frame", "dx_um", "dy_um", "n_particles", "time_s"]
        )
        return empty

    median_disp = median_disp.sort_values("frame").reset_index(drop=True)

    def _robust_sigma(series: pd.Series) -> float:
        med = series.median()
        mad = (series - med).abs().median()
        if mad == 0 or np.isnan(mad):
            return np.std(series.values) or 1.0
        return 1.4826 * mad

    dx_sigma = _robust_sigma(median_disp["dx_um"])
    dy_sigma = _robust_sigma(median_disp["dy_um"])
    dx_median = median_disp["dx_um"].median()
    dy_median = median_disp["dy_um"].median()

    valid_mask = (
        (median_disp["dx_um"] - dx_median).abs() <= outlier_sigma * dx_sigma
    ) & (
        (median_disp["dy_um"] - dy_median).abs() <= outlier_sigma * dy_sigma
    )
    valid_data = median_disp[valid_mask].copy()

    if len(valid_data) < 3:
        valid_data = median_disp.copy()

    frames = valid_data["frame"].values.astype(float)
    times = valid_data["time_s"].values.astype(float)

    cumsum_dx = np.cumsum(valid_data["dx_um"].values)
    cumsum_dy = np.cumsum(valid_data["dy_um"].values)

    valid_data["cum_dx"] = cumsum_dx
    valid_data["cum_dy"] = cumsum_dy

    try:
        slope_x, intercept_x, _, _ = theilslopes(cumsum_dx, times)
        slope_y, intercept_y, _, _ = theilslopes(cumsum_dy, times)
    except Exception:
        if len(times) >= 2:
            slope_x, intercept_x = np.polyfit(times, cumsum_dx, 1)
            slope_y, intercept_y = np.polyfit(times, cumsum_dy, 1)
        else:
            slope_x, intercept_x = 0.0, 0.0
            slope_y, intercept_y = 0.0, 0.0

    all_frames = median_disp["frame"].values
    all_times = median_disp["time_s"].values

    fitted_dx = slope_x * all_times + intercept_x
    fitted_dy = slope_y * all_times + intercept_y

    drift_df = pd.DataFrame(
        {
            "frame": all_frames.astype(int),
            "time_s": all_times,
            "dx_um": fitted_dx,
            "dy_um": fitted_dy,
            "n_particles": median_disp["n_particles"].values,
        }
    )

    drift_df = drift_df.sort_values("frame").reset_index(drop=True)
    return drift_df[["frame", "dx_um", "dy_um", "n_particles", "time_s"]]


def subtract_drift(points_df: pd.DataFrame, drift_df: pd.DataFrame) -> pd.DataFrame:
    result_df = points_df.copy()

    if drift_df.empty:
        result_df["x_corrected_um"] = result_df["x_um"]
        result_df["y_corrected_um"] = result_df["y_um"]
        return result_df

    drift_sorted = drift_df.sort_values("frame").reset_index(drop=True)
    drift_frames = drift_sorted["frame"].values
    drift_dx = drift_sorted["dx_um"].values
    drift_dy = drift_sorted["dy_um"].values
    drift_times = drift_sorted["time_s"].values

    def _interp_drift(target_frames: np.ndarray, target_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(drift_frames) == 1:
            return (
                np.full_like(target_frames, drift_dx[0], dtype=float),
                np.full_like(target_frames, drift_dy[0], dtype=float),
            )

        dx_interp = np.interp(target_times, drift_times, drift_dx)
        dy_interp = np.interp(target_times, drift_times, drift_dy)
        return dx_interp, dy_interp

    target_frames = result_df["frame"].values
    target_times = result_df["time_s"].values
    dx_corr, dy_corr = _interp_drift(target_frames, target_times)

    result_df["x_corrected_um"] = result_df["x_um"].values - dx_corr
    result_df["y_corrected_um"] = result_df["y_um"].values - dy_corr

    return result_df


def compute_drift_velocity(
    drift_df: pd.DataFrame,
) -> tuple[float, float, pd.DataFrame]:
    if drift_df.empty or len(drift_df) < 2:
        return 0.0, 0.0, drift_df.copy()

    drift_sorted = drift_df.sort_values("frame").reset_index(drop=True)
    times = drift_sorted["time_s"].values.astype(float)
    dx = drift_sorted["dx_um"].values.astype(float)
    dy = drift_sorted["dy_um"].values.astype(float)

    try:
        vx, _, _, _ = theilslopes(dx, times)
        vy, _, _, _ = theilslopes(dy, times)
    except Exception:
        vx, _ = np.polyfit(times, dx, 1)
        vy, _ = np.polyfit(times, dy, 1)

    if np.isnan(vx):
        vx = 0.0
    if np.isnan(vy):
        vy = 0.0

    drift_out = drift_sorted.copy()
    return float(vx), float(vy), drift_out
