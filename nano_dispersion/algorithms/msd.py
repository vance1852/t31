from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _msd_single_trajectory(
    x: np.ndarray, y: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    if n < 2:
        return np.array([]), np.array([]), np.array([])

    all_dt: list[float] = []
    all_msd: list[float] = []
    all_count: list[int] = []

    for lag in range(1, n):
        for i in range(n - lag):
            j = i + lag
            dt = times[j] - times[i]
            if dt <= 0:
                continue
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            msd_val = dx * dx + dy * dy
            all_dt.append(float(dt))
            all_msd.append(float(msd_val))
            all_count.append(1)

    if not all_dt:
        return np.array([]), np.array([]), np.array([])

    dt_arr = np.array(all_dt)
    msd_arr = np.array(all_msd)

    sorted_idx = np.argsort(dt_arr)
    dt_sorted = dt_arr[sorted_idx]
    msd_sorted = msd_arr[sorted_idx]

    if len(dt_sorted) == 0:
        return np.array([]), np.array([]), np.array([])

    return dt_sorted, msd_sorted, np.ones_like(dt_sorted, dtype=int)


def _bin_msd_by_time(
    dts: np.ndarray, msds: np.ndarray, counts: np.ndarray, n_bins: int = 50
) -> pd.DataFrame:
    if len(dts) == 0:
        return pd.DataFrame(
            columns=["lag_frames", "lag_time_s", "msd_x", "msd_y", "msd", "count"]
        )

    dt_min, dt_max = dts.min(), dts.max()
    if dt_max <= dt_min:
        bins = np.array([dt_min, dt_max + 1e-12])
    else:
        bin_edges = np.logspace(np.log10(dt_min), np.log10(dt_max * 1.001), n_bins + 1)
        unique_edges = np.unique(bin_edges)
        if len(unique_edges) < 2:
            bins = np.array([dt_min, dt_max + 1e-12])
        else:
            bins = unique_edges

    bin_centers = []
    msd_binned = []
    count_binned = []

    for k in range(len(bins) - 1):
        mask = (dts >= bins[k]) & (dts < bins[k + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append(float(np.mean(dts[mask])))
        msd_binned.append(float(np.mean(msds[mask])))
        count_binned.append(int(mask.sum()))

    if not bin_centers:
        return pd.DataFrame(
            columns=["lag_frames", "lag_time_s", "msd_x", "msd_y", "msd", "count"]
        )

    result = pd.DataFrame(
        {
            "lag_time_s": bin_centers,
            "msd": msd_binned,
            "count": count_binned,
        }
    )
    result = result.sort_values("lag_time_s").reset_index(drop=True)

    if len(result) > 0:
        dt0 = result["lag_time_s"].iloc[0] if result["lag_time_s"].iloc[0] > 0 else 1.0
        result["lag_frames"] = (result["lag_time_s"] / dt0).round().astype(int)
    else:
        result["lag_frames"] = pd.Series(dtype=int)

    result["msd_x"] = result["msd"] / 2.0
    result["msd_y"] = result["msd"] / 2.0

    return result[["lag_frames", "lag_time_s", "msd_x", "msd_y", "msd", "count"]]


def compute_msd(
    points_df: pd.DataFrame,
    use_corrected: bool = True,
    max_lags: Optional[int] = None,
) -> pd.DataFrame:
    x_col = "x_corrected_um" if use_corrected and "x_corrected_um" in points_df.columns else "x_um"
    y_col = "y_corrected_um" if use_corrected and "y_corrected_um" in points_df.columns else "y_um"

    all_dts: list[np.ndarray] = []
    all_msds: list[np.ndarray] = []

    for pid, grp in points_df.groupby("particle_id"):
        traj = grp.sort_values("frame")
        if len(traj) < 3:
            continue

        is_outlier_col = "is_outlier" if "is_outlier" in traj.columns else None
        if is_outlier_col is not None:
            valid = ~traj["is_outlier"].values.astype(bool)
        else:
            valid = np.ones(len(traj), dtype=bool)

        if valid.sum() < 3:
            continue

        traj_valid = traj.loc[valid]
        if len(traj_valid) < 3:
            continue

        x = traj_valid[x_col].values.astype(float)
        y = traj_valid[y_col].values.astype(float)
        times = traj_valid["time_s"].values.astype(float)

        dts, msds, cnts = _msd_single_trajectory(x, y, times)
        if len(dts) == 0:
            continue

        if max_lags is not None and len(dts) > 0:
            n_max = int(max_lags)
            if len(dts) > n_max:
                dts = dts[:n_max]
                msds = msds[:n_max]

        all_dts.append(dts)
        all_msds.append(msds)

    if not all_dts:
        return pd.DataFrame(
            columns=["lag_frames", "lag_time_s", "msd_x", "msd_y", "msd", "count"]
        )

    concat_dts = np.concatenate(all_dts)
    concat_msds = np.concatenate(all_msds)
    concat_counts = np.ones_like(concat_dts, dtype=int)

    return _bin_msd_by_time(concat_dts, concat_msds, concat_counts)


def compute_ensemble_msd(
    trajectories_msd_list: list[pd.DataFrame],
) -> pd.DataFrame:
    if not trajectories_msd_list:
        return pd.DataFrame(
            columns=["lag_frames", "lag_time_s", "msd_x", "msd_y", "msd", "count"]
        )

    all_dts: list[np.ndarray] = []
    all_msds: list[np.ndarray] = []

    for msd_df in trajectories_msd_list:
        if msd_df.empty:
            continue
        if "lag_time_s" not in msd_df.columns or "msd" not in msd_df.columns:
            continue

        times = msd_df["lag_time_s"].values.astype(float)
        msds = msd_df["msd"].values.astype(float)
        counts = msd_df.get("count", pd.Series(np.ones(len(msd_df), dtype=int))).values.astype(int)

        for t, m, c in zip(times, msds, counts):
            all_dts.append(np.full(int(c), t))
            all_msds.append(np.full(int(c), m))

    if not all_dts:
        return pd.DataFrame(
            columns=["lag_frames", "lag_time_s", "msd_x", "msd_y", "msd", "count"]
        )

    concat_dts = np.concatenate(all_dts)
    concat_msds = np.concatenate(all_msds)
    concat_counts = np.ones_like(concat_dts, dtype=int)

    return _bin_msd_by_time(concat_dts, concat_msds, concat_counts)
