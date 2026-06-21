from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree


def sort_and_validate(points_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    sorted_df = points_df.sort_values(["particle_id", "frame"]).copy()
    issues: list[str] = []

    required_cols = {"particle_id", "frame", "time_s", "x_um", "y_um"}
    missing = required_cols - set(sorted_df.columns)
    if missing:
        issues.append(f"缺少必需列: {missing}")
        return sorted_df, issues

    for pid, grp in sorted_df.groupby("particle_id"):
        frames = grp["frame"].values
        if len(frames) < 2:
            continue
        frame_diffs = np.diff(frames)
        gaps = np.where(frame_diffs > 1)[0]
        for gap_idx in gaps:
            gap_start = int(frames[gap_idx])
            gap_end = int(frames[gap_idx + 1])
            missing_count = gap_end - gap_start - 1
            issues.append(
                f"particle {pid}: frame {gap_start}到{gap_end}之间缺失{missing_count}帧"
            )

        if np.any(frame_diffs <= 0):
            issues.append(f"particle {pid}: 存在帧号不单调")

        times = grp["time_s"].values
        if np.any(np.diff(times) <= 0):
            issues.append(f"particle {pid}: 存在时间戳不单调")

    if "is_outlier" not in sorted_df.columns:
        sorted_df["is_outlier"] = False
    if "anomaly_tag" not in sorted_df.columns:
        sorted_df["anomaly_tag"] = None

    sorted_df = sorted_df.reset_index(drop=True)
    return sorted_df, issues


def detect_short_trajectories(
    trajectories: dict[int, pd.DataFrame], min_frames: int
) -> list[int]:
    short_ids: list[int] = []
    for pid, df in trajectories.items():
        if len(df) < min_frames:
            short_ids.append(pid)
    return short_ids


def find_break_points(
    points_df: pd.DataFrame, particle_id: int
) -> list[tuple[int, int]]:
    sub = points_df[points_df["particle_id"] == particle_id].sort_values("frame")
    if len(sub) < 2:
        return []

    frames = sub["frame"].values
    frame_diffs = np.diff(frames)
    break_points: list[tuple[int, int]] = []

    for i in range(len(frame_diffs)):
        if frame_diffs[i] > 1:
            break_points.append((int(frames[i]), int(frames[i + 1])))

    return break_points


def reconnect_broken_trajectories(
    trajectories_dict: dict[int, pd.DataFrame],
    max_gap_frames: int = 5,
    max_distance_um: float = 3.0,
    max_velocity_um_per_s: float = 50.0,
) -> tuple[dict[int, int], list[dict]]:
    reconnected_mapping: dict[int, int] = {}
    reconnection_log: list[dict] = []

    pid_list = list(trajectories_dict.keys())
    endpoints: list[dict] = []

    for pid in pid_list:
        traj = trajectories_dict[pid].sort_values("frame")
        if len(traj) < 2:
            continue

        last_row = traj.iloc[-1]
        first_row = traj.iloc[0]

        endpoints.append(
            {
                "pid": pid,
                "is_end": True,
                "frame": int(last_row["frame"]),
                "time_s": float(last_row["time_s"]),
                "x_um": float(last_row["x_um"]),
                "y_um": float(last_row["y_um"]),
            }
        )
        endpoints.append(
            {
                "pid": pid,
                "is_end": False,
                "frame": int(first_row["frame"]),
                "time_s": float(first_row["time_s"]),
                "x_um": float(first_row["x_um"]),
                "y_um": float(first_row["y_um"]),
            }
        )

    if len(endpoints) < 2:
        return reconnected_mapping, reconnection_log

    ends = [e for e in endpoints if e["is_end"]]
    starts = [e for e in endpoints if not e["is_end"]]

    if not ends or not starts:
        return reconnected_mapping, reconnection_log

    used_source: set[int] = set()
    used_target: set[int] = set()

    for end_pt in ends:
        if end_pt["pid"] in used_source:
            continue
        if end_pt["pid"] in used_target:
            continue

        candidates = []
        for start_pt in starts:
            if start_pt["pid"] == end_pt["pid"]:
                continue
            if start_pt["pid"] in used_source:
                continue
            if start_pt["pid"] in used_target:
                continue

            frame_gap = start_pt["frame"] - end_pt["frame"]
            if frame_gap <= 0 or frame_gap > max_gap_frames:
                continue

            time_gap = start_pt["time_s"] - end_pt["time_s"]
            if time_gap <= 0:
                continue

            dx = start_pt["x_um"] - end_pt["x_um"]
            dy = start_pt["y_um"] - end_pt["y_um"]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > max_distance_um:
                continue

            velocity = dist / time_gap
            if velocity > max_velocity_um_per_s:
                continue

            candidates.append(
                    {
                        "target": start_pt,
                        "dist": dist,
                        "velocity": velocity,
                        "frame_gap": frame_gap,
                        "time_gap": time_gap,
                    }
                )

        if not candidates:
            continue

        candidates.sort(key=lambda c: (c["dist"], c["velocity"]))
        best = candidates[0]
        target_pid = best["target"]["pid"]

        reconnected_mapping[target_pid] = end_pt["pid"]
        used_source.add(end_pt["pid"])
        used_target.add(target_pid)

        reconnection_log.append(
            {
                "source_pid": end_pt["pid"],
                "target_pid": target_pid,
                "distance_um": best["dist"],
                "velocity_um_per_s": best["velocity"],
                "frame_gap": best["frame_gap"],
                "time_gap_s": best["time_gap"],
            }
        )

    return reconnected_mapping, reconnection_log


def detect_outliers(points_df: pd.DataFrame, sigma: float = 3.0) -> pd.DataFrame:
    result_df = points_df.copy()

    if "is_outlier" not in result_df.columns:
        result_df["is_outlier"] = False
    if "anomaly_tag" not in result_df.columns:
        result_df["anomaly_tag"] = None

    def _mad_outliers(series: pd.Series) -> pd.Series:
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0 or np.isnan(mad):
            return pd.Series([False] * len(series), index=series.index)
        modified_zscore = 0.6745 * (series - median) / mad
        return modified_zscore.abs() > sigma

    for pid, grp in result_df.groupby("particle_id"):
        if len(grp) < 5:
            continue

        idx = grp.index
        x_outliers = _mad_outliers(grp["x_um"])
        y_outliers = _mad_outliers(grp["y_um"])

        is_outlier = x_outliers | y_outliers
        result_df.loc[idx[is_outlier], "is_outlier"] = True
        tags = []
        for i, (xo, yo) in enumerate(zip(x_outliers, y_outliers)):
            parts = []
            if xo:
                parts.append("x_jump")
            if yo:
                parts.append("y_jump")
            tags.append("|".join(parts) if parts else None)
        for i, t in enumerate(tags):
            if t is not None:
                existing = result_df.loc[idx[i], "anomaly_tag"]
                is_none = existing is None
                is_nan_float = isinstance(existing, float) and np.isnan(existing)
                if is_none or is_nan_float:
                    result_df.loc[idx[i], "anomaly_tag"] = t
                else:
                    result_df.loc[idx[i], "anomaly_tag"] = str(existing) + "|" + t

    return result_df


def detect_cross_channel(
    points_df: pd.DataFrame, channel_bounds: dict[int, tuple[float, float, float, float]]
) -> list[int]:
    cross_pids: list[int] = []

    for pid, grp in points_df.groupby("particle_id"):
        if len(grp) < 2:
            continue
        x_min, x_max = grp["x_um"].min(), grp["x_um"].max()
        y_min, y_max = grp["y_um"].min(), grp["y_um"].max()

        in_any_channel = False
        for ch_id, (cx_min, cx_max, cy_min, cy_max) in channel_bounds.items():
            if (
                x_min >= cx_min and x_max <= cx_max and y_min >= cy_min and y_max <= cy_max
            ):
                in_any_channel = True
                break

        if not in_any_channel and channel_bounds:
            cross_pids.append(pid)

    return cross_pids


def detect_intensity_drop(
    points_df: pd.DataFrame, drop_ratio: float = 0.5, window: int = 5
) -> list[int]:
    if "intensity" not in points_df.columns:
        return []

    drop_pids: list[int] = []
    half = window // 2

    for pid, grp in points_df.groupby("particle_id"):
        if len(grp) < window:
            continue
        grp_sorted = grp.sort_values("frame")
        intensities = grp_sorted["intensity"].values
        has_drop = False

        for i in range(half, len(intensities) - half):
            before_mean = np.nanmean(intensities[i - half : i])
            after_mean = np.nanmean(intensities[i : i + half])
            if before_mean > 0 and after_mean / before_mean < drop_ratio:
                has_drop = True
                break

        if has_drop:
            drop_pids.append(pid)

    return drop_pids
