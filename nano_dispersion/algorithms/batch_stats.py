from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import numpy as np
import pandas as pd


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def aggregate_trajectory_results(
    results_list: list[Any], exclude_short: bool = True
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    exclude_reasons_counter: Counter[str] = Counter()

    D_vals: list[float] = []
    radius_vals: list[float] = []
    r2_vals: list[float] = []
    alpha_vals: list[float] = []
    valid_indices: list[int] = []

    per_channel_D: dict[int, list[float]] = {}
    per_channel_radius: dict[int, list[float]] = {}
    per_channel_count: dict[int, int] = {}
    per_channel_valid_count: dict[int, int] = {}

    n_total = len(results_list)
    n_excluded = 0

    drift_vx_list: list[float] = []
    drift_vy_list: list[float] = []

    for idx, res in enumerate(results_list):
        excluded = bool(_get_val(res, "excluded_from_distribution", False))
        exclude_reason = _get_val(res, "exclude_reason", None)
        n_frames = _get_val(res, "num_frames", None)

        channel_id = _get_val(res, "channel_id", None)
        if channel_id is not None:
            ch_key = str(channel_id)
            per_channel_count[ch_key] = per_channel_count.get(ch_key, 0) + 1

        if exclude_reason:
            exclude_reasons_counter[str(exclude_reason)] += 1

        if excluded and exclude_short:
            n_excluded += 1
            continue

        D = _get_val(res, "diffusion_D_um2_s", None)
        if D is None:
            D = _get_val(res, "D", None)
        radius = _get_val(res, "hydro_radius_nm", None)
        r2 = _get_val(res, "fit_r2", None)
        alpha = _get_val(res, "alpha_exponent", None)
        if alpha is None:
            alpha = _get_val(res, "alpha", None)

        vx = _get_val(res, "drift_velocity_x", None)
        vy = _get_val(res, "drift_velocity_y", None)
        if vx is not None and not (isinstance(vx, float) and np.isnan(vx)):
            drift_vx_list.append(float(vx))
        if vy is not None and not (isinstance(vy, float) and np.isnan(vy)):
            drift_vy_list.append(float(vy))

        D_valid = D is not None and not (isinstance(D, float) and np.isnan(D)) and D > 0
        r_valid = radius is not None and not (isinstance(radius, float) and np.isnan(radius)) and radius > 0

        if not D_valid:
            if exclude_short:
                n_excluded += 1
                exclude_reasons_counter["invalid_D"] += 1
                continue

        if D_valid:
            D_f = float(D)
            D_vals.append(D_f)
            valid_indices.append(idx)

            if channel_id is not None:
                ch_key = str(channel_id)
                if ch_key not in per_channel_D:
                    per_channel_D[ch_key] = []
                per_channel_D[ch_key].append(D_f)
                per_channel_valid_count[ch_key] = per_channel_valid_count.get(ch_key, 0) + 1

        if r_valid:
            r_f = float(radius)
            radius_vals.append(r_f)
            if channel_id is not None:
                ch_key = str(channel_id)
                if ch_key not in per_channel_radius:
                    per_channel_radius[ch_key] = []
                per_channel_radius[ch_key].append(r_f)

        if r2 is not None and not (isinstance(r2, float) and np.isnan(r2)):
            r2_vals.append(float(r2))
        if alpha is not None and not (isinstance(alpha, float) and np.isnan(alpha)):
            alpha_vals.append(float(alpha))

    stats["n_total"] = n_total
    stats["n_valid"] = len(D_vals)
    stats["n_excluded"] = n_excluded
    stats["exclude_reasons_counter"] = dict(exclude_reasons_counter)

    if D_vals:
        D_arr = np.array(D_vals, dtype=float)
        stats["mean_D"] = float(np.mean(D_arr))
        stats["median_D"] = float(np.median(D_arr))
        stats["std_D"] = float(np.std(D_arr, ddof=1)) if len(D_arr) > 1 else 0.0
        stats["percentile_25_D"] = float(np.percentile(D_arr, 25))
        stats["percentile_75_D"] = float(np.percentile(D_arr, 75))
        stats["min_D"] = float(np.min(D_arr))
        stats["max_D"] = float(np.max(D_arr))
    else:
        for k in ["mean_D", "median_D", "std_D", "percentile_25_D", "percentile_75_D", "min_D", "max_D"]:
            stats[k] = float("nan")

    if radius_vals:
        R_arr = np.array(radius_vals, dtype=float)
        stats["mean_radius_nm"] = float(np.mean(R_arr))
        stats["median_radius_nm"] = float(np.median(R_arr))
        stats["std_radius_nm"] = float(np.std(R_arr, ddof=1)) if len(R_arr) > 1 else 0.0
        stats["percentile_25_radius_nm"] = float(np.percentile(R_arr, 25))
        stats["percentile_75_radius_nm"] = float(np.percentile(R_arr, 75))
    else:
        for k in ["mean_radius_nm", "median_radius_nm", "std_radius_nm",
                  "percentile_25_radius_nm", "percentile_75_radius_nm"]:
            stats[k] = float("nan")

    if r2_vals:
        stats["mean_r2"] = float(np.mean(r2_vals))
        stats["median_r2"] = float(np.median(r2_vals))
    else:
        stats["mean_r2"] = float("nan")
        stats["median_r2"] = float("nan")

    if alpha_vals:
        stats["mean_alpha"] = float(np.mean(alpha_vals))
        stats["median_alpha"] = float(np.median(alpha_vals))
    else:
        stats["mean_alpha"] = float("nan")
        stats["median_alpha"] = float("nan")

    per_channel_stats: dict[int, dict[str, Any]] = {}
    all_ch_keys = set(list(per_channel_D.keys()) + list(per_channel_count.keys()))
    for ch_key in sorted(all_ch_keys):
        ch_stats: dict[str, Any] = {}
        ch_stats["count_total"] = per_channel_count.get(ch_key, 0)
        ch_stats["count_valid"] = per_channel_valid_count.get(ch_key, 0)

        ch_D = per_channel_D.get(ch_key, [])
        if ch_D:
            Dc = np.array(ch_D, dtype=float)
            ch_stats["mean_D"] = float(np.mean(Dc))
            ch_stats["median_D"] = float(np.median(Dc))
            ch_stats["std_D"] = float(np.std(Dc, ddof=1)) if len(Dc) > 1 else 0.0
        else:
            ch_stats["mean_D"] = float("nan")
            ch_stats["median_D"] = float("nan")
            ch_stats["std_D"] = float("nan")

        ch_R = per_channel_radius.get(ch_key, [])
        if ch_R:
            Rc = np.array(ch_R, dtype=float)
            ch_stats["mean_radius_nm"] = float(np.mean(Rc))
            ch_stats["median_radius_nm"] = float(np.median(Rc))
        else:
            ch_stats["mean_radius_nm"] = float("nan")
            ch_stats["median_radius_nm"] = float("nan")

        per_channel_stats[ch_key] = ch_stats

    stats["per_channel_stats"] = per_channel_stats

    drift_stats: dict[str, Any] = {}
    if drift_vx_list:
        drift_stats["mean_vx_um_per_s"] = float(np.mean(drift_vx_list))
        drift_stats["std_vx_um_per_s"] = float(np.std(drift_vx_list, ddof=1)) if len(drift_vx_list) > 1 else 0.0
    else:
        drift_stats["mean_vx_um_per_s"] = float("nan")
        drift_stats["std_vx_um_per_s"] = float("nan")

    if drift_vy_list:
        drift_stats["mean_vy_um_per_s"] = float(np.mean(drift_vy_list))
        drift_stats["std_vy_um_per_s"] = float(np.std(drift_vy_list, ddof=1)) if len(drift_vy_list) > 1 else 0.0
    else:
        drift_stats["mean_vy_um_per_s"] = float("nan")
        drift_stats["std_vy_um_per_s"] = float("nan")

    drift_stats["n_samples"] = max(len(drift_vx_list), len(drift_vy_list))
    stats["drift_stats"] = drift_stats

    return stats


def _compute_worst_score(res: Any, idx: int) -> tuple[float, dict[str, float]]:
    score_components: dict[str, float] = {}

    r2 = _get_val(res, "fit_r2", None)
    if r2 is None or (isinstance(r2, float) and np.isnan(r2)):
        r2_score = 1.0
    else:
        r2_score = max(0.0, 1.0 - float(r2))
    score_components["r2"] = r2_score * 3.0

    alpha = _get_val(res, "alpha_exponent", None)
    if alpha is None:
        alpha = _get_val(res, "alpha", None)
    if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)):
        alpha_score = 0.5
    else:
        alpha_score = min(1.0, abs(float(alpha) - 1.0))
    score_components["alpha"] = alpha_score * 2.0

    n_frames = _get_val(res, "num_frames", None)
    if n_frames is None or (isinstance(n_frames, float) and np.isnan(n_frames)):
        len_score = 0.5
    else:
        nf = int(n_frames)
        len_score = max(0.0, min(1.0, 1.0 - nf / 100.0)) if nf < 100 else 0.0
    score_components["short"] = len_score * 2.0

    excluded = bool(_get_val(res, "excluded_from_distribution", False))
    excl_score = 1.0 if excluded else 0.0
    score_components["excluded"] = excl_score * 1.5

    outliers = _get_val(res, "n_outliers", None)
    if outliers is None or (isinstance(outliers, float) and np.isnan(outliers)):
        outlier_score = 0.0
    else:
        n_out = int(outliers)
        tot = n_frames if n_frames is not None and not (isinstance(n_frames, float) and np.isnan(n_frames)) else max(1, n_out)
        outlier_score = min(1.0, n_out / max(1, tot))
    score_components["outliers"] = outlier_score * 1.5

    total = sum(score_components.values())
    return total, score_components


def flag_worst_trajectories(results_list: list[Any], n: int = 10) -> list[dict[str, Any]]:
    if not results_list:
        return []

    scored: list[tuple[float, int, dict[str, float], Any]] = []
    for idx, res in enumerate(results_list):
        total, comps = _compute_worst_score(res, idx)
        scored.append((total, idx, comps, res))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_n = scored[:n]

    worst_list: list[dict[str, Any]] = []
    for total, idx, comps, res in top_n:
        traj_id = _get_val(res, "trajectory_id", None)
        if traj_id is None:
            traj_id = _get_val(res, "id", idx)
        particle_id = _get_val(res, "particle_id", None)

        item: dict[str, Any] = {
            "index": idx,
            "trajectory_id": traj_id,
            "particle_id": particle_id,
            "total_score": float(total),
            "score_components": comps,
            "diffusion_D_um2_s": _get_val(res, "diffusion_D_um2_s", _get_val(res, "D", None)),
            "alpha_exponent": _get_val(res, "alpha_exponent", _get_val(res, "alpha", None)),
            "fit_r2": _get_val(res, "fit_r2", None),
            "hydro_radius_nm": _get_val(res, "hydro_radius_nm", None),
            "num_frames": _get_val(res, "num_frames", None),
            "excluded_from_distribution": _get_val(res, "excluded_from_distribution", False),
            "exclude_reason": _get_val(res, "exclude_reason", None),
            "model_type": _get_val(res, "model_type", None),
        }
        worst_list.append(item)

    return worst_list
