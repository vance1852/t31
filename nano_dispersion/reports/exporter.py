from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class ReportPaths:
    json: Path
    csv: dict[str, Path] = field(default_factory=dict)
    markdown: Optional[Path] = None


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and np.isnan(value):
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _trajectory_score(traj_dict: dict) -> tuple[float, dict[str, float]]:
    score_components: dict[str, float] = {}

    r2 = _get_val(traj_dict, "fit_r2", None)
    if r2 is None or (isinstance(r2, float) and np.isnan(r2)):
        r2_score = 1.0
    else:
        r2_score = max(0.0, 1.0 - float(r2))
    score_components["r2"] = r2_score * 3.0

    result = _get_val(traj_dict, "result", traj_dict)
    alpha = _get_val(result, "alpha", None)
    if alpha is None:
        alpha = _get_val(result, "alpha_exponent", None)
    if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)):
        alpha_score = 0.5
    else:
        alpha_score = min(1.0, abs(float(alpha) - 1.0))
    score_components["alpha"] = alpha_score * 2.0

    n_frames = _get_val(traj_dict, "num_frames", None)
    if n_frames is None or (isinstance(n_frames, float) and np.isnan(n_frames)):
        len_score = 0.5
    else:
        nf = int(n_frames)
        len_score = max(0.0, min(1.0, 1.0 - nf / 100.0)) if nf < 100 else 0.0
    score_components["short"] = len_score * 2.0

    excluded = _get_val(traj_dict, "excluded_from_distribution", None)
    if excluded is None and result is not traj_dict:
        excluded = _get_val(result, "excluded_from_distribution", False)
    excl_score = 1.0 if excluded else 0.0
    score_components["excluded"] = excl_score * 1.5

    flags = _get_val(traj_dict, "flags", []) or []
    if isinstance(flags, list):
        n_flags = len(flags)
    elif isinstance(flags, dict):
        n_flags = sum(1 for v in flags.values() if v)
    else:
        n_flags = 0
    flag_score = min(1.0, n_flags / 5.0)
    score_components["flags"] = flag_score * 1.0

    total = sum(score_components.values())
    return total, score_components


def _model_statistics(trajectories: list[dict]) -> dict[str, int]:
    stats: dict[str, int] = {
        "brownian": 0,
        "confined": 0,
        "directed": 0,
        "subdiffusive": 0,
        "superdiffusive": 0,
        "anomalous": 0,
        "unknown": 0,
    }
    for traj in trajectories:
        result = _get_val(traj, "result", traj)
        model_type = _get_val(result, "model_type", None)
        if model_type is None:
            stats["unknown"] += 1
        else:
            mt = str(model_type).lower()
            if mt in stats:
                stats[mt] += 1
            else:
                stats["unknown"] += 1
    return stats


def _channel_comparison(trajectories: list[dict]) -> list[dict[str, Any]]:
    per_channel_D: dict[str, list[float]] = {}
    per_channel_count: dict[str, int] = {}
    per_channel_valid: dict[str, int] = {}

    for traj in trajectories:
        ch_id = _get_val(traj, "channel_id", None)
        if ch_id is None:
            continue
        ch_key = str(ch_id)

        per_channel_count[ch_key] = per_channel_count.get(ch_key, 0) + 1

        result = _get_val(traj, "result", traj)
        excluded = _get_val(result, "excluded_from_distribution", False)
        D = _get_val(result, "D_um2_s", None)
        if D is None:
            D = _get_val(result, "diffusion_D_um2_s", None)
        if not excluded and D is not None and not (isinstance(D, float) and np.isnan(D)):
            if ch_key not in per_channel_D:
                per_channel_D[ch_key] = []
            per_channel_D[ch_key].append(float(D))
            per_channel_valid[ch_key] = per_channel_valid.get(ch_key, 0) + 1

    channel_list: list[dict[str, Any]] = []
    all_keys = sorted(set(list(per_channel_D.keys()) + list(per_channel_count.keys())))
    for ch in all_keys:
        D_list = per_channel_D.get(ch, [])
        if D_list:
            D_arr = np.array(D_list, dtype=float)
            channel_list.append({
                "channel_id": ch,
                "n_total": per_channel_count.get(ch, 0),
                "n_valid": per_channel_valid.get(ch, 0),
                "mean_D": float(np.mean(D_arr)),
                "median_D": float(np.median(D_arr)),
                "std_D": float(np.std(D_arr, ddof=1)) if len(D_arr) > 1 else 0.0,
            })
        else:
            channel_list.append({
                "channel_id": ch,
                "n_total": per_channel_count.get(ch, 0),
                "n_valid": per_channel_valid.get(ch, 0),
                "mean_D": float("nan"),
                "median_D": float("nan"),
                "std_D": float("nan"),
            })
    return channel_list


def _get_retest_suggestions(
    qc: dict, summary: dict, calibration: Optional[dict]) -> list[str]:
    suggestions: list[str] = []

    total = _get_val(summary, "total_trajectories", None)
    if total is None:
        total = _get_val(qc, "total_trajectories", None)
    short_count = _get_val(qc, "short_trajectories", 0) or 0
    valid = _get_val(summary, "valid_trajectories", None)
    if valid is None:
        valid = _get_val(summary, "n_valid", None)

    if total and total > 0:
        short_ratio = short_count / total
        if short_ratio > 0.25:
            suggestions.append(f"短轨迹比例过高 ({short_ratio * 100:.1f}% > 25%)，建议优化成像参数或轨迹提取算法")

    if valid is not None and valid < 10:
        suggestions.append(f"有效轨迹数量不足 ({valid}条 < 10条)，统计结果可靠性差，建议增加数据量")

    drift = _get_val(qc, "drift_estimate", None)
    if drift is None:
        drift_summary = _get_val(summary, "drift_summary", None) or _get_val(summary, "drift_stats", None)
        if drift_summary:
            vx = _get_val(drift_summary, "mean_vx_um_per_s", 0.0) or 0.0
            vy = _get_val(drift_summary, "mean_vy_um_per_s", 0.0) or 0.0
            drift_mag = np.sqrt(vx ** 2 + vy ** 2) if (vx or vy) else None
            if drift_mag is not None and drift_mag > 1.0:
                suggestions.append(f"整体漂移过大 ({drift_mag:.3f} um/s > 1 um/s)，建议检查样品稳定性")

    if drift is not None and not (isinstance(drift, float) and np.isnan(drift)):
        if float(drift) > 1.0:
            suggestions.append(f"漂移估计过大 ({drift:.3f} um/s > 1 um/s)，建议检查样品稳定性")

    channel_stats = _get_val(summary, "per_channel_stats", None)
    if not channel_stats:
        channel_list = _channel_comparison([])
    else:
        channel_list = []
        if isinstance(channel_stats, dict):
            for ch_key, ch_data in channel_stats.items():
                entry = dict(ch_data)
                entry["channel_id"] = ch_key
                channel_list.append(entry)

    if len(channel_list) >= 2:
        mean_Ds = [c["mean_D"] for c in channel_list if c.get("mean_D") is not None and not (isinstance(c.get("mean_D"), float) and np.isnan(c.get("mean_D")))]
        if len(mean_Ds) >= 2:
            mean_arr = np.array(mean_Ds, dtype=float)
            cv = np.std(mean_arr, ddof=1) / np.mean(mean_arr) * 100 if np.mean(mean_arr) != 0 else 0
            if cv > 30:
                suggestions.append(f"通道间 D 值差异过大 (CV={cv:.1f}% > 30%)，建议检查通道一致性")

    metadata_conflicts = _get_val(qc, "metadata_conflicts", None)
    if metadata_conflicts and isinstance(metadata_conflicts, dict) and metadata_conflicts:
        for key, val in metadata_conflicts.items():
            if "temperature" in str(key).lower():
                suggestions.append(f"温度记录存在冲突: {val}")

    cc_count = _get_val(qc, "cross_channel_count", 0) or 0
    if cc_count > 0:
        suggestions.append(f"存在 {cc_count} 条跨通道粒子，为阻塞级异常，建议检查通道边界或粒子追踪参数")

    return suggestions


class ReportExporter:
    def __init__(self, result_dir: Path):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        batch_id: str,
        task_id: str,
        batch_data: dict,
        trajectories: list[dict],
        qc_report: dict,
        calibration: Optional[dict],
    ) -> ReportPaths:
        json_path = self.export_json(batch_id, task_id, batch_data, trajectories, qc_report, calibration)
        csv_paths = self.export_csv(batch_id, task_id, trajectories, qc_report, calibration)
        md_path = self.export_markdown(batch_id, task_id, batch_data, trajectories, qc_report, calibration)
        return ReportPaths(json=json_path, csv=csv_paths, markdown=md_path)

    def _extract_trajectory_result(self, traj: dict) -> dict:
        result = _get_val(traj, "result", traj)
        msd_points = _get_val(result, "msd_points", None)
        if msd_points:
            msd_serializable = []
            for pt in msd_points:
                if isinstance(pt, dict):
                    msd_serializable.append([
                        _get_val(pt, "lag_s", _get_val(pt, "time_s", _get_val(pt, "lag_time_s", 0.0))),
                        _get_val(pt, "msd", 0.0),
                        _get_val(pt, "count", _get_val(pt, "n_points", 1)),
                    ])
                else:
                    try:
                        msd_serializable.append([float(pt[0]), float(pt[1]), int(pt[2]) if len(pt) > 2 else 1])
                    except Exception:
                        pass
        else:
            msd_serializable = []

        fit_interval_start = _get_val(result, "fit_interval", None)
        if fit_interval_start is None:
            fs = _get_val(result, "lag_start_s", _get_val(result, "fit_lag_start", None))
            fe = _get_val(result, "lag_end_s", _get_val(result, "fit_lag_end", None))
            if fs is not None and fe is not None:
                fit_interval = [fs, fe]
            else:
                fit_interval = None
        else:
            fit_interval = list(fit_interval_start) if isinstance(fit_interval_start, (list, tuple)) else None

        ci_95 = _get_val(result, "ci_95", None)
        if ci_95 is None:
            cl = _get_val(result, "ci_low", None)
            ch = _get_val(result, "ci_high", None)
            if cl is not None or ch is not None:
                ci_95 = [cl if cl is not None else float("nan"), ch if ch is not None else float("nan")]

        drift_vx = _get_val(result, "drift_velocity_x", None)
        drift_vy = _get_val(result, "drift_velocity_y", None)
        drift_vel = _get_val(result, "drift_velocity", None)
        if drift_vel is None:
            if drift_vx is not None or drift_vy is not None:
                drift_velocity = [
                    drift_vx if drift_vx is not None else 0.0,
                    drift_vy if drift_vy is not None else 0.0,
                ]
        else:
            if isinstance(drift_vel, (list, tuple)) and len(drift_vel) >= 2:
                drift_velocity = list(drift_vel)
            else:
                drift_velocity = [drift_vx if drift_vx is not None else 0.0, drift_vy if drift_vy is not None else 0.0]

        D_val = _get_val(result, "D_um2_s", None)
        if D_val is None:
            D_val = _get_val(result, "diffusion_D_um2_s", None)

        alpha_val = _get_val(result, "alpha", None)
        if alpha_val is None:
            alpha_val = _get_val(result, "alpha_exponent", None)

        return {
            "D_um2_s": D_val,
            "alpha": alpha_val,
            "R2": _get_val(result, "R2", _get_val(result, "fit_r2", None)),
            "model_type": _get_val(result, "model_type", None),
            "model_reason": _get_val(result, "model_reason", None),
            "hydro_radius_nm": _get_val(result, "hydro_radius_nm", None),
            "fit_interval": fit_interval,
            "ci_95": ci_95,
            "drift_velocity": drift_velocity,
            "excluded_from_distribution": _get_val(result, "excluded_from_distribution", False),
            "exclude_reason": _get_val(result, "exclude_reason", None),
            "msd_points": msd_serializable,
        }

    def _build_batch_summary(self, trajectories: list[dict], qc_report: dict) -> dict:
        D_vals: list[float] = []
        radius_vals: list[float] = []
        n_valid = 0
        n_total = len(trajectories)
        exclude_counter: dict[str, int] = {}

        for traj in trajectories:
            result = self._extract_trajectory_result(traj)
            excluded = result["excluded_from_distribution"]
            reason = result["exclude_reason"]
            if reason:
                exclude_counter[str(reason)] = exclude_counter.get(str(reason), 0) + 1
            if not excluded:
                D = result["D_um2_s"]
                r = result["hydro_radius_nm"]
                if D is not None and not (isinstance(D, float) and np.isnan(D)) and D > 0:
                    D_vals.append(float(D))
                    n_valid += 1
                if r is not None and not (isinstance(r, float) and np.isnan(r)) and r > 0:
                    radius_vals.append(float(r))

        stats: dict[str, Any] = {
            "total_trajectories": n_total,
            "valid_trajectories": n_valid,
            "excluded_reasons": exclude_counter,
        }

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

        drift_vx_list: list[float] = []
        drift_vy_list: list[float] = []
        for traj in trajectories:
            result = self._extract_trajectory_result(traj)
            dv = result["drift_velocity"]
            if dv and len(dv) >= 2:
                vx, vy = dv[0], dv[1]
                if vx is not None and not (isinstance(vx, float) and np.isnan(vx)):
                    drift_vx_list.append(float(vx))
                if vy is not None and not (isinstance(vy, float) and np.isnan(vy)):
                    drift_vy_list.append(float(vy))

        drift_summary: dict[str, Any] = {}
        if drift_vx_list:
            drift_summary["mean_vx_um_per_s"] = float(np.mean(drift_vx_list))
            drift_summary["std_vx_um_per_s"] = float(np.std(drift_vx_list, ddof=1)) if len(drift_vx_list) > 1 else 0.0
        else:
            drift_summary["mean_vx_um_per_s"] = float("nan")
            drift_summary["std_vx_um_per_s"] = float("nan")
        if drift_vy_list:
            drift_summary["mean_vy_um_per_s"] = float(np.mean(drift_vy_list))
            drift_summary["std_vy_um_per_s"] = float(np.std(drift_vy_list, ddof=1)) if len(drift_vy_list) > 1 else 0.0
        else:
            drift_summary["mean_vy_um_per_s"] = float("nan")
            drift_summary["std_vy_um_per_s"] = float("nan")
        drift_summary["n_samples"] = max(len(drift_vx_list), len(drift_vy_list))
        stats["drift_summary"] = drift_summary

        per_channel_stats: dict[str, dict[str, Any]] = {}
        channel_list = _channel_comparison(trajectories)
        for c in channel_list:
            ch_id = c["channel_id"]
            per_channel_stats[str(ch_id)] = {
                "count_total": c["n_total"],
                "count_valid": c["n_valid"],
                "mean_D": c["mean_D"],
                "median_D": c["median_D"],
                "std_D": c["std_D"],
            }
        stats["per_channel_stats"] = per_channel_stats

        stats["model_statistics"] = _model_statistics(trajectories)

        return stats

    def export_json(
        self,
        batch_id: str,
        task_id: str,
        batch_data: dict,
        trajectories: list[dict],
        qc_report: dict,
        calibration: Optional[dict],
    ) -> Path:
        batch_serializable = {}
        for k, v in (batch_data or {}).items():
            batch_serializable[k] = v
        batch_serializable["batch_id"] = batch_id
        batch_serializable["task_id"] = task_id

        traj_list: list[dict] = []
        for traj in trajectories:
            flags = _get_val(traj, "flags", []) or []
            flags_list = []
            if isinstance(flags, dict):
                for fk, fv in flags.items():
                    if fv:
                        flags_list.append(fk)
            elif isinstance(flags, list):
                flags_list = list(flags)

            result = self._extract_trajectory_result(traj)

            traj_entry = {
                "particle_id": _get_val(traj, "particle_id", None),
                "trajectory_id": _get_val(traj, "trajectory_id", _get_val(traj, "id", None)),
                "channel_id": _get_val(traj, "channel_id", None),
                "num_frames": _get_val(traj, "num_frames", None),
                "qc_passed": _get_val(traj, "qc_passed", True),
                "flags": flags_list,
                "result": result,
            }
            traj_list.append(traj_entry)

        batch_summary = self._build_batch_summary(trajectories, qc_report)

        calib_serializable: Optional[dict] = None
        if calibration:
            calib_serializable = {}
            for k, v in calibration.items():
                calib_serializable[k] = v

        report = {
            "batch": batch_serializable,
            "qc_report": qc_report if qc_report else {},
            "trajectories": traj_list,
            "batch_summary": batch_summary,
            "calibration": calib_serializable,
        }

        filename = f"batch_{batch_id}_report.json"
        filepath = self.result_dir / filename

        class _Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                        return int(obj)
                if isinstance(obj, np.floating):
                    if np.isnan(obj):
                        return None
                    return float(obj)
                if isinstance(obj, Path):
                    return str(obj)
                if pd.isna(obj) if isinstance(obj, float) else False:
                    return None
                return super().default(obj)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, cls=_Encoder)

        return filepath

    def export_csv(
        self,
        batch_id: str,
        task_id: str,
        trajectories: list[dict],
        qc_report: dict,
        calibration: Optional[dict],
    ) -> dict[str, Path]:
        result_paths: dict[str, Path] = {}

        traj_rows: list[dict] = []
        for idx, traj in enumerate(trajectories):
            result = self._extract_trajectory_result(traj)
            flags = _get_val(traj, "flags", []) or []
            flags_str = ", ".join(str(f) for f in flags) if isinstance(flags, list) else str(flags)

            dv = result["drift_velocity"] or [None, None]
            ci = result["ci_95"] or [None, None]
            fi = result["fit_interval"] or [None, None]

            row = {
                "index": idx,
                "particle_id": _get_val(traj, "particle_id", None),
                "trajectory_id": _get_val(traj, "trajectory_id", _get_val(traj, "id", None)),
                "channel_id": _get_val(traj, "channel_id", None),
                "num_frames": _get_val(traj, "num_frames", None),
                "qc_passed": _get_val(traj, "qc_passed", True),
                "flags": flags_str,
                "D_um2_s": result["D_um2_s"],
                "alpha": result["alpha"],
                "R2": result["R2"],
                "model_type": result["model_type"],
                "model_reason": result["model_reason"],
                "hydro_radius_nm": result["hydro_radius_nm"],
                "fit_interval_start_s": fi[0],
                "fit_interval_end_s": fi[1],
                "ci_95_low": ci[0],
                "ci_95_high": ci[1],
                "drift_vx_um_s": dv[0],
                "drift_vy_um_s": dv[1],
                "excluded_from_distribution": result["excluded_from_distribution"],
                "exclude_reason": result["exclude_reason"],
            }
            traj_rows.append(row)

        traj_df = pd.DataFrame(traj_rows)
        traj_filename = f"batch_{batch_id}_trajectories.csv"
        traj_filepath = self.result_dir / traj_filename
        traj_df.to_csv(traj_filepath, index=False, encoding="utf-8-sig")
        result_paths["trajectories"] = traj_filepath

        anomaly_rows: list[dict] = []
        for idx, traj in enumerate(trajectories):
            flags = _get_val(traj, "flags", []) or []
            flags_list = flags if isinstance(flags, list) else ([k for k, v in flags.items() if v] if isinstance(flags, dict) else [])
            particle_id = _get_val(traj, "particle_id", None)
            trajectory_id = _get_val(traj, "trajectory_id", _get_val(traj, "id", None))
            channel_id = _get_val(traj, "channel_id", None)
            num_frames = _get_val(traj, "num_frames", None)

            result = self._extract_trajectory_result(traj)
            if result["excluded_from_distribution"]:
                reason = result["exclude_reason"] or "unknown"
                severity = "blocking" if "cross" in str(reason).lower() or "跨通道" in str(reason) else "warning"
                anomaly_rows.append({
                    "trajectory_index": idx,
                    "particle_id": particle_id,
                    "trajectory_id": trajectory_id,
                    "channel_id": channel_id,
                    "severity": severity,
                    "type": "exclude_from_distribution",
                    "description": f"排除原因: {reason}",
                    "details": f"num_frames={num_frames}",
                })

            for flag in flags_list:
                flag_str = str(flag)
                severity = "blocking" if "cross" in flag_str.lower() or "跨通道" in flag_str else "warning"
                anomaly_rows.append({
                    "trajectory_index": idx,
                    "particle_id": particle_id,
                    "trajectory_id": trajectory_id,
                    "channel_id": channel_id,
                    "severity": severity,
                    "type": "qc_flag",
                    "description": flag_str,
                    "details": f"num_frames={num_frames}",
                })

        if anomaly_rows:
            anomaly_df = pd.DataFrame(anomaly_rows)
        else:
            anomaly_df = pd.DataFrame(columns=[
                "trajectory_index", "particle_id", "trajectory_id",
                "channel_id", "severity", "type", "description", "details"
            ])
        anomaly_filename = f"batch_{batch_id}_anomalies.csv"
        anomaly_filepath = self.result_dir / anomaly_filename
        anomaly_df.to_csv(anomaly_filepath, index=False, encoding="utf-8-sig")
        result_paths["anomalies"] = anomaly_filepath

        calib_rows: list[dict] = []
        if calibration:
            calib_points = _get_val(calibration, "calibration_points", None)
            if calib_points and isinstance(calib_points, list):
                for cp in calib_points:
                    calib_rows.append({
                        "channel_id": _get_val(cp, "channel_id", None),
                        "nominal_nm": _get_val(cp, "nominal_nm", None),
                        "measured_nm": _get_val(cp, "measured_nm", None),
                        "bias_pct": _get_val(cp, "bias_pct", None),
                        "D_um2_s": _get_val(cp, "D_um2_s", None),
                    })
        if calib_rows:
            calib_df = pd.DataFrame(calib_rows)
        else:
            calib_df = pd.DataFrame(columns=["channel_id", "nominal_nm", "measured_nm", "bias_pct", "D_um2_s"])
        calib_filename = f"batch_{batch_id}_calibration.csv"
        calib_filepath = self.result_dir / calib_filename
        calib_df.to_csv(calib_filepath, index=False, encoding="utf-8-sig")
        result_paths["calibration"] = calib_filepath

        msd_summary_rows: list[dict] = []
        for traj in trajectories:
            result = self._extract_trajectory_result(traj)
            trajectory_id = _get_val(traj, "trajectory_id", _get_val(traj, "id", None))
            particle_id = _get_val(traj, "particle_id", None)
            for pt in result["msd_points"]:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    msd_summary_rows.append({
                        "particle_id": particle_id,
                        "trajectory_id": trajectory_id,
                        "lag_s": pt[0],
                        "msd_um2": pt[1],
                        "count": pt[2] if len(pt) > 2 else 1,
                    })
        if msd_summary_rows:
            msd_df = pd.DataFrame(msd_summary_rows)
        else:
            msd_df = pd.DataFrame(columns=["particle_id", "trajectory_id", "lag_s", "msd_um2", "count"])
        msd_filename = f"batch_{batch_id}_msd_summary.csv"
        msd_filepath = self.result_dir / msd_filename
        msd_df.to_csv(msd_filepath, index=False, encoding="utf-8-sig")
        result_paths["msd_summary"] = msd_filepath

        return result_paths

    def _md_table(self, headers: list[str], rows: list[list[Any]], align: Optional[list[str]] = None) -> str:
        if align is None:
            align = ["---"] * len(headers)
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(align) + " |"
        lines = [header_line, sep_line]
        for row in rows:
            formatted = [str(c) if c is not None else "N/A" for c in row]
            lines.append("| " + " | ".join(formatted) + " |")
        return "\n".join(lines)

    def export_markdown(
        self,
        batch_id: str,
        task_id: str,
        batch_data: dict,
        trajectories: list[dict],
        qc_report: dict,
        calibration: Optional[dict],
    ) -> Path:
        batch_name = _get_val(batch_data, "name", f"Batch {batch_id}")
        batch_summary = self._build_batch_summary(trajectories, qc_report)
        model_stats = _model_statistics(trajectories)
        channel_list = _channel_comparison(trajectories)
        retest_suggestions = _get_retest_suggestions(qc_report, batch_summary, calibration)

        md_lines: list[str] = []

        md_lines.append(f"# 批次分析报告 {batch_name}")
        md_lines.append("")
        md_lines.append(f"> 批次ID: `{batch_id}` | 任务ID: `{task_id}`")
        md_lines.append("")

        md_lines.append("## 1. 实验参数")
        md_lines.append("")
        md_lines.append("")
        param_rows = [
            ["像素尺寸 (μm/px)", _format_float(_get_val(batch_data, "pixel_size_um", None))],
            ["帧率 (Hz)", _format_float(_get_val(batch_data, "frame_rate_hz", None))],
            ["温度 (°C)", _format_float(_get_val(batch_data, "temperature_c", None))],
            ["黏度 (Pa·s)", _format_float(_get_val(batch_data, "viscosity_pa_s", None))],
            ["标称粒径 (nm)", _format_float(_get_val(batch_data, "nominal_diameter_nm", None))],
            ["通道宽度 (μm)", _format_float(_get_val(batch_data, "channel_width_um", None))],
            ["通道高度 (μm)", _format_float(_get_val(batch_data, "channel_height_um", None))],
        ]
        md_lines.append(self._md_table(["参数", "数值"], param_rows, [":---", "---:"]))
        md_lines.append("")

        md_lines.append("## 2. 质控摘要")
        md_lines.append("")
        total = _get_val(qc_report, "total_trajectories", None)
        if total is None:
            total = _get_val(batch_summary, "total_trajectories", len(trajectories))
        valid = _get_val(batch_summary, "valid_trajectories", 0)
        short_count = _get_val(qc_report, "short_trajectories", 0) or 0
        broken = _get_val(qc_report, "broken_trajectories", 0) or 0
        reconnected = _get_val(qc_report, "reconnected_trajectories", 0) or 0
        frame_missing = _get_val(qc_report, "frame_missing_count", 0) or 0
        cross_channel = _get_val(qc_report, "cross_channel_count", 0) or 0
        intensity_drop = _get_val(qc_report, "intensity_drop_count", 0) or 0

        qc_rows = [
            ["总轨迹数", str(total)],
            ["有效轨迹数", str(valid)],
            ["短轨迹数 (不参与D分布)", str(short_count)],
            ["修复轨迹数", str(reconnected)],
            ["断裂轨迹数", str(broken)],
            ["缺失帧数", str(frame_missing)],
        ]
        md_lines.append(self._md_table(["质控项", "数量"], qc_rows, [":---", "---:"]))
        md_lines.append("")

        drift_summary = _get_val(batch_summary, "drift_summary", None) or _get_val(qc_report, "drift_estimate", None)
        if drift_summary and isinstance(drift_summary, dict):
            vx = _format_float(_get_val(drift_summary, "mean_vx_um_per_s", None))
            vy = _format_float(_get_val(drift_summary, "mean_vy_um_per_s", None))
            md_lines.append(f"- **漂移估计**: vx = {vx} μm/s, vy = {vy} μm/s")
        else:
            drift_est = _get_val(qc_report, "drift_estimate", None)
            if drift_est is not None and not (isinstance(drift_est, dict)):
                md_lines.append(f"- **漂移估计 (量级)**: {_format_float(drift_est)} μm/s")
        md_lines.append("")

        anomaly_rows_md = []
        if cross_channel > 0:
            anomaly_rows_md.append(["跨通道异常 (**阻塞级**)", f"**{cross_channel}**"])
        jump_points = broken + frame_missing
        if jump_points > 0:
            anomaly_rows_md.append(["跳点/断裂", str(jump_points)])
        if intensity_drop > 0:
            anomaly_rows_md.append(["强度骤降", str(intensity_drop)])
        if anomaly_rows_md:
            md_lines.append(self._md_table(["异常类型", "数量"], anomaly_rows_md, [":---", "---:"]))
            md_lines.append("")

        metadata_conflicts = _get_val(qc_report, "metadata_conflicts", None)
        if metadata_conflicts and isinstance(metadata_conflicts, dict) and metadata_conflicts:
            md_lines.append("**元数据冲突:**")
            md_lines.append("")
            for ck, cv in metadata_conflicts.items():
                md_lines.append(f"- `{ck}`: {cv}")
            md_lines.append("")

        md_lines.append("## 3. 模型判别统计")
        md_lines.append("")
        model_rows = [
            ["布朗扩散 (brownian)", model_stats.get("brownian", 0)],
            ["受限扩散 (confined)", model_stats.get("confined", 0)],
            ["定向扩散 (directed)", model_stats.get("directed", 0)],
            ["次扩散 (subdiffusive)", model_stats.get("subdiffusive", 0)],
            ["超扩散 (superdiffusive)", model_stats.get("superdiffusive", 0)],
            ["异常扩散 (anomalous)", model_stats.get("anomalous", 0)],
            ["未知 (unknown)", model_stats.get("unknown", 0)],
        ]
        md_lines.append(self._md_table(["模型类型", "轨迹数"], model_rows, [":---", "---:"]))
        md_lines.append("")

        md_lines.append("## 4. 批次统计结果")
        md_lines.append("")
        md_lines.append("### 4.1 扩散系数 D 分布")
        md_lines.append("")
        dist_rows = [
            ["均值 ± 标准差 (μm²/s)", f"{_format_float(_get_val(batch_summary, 'mean_D', None))} ± {_format_float(_get_val(batch_summary, 'std_D', None))}"],
            ["中位数 (μm²/s)", _format_float(_get_val(batch_summary, 'median_D', None))],
            ["P25 (μm²/s)", _format_float(_get_val(batch_summary, 'percentile_25_D', None))],
            ["P75 (μm²/s)", _format_float(_get_val(batch_summary, 'percentile_75_D', None))],
        ]
        md_lines.append(self._md_table(["统计量", "数值"], dist_rows, [":---", "---:"]))
        md_lines.append("")

        md_lines.append("### 4.2 水力学半径")
        md_lines.append("")
        radius_rows = [
            ["均值 ± 标准差 (nm)", f"{_format_float(_get_val(batch_summary, 'mean_radius_nm', None))} ± {_format_float(_get_val(batch_summary, 'std_radius_nm', None))}"],
            ["中位数 (nm)", _format_float(_get_val(batch_summary, 'median_radius_nm', None))],
        ]
        md_lines.append(self._md_table(["统计量", "数值"], radius_rows, [":---", "---:"]))
        md_lines.append("")

        if channel_list:
            md_lines.append("### 4.3 通道间差异")
            md_lines.append("")
            ch_rows = []
            for c in channel_list:
                ch_rows.append([
                    f"通道 {c['channel_id']}",
                    _format_float(c["mean_D"]),
                    _format_float(c["median_D"]),
                    str(c["n_valid"]),
                    str(c["n_total"]),
                ])
            md_lines.append(self._md_table(
                ["通道", "mean_D (μm²/s)", "median_D (μm²/s)", "有效数", "总数"],
                ch_rows,
                [":---", "---:", "---:", "---:", "---:"]
            ))
            md_lines.append("")

        if calibration:
            md_lines.append("## 5. 校准曲线数据")
            md_lines.append("")
            calib_points = _get_val(calibration, "calibration_points", None)
            if calib_points and isinstance(calib_points, list):
                calib_rows_md = []
                for cp in calib_points:
                    ch_id = _get_val(cp, "channel_id", None)
                    nominal = _format_float(_get_val(cp, "nominal_nm", None))
                    measured = _format_float(_get_val(cp, "measured_nm", None))
                    bias = _format_float(_get_val(cp, "bias_pct", None))
                    D_cal = _format_float(_get_val(cp, "D_um2_s", None))
                    calib_rows_md.append([
                        f"通道 {ch_id}" if ch_id is not None else "N/A",
                        nominal,
                        measured,
                        bias,
                        D_cal,
                    ])
                md_lines.append(self._md_table(
                    ["通道", "标称 (nm)", "实测 (nm)", "偏差 (%)", "D (μm²/s)"],
                    calib_rows_md,
                    [":---", "---:", "---:", "---:", "---:"]
                ))
                md_lines.append("")
                mean_bias = _format_float(_get_val(calibration, "mean_bias_pct", None))
                overall_r2 = _format_float(_get_val(calibration, "overall_r2", None))
                md_lines.append(f"- 平均偏差: {mean_bias}%")
                md_lines.append(f"- 整体拟合 R²: {overall_r2}")
                md_lines.append("")

        md_lines.append("## 6. 最差轨迹 Top-10")
        md_lines.append("")
        scored_trajectories = []
        for idx, traj in enumerate(trajectories):
            score, comps = _trajectory_score(traj)
            scored_trajectories.append((score, idx, traj, comps))
        scored_trajectories.sort(key=lambda x: x[0], reverse=True)
        top10 = scored_trajectories[:10]
        worst_rows = []
        for score, idx, traj, comps in top10:
            result = self._extract_trajectory_result(traj)
            particle_id = _get_val(traj, "particle_id", None)
            channel_id = _get_val(traj, "channel_id", None)
            num_frames = _get_val(traj, "num_frames", None)
            D = _format_float(result["D_um2_s"])
            R2 = _format_float(result["R2"])
            model = result["model_type"] or "N/A"
            exclude_note = ""
            if result["excluded_from_distribution"]:
                exclude_note = result["exclude_reason"] or "已排除"
            flags = _get_val(traj, "flags", []) or []
            flag_str = ", ".join(str(f) for f in flags) if isinstance(flags, list) else str(flags)
            anomaly_note_parts = []
            if exclude_note:
                anomaly_note_parts.append(exclude_note)
            if flag_str:
                anomaly_note_parts.append(flag_str)
            worst_rows.append([
                str(particle_id),
                f"通道 {channel_id}" if channel_id is not None else "N/A",
                str(num_frames),
                D,
                R2,
                model,
                "; ".join(anomaly_note_parts) if anomaly_note_parts else "—",
            ])
        if worst_rows:
            md_lines.append(self._md_table(
                ["Particle ID", "通道", "帧数", "D (μm²/s)", "R²", "模型", "排除原因/异常"],
                worst_rows,
                [":---", ":---", "---:", "---:", "---:", ":---", ":---"]
            ))
        else:
            md_lines.append("*无数据*")
        md_lines.append("")

        md_lines.append("## 7. 复测建议")
        md_lines.append("")
        if retest_suggestions:
            for i, s in enumerate(retest_suggestions, 1):
                md_lines.append(f"{i}. {s}")
        else:
            md_lines.append("*暂未发现需复测的问题*")
        md_lines.append("")

        md_lines.append("## 8. 建议与备注")
        md_lines.append("")
        notes: list[str] = []
        if short_count > 0:
            notes.append(f"- 短轨迹共 {short_count} 条未参与粒径分布统计，仅在质控报告中保留记录。")
        if cross_channel > 0:
            notes.append(f"- 跨通道粒子标记为 **阻塞级** 异常，需特别注意排查通道边界定义或粒子追踪参数。")
        drift_data = _get_val(batch_summary, "drift_summary", None)
        if isinstance(drift_data, dict):
            vx_val = _get_val(drift_data, "mean_vx_um_per_s", None)
            vy_val = _get_val(drift_data, "mean_vy_um_per_s", None)
            if vx_val is not None and vy_val is not None and not (isinstance(vx_val, float) and np.isnan(vx_val)) and not (isinstance(vy_val, float) and np.isnan(vy_val)):
                drift_mag = np.sqrt(float(vx_val) ** 2 + float(vy_val) ** 2)
                notes.append(f"- 漂移扣除前后结果均已保留在轨迹结果字段中。")
        if valid < 30 and valid > 0:
            notes.append(f"- 有效轨迹数较少 ({valid}) 建议补充数据以提高统计可靠性。")
        if not notes:
            notes.append("- 本批次数据质量良好，各指标在正常范围内。")
        for note in notes:
            md_lines.append(note)
        md_lines.append("")

        filename = f"batch_{batch_id}_report.md"
        filepath = self.result_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        return filepath
