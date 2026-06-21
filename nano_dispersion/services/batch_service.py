from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from nano_dispersion import algorithms
from nano_dispersion.algorithms.batch_stats import aggregate_trajectory_results
from nano_dispersion.algorithms.diffusion_fitting import (
    FitResult as DiffFitResult,
    discriminate_model,
    fit_brownian,
    fit_confined,
    fit_directed,
)
from nano_dispersion.algorithms.drift_estimation import (
    compute_drift_velocity,
    estimate_global_drift,
    subtract_drift,
)
from nano_dispersion.algorithms.msd import compute_msd
from nano_dispersion.algorithms.stokes_einstein import (
    CalibrationResult as StokesCalibrationResult,
    calibration_curve,
    compute_hydrodynamic_radius,
)
from nano_dispersion.algorithms.trajectory_qc import (
    detect_cross_channel,
    detect_intensity_drop,
    detect_outliers,
    detect_short_trajectories,
    find_break_points,
    reconnect_broken_trajectories,
    sort_and_validate,
)
from nano_dispersion.config import Settings
from nano_dispersion.models.orm import (
    AnomalyRecord,
    AnalysisTask,
    BatchSummary,
    CalibrationPoint,
    ExperimentBatch,
    ParticleTrajectory,
    QCReport,
    TrajectoryPoint,
    TrajectoryResult,
)
from nano_dispersion.reports.exporter import ReportExporter


class BatchService:
    def __init__(self, db_factory, task_manager, settings: Settings):
        self._db_factory = db_factory
        self._task_manager = task_manager
        self._settings = settings

    def _get_session(self) -> Session:
        return self._db_factory()

    def import_batch_from_directory(
        self,
        dir_path: str | Path,
        description: str = "",
    ) -> ExperimentBatch:
        """
        导入本地批次目录：
        1. 读metadata.json
        2. 遍历trajectories/目录下所有CSV
        3. 解析CSV -> ParticleTrajectory + TrajectoryPoint记录
        4. 合并重复particle_id（不同channel可以同号，要区分）
        5. 检查metadata温度冲突
        返回 ExperimentBatch对象
        """
        dir_path = Path(dir_path)
        metadata_path = dir_path / "metadata.json"
        trajectories_dir = dir_path / "trajectories"

        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        batch_name = metadata.get("batch_name") or metadata.get("name") or dir_path.name
        pixel_size_um = metadata.get("pixel_size_um") or metadata.get("pixel_size")
        frame_rate_hz = metadata.get("frame_rate_hz") or metadata.get("fps")
        temperature_c = metadata.get("temperature_c") or metadata.get("temperature")
        viscosity_pa_s = metadata.get("viscosity_pa_s") or metadata.get("viscosity")
        if viscosity_pa_s is None:
            viscosity_pa_s = self._settings.default_water_viscosity_pa_s
        nominal_diameter_nm = metadata.get("nominal_diameter_nm") or metadata.get("nominal_diameter")
        channel_width_um = metadata.get("channel_width_um")
        channel_height_um = metadata.get("channel_height_um")

        channel_bounds: dict[str, tuple[float, float, float, float]] = {}
        channels_meta = metadata.get("channels")
        if channels_meta and isinstance(channels_meta, list):
            channel_width_um_val = metadata.get("channel_width_um", 200.0)
            channel_height_um_val = metadata.get("channel_height_um", 50.0)
            for ch_idx, ch_id_str in enumerate(channels_meta):
                x_min = float(ch_idx * channel_width_um_val)
                x_max = float((ch_idx + 1) * channel_width_um_val)
                y_min = 0.0
                y_max = float(channel_height_um_val)
                channel_bounds[str(ch_id_str)] = (x_min, x_max, y_min, y_max)
        elif channels_meta and isinstance(channels_meta, dict):
            for ch_id_str, ch_data in channels_meta.items():
                if isinstance(ch_data, dict):
                    x_min = ch_data.get("x_min")
                    x_max = ch_data.get("x_max")
                    y_min = ch_data.get("y_min")
                    y_max = ch_data.get("y_max")
                    if all(v is not None for v in [x_min, x_max, y_min, y_max]):
                        channel_bounds[str(ch_id_str)] = (float(x_min), float(x_max), float(y_min), float(y_max))

        db = self._get_session()
        try:
            batch = ExperimentBatch(
                name=batch_name,
                description=description,
                import_path=str(dir_path.absolute()),
                metadata_json=metadata,
                status="imported",
                pixel_size_um=float(pixel_size_um) if pixel_size_um is not None else None,
                frame_rate_hz=float(frame_rate_hz) if frame_rate_hz is not None else None,
                temperature_c=float(temperature_c) if temperature_c is not None else None,
                viscosity_pa_s=float(viscosity_pa_s) if viscosity_pa_s is not None else None,
                nominal_diameter_nm=float(nominal_diameter_nm) if nominal_diameter_nm is not None else None,
                channel_width_um=float(channel_width_um) if channel_width_um is not None else None,
                channel_height_um=float(channel_height_um) if channel_height_um is not None else None,
            )
            db.add(batch)
            db.flush()

            csv_files: list[Path] = []
            if trajectories_dir.exists() and trajectories_dir.is_dir():
                csv_files = sorted(trajectories_dir.glob("*.csv"))

            all_frames_data: list[pd.DataFrame] = []
            csv_channel_map: dict[Path, Optional[int]] = {}

            for csv_file in csv_files:
                ch_id: Optional[int] = None
                stem = csv_file.stem.lower()
                for prefix in ["channel_", "ch_", "channel", "ch"]:
                    if stem.startswith(prefix):
                        rem = stem[len(prefix):]
                        if rem.isdigit():
                            ch_id = int(rem)
                            break
                if ch_id is None:
                    csv_channel_map[csv_file] = None
                else:
                    csv_channel_map[csv_file] = ch_id

                try:
                    df = pd.read_csv(csv_file)
                    if ch_id is not None and "channel_id" not in df.columns:
                        df["channel_id"] = ch_id
                    all_frames_data.append(df)
                except Exception:
                    continue

            if all_frames_data:
                combined_df = pd.concat(all_frames_data, ignore_index=True)
            else:
                combined_df = pd.DataFrame()

            required_cols = {"particle_id", "frame", "time_s", "x_um", "y_um"}
            if combined_df.empty:
                pass
            else:
                for col in required_cols:
                    if col not in combined_df.columns:
                        if col == "time_s" and "frame" in combined_df.columns and frame_rate_hz:
                            combined_df["time_s"] = combined_df["frame"].astype(float) / float(frame_rate_hz)
                        elif col == "x_um" and "x_px" in combined_df.columns and pixel_size_um:
                            combined_df["x_um"] = combined_df["x_px"].astype(float) * float(pixel_size_um)
                        elif col == "y_um" and "y_px" in combined_df.columns and pixel_size_um:
                            combined_df["y_um"] = combined_df["y_px"].astype(float) * float(pixel_size_um)

            if not combined_df.empty and required_cols.issubset(set(combined_df.columns)):
                if "channel_id" not in combined_df.columns:
                    combined_df["channel_id"] = None

                combined_df["_ch_id"] = combined_df["channel_id"].apply(
                    lambda x: str(x) if pd.notna(x) and x is not None else None
                )

                pid_groups = combined_df.groupby(["_ch_id", "particle_id"], sort=False)

                for (ch_id, particle_id), grp in pid_groups:
                    grp_sorted = grp.sort_values("frame").copy()
                    num_frames = len(grp_sorted)
                    times = grp_sorted["time_s"].astype(float).values
                    duration_s = float(times[-1] - times[0]) if len(times) >= 2 else None

                    flags: dict[str, Any] = {}
                    qc_passed = True

                    traj = ParticleTrajectory(
                        batch_id=batch.id,
                        particle_id=int(particle_id),
                        channel_id=ch_id,
                        num_frames=num_frames,
                        duration_s=duration_s,
                        flags=flags,
                        qc_passed=qc_passed,
                    )
                    db.add(traj)
                    db.flush()
                    trajectory_id = traj.id

                    points_to_add: list[TrajectoryPoint] = []
                    for _, row in grp_sorted.iterrows():
                        is_outlier = bool(row.get("is_outlier", False)) if pd.notna(row.get("is_outlier")) else False
                        anomaly_tag_val = row.get("anomaly_tag")
                        anomaly_tag = str(anomaly_tag_val) if pd.notna(anomaly_tag_val) else None

                        x_corr = row.get("x_corrected_um")
                        y_corr = row.get("y_corrected_um")
                        x_corrected = float(x_corr) if pd.notna(x_corr) else None
                        y_corrected = float(y_corr) if pd.notna(y_corr) else None

                        intensity_val = row.get("intensity")
                        intensity = float(intensity_val) if pd.notna(intensity_val) else None

                        pt = TrajectoryPoint(
                            trajectory_id=trajectory_id,
                            frame=int(row["frame"]),
                            time_s=float(row["time_s"]),
                            x_um=float(row["x_um"]),
                            y_um=float(row["y_um"]),
                            x_corrected_um=x_corrected,
                            y_corrected_um=y_corrected,
                            intensity=intensity,
                            is_outlier=is_outlier,
                            anomaly_tag=anomaly_tag,
                        )
                        points_to_add.append(pt)

                    db.bulk_save_objects(points_to_add)

            db.commit()
            db.refresh(batch)
            return batch
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def submit_analysis(self, batch_id: int) -> int:
        """提交分析任务到TaskManager，返回task_id"""
        def analyze_fn(progress_callback: Callable, **kwargs) -> dict:
            return self._run_full_analysis(batch_id, progress_callback)
        return self._task_manager.submit_task(batch_id, "full_analysis", analyze_fn)

    def _load_trajectories_to_dataframe(self, batch_id: int) -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
        """加载批次所有轨迹点为DataFrame，同时返回trajectory元信息"""
        db = self._get_session()
        try:
            traj_info: dict[int, dict[str, Any]] = {}
            all_rows: list[dict[str, Any]] = []

            trajectories = db.query(ParticleTrajectory).filter(
                ParticleTrajectory.batch_id == batch_id
            ).all()

            for traj in trajectories:
                traj_info[traj.id] = {
                    "trajectory_id": traj.id,
                    "particle_id": traj.particle_id,
                    "channel_id": traj.channel_id,
                    "num_frames": traj.num_frames,
                    "qc_passed": traj.qc_passed,
                    "flags": traj.flags or {},
                }

                for pt in traj.points:
                    all_rows.append({
                        "trajectory_id": traj.id,
                        "particle_id": traj.particle_id,
                        "channel_id": traj.channel_id,
                        "frame": pt.frame,
                        "time_s": pt.time_s,
                        "x_um": pt.x_um,
                        "y_um": pt.y_um,
                        "x_corrected_um": pt.x_corrected_um,
                        "y_corrected_um": pt.y_corrected_um,
                        "intensity": pt.intensity,
                        "is_outlier": pt.is_outlier,
                        "anomaly_tag": pt.anomaly_tag,
                    })

            df = pd.DataFrame(all_rows)
            return df, traj_info
        finally:
            db.close()

    def _run_full_analysis(self, batch_id: int, progress_callback: Callable) -> dict:
        """
        完整分析流程（调用algorithms模块）：
        1. 加载批次所有轨迹点 -> one big DataFrame
        2. 轨迹质控 (algorithms.trajectory_qc.*):
           progress 5% -> 20%
        3. 漂移估计与扣除:
           progress 20% -> 40%
        4. 每条轨迹:
           - compute_msd
           - fit_brownian / fit_confined / fit_directed
           - discriminate_model
           - compute_hydrodynamic_radius
           记录TrajectoryResult
           progress 40% -> 75%
        5. 批次统计 + 校准:
           aggregate_trajectory_results
           calibration_curve
           记录BatchSummary, CalibrationPoint
           progress 75% -> 90%
        6. 生成AnomalyRecord（所有异常）
           生成QCReport
           progress 90% -> 98%
        7. 返回结果dict
        """
        result_summary: dict[str, Any] = {"batch_id": batch_id}

        progress_callback(1.0, "开始加载轨迹数据...")
        all_points_df, traj_info = self._load_trajectories_to_dataframe(batch_id)
        progress_callback(5.0, f"加载完成，共 {len(traj_info)} 条轨迹")

        if all_points_df.empty or not traj_info:
            result_summary["error"] = "无轨迹数据"
            return result_summary

        db = self._get_session()
        batch = db.get(ExperimentBatch, batch_id)
        if not batch:
            db.close()
            return {"error": f"Batch {batch_id} 不存在"}
        task = db.query(AnalysisTask).filter(
            AnalysisTask.batch_id == batch_id,
            AnalysisTask.status == "running",
        ).order_by(AnalysisTask.id.desc()).first()
        task_id = task.id if task else None
        db.close()

        temperature_c = batch.temperature_c if batch.temperature_c is not None else 25.0
        viscosity_pa_s = batch.viscosity_pa_s if batch.viscosity_pa_s is not None else self._settings.default_water_viscosity_pa_s
        temperature_k = temperature_c + 273.15

        progress_callback(6.0, "步骤1/6: 轨迹质控检查...")
        sorted_df, issues = sort_and_validate(all_points_df)

        trajectories_dict: dict[int, pd.DataFrame] = {}
        for tid in traj_info:
            sub = sorted_df[sorted_df["trajectory_id"] == tid]
            if not sub.empty:
                trajectories_dict[tid] = sub

        min_frames = self._settings.min_trajectory_frames
        short_ids = detect_short_trajectories(trajectories_dict, min_frames)

        broken_count = 0
        all_break_points: dict[int, list[tuple[int, int]]] = {}
        for tid in traj_info:
            pid = traj_info[tid]["particle_id"]
            breaks = find_break_points(sorted_df, pid)
            if breaks:
                all_break_points[tid] = breaks
                broken_count += len(breaks)

        reconnected_map, reconnection_log = reconnect_broken_trajectories(
            trajectories_dict,
            max_gap_frames=self._settings.max_gap_frames_for_reconnect,
            max_distance_um=self._settings.max_neighbor_distance_um,
            max_velocity_um_per_s=self._settings.max_velocity_jump_um_per_s,
        )

        progress_callback(12.0, "检测异常点...")
        sorted_df = detect_outliers(sorted_df, sigma=self._settings.drift_outlier_sigma)

        frame_missing_count = len([i for i in issues if "缺失" in i])

        channel_bounds: dict[int, tuple[float, float, float, float]] = {}
        cross_ids: list[int] = detect_cross_channel(sorted_df, channel_bounds)

        intensity_drop_ids: list[int] = detect_intensity_drop(sorted_df)

        for tid in short_ids:
            if tid in traj_info:
                traj_info[tid]["flags"]["short_trajectory"] = True
        for tid, breaks in all_break_points.items():
            if tid in traj_info:
                traj_info[tid]["flags"]["broken"] = True
                traj_info[tid]["flags"]["break_points"] = [list(b) for b in breaks]
        if reconnected_map:
            for target_tid, source_tid in reconnected_map.items():
                if target_tid in traj_info:
                    traj_info[target_tid]["flags"]["reconnected"] = True
                    traj_info[target_tid]["flags"]["reconnected_from"] = source_tid
        for tid in cross_ids:
            for info_tid in traj_info:
                if traj_info[info_tid]["particle_id"] == tid:
                    traj_info[info_tid]["flags"]["cross_channel"] = True
        for tid in intensity_drop_ids:
            for info_tid in traj_info:
                if traj_info[info_tid]["particle_id"] == tid:
                    traj_info[info_tid]["flags"]["intensity_drop"] = True

        progress_callback(20.0, "步骤2/6: 漂移估计与扣除...")
        drift_df = estimate_global_drift(sorted_df, outlier_sigma=self._settings.drift_outlier_sigma)
        drift_vx, drift_vy, drift_df_out = compute_drift_velocity(drift_df)

        sorted_df = subtract_drift(sorted_df, drift_df)

        for tid in traj_info:
            sub = sorted_df[sorted_df["trajectory_id"] == tid]
            trajectories_dict[tid] = sub

        drift_summary_data = {
            "vx_um_per_s": drift_vx,
            "vy_um_per_s": drift_vy,
            "magnitude_um_per_s": float(np.sqrt(drift_vx**2 + drift_vy**2)) if (drift_vx or drift_vy) else 0.0,
        }
        progress_callback(40.0, "步骤3/6: 逐条轨迹分析...")

        traj_ids = list(traj_info.keys())
        total_traj = len(traj_ids)

        trajectory_results_to_add: list[TrajectoryResult] = []
        trajectory_results_data: list[dict[str, Any]] = []

        for idx, tid in enumerate(traj_ids):
            info = traj_info[tid]
            sub = trajectories_dict.get(tid)
            if sub is None or sub.empty:
                continue

            progress_pct = 40.0 + (idx / max(1, total_traj)) * 35.0
            if idx % max(1, total_traj // 20) == 0:
                progress_callback(progress_pct, f"分析轨迹 {idx + 1}/{total_traj}...")

            num_frames = info["num_frames"]
            is_short = num_frames < min_frames
            is_cross = bool(info["flags"].get("cross_channel", False))

            excluded = False
            exclude_reason: Optional[str] = None

            if is_cross:
                excluded = True
                exclude_reason = "cross_channel_particle"
            elif is_short:
                excluded = True
                exclude_reason = f"short_trajectory_{num_frames}_frames_less_than_{min_frames}"

            try:
                msd_df = compute_msd(
                    sub,
                    use_corrected=True,
                )
            except Exception:
                msd_df = pd.DataFrame()

            fit_b: DiffFitResult = fit_brownian(
                msd_df,
                min_lags=self._settings.msd_fit_min_lags,
                max_ratio=self._settings.msd_fit_max_lags_ratio,
                confidence=self._settings.confidence_level,
            )
            fit_c: Optional[DiffFitResult] = None
            fit_d: Optional[DiffFitResult] = None
            try:
                fit_c = fit_confined(
                    msd_df,
                    min_lags=self._settings.msd_fit_min_lags,
                    max_ratio=0.8,
                )
            except Exception:
                pass
            try:
                fit_d = fit_directed(
                    msd_df,
                    min_lags=self._settings.msd_fit_min_lags,
                    max_ratio=0.6,
                )
            except Exception:
                pass

            try:
                model_disc = discriminate_model(msd_df, fit_b, fit_c, fit_d)
                model_type = model_disc.model_type
                model_reason = model_disc.reason
            except Exception:
                model_type = None
                model_reason = None

            D_val = None
            alpha_val = None
            fit_r2_val = None
            fit_lag_start = None
            fit_lag_end = None
            ci_low_val = None
            ci_high_val = None
            hydro_radius_nm_val = None
            msd_points_json: Optional[list[dict[str, Any]]] = None

            if fit_b and not (isinstance(fit_b.D, float) and np.isnan(fit_b.D)):
                D_val = float(fit_b.D)
                alpha_val = float(fit_b.alpha)
                fit_r2_val = float(fit_b.r2)
                fit_lag_start_s = float(fit_b.lag_start_s)
                fit_lag_end_s = float(fit_b.lag_end_s)

                frame_rate = batch.frame_rate_hz if batch.frame_rate_hz else 1.0
                fit_lag_start = max(1, int(round(fit_lag_start_s * frame_rate)))
                fit_lag_end = max(fit_lag_start + 1, int(round(fit_lag_end_s * frame_rate)))

                ci_low_val = float(fit_b.ci_low) if fit_b.ci_low is not None else None
                ci_high_val = float(fit_b.ci_high) if fit_b.ci_high is not None else None

                try:
                    D_m2_s = D_val * 1e-12
                    hydro_radius_nm_val = compute_hydrodynamic_radius(
                        D_m2_s, temperature_k, viscosity_pa_s,
                        k_B=self._settings.boltzmann_constant,
                    )
                    if isinstance(hydro_radius_nm_val, float) and np.isnan(hydro_radius_nm_val):
                        hydro_radius_nm_val = None
                except Exception:
                    hydro_radius_nm_val = None

                if excluded and exclude_reason and "short" in exclude_reason.lower():
                    pass
                elif D_val is not None and (D_val <= 0 or fit_r2_val < 0.5):
                    if not excluded:
                        excluded = True
                        exclude_reason = f"poor_fit_r2={fit_r2_val:.3f}"

            if not msd_df.empty:
                msd_points_json = []
                for _, mrow in msd_df.iterrows():
                    try:
                        msd_points_json.append({
                            "lag": int(mrow.get("lag_frames", 0)),
                            "time_s": float(mrow.get("lag_time_s", 0.0)),
                            "msd_um2": float(mrow.get("msd", 0.0)),
                            "count": int(mrow.get("count", 1)),
                        })
                    except Exception:
                        continue

            if not model_type and excluded:
                if exclude_reason and "cross" in exclude_reason.lower():
                    model_type = "excluded"
                    model_reason = exclude_reason

            tdr = TrajectoryResult(
                trajectory_id=tid,
                task_id=task_id,
                diffusion_D_um2_s=D_val,
                alpha_exponent=alpha_val,
                fit_r2=fit_r2_val,
                fit_lag_start=fit_lag_start,
                fit_lag_end=fit_lag_end,
                ci_low=ci_low_val,
                ci_high=ci_high_val,
                hydro_radius_nm=hydro_radius_nm_val,
                model_type=model_type,
                model_reason=model_reason,
                excluded_from_distribution=excluded,
                exclude_reason=exclude_reason,
                msd_points=msd_points_json,
                drift_velocity_x=drift_vx if drift_vx else None,
                drift_velocity_y=drift_vy if drift_vy else None,
            )
            trajectory_results_to_add.append(tdr)

            res_data: dict[str, Any] = {
                "trajectory_id": tid,
                "particle_id": info["particle_id"],
                "channel_id": info["channel_id"],
                "num_frames": info["num_frames"],
                "diffusion_D_um2_s": D_val,
                "alpha_exponent": alpha_val,
                "fit_r2": fit_r2_val,
                "hydro_radius_nm": hydro_radius_nm_val,
                "model_type": model_type,
                "excluded_from_distribution": excluded,
                "exclude_reason": exclude_reason,
                "drift_velocity_x": drift_vx if drift_vx else None,
                "drift_velocity_y": drift_vy if drift_vy else None,
                "channel": info["channel_id"],
                "D": D_val,
                "alpha": alpha_val,
            }
            trajectory_results_data.append(res_data)

        db = self._get_session()
        try:
            for tid in traj_info:
                traj_db = db.get(ParticleTrajectory, tid)
                if traj_db:
                    traj_db.flags = traj_info[tid]["flags"]
                    if traj_info[tid]["flags"].get("cross_channel"):
                        traj_db.qc_passed = False

            db.bulk_save_objects(trajectory_results_to_add)
            db.flush()

            trajectory_ids_updated: set[int] = {tr.trajectory_id for tr in trajectory_results_to_add}
            points_update_data: dict[int, list[TrajectoryPoint]] = {}
            for tid in trajectory_ids_updated:
                sub_df = sorted_df[sorted_df["trajectory_id"] == tid]
                if sub_df.empty:
                    continue
                pts = db.query(TrajectoryPoint).filter(
                    TrajectoryPoint.trajectory_id == tid
                ).order_by(TrajectoryPoint.frame).all()
                for pt in pts:
                    match_row = sub_df[sub_df["frame"] == pt.frame]
                    if not match_row.empty:
                        row = match_row.iloc[0]
                        xc = row.get("x_corrected_um")
                        yc = row.get("y_corrected_um")
                        if xc is not None and not (isinstance(xc, float) and np.isnan(xc)):
                            pt.x_corrected_um = float(xc)
                        if yc is not None and not (isinstance(yc, float) and np.isnan(yc)):
                            pt.y_corrected_um = float(yc)
                        if bool(row.get("is_outlier", False)):
                            pt.is_outlier = True
                        tag_val = row.get("anomaly_tag")
                        if tag_val is not None and not (isinstance(tag_val, float) and np.isnan(tag_val)):
                            if not pt.anomaly_tag:
                                pt.anomaly_tag = str(tag_val)
                            else:
                                pt.anomaly_tag = str(pt.anomaly_tag) + "|" + str(tag_val)

            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        progress_callback(76.0, "步骤4/6: 批次统计与校准...")
        agg_stats = aggregate_trajectory_results(trajectory_results_data, exclude_short=True)

        valid_radius_vals = [
            r["hydro_radius_nm"] for r in trajectory_results_data
            if not r["excluded_from_distribution"]
            and r["hydro_radius_nm"] is not None
            and not (isinstance(r["hydro_radius_nm"], float) and np.isnan(r["hydro_radius_nm"]))
        ]
        mean_radius = float(np.mean(valid_radius_vals)) if valid_radius_vals else None

        calibration_points_list: list[tuple[float, float, Optional[int]]] = []
        nominal = batch.nominal_diameter_nm
        if nominal and nominal > 0 and mean_radius and mean_radius > 0:
            ch_groups: dict[Optional[int], list[float]] = {}
            for r in trajectory_results_data:
                if not r["excluded_from_distribution"] and r["hydro_radius_nm"] is not None:
                    ch_key = r.get("channel_id")
                    if ch_key not in ch_groups:
                        ch_groups[ch_key] = []
                    if not (isinstance(r["hydro_radius_nm"], float) and np.isnan(r["hydro_radius_nm"])):
                        ch_groups[ch_key].append(float(r["hydro_radius_nm"]))

            for ch_key, radii in ch_groups.items():
                if radii:
                    ch_mean = float(np.mean(radii))
                    calibration_points_list.append((float(nominal), ch_mean, ch_key))

        calib_result: Optional[StokesCalibrationResult] = None
        if len(calibration_points_list) >= 2:
            calib_result = calibration_curve(calibration_points_list)

        db = self._get_session()
        try:
            batch_summary = BatchSummary(
                task_id=task_id,
                batch_id=batch_id,
                mean_D=agg_stats.get("mean_D"),
                median_D=agg_stats.get("median_D"),
                std_D=agg_stats.get("std_D"),
                mean_radius_nm=agg_stats.get("mean_radius_nm"),
                valid_trajectories=int(agg_stats.get("n_valid", 0)),
                total_trajectories=int(agg_stats.get("n_total", total_traj)),
                drift_summary=drift_summary_data,
                distribution_stats={
                    "D_percentiles": {
                        "p25": agg_stats.get("percentile_25_D"),
                        "p75": agg_stats.get("percentile_75_D"),
                        "min": agg_stats.get("min_D"),
                        "max": agg_stats.get("max_D"),
                    },
                    "radius_percentiles": {
                        "p25": agg_stats.get("percentile_25_radius_nm"),
                        "p75": agg_stats.get("percentile_75_radius_nm"),
                    },
                    "per_channel": agg_stats.get("per_channel_stats", {}),
                    "exclude_reasons": agg_stats.get("exclude_reasons_counter", {}),
                },
            )
            db.add(batch_summary)

            if calib_result is not None and calib_result.calibration_points:
                for cp in calib_result.calibration_points:
                    ch_id_val = cp.get("channel_id")
                    cp_nominal = cp.get("nominal_nm")
                    cp_measured = cp.get("measured_nm")
                    bias_pct = None
                    D_um2_s = None
                    if cp_nominal and cp_measured and cp_nominal > 0:
                        bias_pct = float((cp_measured - cp_nominal) / cp_nominal * 100.0)

                    for r in trajectory_results_data:
                        if r.get("channel_id") == ch_id_val and not r["excluded_from_distribution"]:
                            D_um2_s = r.get("diffusion_D_um2_s")
                            break

                    db_cp = CalibrationPoint(
                        batch_id=batch_id,
                        task_id=task_id,
                        channel_id=str(ch_id_val) if ch_id_val is not None else None,
                        nominal_nm=float(cp_nominal) if cp_nominal else None,
                        measured_nm=float(cp_measured) if cp_measured else None,
                        bias_pct=bias_pct,
                        D_um2_s=float(D_um2_s) if D_um2_s is not None else None,
                    )
                    db.add(db_cp)
            elif nominal and mean_radius:
                cp_nominal = float(nominal)
                cp_measured = float(mean_radius)
                bias_pct = float((cp_measured - cp_nominal) / cp_nominal * 100.0) if cp_nominal > 0 else None
                D_um2_s = agg_stats.get("mean_D")
                db_cp = CalibrationPoint(
                    batch_id=batch_id,
                    task_id=task_id,
                    channel_id=None,
                    nominal_nm=cp_nominal,
                    measured_nm=cp_measured,
                    bias_pct=bias_pct,
                    D_um2_s=float(D_um2_s) if D_um2_s is not None else None,
                )
                db.add(db_cp)

            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        progress_callback(91.0, "步骤5/6: 生成异常记录与质控报告...")
        db = self._get_session()
        try:
            qc_report = QCReport(
                batch_id=batch_id,
                task_id=task_id,
                short_trajectories=len(short_ids),
                broken_trajectories=len(all_break_points),
                reconnected_trajectories=len(reconnected_map),
                frame_missing_count=frame_missing_count,
                drift_estimate=drift_summary_data.get("magnitude_um_per_s"),
                cross_channel_count=len([
                    tid for tid in traj_info
                    if traj_info[tid]["flags"].get("cross_channel")
                ]),
                intensity_drop_count=len([
                    tid for tid in traj_info
                    if traj_info[tid]["flags"].get("intensity_drop")
                ]),
                metadata_conflicts=None,
            )
            db.add(qc_report)

            anomaly_records: list[AnomalyRecord] = []

            for tid in traj_info:
                info = traj_info[tid]
                flags = info["flags"]

                if flags.get("cross_channel"):
                    anomaly_records.append(AnomalyRecord(
                        trajectory_id=tid,
                        task_id=task_id,
                        severity="blocking",
                        type="cross_channel",
                        description="粒子轨迹跨越多个通道边界",
                        details={"particle_id": info["particle_id"], "channel_id": info["channel_id"]},
                    ))

                if flags.get("short_trajectory"):
                    anomaly_records.append(AnomalyRecord(
                        trajectory_id=tid,
                        task_id=task_id,
                        severity="warning",
                        type="short_trajectory",
                        description=f"轨迹过短: {info['num_frames']}帧 < 最小{min_frames}帧",
                        details={"num_frames": info["num_frames"], "min_required": min_frames},
                    ))

                if flags.get("broken"):
                    anomaly_records.append(AnomalyRecord(
                        trajectory_id=tid,
                        task_id=task_id,
                        severity="warning",
                        type="broken_trajectory",
                        description="轨迹存在断点/缺失帧",
                        details={"break_points": flags.get("break_points", [])},
                    ))

                if flags.get("reconnected"):
                    anomaly_records.append(AnomalyRecord(
                        trajectory_id=tid,
                        task_id=task_id,
                        severity="info",
                        type="reconnected",
                        description="断裂轨迹已自动重连",
                        details={"reconnected_from": flags.get("reconnected_from")},
                    ))

                if flags.get("intensity_drop"):
                    anomaly_records.append(AnomalyRecord(
                        trajectory_id=tid,
                        task_id=task_id,
                        severity="warning",
                        type="intensity_drop",
                        description="轨迹中存在强度骤降（可能是光漂白或离焦）",
                        details={},
                    ))

            db.bulk_save_objects(anomaly_records)

            batch.status = "analyzed"
            db.commit()

            qc_report_dict = {
                "id": qc_report.id,
                "short_trajectories": qc_report.short_trajectories,
                "broken_trajectories": qc_report.broken_trajectories,
                "reconnected_trajectories": qc_report.reconnected_trajectories,
                "frame_missing_count": qc_report.frame_missing_count,
                "drift_estimate": qc_report.drift_estimate,
                "cross_channel_count": qc_report.cross_channel_count,
                "intensity_drop_count": qc_report.intensity_drop_count,
                "metadata_conflicts": qc_report.metadata_conflicts,
                "issues_count": len(anomaly_records),
            }
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

        calibration_summary: Optional[dict[str, Any]] = None
        if calib_result is not None:
            calibration_summary = {
                "slope": calib_result.slope,
                "intercept": calib_result.intercept,
                "r2": calib_result.r2,
                "per_channel_bias": calib_result.per_channel_bias,
                "calibration_points": calib_result.calibration_points,
            }

        result_summary = {
            "batch_id": batch_id,
            "task_id": task_id,
            "total_trajectories": total_traj,
            "valid_trajectories": int(agg_stats.get("n_valid", 0)),
            "short_trajectories": len(short_ids),
            "cross_channel_count": len([tid for tid in traj_info if traj_info[tid]["flags"].get("cross_channel")]),
            "statistics": {
                "mean_D_um2_s": agg_stats.get("mean_D"),
                "median_D_um2_s": agg_stats.get("median_D"),
                "std_D_um2_s": agg_stats.get("std_D"),
                "mean_radius_nm": agg_stats.get("mean_radius_nm"),
                "median_radius_nm": agg_stats.get("median_radius_nm"),
            },
            "drift": drift_summary_data,
            "qc_report": qc_report_dict if "qc_report_dict" in locals() else None,
            "calibration": calibration_summary,
            "anomalies_count": len(anomaly_records) if "anomaly_records" in locals() else 0,
        }

        progress_callback(98.0, "分析完成，整理结果...")
        return result_summary

    def list_batches(self, skip: int = 0, limit: int = 100) -> list[ExperimentBatch]:
        db = self._get_session()
        try:
            return db.query(ExperimentBatch).order_by(
                ExperimentBatch.created_at.desc()
            ).offset(skip).limit(limit).all()
        finally:
            db.close()

    def get_batch(self, batch_id: int) -> Optional[ExperimentBatch]:
        db = self._get_session()
        try:
            return db.get(ExperimentBatch, batch_id)
        finally:
            db.close()

    def _resolve_task_id(self, batch_id: int, task_id: Optional[int]) -> Optional[int]:
        if task_id is not None:
            return task_id
        db = self._get_session()
        try:
            task = db.query(AnalysisTask).filter(
                AnalysisTask.batch_id == batch_id,
                AnalysisTask.status == "completed",
            ).order_by(AnalysisTask.id.desc()).first()
            if task:
                return task.id
            task = db.query(AnalysisTask).filter(
                AnalysisTask.batch_id == batch_id,
            ).order_by(AnalysisTask.id.desc()).first()
            return task.id if task else None
        finally:
            db.close()

    def get_qc_summary(self, batch_id: int, task_id: Optional[int] = None) -> dict:
        resolved_task_id = self._resolve_task_id(batch_id, task_id)
        db = self._get_session()
        try:
            qc = db.query(QCReport).filter(
                QCReport.batch_id == batch_id,
            )
            if resolved_task_id is not None:
                qc = qc.filter(QCReport.task_id == resolved_task_id)
            qc = qc.order_by(QCReport.id.desc()).first()

            summary = db.query(BatchSummary).filter(
                BatchSummary.batch_id == batch_id,
            )
            if resolved_task_id is not None:
                summary = summary.filter(BatchSummary.task_id == resolved_task_id)
            summary = summary.order_by(BatchSummary.id.desc()).first()

            batch = db.get(ExperimentBatch, batch_id)

            result: dict[str, Any] = {
                "batch_id": batch_id,
                "task_id": resolved_task_id,
                "qc_report": None,
                "batch_summary": None,
                "batch": {
                    "name": batch.name if batch else None,
                    "status": batch.status if batch else None,
                },
            }

            if qc:
                result["qc_report"] = {
                    "id": qc.id,
                    "short_trajectories": qc.short_trajectories,
                    "broken_trajectories": qc.broken_trajectories,
                    "reconnected_trajectories": qc.reconnected_trajectories,
                    "frame_missing_count": qc.frame_missing_count,
                    "drift_estimate": qc.drift_estimate,
                    "cross_channel_count": qc.cross_channel_count,
                    "intensity_drop_count": qc.intensity_drop_count,
                    "metadata_conflicts": qc.metadata_conflicts,
                    "created_at": qc.created_at.isoformat() if qc.created_at else None,
                }

            if summary:
                result["batch_summary"] = {
                    "id": summary.id,
                    "mean_D": summary.mean_D,
                    "median_D": summary.median_D,
                    "std_D": summary.std_D,
                    "mean_radius_nm": summary.mean_radius_nm,
                    "valid_trajectories": summary.valid_trajectories,
                    "total_trajectories": summary.total_trajectories,
                    "drift_summary": summary.drift_summary,
                    "distribution_stats": summary.distribution_stats,
                }

            return result
        finally:
            db.close()

    def get_trajectory_results(
        self,
        batch_id: int,
        task_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
        channel_id: Optional[int] = None,
        only_valid: bool = False,
    ) -> list[dict]:
        resolved_task_id = self._resolve_task_id(batch_id, task_id)
        if resolved_task_id is None:
            return []

        db = self._get_session()
        try:
            query = db.query(TrajectoryResult).join(
                ParticleTrajectory,
                TrajectoryResult.trajectory_id == ParticleTrajectory.id,
            ).filter(
                ParticleTrajectory.batch_id == batch_id,
                TrajectoryResult.task_id == resolved_task_id,
            )

            if channel_id is not None:
                query = query.filter(ParticleTrajectory.channel_id == channel_id)

            if only_valid:
                query = query.filter(TrajectoryResult.excluded_from_distribution == False)

            query = query.order_by(TrajectoryResult.id).offset(skip).limit(limit)
            results = query.all()

            output: list[dict] = []
            for tr in results:
                traj = db.get(ParticleTrajectory, tr.trajectory_id)
                output.append({
                    "id": tr.id,
                    "trajectory_id": tr.trajectory_id,
                    "task_id": tr.task_id,
                    "particle_id": traj.particle_id if traj else None,
                    "channel_id": traj.channel_id if traj else None,
                    "num_frames": traj.num_frames if traj else None,
                    "diffusion_D_um2_s": tr.diffusion_D_um2_s,
                    "alpha_exponent": tr.alpha_exponent,
                    "fit_r2": tr.fit_r2,
                    "fit_lag_start": tr.fit_lag_start,
                    "fit_lag_end": tr.fit_lag_end,
                    "ci_low": tr.ci_low,
                    "ci_high": tr.ci_high,
                    "hydro_radius_nm": tr.hydro_radius_nm,
                    "model_type": tr.model_type,
                    "model_reason": tr.model_reason,
                    "excluded_from_distribution": tr.excluded_from_distribution,
                    "exclude_reason": tr.exclude_reason,
                    "drift_velocity_x": tr.drift_velocity_x,
                    "drift_velocity_y": tr.drift_velocity_y,
                })
            return output
        finally:
            db.close()

    def get_batch_summary(self, batch_id: int, task_id: Optional[int] = None) -> dict:
        resolved_task_id = self._resolve_task_id(batch_id, task_id)
        db = self._get_session()
        try:
            bs = db.query(BatchSummary).filter(
                BatchSummary.batch_id == batch_id,
            )
            if resolved_task_id is not None:
                bs = bs.filter(BatchSummary.task_id == resolved_task_id)
            bs = bs.order_by(BatchSummary.id.desc()).first()

            if not bs:
                return {"batch_id": batch_id, "task_id": resolved_task_id, "summary": None}

            return {
                "batch_id": batch_id,
                "task_id": resolved_task_id,
                "summary": {
                    "id": bs.id,
                    "mean_D": bs.mean_D,
                    "median_D": bs.median_D,
                    "std_D": bs.std_D,
                    "mean_radius_nm": bs.mean_radius_nm,
                    "valid_trajectories": bs.valid_trajectories,
                    "total_trajectories": bs.total_trajectories,
                    "drift_summary": bs.drift_summary,
                    "distribution_stats": bs.distribution_stats,
                },
            }
        finally:
            db.close()

    def get_calibration(self, batch_id: int, task_id: Optional[int] = None) -> Optional[dict]:
        resolved_task_id = self._resolve_task_id(batch_id, task_id)
        if resolved_task_id is None:
            return None

        db = self._get_session()
        try:
            cps = db.query(CalibrationPoint).filter(
                CalibrationPoint.batch_id == batch_id,
                CalibrationPoint.task_id == resolved_task_id,
            ).all()

            if not cps:
                return None

            points_list: list[dict[str, Any]] = []
            bias_vals: list[float] = []
            for cp in cps:
                pd_dict = {
                    "id": cp.id,
                    "channel_id": cp.channel_id,
                    "nominal_nm": cp.nominal_nm,
                    "measured_nm": cp.measured_nm,
                    "bias_pct": cp.bias_pct,
                    "D_um2_s": cp.D_um2_s,
                }
                points_list.append(pd_dict)
                if cp.bias_pct is not None and not (isinstance(cp.bias_pct, float) and np.isnan(cp.bias_pct)):
                    bias_vals.append(float(cp.bias_pct))

            cal_tuples: list[tuple[float, float, Optional[int]]] = []
            for cp in cps:
                if cp.nominal_nm and cp.measured_nm:
                    cal_tuples.append((float(cp.nominal_nm), float(cp.measured_nm), cp.channel_id))

            slope = intercept = r2 = None
            if len(cal_tuples) >= 2:
                cal = calibration_curve(cal_tuples)
                slope = cal.slope
                intercept = cal.intercept
                r2 = cal.r2

            return {
                "batch_id": batch_id,
                "task_id": resolved_task_id,
                "calibration_points": points_list,
                "mean_bias_pct": float(np.mean(bias_vals)) if bias_vals else None,
                "std_bias_pct": float(np.std(bias_vals, ddof=1)) if len(bias_vals) > 1 else None,
                "overall_r2": r2,
                "calibration_factor": slope if slope is not None and not np.isnan(slope) else None,
                "intercept": intercept if intercept is not None and not (isinstance(intercept, float) and np.isnan(intercept)) else None,
            }
        finally:
            db.close()

    def get_anomalies(
        self,
        batch_id: int,
        task_id: Optional[int] = None,
        severity: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[dict]:
        resolved_task_id = self._resolve_task_id(batch_id, task_id)
        if resolved_task_id is None:
            return []

        db = self._get_session()
        try:
            query = db.query(AnomalyRecord).join(
                ParticleTrajectory,
                AnomalyRecord.trajectory_id == ParticleTrajectory.id,
            ).filter(
                ParticleTrajectory.batch_id == batch_id,
                AnomalyRecord.task_id == resolved_task_id,
            )

            if severity:
                query = query.filter(AnomalyRecord.severity == severity)

            query = query.order_by(
                AnomalyRecord.severity.desc(),
                AnomalyRecord.id,
            ).offset(skip).limit(limit)

            records = query.all()

            output: list[dict] = []
            for rec in records:
                traj = db.get(ParticleTrajectory, rec.trajectory_id)
                output.append({
                    "id": rec.id,
                    "trajectory_id": rec.trajectory_id,
                    "task_id": rec.task_id,
                    "particle_id": traj.particle_id if traj else None,
                    "channel_id": traj.channel_id if traj else None,
                    "severity": rec.severity,
                    "type": rec.type,
                    "description": rec.description,
                    "details": rec.details,
                    "created_at": rec.created_at.isoformat() if rec.created_at else None,
                })
            return output
        finally:
            db.close()

    def explain_trajectory(self, trajectory_id: int, task_id: Optional[int] = None) -> dict:
        db = self._get_session()
        try:
            traj = db.get(ParticleTrajectory, trajectory_id)
            if not traj:
                return {"error": f"Trajectory {trajectory_id} 不存在"}

            batch_id = traj.batch_id
            resolved_task_id = self._resolve_task_id(batch_id, task_id)

            result_query = db.query(TrajectoryResult).filter(
                TrajectoryResult.trajectory_id == trajectory_id,
            )
            if resolved_task_id is not None:
                result_query = result_query.filter(TrajectoryResult.task_id == resolved_task_id)
            tr = result_query.order_by(TrajectoryResult.id.desc()).first()

            anomalies_query = db.query(AnomalyRecord).filter(
                AnomalyRecord.trajectory_id == trajectory_id,
            )
            if resolved_task_id is not None:
                anomalies_query = anomalies_query.filter(AnomalyRecord.task_id == resolved_task_id)
            anomalies = anomalies_query.all()

            flags = traj.flags or {}
            flag_list = [k for k, v in flags.items() if v]

            anomaly_list: list[dict[str, Any]] = []
            for a in anomalies:
                anomaly_list.append({
                    "severity": a.severity,
                    "type": a.type,
                    "description": a.description,
                    "details": a.details,
                })

            excluded = tr.excluded_from_distribution if tr else False
            exclude_reason = tr.exclude_reason if tr else None

            if flags.get("cross_channel"):
                excluded = True
                exclude_reason = exclude_reason or "cross_channel_particle"

            return {
                "trajectory_id": trajectory_id,
                "particle_id": traj.particle_id,
                "channel_id": traj.channel_id,
                "num_frames": traj.num_frames,
                "duration_s": traj.duration_s,
                "qc_passed": traj.qc_passed,
                "excluded": excluded,
                "exclude_reason": exclude_reason,
                "flags": flag_list,
                "anomalies": anomaly_list,
                "model_type": tr.model_type if tr else None,
                "model_reason": tr.model_reason if tr else None,
                "fit_quality": {
                    "diffusion_D_um2_s": tr.diffusion_D_um2_s if tr else None,
                    "alpha_exponent": tr.alpha_exponent if tr else None,
                    "fit_r2": tr.fit_r2 if tr else None,
                    "hydro_radius_nm": tr.hydro_radius_nm if tr else None,
                    "fit_lag_start": tr.fit_lag_start if tr else None,
                    "fit_lag_end": tr.fit_lag_end if tr else None,
                    "ci_low": tr.ci_low if tr else None,
                    "ci_high": tr.ci_high if tr else None,
                } if tr else None,
            }
        finally:
            db.close()

    def export_reports(
        self,
        batch_id: int,
        task_id: int,
        export_dir: Optional[str | Path] = None,
    ) -> dict:
        db = self._get_session()
        try:
            batch = db.get(ExperimentBatch, batch_id)
            if not batch:
                return {"error": f"Batch {batch_id} 不存在"}

            task = db.get(AnalysisTask, task_id)
            if not task:
                return {"error": f"Task {task_id} 不存在"}

            batch_data = {
                "id": batch.id,
                "name": batch.name,
                "description": batch.description,
                "pixel_size_um": batch.pixel_size_um,
                "frame_rate_hz": batch.frame_rate_hz,
                "temperature_c": batch.temperature_c,
                "viscosity_pa_s": batch.viscosity_pa_s,
                "nominal_diameter_nm": batch.nominal_diameter_nm,
                "channel_width_um": batch.channel_width_um,
                "channel_height_um": batch.channel_height_um,
                "metadata_json": batch.metadata_json,
                "status": batch.status,
            }

            trajectories = db.query(ParticleTrajectory).filter(
                ParticleTrajectory.batch_id == batch_id
            ).all()

            trajectories_export: list[dict[str, Any]] = []
            for traj in trajectories:
                tr = db.query(TrajectoryResult).filter(
                    TrajectoryResult.trajectory_id == traj.id,
                    TrajectoryResult.task_id == task_id,
                ).first()

                result_dict: dict[str, Any] = {}
                if tr:
                    result_dict = {
                        "D_um2_s": tr.diffusion_D_um2_s,
                        "diffusion_D_um2_s": tr.diffusion_D_um2_s,
                        "alpha": tr.alpha_exponent,
                        "alpha_exponent": tr.alpha_exponent,
                        "R2": tr.fit_r2,
                        "fit_r2": tr.fit_r2,
                        "model_type": tr.model_type,
                        "model_reason": tr.model_reason,
                        "hydro_radius_nm": tr.hydro_radius_nm,
                        "fit_lag_start": tr.fit_lag_start,
                        "fit_lag_end": tr.fit_lag_end,
                        "lag_start_s": (
                            (tr.fit_lag_start / batch.frame_rate_hz)
                            if tr.fit_lag_start and batch.frame_rate_hz
                            else None
                        ),
                        "lag_end_s": (
                            (tr.fit_lag_end / batch.frame_rate_hz)
                            if tr.fit_lag_end and batch.frame_rate_hz
                            else None
                        ),
                        "ci_low": tr.ci_low,
                        "ci_high": tr.ci_high,
                        "drift_velocity_x": tr.drift_velocity_x,
                        "drift_velocity_y": tr.drift_velocity_y,
                        "excluded_from_distribution": tr.excluded_from_distribution,
                        "exclude_reason": tr.exclude_reason,
                        "msd_points": tr.msd_points,
                    }

                anomalies = db.query(AnomalyRecord).filter(
                    AnomalyRecord.trajectory_id == traj.id,
                    AnomalyRecord.task_id == task_id,
                ).all()

                traj_dict = {
                    "trajectory_id": traj.id,
                    "id": traj.id,
                    "particle_id": traj.particle_id,
                    "channel_id": traj.channel_id,
                    "num_frames": traj.num_frames,
                    "duration_s": traj.duration_s,
                    "qc_passed": traj.qc_passed,
                    "flags": traj.flags or {},
                    "result": result_dict,
                    "n_outliers": sum(
                        1 for pt in traj.points if pt.is_outlier
                    ),
                    "anomalies": [
                        {
                            "severity": a.severity,
                            "type": a.type,
                            "description": a.description,
                            "details": a.details,
                        }
                        for a in anomalies
                    ],
                }
                trajectories_export.append(traj_dict)

            qc = db.query(QCReport).filter(
                QCReport.batch_id == batch_id,
                QCReport.task_id == task_id,
            ).first()

            qc_report_dict: dict[str, Any] = {}
            if qc:
                qc_report_dict = {
                    "id": qc.id,
                    "short_trajectories": qc.short_trajectories,
                    "broken_trajectories": qc.broken_trajectories,
                    "reconnected_trajectories": qc.reconnected_trajectories,
                    "frame_missing_count": qc.frame_missing_count,
                    "drift_estimate": qc.drift_estimate,
                    "cross_channel_count": qc.cross_channel_count,
                    "intensity_drop_count": qc.intensity_drop_count,
                    "metadata_conflicts": qc.metadata_conflicts,
                    "total_trajectories": len(trajectories),
                }

            calibration_data = self.get_calibration(batch_id, task_id)

            if export_dir is None:
                export_path = self._settings.result_dir
            else:
                export_path = Path(export_dir)

            exporter = ReportExporter(Path(export_path))
            try:
                paths = exporter.export_all(
                    batch_id=str(batch_id),
                    task_id=str(task_id),
                    batch_data=batch_data,
                    trajectories=trajectories_export,
                    qc_report=qc_report_dict,
                    calibration=calibration_data,
                )
            except Exception as e:
                return {
                    "error": f"导出报告失败: {str(e)}",
                    "traceback": traceback.format_exc(),
                }

            return {
                "batch_id": batch_id,
                "task_id": task_id,
                "export_dir": str(export_path),
                "json_report": str(paths.json),
                "csv_reports": {k: str(v) for k, v in paths.csv.items()},
                "markdown_report": str(paths.markdown) if paths.markdown else None,
                "trajectory_count": len(trajectories_export),
                "generated_at": datetime.utcnow().isoformat(),
            }
        finally:
            db.close()
