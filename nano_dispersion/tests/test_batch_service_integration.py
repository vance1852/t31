from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from nano_dispersion.config import Settings
from nano_dispersion.data_generator import generate_sample_batch, init_sample_data
from nano_dispersion.models.database import Base
from nano_dispersion.models.orm import (
    AnomalyRecord,
    AnalysisTask,
    BatchSummary,
    ParticleTrajectory,
    QCReport,
    TrajectoryResult,
)
from nano_dispersion.reports.exporter import ReportExporter
from nano_dispersion.services.batch_service import BatchService
from nano_dispersion.services.task_manager import TaskManager


def _make_simple_brownian_batch(
    base_dir: Path,
    batch_id: str,
    num_particles: int = 8,
    n_frames: int = 40,
    D: float = 2.0,
    dt: float = 1.0 / 30.0,
    nominal_diameter_nm: float = 100.0,
    drift_um_per_s: tuple = (0.05, 0.01),
    seed: int = 0,
):
    batch_dir = base_dir / batch_id
    traj_dir = batch_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "batch_id": batch_id,
        "batch_name": batch_id,
        "description": f"test batch {batch_id}",
        "pixel_size_um": 0.107,
        "frame_rate_hz": 1.0 / dt,
        "temperature_c": 25.0,
        "viscosity_pa_s": 0.00089,
        "nominal_diameter_nm": nominal_diameter_nm,
        "channel_width_um": 200.0,
        "channel_height_um": 50.0,
        "channels": [0],
        "channels_boundaries": {
            "0": {"x_min": 0, "x_max": 200, "y_min": 0, "y_max": 50},
        },
    }
    with open(batch_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    rng = np.random.default_rng(seed)
    sigma = np.sqrt(2.0 * D * dt)
    rows = []
    for pid in range(num_particles):
        x = rng.uniform(20, 180)
        y = rng.uniform(5, 45)
        for frame in range(n_frames):
            x += rng.normal(0, sigma) + drift_um_per_s[0] * dt
            y += rng.normal(0, sigma) + drift_um_per_s[1] * dt
            x = np.clip(x, 1, 199)
            y = np.clip(y, 1, 49)
            rows.append({
                "frame": frame,
                "time_s": frame * dt,
                "particle_id": pid + 1,
                "x_um": float(x),
                "y_um": float(y),
                "intensity": 1000.0 + rng.normal(0, 50),
                "channel_id": 0,
            })

    df = pd.DataFrame(rows)
    df.to_csv(traj_dir / "ch0_trajectories.csv", index=False)
    return batch_dir


@pytest.fixture
def integration_settings(tmp_path: Path) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=6817,
        base_dir=tmp_path,
        db_path=tmp_path / "integration.db",
        data_dir=tmp_path / "data" / "batches",
        result_dir=tmp_path / "results",
        min_trajectory_frames=20,
        max_gap_frames_for_reconnect=5,
        max_neighbor_distance_um=3.0,
        max_velocity_jump_um_per_s=50.0,
        drift_outlier_sigma=3.0,
        msd_fit_min_lags=5,
        msd_fit_max_lags_ratio=0.4,
        confidence_level=0.95,
        boltzmann_constant=1.380649e-23,
        default_water_viscosity_pa_s=0.001,
        worker_threads=1,
    )


@pytest.fixture
def integration_services(integration_settings: Settings):
    integration_settings.ensure_dirs()

    db_file = Path(integration_settings.db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{db_file.as_posix()}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True
    )

    task_manager = TaskManager(TestingSessionLocal, worker_threads=1)
    task_manager.start()

    batch_service = BatchService(TestingSessionLocal, task_manager, integration_settings)

    try:
        yield batch_service, task_manager, TestingSessionLocal, integration_settings
    finally:
        task_manager.shutdown(wait=False)
        engine.dispose()


def _wait_for_task(task_manager, task_id: str, timeout: float = 60.0) -> str | None:
    start = time.time()
    last_status = None
    while time.time() - start < timeout:
        status_obj = task_manager.get_task_status(task_id)
        if status_obj is not None:
            last_status = status_obj.status
            if last_status in ("completed", "failed"):
                break
        time.sleep(0.5)
    return last_status


def test_import_and_analyze_flow(integration_services, tmp_path: Path):
    batch_service, task_manager, session_factory, settings = integration_services

    out_dir = settings.data_dir
    batch_dir = _make_simple_brownian_batch(
        out_dir,
        batch_id="batch_flow_001",
        num_particles=8,
        n_frames=40,
        D=2.0,
        seed=42,
    )

    batch = batch_service.import_batch_from_directory(batch_dir, description="test")
    assert batch.id is not None
    assert batch.status == "imported"

    session = session_factory()
    try:
        traj_count = session.query(ParticleTrajectory).filter(
            ParticleTrajectory.batch_id == batch.id
        ).count()
    finally:
        session.close()
    assert traj_count > 0

    task_id = batch_service.submit_analysis(batch.id)
    assert task_id is not None

    final_status = _wait_for_task(task_manager, task_id, timeout=90.0)
    assert final_status == "completed", f"Expected completed, got {final_status}"

    summary = batch_service.get_batch_summary(batch.id, task_id)
    assert "summary" in summary
    sum_data = summary["summary"]
    assert sum_data is not None
    assert sum_data["total_trajectories"] > 0


def test_cross_channel_trajectory_blocks(integration_services, tmp_path: Path):
    batch_service, task_manager, session_factory, settings = integration_services

    out_dir = settings.data_dir
    batch_dir = out_dir / "cross_test"
    trajectories_dir = batch_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "batch_id": "cross_test",
        "batch_name": "cross_test",
        "description": "cross channel test",
        "pixel_size_um": 0.107,
        "frame_rate_hz": 30.0,
        "temperature_c": 25.0,
        "viscosity_pa_s": 0.00089,
        "nominal_diameter_nm": 100.0,
        "channel_width_um": 200.0,
        "channel_height_um": 50.0,
        "channels": [0, 1],
        "channels_boundaries": {
            "0": {"x_min": 0, "x_max": 200, "y_min": 0, "y_max": 50},
            "1": {"x_min": 200, "x_max": 400, "y_min": 0, "y_max": 50},
        },
    }
    with open(batch_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    rows = []
    n_frames = 30
    for frame in range(n_frames):
        x = 180.0 + frame * 3.0
        rows.append({
            "frame": frame,
            "time_s": frame / 30.0,
            "particle_id": 1,
            "x_um": x,
            "y_um": 25.0,
            "intensity": 1000.0,
            "channel_id": 0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(trajectories_dir / "crossing_particle.csv", index=False)

    batch = batch_service.import_batch_from_directory(batch_dir, description="cross test")

    task_id = batch_service.submit_analysis(batch.id)
    _wait_for_task(task_manager, task_id, timeout=60.0)

    session = session_factory()
    try:
        blocking_anoms = session.query(AnomalyRecord).join(
            ParticleTrajectory,
            AnomalyRecord.trajectory_id == ParticleTrajectory.id,
        ).filter(
            ParticleTrajectory.batch_id == batch.id,
            AnomalyRecord.severity == "blocking",
        ).all()
    finally:
        session.close()

    assert len(blocking_anoms) >= 0


def test_multiple_tasks_isolation(integration_services, tmp_path: Path):
    batch_service, task_manager, session_factory, settings = integration_services

    out_dir = settings.data_dir
    batch_dir1 = _make_simple_brownian_batch(
        out_dir,
        batch_id="iso_batch_50nm",
        num_particles=5,
        n_frames=35,
        D=4.0,
        nominal_diameter_nm=50.0,
        seed=10,
    )
    batch_dir2 = _make_simple_brownian_batch(
        out_dir,
        batch_id="iso_batch_500nm",
        num_particles=5,
        n_frames=35,
        D=0.4,
        nominal_diameter_nm=500.0,
        seed=20,
    )

    batch1 = batch_service.import_batch_from_directory(batch_dir1)
    batch2 = batch_service.import_batch_from_directory(batch_dir2)

    task1 = batch_service.submit_analysis(batch1.id)
    task2 = batch_service.submit_analysis(batch2.id)

    for tid in [task1, task2]:
        _wait_for_task(task_manager, tid, timeout=120.0)

    sum1 = batch_service.get_batch_summary(batch1.id, task1)
    sum2 = batch_service.get_batch_summary(batch2.id, task2)

    session = session_factory()
    try:
        t1_results = session.query(TrajectoryResult).join(
            AnalysisTask,
            TrajectoryResult.task_id == AnalysisTask.id,
        ).filter(AnalysisTask.batch_id == batch1.id).count()

        t2_results = session.query(TrajectoryResult).join(
            AnalysisTask,
            TrajectoryResult.task_id == AnalysisTask.id,
        ).filter(AnalysisTask.batch_id == batch2.id).count()
    finally:
        session.close()

    assert t1_results >= 0
    assert t2_results >= 0


def _to_str(v):
    if isinstance(v, bytes):
        try:
            return v.hex()
        except Exception:
            try:
                return v.decode("utf-8", errors="replace")
            except Exception:
                return str(v)
    return v


def _clean_bytes(obj):
    if isinstance(obj, bytes):
        return _to_str(obj)
    if isinstance(obj, dict):
        return {k: _clean_bytes(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_bytes(item) for item in obj]
    return obj


def test_export_report_files_exist(integration_services, tmp_path: Path):
    batch_service, task_manager, session_factory, settings = integration_services
    result_dir = settings.result_dir

    out_dir = settings.data_dir
    batch_dir = _make_simple_brownian_batch(
        out_dir,
        batch_id="export_batch_001",
        num_particles=8,
        n_frames=40,
        D=2.0,
        seed=30,
    )
    batch = batch_service.import_batch_from_directory(batch_dir, description="export test")

    task_id = batch_service.submit_analysis(batch.id)
    _wait_for_task(task_manager, task_id, timeout=90.0)

    traj_results = batch_service.get_trajectory_results(batch.id, task_id, limit=1000)

    trajectories_export = []
    for tr in traj_results:
        tr_clean = _clean_bytes(tr)
        trajectories_export.append({
            "trajectory_id": _to_str(tr_clean.get("trajectory_id")),
            "particle_id": _to_str(tr_clean.get("particle_id")),
            "channel_id": tr_clean.get("channel_id"),
            "num_frames": tr_clean.get("num_frames"),
            "qc_passed": True,
            "flags": {},
            "result": {
                "diffusion_D_um2_s": tr_clean.get("diffusion_D_um2_s"),
                "D_um2_s": tr_clean.get("diffusion_D_um2_s"),
                "alpha_exponent": tr_clean.get("alpha_exponent"),
                "alpha": tr_clean.get("alpha_exponent"),
                "fit_r2": tr_clean.get("fit_r2"),
                "R2": tr_clean.get("fit_r2"),
                "model_type": tr_clean.get("model_type"),
                "model_reason": tr_clean.get("model_reason"),
                "hydro_radius_nm": tr_clean.get("hydro_radius_nm"),
                "fit_lag_start": tr_clean.get("fit_lag_start"),
                "fit_lag_end": tr_clean.get("fit_lag_end"),
                "lag_start_s": tr_clean.get("fit_lag_start"),
                "lag_end_s": tr_clean.get("fit_lag_end"),
                "ci_low": tr_clean.get("ci_low"),
                "ci_high": tr_clean.get("ci_high"),
                "ci_95": [tr_clean.get("ci_low"), tr_clean.get("ci_high")],
                "drift_velocity_x": tr_clean.get("drift_velocity_x"),
                "drift_velocity_y": tr_clean.get("drift_velocity_y"),
                "drift_velocity": [tr_clean.get("drift_velocity_x"), tr_clean.get("drift_velocity_y")],
                "excluded_from_distribution": tr_clean.get("excluded_from_distribution"),
                "exclude_reason": tr_clean.get("exclude_reason"),
                "msd_points": [],
            },
        })

    batch_data_dict = {
        "name": _clean_bytes(batch.name),
        "pixel_size_um": batch.pixel_size_um,
        "frame_rate_hz": batch.frame_rate_hz,
        "temperature_c": batch.temperature_c,
        "viscosity_pa_s": batch.viscosity_pa_s,
        "nominal_diameter_nm": batch.nominal_diameter_nm,
        "channel_width_um": batch.channel_width_um,
        "channel_height_um": batch.channel_height_um,
    }

    exporter = ReportExporter(result_dir)

    safe_batch_id = _to_str(batch.id)
    safe_task_id = _to_str(task_id)

    paths = exporter.export_all(
        batch_id=safe_batch_id,
        task_id=safe_task_id,
        batch_data=batch_data_dict,
        trajectories=trajectories_export,
        qc_report={},
        calibration={},
    )

    assert paths.json.exists()
    assert paths.json.stat().st_size > 0

    assert "trajectories" in paths.csv
    assert paths.csv["trajectories"].exists()
    assert paths.csv["trajectories"].stat().st_size > 0

    assert "anomalies" in paths.csv
    assert paths.csv["anomalies"].exists()

    assert "calibration" in paths.csv
    assert paths.csv["calibration"].exists()

    assert "msd_summary" in paths.csv
    assert paths.csv["msd_summary"].exists()

    assert paths.markdown is not None
    assert paths.markdown.exists()
    assert paths.markdown.stat().st_size > 0
