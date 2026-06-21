from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from nano_dispersion.models.database import Base


class TimestampMixin:
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class ExperimentBatch(TimestampMixin, Base):
    __tablename__ = "experiment_batches"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    import_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    metadata_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending", index=True)
    pixel_size_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    frame_rate_hz: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    temperature_c: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    viscosity_pa_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nominal_diameter_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    channel_width_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    channel_height_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    trajectories: Mapped[list["ParticleTrajectory"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan", lazy="selectin"
    )
    tasks: Mapped[list["AnalysisTask"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan", lazy="selectin"
    )
    summaries: Mapped[list["BatchSummary"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan", lazy="selectin"
    )
    calibrations: Mapped[list["CalibrationPoint"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan", lazy="selectin"
    )
    qc_reports: Mapped[list["QCReport"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan", lazy="selectin"
    )


class ParticleTrajectory(TimestampMixin, Base):
    __tablename__ = "particle_trajectories"

    batch_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_batches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    particle_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    channel_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    num_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    flags: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    qc_passed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    batch: Mapped["ExperimentBatch"] = relationship(back_populates="trajectories")
    points: Mapped[list["TrajectoryPoint"]] = relationship(
        back_populates="trajectory", cascade="all, delete-orphan", lazy="selectin"
    )
    results: Mapped[list["TrajectoryResult"]] = relationship(
        back_populates="trajectory", cascade="all, delete-orphan", lazy="selectin"
    )
    anomalies: Mapped[list["AnomalyRecord"]] = relationship(
        back_populates="trajectory", cascade="all, delete-orphan", lazy="selectin"
    )


class TrajectoryPoint(TimestampMixin, Base):
    __tablename__ = "trajectory_points"

    trajectory_id: Mapped[int] = mapped_column(
        ForeignKey("particle_trajectories.id", ondelete="CASCADE"), nullable=False, index=True
    )
    frame: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    time_s: Mapped[float] = mapped_column(Float, nullable=False)
    x_um: Mapped[float] = mapped_column(Float, nullable=False)
    y_um: Mapped[float] = mapped_column(Float, nullable=False)
    x_corrected_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    y_corrected_um: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    intensity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_outlier: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    anomaly_tag: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    trajectory: Mapped["ParticleTrajectory"] = relationship(back_populates="points")


class AnalysisTask(TimestampMixin, Base):
    __tablename__ = "analysis_tasks"

    batch_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_batches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    task_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending", index=True)
    progress_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    result_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    batch: Mapped["ExperimentBatch"] = relationship(back_populates="tasks")
    trajectory_results: Mapped[list["TrajectoryResult"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", lazy="selectin"
    )
    batch_summaries: Mapped[list["BatchSummary"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", lazy="selectin"
    )
    calibrations: Mapped[list["CalibrationPoint"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", lazy="selectin"
    )
    anomalies: Mapped[list["AnomalyRecord"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", lazy="selectin"
    )
    qc_reports: Mapped[list["QCReport"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", lazy="selectin"
    )


class TrajectoryResult(TimestampMixin, Base):
    __tablename__ = "trajectory_results"

    trajectory_id: Mapped[int] = mapped_column(
        ForeignKey("particle_trajectories.id", ondelete="CASCADE"), nullable=False, index=True
    )
    task_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    diffusion_D_um2_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    alpha_exponent: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fit_r2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fit_lag_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    fit_lag_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    ci_low: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ci_high: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hydro_radius_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    model_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    excluded_from_distribution: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    exclude_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    msd_points: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    drift_velocity_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    drift_velocity_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    trajectory: Mapped["ParticleTrajectory"] = relationship(back_populates="results")
    task: Mapped["AnalysisTask"] = relationship(back_populates="trajectory_results")


class BatchSummary(TimestampMixin, Base):
    __tablename__ = "batch_summaries"

    task_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_batches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    mean_D: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    median_D: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    std_D: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mean_radius_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    valid_trajectories: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_trajectories: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    drift_summary: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    distribution_stats: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    task: Mapped["AnalysisTask"] = relationship(back_populates="batch_summaries")
    batch: Mapped["ExperimentBatch"] = relationship(back_populates="summaries")


class CalibrationPoint(TimestampMixin, Base):
    __tablename__ = "calibration_points"

    batch_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_batches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    task_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    channel_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    nominal_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    measured_nm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bias_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    D_um2_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    batch: Mapped["ExperimentBatch"] = relationship(back_populates="calibrations")
    task: Mapped["AnalysisTask"] = relationship(back_populates="calibrations")


class AnomalyRecord(TimestampMixin, Base):
    __tablename__ = "anomaly_records"

    trajectory_id: Mapped[int] = mapped_column(
        ForeignKey("particle_trajectories.id", ondelete="CASCADE"), nullable=False, index=True
    )
    task_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    severity: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    trajectory: Mapped["ParticleTrajectory"] = relationship(back_populates="anomalies")
    task: Mapped["AnalysisTask"] = relationship(back_populates="anomalies")


class QCReport(TimestampMixin, Base):
    __tablename__ = "qc_reports"

    batch_id: Mapped[int] = mapped_column(
        ForeignKey("experiment_batches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    task_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_tasks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    short_trajectories: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    broken_trajectories: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reconnected_trajectories: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    frame_missing_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    drift_estimate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cross_channel_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    intensity_drop_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_conflicts: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    batch: Mapped["ExperimentBatch"] = relationship(back_populates="qc_reports")
    task: Mapped["AnalysisTask"] = relationship(back_populates="qc_reports")
