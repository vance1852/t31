from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TimestampedBase(BaseModel):
    id: int
    created_at: datetime
    updated_at: datetime


class BatchCreate(BaseModel):
    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    import_path: Optional[str] = Field(None, max_length=512)
    metadata_json: Optional[dict[str, Any]] = None
    status: str = Field(default="pending", max_length=50)
    pixel_size_um: Optional[float] = None
    frame_rate_hz: Optional[float] = None
    temperature_c: Optional[float] = None
    viscosity_pa_s: Optional[float] = None
    nominal_diameter_nm: Optional[float] = None
    channel_width_um: Optional[float] = None
    channel_height_um: Optional[float] = None


class BatchResponse(TimestampedBase):
    model_config = ConfigDict(from_attributes=True)

    name: str
    description: Optional[str]
    import_path: Optional[str]
    metadata_json: Optional[dict[str, Any]]
    status: str
    pixel_size_um: Optional[float]
    frame_rate_hz: Optional[float]
    temperature_c: Optional[float]
    viscosity_pa_s: Optional[float]
    nominal_diameter_nm: Optional[float]
    channel_width_um: Optional[float]
    channel_height_um: Optional[float]


class TrajectoryPointResponse(TimestampedBase):
    model_config = ConfigDict(from_attributes=True)

    trajectory_id: int
    frame: int
    time_s: float
    x_um: float
    y_um: float
    x_corrected_um: Optional[float]
    y_corrected_um: Optional[float]
    intensity: Optional[float]
    is_outlier: bool
    anomaly_tag: Optional[str]


class TrajectoryResponse(TimestampedBase):
    model_config = ConfigDict(from_attributes=True)

    batch_id: int
    particle_id: int
    channel_id: Optional[str]
    num_frames: int
    duration_s: Optional[float]
    flags: Optional[dict[str, Any]]
    qc_passed: bool
    points: Optional[list[TrajectoryPointResponse]] = None


class TaskResponse(TimestampedBase):
    model_config = ConfigDict(from_attributes=True)

    batch_id: int
    task_type: str
    status: str
    progress_pct: float
    message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    result_json: Optional[dict[str, Any]]


class MSDPoint(BaseModel):
    lag: int
    time_s: float
    msd_um2: float
    sem: Optional[float] = None
    n_points: Optional[int] = None


class FitResult(BaseModel):
    diffusion_D_um2_s: float
    alpha_exponent: float
    fit_r2: float
    fit_lag_start: int
    fit_lag_end: int
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    msd_points: Optional[list[MSDPoint]] = None


class StokesEinsteinResult(BaseModel):
    hydro_radius_nm: float
    diffusion_D_um2_s: float
    temperature_c: float
    viscosity_pa_s: float
    boltzmann_constant: float = 1.380649e-23


class ModelDiscrimination(BaseModel):
    model_type: str
    model_reason: str
    brownian_r2: Optional[float] = None
    confined_r2: Optional[float] = None
    directed_r2: Optional[float] = None
    anomalous_r2: Optional[float] = None
    subdiffusive_r2: Optional[float] = None
    drift_velocity_x: Optional[float] = None
    drift_velocity_y: Optional[float] = None


class TrajectoryResultResponse(TimestampedBase):
    model_config = ConfigDict(from_attributes=True)

    trajectory_id: int
    task_id: int
    diffusion_D_um2_s: Optional[float]
    alpha_exponent: Optional[float]
    fit_r2: Optional[float]
    fit_lag_start: Optional[int]
    fit_lag_end: Optional[int]
    ci_low: Optional[float]
    ci_high: Optional[float]
    hydro_radius_nm: Optional[float]
    model_type: Optional[str]
    model_reason: Optional[str]
    excluded_from_distribution: bool
    exclude_reason: Optional[str]
    msd_points: Optional[list[MSDPoint]] = None
    drift_velocity_x: Optional[float]
    drift_velocity_y: Optional[float]


class CalibrationPointResponse(TimestampedBase):
    model_config = ConfigDict(from_attributes=True)

    batch_id: int
    task_id: int
    channel_id: Optional[str]
    nominal_nm: Optional[float]
    measured_nm: Optional[float]
    bias_pct: Optional[float]
    D_um2_s: Optional[float]


class CalibrationResult(BaseModel):
    batch_id: int
    task_id: int
    calibration_points: list[CalibrationPointResponse]
    mean_bias_pct: Optional[float] = None
    std_bias_pct: Optional[float] = None
    calibration_factor: Optional[float] = None
    overall_r2: Optional[float] = None


class AnomalyResponse(TimestampedBase):
    model_config = ConfigDict(from_attributes=True)

    trajectory_id: int
    task_id: int
    severity: str
    type: str
    description: Optional[str]
    details: Optional[dict[str, Any]]


class QCResponse(TimestampedBase):
    model_config = ConfigDict(from_attributes=True)

    batch_id: int
    task_id: int
    short_trajectories: int
    broken_trajectories: int
    reconnected_trajectories: int
    frame_missing_count: int
    drift_estimate: Optional[float]
    cross_channel_count: int
    intensity_drop_count: int
    metadata_conflicts: Optional[dict[str, Any]]


class BatchSummaryResponse(TimestampedBase):
    model_config = ConfigDict(from_attributes=True)

    task_id: int
    batch_id: int
    mean_D: Optional[float]
    median_D: Optional[float]
    std_D: Optional[float]
    mean_radius_nm: Optional[float]
    valid_trajectories: int
    total_trajectories: int
    drift_summary: Optional[dict[str, Any]]
    distribution_stats: Optional[dict[str, Any]]


class ReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    batch: BatchResponse
    task: TaskResponse
    summary: Optional[BatchSummaryResponse] = None
    qc_report: Optional[QCResponse] = None
    calibration: Optional[CalibrationResult] = None
    trajectory_count: int = 0
    valid_trajectory_count: int = 0
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    anomalies: list[AnomalyResponse] = []
