from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CalibrationResult:
    slope: float
    intercept: float
    r2: float
    per_channel_bias: dict[int, dict[str, float]] = field(default_factory=dict)
    residuals: list[float] = field(default_factory=list)
    calibration_points: list[dict] = field(default_factory=list)


def _um2s_to_m2s(D_um2_s: float) -> float:
    return D_um2_s * 1e-12


def _m2s_to_um2s(D_m2_s: float) -> float:
    return D_m2_s * 1e12


def compute_hydrodynamic_radius(
    D_m2_s: float,
    temperature_K: float,
    viscosity_Pa_s: float,
    k_B: float = 1.380649e-23,
) -> float:
    if D_m2_s <= 0 or temperature_K <= 0 or viscosity_Pa_s <= 0:
        return float("nan")

    eta = viscosity_Pa_s
    R_m = k_B * temperature_K / (6.0 * np.pi * eta * D_m2_s)
    R_nm = R_m * 1e9
    return float(R_nm)


def compute_diffusion_from_radius(
    radius_nm: float,
    temperature_K: float,
    viscosity_Pa_s: float,
    k_B: float = 1.380649e-23,
) -> float:
    if radius_nm <= 0 or temperature_K <= 0 or viscosity_Pa_s <= 0:
        return float("nan")

    R_m = radius_nm * 1e-9
    eta = viscosity_Pa_s
    D_m2_s = k_B * temperature_K / (6.0 * np.pi * eta * R_m)
    return float(D_m2_s)


def _calculate_linreg_r2(
    x: np.ndarray, y: np.ndarray, slope: float, intercept: float
) -> float:
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def calibration_curve(
    calibration_points: list[tuple[float, float, Optional[int]]],
) -> CalibrationResult:
    if len(calibration_points) < 2:
        return CalibrationResult(
            slope=np.nan,
            intercept=np.nan,
            r2=0.0,
        )

    nominal_vals: list[float] = []
    measured_vals: list[float] = []
    channel_ids: list[Optional[int]] = []
    point_records: list[dict] = []

    for nom, meas, ch in calibration_points:
        if np.isnan(nom) or np.isnan(meas) or nom <= 0 or meas <= 0:
            continue
        nominal_vals.append(float(nom))
        measured_vals.append(float(meas))
        channel_ids.append(ch)
        point_records.append(
            {"nominal_nm": float(nom), "measured_nm": float(meas), "channel_id": ch}
        )

    if len(nominal_vals) < 2:
        return CalibrationResult(
            slope=np.nan,
            intercept=np.nan,
            r2=0.0,
            calibration_points=point_records,
        )

    x = np.array(nominal_vals, dtype=float)
    y = np.array(measured_vals, dtype=float)

    A = np.vstack([x, np.ones_like(x)]).T
    try:
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])

    slope = float(slope)
    intercept = float(intercept)
    r2 = _calculate_linreg_r2(x, y, slope, intercept)

    y_pred = slope * x + intercept
    residuals = (y - y_pred).tolist()

    per_channel_bias: dict[str, dict[str, float]] = {}
    channel_points: dict[str, list[tuple[float, float]]] = {}

    for nom, meas, ch in zip(nominal_vals, measured_vals, channel_ids):
        if ch is None:
            continue
        ch_key = str(ch)
        if ch_key not in channel_points:
            channel_points[ch_key] = []
        channel_points[ch_key].append((nom, meas))

    for ch_key, points in channel_points.items():
        if len(points) == 0:
            continue
        noms = np.array([p[0] for p in points])
        meas = np.array([p[1] for p in points])
        biases_pct = (meas - noms) / noms * 100.0
        per_channel_bias[ch_key] = {
            "mean_bias_pct": float(np.mean(biases_pct)),
            "std_bias_pct": float(np.std(biases_pct)),
            "count": int(len(points)),
            "mean_nominal_nm": float(np.mean(noms)),
            "mean_measured_nm": float(np.mean(meas)),
        }

    return CalibrationResult(
        slope=slope,
        intercept=intercept,
        r2=float(r2),
        per_channel_bias=per_channel_bias,
        residuals=[float(r) for r in residuals],
        calibration_points=point_records,
    )
