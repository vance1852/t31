from __future__ import annotations

import numpy as np
import pytest

from nano_dispersion.algorithms.stokes_einstein import (
    _um2s_to_m2s,
    calibration_curve,
    compute_diffusion_from_radius,
    compute_hydrodynamic_radius,
)


def test_radius_round_trip():
    k_B = 1.380649e-23
    T_K = 298.15
    eta = 0.001

    for R_nm_true in [10.0, 50.0, 100.0, 500.0, 1000.0]:
        D_m2_s = compute_diffusion_from_radius(R_nm_true, T_K, eta, k_B)
        assert D_m2_s > 0
        assert np.isfinite(D_m2_s)

        R_nm_recovered = compute_hydrodynamic_radius(D_m2_s, T_K, eta, k_B)
        assert np.isfinite(R_nm_recovered)
        assert abs(R_nm_recovered - R_nm_true) / R_nm_true < 1e-6


def test_temperature_effect():
    R_nm = 100.0
    eta = 0.001
    k_B = 1.380649e-23

    D_25 = compute_diffusion_from_radius(R_nm, 25 + 273.15, eta, k_B)
    D_50 = compute_diffusion_from_radius(R_nm, 50 + 273.15, eta, k_B)

    assert D_50 > D_25


def test_viscosity_effect():
    R_nm = 100.0
    T_K = 298.15
    k_B = 1.380649e-23

    D_low_eta = compute_diffusion_from_radius(R_nm, T_K, 0.0005, k_B)
    D_high_eta = compute_diffusion_from_radius(R_nm, T_K, 0.005, k_B)

    assert D_low_eta > D_high_eta


def test_calibration_linear():
    rng = np.random.default_rng(42)
    calibration_points = []
    nominal_nm_list = [25.0, 50.0, 100.0, 200.0, 500.0]

    for nom in nominal_nm_list:
        noise = rng.normal(0, nom * 0.02)
        meas = nom * 1.05 + noise
        channel_id = int(rng.integers(0, 3))
        calibration_points.append((nom, meas, channel_id))

    result = calibration_curve(calibration_points)

    assert np.isfinite(result.slope)
    assert np.isfinite(result.intercept)
    assert 0.0 < result.r2 <= 1.0
    assert result.r2 > 0.95
