from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def compute_diffusion_coefficient_um2s(
    diameter_nm: float,
    temperature_c: float,
    viscosity_pa_s: float,
    k_B: float = 1.380649e-23,
) -> float:
    radius_m = (diameter_nm * 1e-9) / 2.0
    temperature_K = temperature_c + 273.15
    D_m2_s = k_B * temperature_K / (6.0 * np.pi * viscosity_pa_s * radius_m)
    return D_m2_s * 1e12


def _brownian_trajectory(
    D_um2_s: float,
    n_frames: int,
    dt_s: float,
    start_xy: Optional[tuple[float, float]] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    sigma = np.sqrt(2.0 * D_um2_s * dt_s)
    if start_xy is None:
        x0, y0 = 0.0, 0.0
    else:
        x0, y0 = start_xy
    dx = rng.normal(0.0, sigma, size=n_frames)
    dy = rng.normal(0.0, sigma, size=n_frames)
    xs = np.cumsum(dx) + x0 - dx[0]
    ys = np.cumsum(dy) + y0 - dy[0]
    xs[0] = x0
    ys[0] = y0
    return xs, ys


def _add_drift(
    xs: np.ndarray,
    ys: np.ndarray,
    vx: float,
    vy: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(xs)
    times = np.arange(n) * dt_s
    xs_drifted = xs + vx * times
    ys_drifted = ys + vy * times
    return xs_drifted, ys_drifted


def _add_jumps(
    xs: np.ndarray,
    ys: np.ndarray,
    D_um2_s: float,
    dt_s: float,
    jump_prob: float = 0.05,
    jump_sigma_factor: float = 10.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    n = len(xs)
    step_sigma = np.sqrt(2.0 * D_um2_s * dt_s)
    jump_sigma = step_sigma * jump_sigma_factor
    mask = rng.random(n) < jump_prob
    mask[0] = False
    jump_dx = rng.normal(0.0, jump_sigma, size=n) * mask
    jump_dy = rng.normal(0.0, jump_sigma, size=n) * mask
    xs_out = xs.copy()
    ys_out = ys.copy()
    xs_out[mask] += jump_dx[mask]
    ys_out[mask] += jump_dy[mask]
    return xs_out, ys_out


def _add_noise(
    xs: np.ndarray,
    ys: np.ndarray,
    localization_error_um: float = 0.02,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    xs_noisy = xs + rng.normal(0.0, localization_error_um, size=len(xs))
    ys_noisy = ys + rng.normal(0.0, localization_error_um, size=len(ys))
    return xs_noisy, ys_noisy


def _break_trajectory(
    xs: np.ndarray,
    ys: np.ndarray,
    times: np.ndarray,
    intensities: np.ndarray,
    frame_ids: np.ndarray,
    break_prob: float = 0.15,
    min_missing: int = 3,
    max_missing: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    n = len(xs)
    if n < 20:
        return xs, ys, times, intensities, frame_ids
    if rng.random() > break_prob:
        return xs, ys, times, intensities, frame_ids
    num_breaks = rng.integers(1, 3)
    keep_mask = np.ones(n, dtype=bool)
    for _ in range(num_breaks):
        lo = 10
        hi = n - max_missing - 5
        if lo >= hi:
            break
        break_start = rng.integers(lo, hi)
        missing_len = rng.integers(min_missing, max_missing + 1)
        break_end = min(break_start + missing_len, n)
        keep_mask[break_start:break_end] = False
    return (
        xs[keep_mask],
        ys[keep_mask],
        times[keep_mask],
        intensities[keep_mask],
        frame_ids[keep_mask],
    )


def _simulate_intensity(
    base_intensity: float,
    n_frames: int,
    photobleach: bool = True,
    decay_ratio_range: tuple[float, float] = (0.1, 0.3),
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    intensities = np.full(n_frames, base_intensity, dtype=float)
    noise = rng.normal(0.0, base_intensity * 0.05, size=n_frames)
    intensities += noise
    if photobleach and rng.random() < 0.25:
        decay_ratio = rng.uniform(*decay_ratio_range)
        start_frame = rng.integers(n_frames // 3, n_frames // 2)
        decay_len = n_frames - start_frame
        t = np.arange(decay_len)
        tau = decay_len / (-np.log(decay_ratio)) if decay_ratio > 0 else 1.0
        decay_curve = np.exp(-t / tau)
        decay_curve = decay_curve * (1.0 - decay_ratio) + decay_ratio
        intensities[start_frame:] *= decay_curve
    return intensities


def _generate_channel_bounds(
    channel_id: str,
    width_um: float,
    height_um: float,
) -> tuple[float, float, float, float]:
    channel_name = channel_id.lower()
    if "a" in channel_name:
        idx = 0
    elif "b" in channel_name:
        idx = 1
    elif "c" in channel_name:
        idx = 2
    elif "d" in channel_name:
        idx = 3
    else:
        try:
            num_part = "".join(filter(str.isdigit, channel_name))
            idx = int(num_part) - 1 if num_part else 0
        except ValueError:
            idx = 0
    xmin = idx * width_um
    xmax = (idx + 1) * width_um
    ymin = 0.0
    ymax = height_um
    return xmin, xmax, ymin, ymax


def _confine_to_channel(
    xs: np.ndarray,
    ys: np.ndarray,
    bounds: tuple[float, float, float, float],
    reflect: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    xs_out = xs.copy()
    ys_out = ys.copy()
    if reflect:
        for i in range(1, len(xs_out)):
            if xs_out[i] < xmin:
                xs_out[i] = xmin + (xmin - xs_out[i])
            elif xs_out[i] > xmax:
                xs_out[i] = xmax - (xs_out[i] - xmax)
            if ys_out[i] < ymin:
                ys_out[i] = ymin + (ymin - ys_out[i])
            elif ys_out[i] > ymax:
                ys_out[i] = ymax - (ys_out[i] - ymax)
    return xs_out, ys_out


def _generate_metadata(
    batch_name: str,
    **kwargs,
) -> dict:
    metadata = {
        "batch_id": kwargs.get("batch_id", batch_name),
        "description": kwargs.get("description", "Synthetic nanoparticle tracking experiment"),
        "pixel_size_um": kwargs.get("pixel_size_um", 0.107),
        "frame_rate_hz": kwargs.get("frame_rate_hz", 30.0),
        "temperature_c": kwargs.get("temperature_c", 25.0),
        "viscosity_pa_s": kwargs.get("viscosity_pa_s", 0.00089),
        "nominal_diameter_nm": kwargs.get("nominal_diameter_nm", 100.0),
        "channel_width_um": kwargs.get("channel_width_um", 200.0),
        "channel_height_um": kwargs.get("channel_height_um", 50.0),
        "channels": kwargs.get("channels", ["channel_A", "channel_B", "channel_C"]),
        "notes": kwargs.get("notes", ""),
        "operator": kwargs.get("operator", "synthetic_generator"),
        "timestamp": kwargs.get("timestamp", datetime.now().isoformat()),
    }
    for key, value in kwargs.items():
        if key not in metadata:
            metadata[key] = value
    return metadata


def _maybe_cross_channel(
    xs: np.ndarray,
    ys: np.ndarray,
    bounds: tuple[float, float, float, float],
    channel_idx: int,
    width_um: float,
    cross_prob: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() > cross_prob:
        return xs, ys, channel_idx
    n = len(xs)
    if n < 30:
        return xs, ys, channel_idx
    lo = n // 2
    hi = n - 10
    if lo >= hi:
        return xs, ys, channel_idx
    cross_start = rng.integers(lo, hi)
    xmin, xmax, ymin, ymax = bounds
    if channel_idx < 2:
        direction = 1
        new_channel_idx = channel_idx + 1
        target_xmin = (channel_idx + 1) * width_um
        target_xmax = (channel_idx + 2) * width_um
    elif channel_idx > 0:
        direction = -1
        new_channel_idx = channel_idx - 1
        target_xmin = (channel_idx - 1) * width_um
        target_xmax = channel_idx * width_um
    else:
        return xs, ys, channel_idx
    xs[cross_start:] = xs[cross_start:] + direction * (width_um * 0.6 + rng.uniform(0, width_um * 0.3))
    for i in range(cross_start, n):
        if direction > 0:
            if xs[i] > target_xmax:
                xs[i] = target_xmax - (xs[i] - target_xmax)
        else:
            if xs[i] < target_xmin:
                xs[i] = target_xmin + (target_xmin - xs[i])
        if ys[i] < ymin:
            ys[i] = ymin + (ymin - ys[i])
        elif ys[i] > ymax:
            ys[i] = ymax - (ys[i] - ymax)
    return xs, ys, new_channel_idx


def generate_sample_batch(
    output_dir: str | os.PathLike,
    batch_index: int = 0,
    num_particles_per_channel: int = 40,
    n_frames_range: tuple[int, int] = (100, 300),
    nominal_diameter_nm: float = 100,
    temperature_c: float = 25.0,
    viscosity_pa_s: float = 0.00089,
    frame_rate: float = 30.0,
    pixel_size_um: float = 0.107,
    channels: Optional[list[str]] = None,
    drift_um_per_s: tuple[float, float] = (0.3, 0.05),
    seed: Optional[int] = None,
    channel_width_um: float = 200.0,
    channel_height_um: float = 50.0,
    notes: str = "",
    description: str = "",
    operator: str = "synthetic_generator",
    wrong_temperature_channel: Optional[str] = None,
    wrong_temperature_value: Optional[float] = None,
    add_temp_column: bool = True,
) -> tuple[dict, list[str]]:
    rng = np.random.default_rng(seed)
    if channels is None:
        channels = ["channel_A", "channel_B", "channel_C"]
    timestamp = datetime.now().isoformat()
    batch_date_str = datetime.now().strftime("%Y%m%d")
    batch_id = f"batch_{batch_date_str}_{batch_index:03d}"
    batch_dir = Path(output_dir) / batch_id
    trajectories_dir = batch_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    dt_s = 1.0 / frame_rate
    D_um2s = compute_diffusion_coefficient_um2s(
        nominal_diameter_nm, temperature_c, viscosity_pa_s
    )
    D_um2s_wrong = None
    if wrong_temperature_value is not None:
        D_um2s_wrong = compute_diffusion_coefficient_um2s(
            nominal_diameter_nm, wrong_temperature_value, viscosity_pa_s
        )
    full_notes = notes
    if wrong_temperature_channel is not None and wrong_temperature_value is not None:
        if full_notes:
            full_notes += "; "
        full_notes += (
            f"Note: {wrong_temperature_channel} may have been recorded at "
            f"{wrong_temperature_value}°C instead of {temperature_c}°C due to "
            "localized heating; verify before analysis."
        )
    metadata = _generate_metadata(
        batch_name=batch_id,
        batch_id=batch_id,
        description=description or f"Synthetic batch {batch_index}: {nominal_diameter_nm}nm @ {temperature_c}°C",
        pixel_size_um=pixel_size_um,
        frame_rate_hz=frame_rate,
        temperature_c=temperature_c,
        viscosity_pa_s=viscosity_pa_s,
        nominal_diameter_nm=nominal_diameter_nm,
        channel_width_um=channel_width_um,
        channel_height_um=channel_height_um,
        channels=channels,
        notes=full_notes,
        operator=operator,
        timestamp=timestamp,
    )
    metadata_path = batch_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    csv_file_list: list[str] = []
    global_particle_counter = 0
    used_particle_ids_per_channel: dict[str, set[int]] = {}
    duplicate_id_pool: set[int] = set()
    for ch_idx, channel_id in enumerate(channels):
        used_particle_ids_per_channel[channel_id] = set()
    for ch_idx, channel_id in enumerate(channels):
        bounds = _generate_channel_bounds(channel_id, channel_width_um, channel_height_um)
        xmin, xmax, ymin, ymax = bounds
        is_wrong_temp_channel = (wrong_temperature_channel is not None) and (
            channel_id == wrong_temperature_channel
            or (wrong_temperature_channel == "random" and ch_idx == len(channels) - 1)
        )
        D_local = D_um2s_wrong if (is_wrong_temp_channel and D_um2s_wrong is not None) else D_um2s
        frames_all: list[int] = []
        times_all: list[float] = []
        particle_ids_all: list[int] = []
        xs_all: list[float] = []
        ys_all: list[float] = []
        intensities_all: list[float] = []
        channel_ids_all: list[str] = []
        temp_all: list[float] = []
        for p in range(num_particles_per_channel):
            n_frames = int(rng.integers(n_frames_range[0], n_frames_range[1] + 1))
            is_short = rng.random() < 0.12
            if is_short:
                n_frames = int(rng.integers(5, 20))
            if duplicate_id_pool and rng.random() < 0.08 and len(duplicate_id_pool) > 0:
                pid = int(sorted(duplicate_id_pool)[rng.integers(0, len(duplicate_id_pool))])
            else:
                global_particle_counter += 1
                pid = global_particle_counter
                if rng.random() < 0.1:
                    duplicate_id_pool.add(pid)
            used_particle_ids_per_channel[channel_id].add(pid)
            start_x = rng.uniform(xmin + channel_width_um * 0.1, xmax - channel_width_um * 0.1)
            start_y = rng.uniform(ymin + channel_height_um * 0.1, ymax - channel_height_um * 0.1)
            xs, ys = _brownian_trajectory(D_local, n_frames, dt_s, (start_x, start_y), rng)
            apply_drift = rng.random() < 0.7
            if apply_drift:
                vx_var = drift_um_per_s[0] * rng.uniform(0.5, 1.5)
                vy_var = drift_um_per_s[1] * rng.uniform(0.5, 1.5)
                xs, ys = _add_drift(xs, ys, vx_var, vy_var, dt_s)
            xs, ys = _confine_to_channel(xs, ys, bounds, reflect=True)
            xs, ys = _add_jumps(xs, ys, D_local, dt_s, jump_prob=0.05, jump_sigma_factor=10.0, rng=rng)
            xs, ys = _add_noise(xs, ys, localization_error_um=0.02, rng=rng)
            xs, ys, final_channel_idx = _maybe_cross_channel(
                xs, ys, bounds, ch_idx, channel_width_um, cross_prob=0.05, rng=rng
            )
            actual_channel_id = channels[final_channel_idx] if 0 <= final_channel_idx < len(channels) else channel_id
            base_intensity = rng.uniform(800.0, 1500.0)
            is_aggregated = rng.random() < 0.03
            if is_aggregated:
                jitter = 1e-4
                xs = xs + jitter
                ys = ys + jitter
                base_intensity *= rng.uniform(1.8, 2.5)
            intensities = _simulate_intensity(base_intensity, n_frames, photobleach=True, rng=rng)
            frame_ids = np.arange(n_frames, dtype=int)
            times = frame_ids.astype(float) * dt_s
            jitter_prob = 0.15
            if rng.random() < jitter_prob:
                jitter_amount = dt_s * rng.uniform(-0.2, 0.2, size=n_frames)
                jitter_amount[0] = 0.0
                times = times + jitter_amount
                sort_idx = np.argsort(times)
                times = times[sort_idx]
                frame_ids = frame_ids[sort_idx]
                xs = xs[sort_idx]
                ys = ys[sort_idx]
                intensities = intensities[sort_idx]
            drop_frame_prob = 0.1
            if rng.random() < drop_frame_prob and n_frames > 20:
                num_drop = rng.integers(1, max(2, int(n_frames * 0.05) + 1))
                drop_candidates = np.arange(5, n_frames - 5)
                if len(drop_candidates) >= num_drop:
                    drop_idx = rng.choice(drop_candidates, size=num_drop, replace=False)
                    keep = np.ones(n_frames, dtype=bool)
                    keep[drop_idx] = False
                    xs = xs[keep]
                    ys = ys[keep]
                    times = times[keep]
                    intensities = intensities[keep]
                    frame_ids = frame_ids[keep]
                    n_frames = len(xs)
            xs, ys, times, intensities, frame_ids = _break_trajectory(
                xs, ys, times, intensities, frame_ids,
                break_prob=0.15, min_missing=3, max_missing=10, rng=rng
            )
            recorded_temp = temperature_c
            if is_wrong_temp_channel and wrong_temperature_value is not None:
                if rng.random() < 0.15:
                    recorded_temp = wrong_temperature_value
            temps_particle = np.full(len(xs), recorded_temp, dtype=float)
            frames_all.extend(frame_ids.tolist())
            times_all.extend(times.tolist())
            particle_ids_all.extend([pid] * len(xs))
            xs_all.extend(xs.tolist())
            ys_all.extend(ys.tolist())
            intensities_all.extend(intensities.tolist())
            channel_ids_all.extend([actual_channel_id] * len(xs))
            temp_all.extend(temps_particle.tolist())
        df = pd.DataFrame({
            "frame": frames_all,
            "time_s": times_all,
            "particle_id": particle_ids_all,
            "x_um": xs_all,
            "y_um": ys_all,
            "intensity": intensities_all,
            "channel_id": channel_ids_all,
        })
        if add_temp_column:
            df["temperature"] = temp_all
        csv_count = 0
        for p_idx, pid in enumerate(sorted(used_particle_ids_per_channel[channel_id])):
            p_df = df[df["particle_id"] == pid].copy()
            if len(p_df) == 0:
                continue
            p_df = p_df.sort_values("frame").reset_index(drop=True)
            csv_filename = f"{channel_id}_{p_idx + 1:03d}.csv"
            csv_path = trajectories_dir / csv_filename
            cols = ["frame", "time_s", "particle_id", "x_um", "y_um", "intensity", "channel_id"]
            if add_temp_column:
                cols.append("temperature")
            p_df[cols].to_csv(csv_path, index=False)
            csv_file_list.append(str(csv_path))
            csv_count += 1
    return metadata, csv_file_list


def init_sample_data(output_dir: str | os.PathLike, num_batches: int = 4) -> list[tuple[dict, list[str]]]:
    results = []
    base_seed = 42
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    batch_configs = [
        {
            "batch_index": 0,
            "nominal_diameter_nm": 100,
            "temperature_c": 25.0,
            "viscosity_pa_s": 0.00089,
            "drift_um_per_s": (0.3, 0.05),
            "notes": "Control batch: 100nm polystyrene beads in water at 25°C",
            "description": "Control batch - 100nm diameter, 25°C, water",
            "wrong_temperature_channel": None,
            "wrong_temperature_value": None,
            "seed_offset": 0,
        },
        {
            "batch_index": 1,
            "nominal_diameter_nm": 50,
            "temperature_c": 37.0,
            "viscosity_pa_s": 0.0018,
            "drift_um_per_s": (0.25, 0.04),
            "notes": "50nm gold nanoparticles in 30% glycerol solution at 37°C; higher viscosity expected to reduce diffusion",
            "description": "High viscosity batch - 50nm diameter, 37°C, 30% glycerol",
            "wrong_temperature_channel": None,
            "wrong_temperature_value": None,
            "seed_offset": 100,
        },
        {
            "batch_index": 2,
            "nominal_diameter_nm": 200,
            "temperature_c": 25.0,
            "viscosity_pa_s": 0.00089,
            "drift_um_per_s": (1.2, 0.25),
            "notes": "200nm microspheres with strong electroosmotic flow drift; check drift correction",
            "description": "High drift batch - 200nm diameter, 25°C, strong flow",
            "wrong_temperature_channel": None,
            "wrong_temperature_value": None,
            "seed_offset": 200,
        },
        {
            "batch_index": 3,
            "nominal_diameter_nm": 100,
            "temperature_c": 25.0,
            "viscosity_pa_s": 0.00089,
            "drift_um_per_s": (0.3, 0.05),
            "notes": "Labeled 25°C but channel_C suspected local heating",
            "description": "Temperature anomaly batch - 100nm, labeled 25°C, channel_C ~30°C",
            "wrong_temperature_channel": "channel_C",
            "wrong_temperature_value": 30.0,
            "seed_offset": 300,
        },
    ]
    for cfg in batch_configs[:num_batches]:
        seed = base_seed + cfg["seed_offset"]
        wrong_ch = cfg["wrong_temperature_channel"]
        wrong_temp = cfg["wrong_temperature_value"]
        metadata, csv_files = generate_sample_batch(
            output_dir=output_dir,
            batch_index=cfg["batch_index"],
            nominal_diameter_nm=cfg["nominal_diameter_nm"],
            temperature_c=cfg["temperature_c"],
            viscosity_pa_s=cfg["viscosity_pa_s"],
            drift_um_per_s=cfg["drift_um_per_s"],
            notes=cfg["notes"],
            description=cfg["description"],
            seed=seed,
            wrong_temperature_channel=wrong_ch,
            wrong_temperature_value=wrong_temp,
        )
        results.append((metadata, csv_files))
    return results


def generate_sample_experiment(
    output_dir: str | os.PathLike,
    batch_name: str,
    config: Optional[dict] = None,
) -> tuple[dict, list[str]]:
    if config is None:
        config = {}
    batch_index = config.get("batch_index", 0)
    metadata, csv_files = generate_sample_batch(
        output_dir=output_dir,
        batch_index=batch_index,
        num_particles_per_channel=config.get("num_particles_per_channel", 40),
        n_frames_range=config.get("n_frames_range", (100, 300)),
        nominal_diameter_nm=config.get("nominal_diameter_nm", 100),
        temperature_c=config.get("temperature_c", 25.0),
        viscosity_pa_s=config.get("viscosity_pa_s", 0.00089),
        frame_rate=config.get("frame_rate", 30.0),
        pixel_size_um=config.get("pixel_size_um", 0.107),
        channels=config.get("channels", None),
        drift_um_per_s=config.get("drift_um_per_s", (0.3, 0.05)),
        seed=config.get("seed", None),
        channel_width_um=config.get("channel_width_um", 200.0),
        channel_height_um=config.get("channel_height_um", 50.0),
        notes=config.get("notes", ""),
        description=config.get("description", f"{batch_name}"),
        operator=config.get("operator", "synthetic_generator"),
        wrong_temperature_channel=config.get("wrong_temperature_channel", None),
        wrong_temperature_value=config.get("wrong_temperature_value", None),
        add_temp_column=config.get("add_temp_column", True),
    )
    return metadata, csv_files
