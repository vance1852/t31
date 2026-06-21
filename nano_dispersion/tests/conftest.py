from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generator

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from nano_dispersion.config import Settings
from nano_dispersion.models.database import Base


@pytest.fixture
def tmp_path(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("nano_dispersion_test")


@pytest.fixture
def tmp_data_dir(tmp_path) -> Path:
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tmp_result_dir(tmp_path) -> Path:
    d = tmp_path / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def db_session(tmp_path) -> Generator[Session, None, None]:
    db_file = tmp_path / "test_nano_dispersion.db"
    engine = create_engine(
        f"sqlite:///{db_file.as_posix()}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def db_factory(db_session) -> Callable[[], Session]:
    def _factory() -> Session:
        return db_session
    return _factory


@pytest.fixture
def settings_override(tmp_path, tmp_data_dir, tmp_result_dir) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=6817,
        base_dir=tmp_path,
        db_path=tmp_path / "data" / "override.db",
        data_dir=tmp_data_dir,
        result_dir=tmp_result_dir,
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


def _generate_brownian_trajectory(
    D: float,
    n_frames: int,
    dt: float,
    start_x: float = 0.0,
    start_y: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()
    sigma = np.sqrt(2.0 * D * dt)
    dx = rng.normal(0.0, sigma, size=n_frames)
    dy = rng.normal(0.0, sigma, size=n_frames)
    xs = np.empty(n_frames)
    ys = np.empty(n_frames)
    xs[0] = start_x
    ys[0] = start_y
    xs[1:] = start_x + np.cumsum(dx[1:])
    ys[1:] = start_y + np.cumsum(dy[1:])
    return xs, ys


@pytest.fixture
def mock_batch_data() -> dict[str, Any]:
    rng = np.random.default_rng(42)
    n_particles = 5
    n_frames = 60
    dt = 0.033
    D_true = 2.0

    rows = []
    for pid in range(n_particles):
        start_x = rng.uniform(50, 150)
        start_y = rng.uniform(10, 40)
        xs, ys = _generate_brownian_trajectory(D_true, n_frames, dt, start_x, start_y, rng)
        for frame in range(n_frames):
            rows.append({
                "particle_id": pid,
                "frame": frame,
                "time_s": frame * dt,
                "x_um": float(xs[frame]),
                "y_um": float(ys[frame]),
                "intensity": rng.uniform(800, 1500),
            })

    df = pd.DataFrame(rows)
    trajectories: dict[int, pd.DataFrame] = {}
    for pid in range(n_particles):
        trajectories[pid] = df[df["particle_id"] == pid].copy().reset_index(drop=True)

    return {
        "dataframe": df,
        "trajectories": trajectories,
        "D_true": D_true,
        "dt": dt,
        "n_particles": n_particles,
        "n_frames": n_frames,
    }


@pytest.fixture
def sample_trajectory_factory() -> Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    def _factory(
        D: float = 2.0,
        n_frames: int = 100,
        dt: float = 0.033,
        start_x: float = 0.0,
        start_y: float = 0.0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        rng = np.random.default_rng(seed)
        xs, ys = _generate_brownian_trajectory(D, n_frames, dt, start_x, start_y, rng)
        times = np.arange(n_frames, dtype=float) * dt
        return xs, ys, times, n_frames
    return _factory
