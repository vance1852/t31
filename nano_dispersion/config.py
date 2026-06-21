"""全局配置模块 - 使用pydantic-settings管理环境变量配置."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置."""

    model_config = SettingsConfigDict(
        env_prefix="NANO_DISPERSION_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 6817

    base_dir: Path = Path(__file__).resolve().parent.parent
    db_path: Path = base_dir / "data" / "nano_dispersion.db"
    data_dir: Path = base_dir / "data" / "batches"
    result_dir: Path = base_dir / "results"

    generate_samples: bool = False

    min_trajectory_frames: int = 20
    max_gap_frames_for_reconnect: int = 5
    max_neighbor_distance_um: float = 3.0
    max_velocity_jump_um_per_s: float = 50.0
    drift_outlier_sigma: float = 3.0

    msd_fit_min_lags: int = 5
    msd_fit_max_lags_ratio: float = 0.4
    confidence_level: float = 0.95

    boltzmann_constant: float = 1.380649e-23
    default_water_viscosity_pa_s: float = 0.001

    worker_threads: int = 2

    @property
    def database_url(self) -> str:
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path.as_posix()}"

    def ensure_dirs(self) -> None:
        for d in [self.data_dir, self.result_dir, Path(self.db_path).parent]:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取单例配置对象."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
