from __future__ import annotations

from typing import Optional

from nano_dispersion.config import get_settings
from nano_dispersion.models.database import SessionLocal, init_db
from nano_dispersion.services.batch_service import BatchService
from nano_dispersion.services.task_manager import TaskManager

__all__ = [
    "TaskManager",
    "BatchService",
    "initialize_services",
    "get_batch_service",
    "get_task_manager",
]

_task_manager: Optional[TaskManager] = None
_batch_service: Optional[BatchService] = None


def initialize_services():
    """初始化所有服务（启动时调用）"""
    global _task_manager, _batch_service
    init_db()
    settings = get_settings()
    _task_manager = TaskManager(SessionLocal, settings.worker_threads)
    _batch_service = BatchService(SessionLocal, _task_manager, settings)
    _task_manager.start()
    return _batch_service, _task_manager


def get_batch_service() -> BatchService:
    """获取全局 BatchService 单例"""
    global _batch_service
    if _batch_service is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _batch_service


def get_task_manager() -> TaskManager:
    """获取全局 TaskManager 单例"""
    global _task_manager
    if _task_manager is None:
        raise RuntimeError("Services not initialized. Call initialize_services() first.")
    return _task_manager
