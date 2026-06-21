from __future__ import annotations

import json
import queue
import threading
import traceback
from collections.abc import Callable
from datetime import datetime
from threading import Thread

from sqlalchemy.orm import Session

from nano_dispersion.models.database import get_db
from nano_dispersion.models.orm import AnalysisTask


class TaskManager:
    """使用threading实现的简单后台任务队列+状态管理"""

    def __init__(self, db_factory, worker_threads: int = 2):
        self._queue: queue.Queue = queue.Queue()
        self._workers: list[Thread] = []
        self._db_factory = db_factory
        self._worker_threads = worker_threads
        self._active_tasks: dict[int, Thread] = {}
        self._lock = threading.Lock()
        self._shutdown = threading.Event()

    def start(self) -> None:
        """启动worker线程池"""
        for _ in range(self._worker_threads):
            t = Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

    def submit_task(
        self,
        batch_id: int,
        task_type: str,
        task_function: Callable,
        **kwargs,
    ) -> int:
        """提交任务，创建AnalysisTask记录并入队，返回task_id"""
        db: Session = self._db_factory()
        try:
            task = AnalysisTask(
                batch_id=batch_id,
                task_type=task_type,
                status="pending",
                progress_pct=0.0,
                message="任务已提交，等待执行...",
            )
            db.add(task)
            db.commit()
            db.refresh(task)
            task_id = task.id
        finally:
            db.close()

        self._queue.put((task_id, task_function, kwargs))
        return task_id

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                task_id, func, kwargs = self._queue.get(timeout=1)
                self._execute_task(task_id, func, kwargs)
            except queue.Empty:
                continue

    def _execute_task(self, task_id: int, func: Callable, kwargs: dict) -> None:
        """执行任务，更新status/running->completed/failed，progress实时更新"""

        def progress_cb(pct: float, msg: str = "") -> None:
            db: Session = self._db_factory()
            try:
                task = db.get(AnalysisTask, task_id)
                if task:
                    task.progress_pct = float(pct)
                    task.message = msg
                    db.commit()
            finally:
                db.close()

        db: Session = self._db_factory()
        try:
            task = db.get(AnalysisTask, task_id)
            if not task:
                return
            task.status = "running"
            task.started_at = datetime.utcnow()
            task.progress_pct = 1.0
            task.message = "任务开始执行..."
            db.commit()
            with self._lock:
                self._active_tasks[task_id] = threading.current_thread()
        finally:
            db.close()

        try:
            result = func(progress_callback=progress_cb, **kwargs)

            db: Session = self._db_factory()
            try:
                task = db.get(AnalysisTask, task_id)
                if task:
                    task.status = "completed"
                    task.completed_at = datetime.utcnow()
                    task.progress_pct = 100.0
                    task.message = "任务执行完成"
                    try:
                        task.result_json = result if isinstance(result, dict) else {"result": result}
                    except Exception:
                        task.result_json = {"success": True}
                    db.commit()
            finally:
                db.close()

        except Exception as e:
            error_tb = traceback.format_exc()
            db: Session = self._db_factory()
            try:
                task = db.get(AnalysisTask, task_id)
                if task:
                    task.status = "failed"
                    task.error_message = f"{str(e)}\n{error_tb}"
                    task.message = f"任务执行失败: {str(e)}"
                    db.commit()
            finally:
                db.close()
        finally:
            with self._lock:
                self._active_tasks.pop(task_id, None)

    def get_task_status(self, task_id: int) -> AnalysisTask | None:
        """从数据库查询任务状态"""
        db: Session = self._db_factory()
        try:
            return db.get(AnalysisTask, task_id)
        finally:
            db.close()

    def shutdown(self, wait: bool = True) -> None:
        self._shutdown.set()
        if wait:
            for t in self._workers:
                if t.is_alive():
                    t.join(timeout=5)
        self._workers.clear()
