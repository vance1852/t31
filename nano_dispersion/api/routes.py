from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import get_settings
from ..models import schemas
from ..models.database import SessionLocal
from ..models.orm import AnalysisTask, ExperimentBatch
from ..services import BatchService, TaskManager


def get_batch_service_dep():
    from ..services import get_batch_service

    return get_batch_service()


def get_task_manager_dep():
    from ..services import get_task_manager

    return get_task_manager()


BatchServiceDep = Annotated[BatchService, Depends(get_batch_service_dep)]
TaskManagerDep = Annotated[TaskManager, Depends(get_task_manager_dep)]

api_router = APIRouter()


class ImportBatchRequest(BaseModel):
    directory_path: str
    description: Optional[str] = None


class ExportReportRequest(BaseModel):
    task_id: int
    export_dir: Optional[str] = None


class GenerateSamplesRequest(BaseModel):
    output_dir: Optional[str] = None
    num_batches: int = 4


class BatchListResponse(BaseModel):
    total: int
    items: list[schemas.BatchResponse]


class ExportReportResponse(BaseModel):
    json_path: str
    csv_paths: dict[str, str]
    markdown_path: Optional[str] = None
    export_dir: str
    generated_at: str


class GenerateSamplesResponse(BaseModel):
    batches: list[dict[str, str]]


class TrajectoryExplainResponse(BaseModel):
    trajectory_id: int
    particle_id: Optional[int]
    channel_id: Optional[int]
    num_frames: int
    duration_s: Optional[float]
    qc_passed: bool
    excluded: bool
    exclude_reason: Optional[str]
    flags: list[str]
    anomalies: list[dict[str, Any]]
    model_type: Optional[str]
    model_reason: Optional[str]
    fit_quality: Optional[dict[str, Any]]


def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@api_router.post("/batches/import", response_model=schemas.BatchResponse, status_code=status.HTTP_201_CREATED)
def import_batch(
    req: ImportBatchRequest,
    batch_service: BatchServiceDep,
):
    dir_path = Path(req.directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"目录不存在或不是有效目录: {req.directory_path}",
        )
    try:
        batch = batch_service.import_batch_from_directory(
            dir_path=dir_path,
            description=req.description or "",
        )
        return schemas.BatchResponse.model_validate(batch)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导入批次失败: {str(e)}",
        )


@api_router.post("/batches/upload", response_model=schemas.BatchResponse, status_code=status.HTTP_201_CREATED)
async def upload_batch(
    batch_service: BatchServiceDep,
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
):
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="只支持ZIP文件上传",
        )
    settings = get_settings()
    upload_dir = settings.data_dir / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix="batch_upload_", dir=str(upload_dir)))
    try:
        zip_path = temp_dir / file.filename
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(extract_dir))

        batch_root = extract_dir
        if (batch_root / "metadata.json").exists():
            pass
        else:
            subdirs = [d for d in batch_root.iterdir() if d.is_dir()]
            if len(subdirs) == 1:
                batch_root = subdirs[0]

        if not (batch_root / "metadata.json").exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ZIP文件中未找到metadata.json，请检查批次结构",
            )

        batch = batch_service.import_batch_from_directory(
            dir_path=batch_root,
            description=description or "",
        )
        return schemas.BatchResponse.model_validate(batch)
    except HTTPException:
        raise
    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ZIP文件损坏或格式不正确",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理上传批次失败: {str(e)}",
        )
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


@api_router.get("/batches/", response_model=BatchListResponse)
def list_batches(
    batch_service: BatchServiceDep,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db=Depends(_get_db),
):
    try:
        total = db.query(ExperimentBatch).count()
        items = batch_service.list_batches(skip=skip, limit=limit)
        return BatchListResponse(
            total=total,
            items=[schemas.BatchResponse.model_validate(b) for b in items],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取批次列表失败: {str(e)}",
        )


@api_router.get("/batches/{batch_id}", response_model=schemas.BatchResponse)
def get_batch(
    batch_id: int,
    batch_service: BatchServiceDep,
):
    batch = batch_service.get_batch(batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"批次 {batch_id} 不存在",
        )
    return schemas.BatchResponse.model_validate(batch)


@api_router.delete("/batches/{batch_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_batch(
    batch_id: int,
    batch_service: BatchServiceDep,
    db=Depends(_get_db),
):
    batch = db.get(ExperimentBatch, batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"批次 {batch_id} 不存在",
        )
    try:
        db.delete(batch)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除批次失败: {str(e)}",
        )
    return None


@api_router.post("/batches/{batch_id}/analyze", response_model=schemas.TaskResponse)
def analyze_batch(
    batch_id: int,
    batch_service: BatchServiceDep,
    task_manager: TaskManagerDep,
    db=Depends(_get_db),
):
    batch = db.get(ExperimentBatch, batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"批次 {batch_id} 不存在",
        )
    try:
        running_task = db.query(AnalysisTask).filter(
            AnalysisTask.batch_id == batch_id,
            AnalysisTask.status.in_(["pending", "running"]),
        ).first()
        if running_task:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"批次 {batch_id} 已有任务正在执行中 (task_id={running_task.id})",
            )

        task_id = batch_service.submit_analysis(batch_id)
        task = task_manager.get_task_status(task_id)
        if not task:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="任务提交后无法获取状态",
            )
        return schemas.TaskResponse.model_validate(task)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"提交分析任务失败: {str(e)}",
        )


@api_router.get("/tasks/{task_id}", response_model=schemas.TaskResponse)
def get_task(
    task_id: int,
    task_manager: TaskManagerDep,
):
    task = task_manager.get_task_status(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务 {task_id} 不存在",
        )
    return schemas.TaskResponse.model_validate(task)


@api_router.get("/batches/{batch_id}/tasks", response_model=list[schemas.TaskResponse])
def list_batch_tasks(
    batch_id: int,
    batch_service: BatchServiceDep,
    db=Depends(_get_db),
):
    batch = db.get(ExperimentBatch, batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"批次 {batch_id} 不存在",
        )
    tasks = db.query(AnalysisTask).filter(
        AnalysisTask.batch_id == batch_id,
    ).order_by(AnalysisTask.created_at.desc()).all()
    return [schemas.TaskResponse.model_validate(t) for t in tasks]


@api_router.get("/batches/{batch_id}/qc")
def get_qc_summary(
    batch_id: int,
    batch_service: BatchServiceDep,
    task_id: Optional[int] = Query(None),
    db=Depends(_get_db),
):
    batch = db.get(ExperimentBatch, batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"批次 {batch_id} 不存在",
        )
    try:
        result = batch_service.get_qc_summary(batch_id, task_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取质控摘要失败: {str(e)}",
        )


@api_router.get("/batches/{batch_id}/trajectories")
def get_trajectories(
    batch_id: int,
    batch_service: BatchServiceDep,
    task_id: Optional[int] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    channel_id: Optional[int] = Query(None),
    only_valid: bool = Query(False),
    db=Depends(_get_db),
):
    batch = db.get(ExperimentBatch, batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"批次 {batch_id} 不存在",
        )
    try:
        results = batch_service.get_trajectory_results(
            batch_id=batch_id,
            task_id=task_id,
            skip=skip,
            limit=limit,
            channel_id=channel_id,
            only_valid=only_valid,
        )
        return {"total": len(results), "items": results}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取轨迹结果失败: {str(e)}",
        )


@api_router.get("/batches/{batch_id}/summary")
def get_batch_summary(
    batch_id: int,
    batch_service: BatchServiceDep,
    task_id: Optional[int] = Query(None),
    db=Depends(_get_db),
):
    batch = db.get(ExperimentBatch, batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"批次 {batch_id} 不存在",
        )
    try:
        result = batch_service.get_batch_summary(batch_id, task_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取批次汇总失败: {str(e)}",
        )


@api_router.get("/batches/{batch_id}/calibration")
def get_calibration(
    batch_id: int,
    batch_service: BatchServiceDep,
    task_id: Optional[int] = Query(None),
    db=Depends(_get_db),
):
    batch = db.get(ExperimentBatch, batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"批次 {batch_id} 不存在",
        )
    try:
        result = batch_service.get_calibration(batch_id, task_id)
        if result is None:
            return {"batch_id": batch_id, "task_id": task_id, "calibration_points": [], "message": "暂无校准数据"}
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取校准结果失败: {str(e)}",
        )


@api_router.get("/batches/{batch_id}/anomalies")
def get_anomalies(
    batch_id: int,
    batch_service: BatchServiceDep,
    task_id: Optional[int] = Query(None),
    severity: Optional[str] = Query(None, pattern="^(info|warning|error|blocking)$"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db=Depends(_get_db),
):
    batch = db.get(ExperimentBatch, batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"批次 {batch_id} 不存在",
        )
    try:
        results = batch_service.get_anomalies(
            batch_id=batch_id,
            task_id=task_id,
            severity=severity,
            skip=skip,
            limit=limit,
        )
        return {"total": len(results), "items": results}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取异常清单失败: {str(e)}",
        )


@api_router.get("/trajectories/{trajectory_id}/explain", response_model=TrajectoryExplainResponse)
def explain_trajectory(
    trajectory_id: int,
    batch_service: BatchServiceDep,
    task_id: Optional[int] = Query(None),
):
    try:
        result = batch_service.explain_trajectory(trajectory_id, task_id)
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"],
            )
        return TrajectoryExplainResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"解释轨迹失败: {str(e)}",
        )


@api_router.post("/batches/{batch_id}/export", response_model=ExportReportResponse)
def export_report(
    batch_id: int,
    req: ExportReportRequest,
    batch_service: BatchServiceDep,
    db=Depends(_get_db),
):
    batch = db.get(ExperimentBatch, batch_id)
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"批次 {batch_id} 不存在",
        )
    task = db.get(AnalysisTask, req.task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务 {req.task_id} 不存在",
        )
    if task.batch_id != batch_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"任务 {req.task_id} 不属于批次 {batch_id}",
        )
    if task.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"任务 {req.task_id} 状态为 '{task.status}'，只有已完成的任务才能导出报告",
        )
    try:
        result = batch_service.export_reports(
            batch_id=batch_id,
            task_id=req.task_id,
            export_dir=req.export_dir,
        )
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"],
            )
        return ExportReportResponse(
            json_path=result["json_report"],
            csv_paths=result["csv_reports"],
            markdown_path=result.get("markdown_report"),
            export_dir=result["export_dir"],
            generated_at=result["generated_at"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"导出报告失败: {str(e)}",
        )


@api_router.get("/reports/{filename}")
def download_report(
    filename: str,
):
    settings = get_settings()
    report_dir = settings.result_dir

    safe_filename = Path(filename).name
    file_path = report_dir / safe_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"报告文件不存在: {safe_filename}",
        )

    if str(file_path.resolve()).startswith(str(report_dir.resolve())):
        return FileResponse(
            path=str(file_path),
            filename=safe_filename,
            media_type="application/octet-stream",
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法文件名",
        )


@api_router.post("/samples/generate", response_model=GenerateSamplesResponse)
def generate_samples(
    req: GenerateSamplesRequest,
):
    try:
        from ..data_generator.sample_generator import init_sample_data

        settings = get_settings()
        output_dir = Path(req.output_dir) if req.output_dir else settings.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if req.num_batches < 1 or req.num_batches > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="num_batches 必须在 1 到 10 之间",
            )

        results = init_sample_data(output_dir=str(output_dir), num_batches=req.num_batches)

        batches_info: list[dict[str, str]] = []
        for metadata, _ in results:
            batch_name = metadata.get("batch_id") or metadata.get("batch_name") or "unknown"
            dir_path = str(output_dir / batch_name)
            batches_info.append({
                "name": batch_name,
                "directory_path": dir_path,
            })

        return GenerateSamplesResponse(batches=batches_info)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"生成样例数据失败: {str(e)}",
        )
