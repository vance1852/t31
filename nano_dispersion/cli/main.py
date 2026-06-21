from __future__ import annotations

import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

if sys.platform.startswith("win") and not any("pytest" in argv.lower() for argv in sys.argv):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

console = Console()

_severity_colors = {
    "info": "blue",
    "warning": "yellow",
    "error": "red",
    "blocking": "bold red",
}

_status_colors = {
    "pending": "yellow",
    "imported": "cyan",
    "running": "blue",
    "analyzed": "green",
    "completed": "green",
    "failed": "red",
}


def _format_float(v, digits: int = 4) -> str:
    if v is None:
        return "N/A"
    try:
        f = float(v)
        import math
        if math.isnan(f):
            return "N/A"
        return f"{f:.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _format_datetime(v) -> str:
    if v is None:
        return "N/A"
    try:
        if isinstance(v, datetime):
            return v.strftime("%Y-%m-%d %H:%M:%S")
        return str(v)
    except Exception:
        return str(v)


def _ensure_services():
    from nano_dispersion import services

    try:
        services.get_batch_service()
    except RuntimeError:
        services.initialize_services()
    return services.get_batch_service(), services.get_task_manager()


def _channel_id_to_name(ch_id) -> str:
    if ch_id is None:
        return "N/A"
    try:
        ch = int(ch_id)
        names = {0: "A", 1: "B", 2: "C", 3: "D"}
        return f"channel_{names.get(ch, str(ch))}"
    except (TypeError, ValueError):
        return str(ch_id)


@click.group()
@click.version_option(version="0.1.0", prog_name="nano-dispersion")
def cli():
    """微流控芯片纳米颗粒扩散分析 CLI"""
    pass


@cli.command("init-samples")
@click.option("--output-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
              help="输出目录 (默认: config.data_dir)")
@click.option("--num-batches", type=click.IntRange(1, 8), default=4, show_default=True, help="批次数量 (1-8)")
@click.option("--seed", type=int, default=None, help="随机种子")
@click.option("--yes/--no", default=False, help="覆盖已存在数据前确认")
def init_samples(output_dir, num_batches, seed, yes):
    """初始化样例数据"""
    try:
        from nano_dispersion.config import get_settings
        from nano_dispersion.data_generator import init_sample_data

        settings = get_settings()
        out_dir = output_dir if output_dir is not None else settings.data_dir

        out_path = Path(out_dir)
        if out_path.exists() and any(out_path.iterdir()):
            if not yes:
                if not click.confirm(f"目录 {out_path} 已存在且非空，是否继续？", default=False):
                    console.print("[yellow]操作已取消[/yellow]")
                    return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("生成样例数据...", total=num_batches)
            if seed is not None:
                import numpy as np
                rng = np.random.default_rng(seed)

            results = init_sample_data(out_path, num_batches=num_batches)

            for i, (metadata, _) in enumerate(results):
                progress.update(task, advance=1, description=f"已生成批次 {i + 1}/{num_batches}")

        table = Table(title="已生成批次列表", show_header=True, header_style="bold magenta")
        table.add_column("序号", justify="right", style="cyan", no_wrap=True)
        table.add_column("批次名称", style="green")
        table.add_column("描述")
        table.add_column("标称粒径(nm)", justify="right")
        table.add_column("温度(°C)", justify="right")

        for i, (metadata, csv_files) in enumerate(results, 1):
            table.add_row(
                str(i),
                metadata.get("batch_id", "N/A"),
                metadata.get("description", "N/A"),
                _format_float(metadata.get("nominal_diameter_nm"), 1),
                _format_float(metadata.get("temperature_c"), 1),
            )

        console.print(table)
        console.print(f"\n[green]✓ 共生成 {len(results)} 个批次，输出目录: {out_path}[/green]")
    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command("import-batch")
@click.argument("dir_path", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option("--description", type=str, default="", help="批次描述")
@click.option("--name", type=str, default=None, help="批次名称 (默认从目录名推断)")
def import_batch(dir_path, description, name):
    """导入实验批次"""
    try:
        batch_service, _ = _ensure_services()

        dir_path = Path(dir_path)
        metadata_path = dir_path / "metadata.json"
        trajectories_dir = dir_path / "trajectories"

        if not trajectories_dir.exists():
            console.print(f"[bold red]错误:[/bold red] 找不到 trajectories/ 目录: {trajectories_dir}")
            raise click.Abort()

        csv_files = sorted(trajectories_dir.glob("*.csv"))
        if not csv_files:
            console.print(f"[bold yellow]警告:[/bold yellow] trajectories/ 目录下没有CSV文件")

        with console.status("[bold green]正在导入批次...", spinner="dots"):
            batch = batch_service.import_batch_from_directory(dir_path, description=description)
            if name is not None:
                from nano_dispersion.models.database import SessionLocal
                db = SessionLocal()
                try:
                    from nano_dispersion.models.orm import ExperimentBatch
                    b = db.get(ExperimentBatch, batch.id)
                    if b:
                        b.name = name
                        db.commit()
                        db.refresh(b)
                        batch = b
                finally:
                    db.close()

            from nano_dispersion.models.database import SessionLocal
            from nano_dispersion.models.orm import ParticleTrajectory, TrajectoryPoint
            db = SessionLocal()
            try:
                from sqlalchemy import func
                traj_count = db.query(func.count(ParticleTrajectory.id)).filter(
                    ParticleTrajectory.batch_id == batch.id
                ).scalar() or 0

                point_count = db.query(func.count(TrajectoryPoint.id)).join(
                    ParticleTrajectory, TrajectoryPoint.trajectory_id == ParticleTrajectory.id
                ).filter(ParticleTrajectory.batch_id == batch.id).scalar() or 0

                channels = db.query(ParticleTrajectory.channel_id).filter(
                    ParticleTrajectory.batch_id == batch.id
                ).distinct().all()
                channel_count = len([c for (c,) in channels if c is not None])
            finally:
                db.close()

        info_table = Table(title="导入完成", show_header=False, show_lines=False)
        info_table.add_column("属性", style="cyan", no_wrap=True)
        info_table.add_column("值", style="green")
        info_table.add_row("批次ID", str(batch.id))
        info_table.add_row("批次名称", batch.name)
        info_table.add_row("状态", f"[{_status_colors.get(batch.status, 'white')}]{batch.status}[/]")
        info_table.add_row("轨迹数", str(traj_count))
        info_table.add_row("数据点数", str(point_count))
        info_table.add_row("通道数", str(channel_count))
        if batch.temperature_c is not None:
            info_table.add_row("温度(°C)", _format_float(batch.temperature_c, 1))
        if batch.nominal_diameter_nm is not None:
            info_table.add_row("标称粒径(nm)", _format_float(batch.nominal_diameter_nm, 1))

        console.print(info_table)
    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command("list-batches")
@click.option("--limit", type=int, default=50, show_default=True)
@click.option("--offset", type=int, default=0, show_default=True)
def list_batches(limit, offset):
    """列出所有批次"""
    try:
        batch_service, _ = _ensure_services()

        batches = batch_service.list_batches(skip=offset, limit=limit)

        if not batches:
            console.print("[yellow]暂无批次数据[/yellow]")
            return

        from nano_dispersion.models.database import SessionLocal
        from nano_dispersion.models.orm import ParticleTrajectory, BatchSummary, AnalysisTask
        from sqlalchemy import func

        table = Table(title="批次列表", show_header=True, header_style="bold magenta")
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Status")
        table.add_column("创建时间", style="blue")
        table.add_column("轨迹数", justify="right")
        table.add_column("温度(°C)", justify="right")
        table.add_column("粒径(nm)", justify="right")

        for b in batches:
            db = SessionLocal()
            try:
                traj_count = db.query(func.count(ParticleTrajectory.id)).filter(
                    ParticleTrajectory.batch_id == b.id
                ).scalar() or 0

                mean_radius = None
                latest_task = db.query(AnalysisTask).filter(
                    AnalysisTask.batch_id == b.id
                ).order_by(AnalysisTask.id.desc()).first()
                if latest_task:
                    summary = db.query(BatchSummary).filter(
                        BatchSummary.task_id == latest_task.id
                    ).first()
                    if summary:
                        mean_radius = summary.mean_radius_nm
            finally:
                db.close()

            status_color = _status_colors.get(b.status, "white")
            table.add_row(
                str(b.id),
                b.name,
                f"[{status_color}]{b.status}[/]",
                _format_datetime(b.created_at),
                str(traj_count),
                _format_float(b.temperature_c, 1),
                _format_float(mean_radius, 1) if mean_radius else "N/A",
            )

        console.print(table)
    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command("analyze")
@click.argument("batch_id", type=int)
@click.option("--wait/--no-wait", default=True, show_default=True, help="等待任务完成并显示进度")
@click.option("--poll-interval", type=float, default=2.0, show_default=True, help="轮询间隔秒数")
def analyze(batch_id, wait, poll_interval):
    """触发分析任务"""
    try:
        batch_service, task_manager = _ensure_services()

        batch = batch_service.get_batch(batch_id)
        if not batch:
            console.print(f"[bold red]错误:[/bold red] 批次 {batch_id} 不存在")
            raise click.Abort()

        task_id = batch_service.submit_analysis(batch_id)
        console.print(f"[green]✓ 任务已提交，任务ID: {task_id}[/green]")

        if not wait:
            console.print(f"查看任务状态: [cyan]nano-dispersion status {task_id}[/cyan]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            pt = progress.add_task("分析中...", total=100)

            last_pct = 0.0
            last_msg = ""

            while True:
                task = task_manager.get_task_status(task_id)
                if not task:
                    time.sleep(poll_interval)
                    continue

                pct = float(task.progress_pct or 0.0)
                msg = task.message or ""

                if pct != last_pct or msg != last_msg:
                    progress.update(pt, completed=min(pct, 100), description=msg or "分析中...")
                    last_pct = pct
                    last_msg = msg

                if task.status == "completed":
                    progress.update(pt, completed=100, description="[green]分析完成[/green]")
                    console.print(f"\n[bold green]✓ 分析任务 {task_id} 执行成功[/bold green]")
                    result = task.result_json or {}
                    if isinstance(result, dict) and "statistics" in result:
                        stats = result["statistics"]
                        console.print("  - 扩散系数 D (μm²/s)")
                        console.print(f"    均值: {_format_float(stats.get('mean_D_um2_s'))}")
                        console.print(f"    中位数: {_format_float(stats.get('median_D_um2_s'))}")
                        console.print("  - 水力学半径 (nm)")
                        console.print(f"    均值: {_format_float(stats.get('mean_radius_nm'))}")
                    break
                elif task.status == "failed":
                    progress.update(pt, completed=min(pct, 100), description="[red]分析失败[/red]")
                    console.print(f"\n[bold red]✗ 任务 {task_id} 执行失败[/bold red]")
                    if task.error_message:
                        err_lines = (task.error_message or "").split("\n")
                        console.print(Panel(err_lines[0] if err_lines else "未知错误"))
                    raise click.Abort()

                time.sleep(poll_interval)

    except click.Abort:
        raise
    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command("status")
@click.argument("task_id", type=int)
def status(task_id):
    """查看任务状态"""
    try:
        _, task_manager = _ensure_services()

        task = task_manager.get_task_status(task_id)
        if not task:
            console.print(f"[bold red]错误:[/bold red] 任务 {task_id} 不存在")
            raise click.Abort()

        status_color = _status_colors.get(task.status, "white")
        duration_str = "N/A"
        if task.started_at and task.completed_at:
            duration = task.completed_at - task.started_at
            duration_str = str(duration)
        elif task.started_at:
            duration = datetime.utcnow() - task.started_at.replace(tzinfo=None)
            duration_str = f"{duration} (进行中)"

        info_table = Table(title=f"任务 #{task_id} 详情", show_header=False)
        info_table.add_column("属性", style="cyan")
        info_table.add_column("值")
        info_table.add_row("状态", f"[{status_color}]{task.status}[/]")
        info_table.add_row("类型", task.task_type)
        info_table.add_row("批次ID", str(task.batch_id))
        info_table.add_row("进度", f"{float(task.progress_pct or 0):.1f}%")
        info_table.add_row("消息", task.message or "N/A")
        info_table.add_row("提交时间", _format_datetime(task.created_at))
        info_table.add_row("开始时间", _format_datetime(task.started_at))
        info_table.add_row("完成时间", _format_datetime(task.completed_at))
        info_table.add_row("耗时", duration_str)

        console.print(info_table)

        if task.status == "failed" and task.error_message:
            err_msg = (task.error_message or "").split("\n")[0]
            console.print(f"\n[bold red]错误信息:[/bold red] {err_msg}")
    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command("qc-summary")
@click.argument("batch_id", type=int)
@click.option("--task-id", type=int, default=None, help="使用特定任务ID，默认最新完成")
def qc_summary(batch_id, task_id):
    """查看质控摘要"""
    try:
        batch_service, _ = _ensure_services()

        batch = batch_service.get_batch(batch_id)
        if not batch:
            console.print(f"[bold red]错误:[/bold red] 批次 {batch_id} 不存在")
            raise click.Abort()

        result = batch_service.get_qc_summary(batch_id, task_id)
        qc = result.get("qc_report")
        summary = result.get("batch_summary")

        if not qc and not summary:
            console.print("[yellow]暂无质控数据，请先执行分析[/yellow]")
            return

        total = summary.get("total_trajectories", 0) if summary else 0
        valid = summary.get("valid_trajectories", 0) if summary else 0
        short = qc.get("short_trajectories", 0) if qc else 0
        broken = qc.get("broken_trajectories", 0) if qc else 0
        reconnected = qc.get("reconnected_trajectories", 0) if qc else 0
        cross = qc.get("cross_channel_count", 0) if qc else 0
        drift = qc.get("drift_estimate") if qc else None
        intensity_drop = qc.get("intensity_drop_count", 0) if qc else 0
        frame_missing = qc.get("frame_missing_count", 0) if qc else 0
        excluded = total - valid if total else 0

        dist_stats = summary.get("distribution_stats", {}) if summary else {}
        exclude_reasons = dist_stats.get("exclude_reasons", {}) if dist_stats else {}

        table = Table(title=f"批次 #{batch_id} 质控摘要", show_header=True, header_style="bold magenta")
        table.add_column("统计项", style="cyan")
        table.add_column("数量", justify="right", style="green")
        table.add_column("说明")
        table.add_row("总轨迹数", str(total), "导入的所有轨迹")
        table.add_row("有效轨迹", f"[bold green]{valid}[/bold green]", "参与D分布统计")
        table.add_row("短轨迹", f"[yellow]{short}[/yellow]", "帧数不足，不参与分布")
        table.add_row("修复轨迹", f"[blue]{reconnected}[/blue]", "断裂后自动重连")
        table.add_row("异常/漂移", str(broken + int(drift is not None and drift > 0)), "存在断点或漂移")

        table.add_row("跨通道", f"[bold red]{cross}[/bold red]", "阻塞级异常")
        table.add_row("强度骤降", str(intensity_drop), "光漂白/离焦")
        table.add_row("缺失帧", str(frame_missing), "帧数据缺失")
        table.add_row("排除总数", str(excluded), f"共{len(exclude_reasons)}种原因")

        console.print(table)

        if exclude_reasons:
            reason_table = Table(title="排除原因统计", show_header=True, header_style="bold yellow")
            reason_table.add_column("原因", style="red")
            reason_table.add_column("数量", justify="right")
            for reason, count in sorted(exclude_reasons.items(), key=lambda x: -x[1]):
                reason_table.add_row(str(reason), str(count))
            console.print(reason_table)

        if drift is not None:
            console.print(f"\n[blue]漂移估计量级: {_format_float(drift, 4)} μm/s[/blue]")

    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command("batch-summary")
@click.argument("batch_id", type=int)
@click.option("--task-id", type=int, default=None)
def batch_summary(batch_id, task_id):
    """查看批次汇总"""
    try:
        batch_service, _ = _ensure_services()

        batch = batch_service.get_batch(batch_id)
        if not batch:
            console.print(f"[bold red]错误:[/bold red] 批次 {batch_id} 不存在")
            raise click.Abort()

        bs_result = batch_service.get_batch_summary(batch_id, task_id)
        summary = bs_result.get("summary")

        if not summary:
            console.print("[yellow]暂无汇总数据，请先执行分析[/yellow]")
            return

        dist_stats = summary.get("distribution_stats", {}) or {}
        d_pct = dist_stats.get("D_percentiles", {}) or {}
        r_pct = dist_stats.get("radius_percentiles", {}) or {}
        per_channel = dist_stats.get("per_channel", {}) or {}
        exclude_reasons = dist_stats.get("exclude_reasons", {}) or {}

        d_table = Table(title="扩散系数 D (μm²/s) 分布统计", show_header=True, header_style="bold blue")
        d_table.add_column("统计量", style="cyan")
        d_table.add_column("数值", justify="right", style="green")
        d_table.add_row("均值 ± 标准差", f"{_format_float(summary.get('mean_D'))} ± {_format_float(summary.get('std_D'))}")
        d_table.add_row("中位数", _format_float(summary.get("median_D")))
        d_table.add_row("P25", _format_float(d_pct.get("p25")))
        d_table.add_row("P75", _format_float(d_pct.get("p75")))
        d_table.add_row("最小值", _format_float(d_pct.get("min")))
        d_table.add_row("最大值", _format_float(d_pct.get("max")))
        console.print(d_table)

        r_table = Table(title="水力学半径 (nm) 分布统计", show_header=True, header_style="bold blue")
        r_table.add_column("统计量", style="cyan")
        r_table.add_column("数值", justify="right", style="green")
        r_table.add_row("均值", _format_float(summary.get("mean_radius_nm")))
        r_table.add_row("P25", _format_float(r_pct.get("p25")))
        r_table.add_row("P75", _format_float(r_pct.get("p75")))
        console.print(r_table)

        if per_channel:
            ch_table = Table(title="通道间差异", show_header=True, header_style="bold cyan")
            ch_table.add_column("通道", style="green")
            ch_table.add_column("总数", justify="right")
            ch_table.add_column("有效", justify="right")
            ch_table.add_column("mean_D", justify="right")
            ch_table.add_column("median_D", justify="right")
            ch_table.add_column("std_D", justify="right")
            for ch_key, ch_data in per_channel.items():
                ch_table.add_row(
                    _channel_id_to_name(ch_key),
                    str(ch_data.get("count_total", "N/A")),
                    str(ch_data.get("count_valid", "N/A")),
                    _format_float(ch_data.get("mean_D")),
                    _format_float(ch_data.get("median_D")),
                    _format_float(ch_data.get("std_D")),
                )
            console.print(ch_table)

        result2 = batch_service.get_qc_summary(batch_id, task_id)
        summary_data = result2.get("batch_summary")
        if summary_data:
            ds2 = summary_data.get("distribution_stats", {}) or {}
            model_stats = {}
            traj_results = batch_service.get_trajectory_results(batch_id, task_id, limit=10000)
            for tr in traj_results:
                mt = tr.get("model_type") or "unknown"
                model_stats[mt] = model_stats.get(mt, 0) + 1

            if model_stats:
                m_table = Table(title="模型判别统计", show_header=True, header_style="bold magenta")
                m_table.add_column("模型类型", style="green")
                m_table.add_column("轨迹数", justify="right")
                model_names = {
                    "brownian": "布朗扩散",
                    "confined": "受限扩散",
                    "directed": "定向扩散",
                    "subdiffusive": "次扩散",
                    "superdiffusive": "超扩散",
                    "anomalous": "异常扩散",
                    "unknown": "未知",
                    "excluded": "已排除",
                }
                for mt, count in sorted(model_stats.items(), key=lambda x: -x[1]):
                    m_table.add_row(model_names.get(mt, mt), str(count))
                console.print(m_table)

    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command("list-trajectories")
@click.argument("batch_id", type=int)
@click.option("--task-id", type=int, default=None)
@click.option("--channel", type=str, default=None, help="过滤通道")
@click.option("--only-valid", is_flag=True, default=False, help="只显示有效轨迹")
@click.option("--limit", type=int, default=50, show_default=True)
@click.option("--worst", is_flag=True, default=False, help="按最差排序 (R²升序+排除优先)")
def list_trajectories(batch_id, task_id, channel, only_valid, limit, worst):
    """列出轨迹结果"""
    try:
        batch_service, _ = _ensure_services()

        batch = batch_service.get_batch(batch_id)
        if not batch:
            console.print(f"[bold red]错误:[/bold red] 批次 {batch_id} 不存在")
            raise click.Abort()

        channel_id = None
        if channel is not None:
            import re
            m = re.search(r"(\d+)", channel)
            if m:
                channel_id = f"channel_{m.group(1)}"
            names_map = {"A": "channel_A", "B": "channel_B", "C": "channel_C", "D": "channel_D"}
            upper = channel.upper()
            if upper in names_map:
                channel_id = names_map[upper]
            if channel.startswith("channel_"):
                channel_id = channel

        results = batch_service.get_trajectory_results(
            batch_id, task_id,
            skip=0, limit=10000,
            channel_id=channel_id, only_valid=only_valid,
        )

        if not results:
            console.print("[yellow]暂无轨迹结果[/yellow]")
            return

        if worst:
            def sort_key(r):
                r2 = float(r.get("fit_r2") or 1.0)
                excluded = 1 if r.get("excluded_from_distribution") else 0
                return (excluded, r2)
            results.sort(key=sort_key)

        results = results[:limit]

        table = Table(title=f"批次 #{batch_id} 轨迹列表", show_header=True, header_style="bold magenta")
        table.add_column("particle_id", justify="right", style="cyan")
        table.add_column("channel", style="blue")
        table.add_column("frames", justify="right")
        table.add_column("D", justify="right")
        table.add_column("alpha", justify="right")
        table.add_column("R²", justify="right")
        table.add_column("模型", style="green")
        table.add_column("排除原因", style="red")

        model_names = {
            "brownian": "布朗",
            "confined": "受限",
            "directed": "定向",
            "subdiffusive": "次扩",
            "superdiffusive": "超扩",
            "anomalous": "异常",
            "unknown": "未知",
            "excluded": "排除",
        }

        for r in results:
            excluded = r.get("excluded_from_distribution")
            reason = r.get("exclude_reason") or ""
            r2 = r.get("fit_r2")
            if r2 is not None and float(r2) < 0.7:
                r2_str = f"[yellow]{_format_float(r2, 3)}[/yellow]"
            else:
                r2_str = _format_float(r2, 3)

            table.add_row(
                str(r.get("particle_id", "N/A")),
                _channel_id_to_name(r.get("channel_id")),
                str(r.get("num_frames", "N/A")),
                _format_float(r.get("diffusion_D_um2_s")),
                _format_float(r.get("alpha_exponent"), 3),
                r2_str,
                model_names.get(r.get("model_type") or "unknown", r.get("model_type") or "N/A"),
                f"[red]{reason}[/red]" if excluded else "",
            )

        console.print(table)
    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command("list-anomalies")
@click.argument("batch_id", type=int)
@click.option("--task-id", type=int, default=None)
@click.option("--severity", type=click.Choice(["info", "warning", "error", "blocking"]), default=None)
@click.option("--limit", type=int, default=100, show_default=True)
def list_anomalies(batch_id, task_id, severity, limit):
    """列出异常"""
    try:
        batch_service, _ = _ensure_services()

        batch = batch_service.get_batch(batch_id)
        if not batch:
            console.print(f"[bold red]错误:[/bold red] 批次 {batch_id} 不存在")
            raise click.Abort()

        anomalies = batch_service.get_anomalies(
            batch_id, task_id, severity=severity, skip=0, limit=limit,
        )

        if not anomalies:
            console.print("[green]✓ 没有发现异常记录[/green]")
            return

        table = Table(title=f"批次 #{batch_id} 异常列表", show_header=True, header_style="bold red")
        table.add_column("severity", style="bold")
        table.add_column("类型", style="cyan")
        table.add_column("particle_id", justify="right")
        table.add_column("描述")

        for a in anomalies:
            sev = a.get("severity", "info")
            color = _severity_colors.get(sev, "white")
            table.add_row(
                f"[{color}]{sev.upper()}[/]",
                a.get("type", "N/A"),
                str(a.get("particle_id", "N/A")),
                a.get("description", "N/A"),
            )

        console.print(table)
    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command("explain-trajectory")
@click.argument("trajectory_id", type=int)
@click.option("--task-id", type=int, default=None)
@click.option("--verbose", is_flag=True, default=False, help="显示详细 MSD 点列表")
def explain_trajectory(trajectory_id, task_id, verbose):
    """解释单条轨迹"""
    try:
        batch_service, _ = _ensure_services()

        result = batch_service.explain_trajectory(trajectory_id, task_id)
        if "error" in result:
            console.print(f"[bold red]错误:[/bold red] {result['error']}")
            raise click.Abort()

        console.print(Panel.fit(
            f"[bold cyan]轨迹 #{trajectory_id} 详细信息[/bold cyan]",
            border_style="cyan",
        ))

        console.print("\n[bold]基本信息[/bold]")
        info_table = Table(show_header=False, show_lines=False)
        info_table.add_column("属性", style="cyan", width=15)
        info_table.add_column("值", style="green")
        info_table.add_row("particle_id", str(result.get("particle_id", "N/A")))
        info_table.add_row("通道", _channel_id_to_name(result.get("channel_id")))
        info_table.add_row("帧数", str(result.get("num_frames", "N/A")))
        duration = result.get("duration_s")
        info_table.add_row("时长(s)", _format_float(duration, 3) if duration is not None else "N/A")
        console.print(info_table)

        console.print("\n[bold]QC 状态[/bold]")
        qc_passed = result.get("qc_passed", True)
        flags = result.get("flags") or []
        qc_status = "[bold green]通过[/bold green]" if qc_passed else "[bold red]失败[/bold red]"
        console.print(f"  结果: {qc_status}")
        if flags:
            console.print(f"  Flags: [yellow]{', '.join(flags)}[/yellow]")
        else:
            console.print("  Flags: 无")

        fit = result.get("fit_quality")
        console.print("\n[bold]拟合结果[/bold]")
        if fit:
            fit_table = Table(show_header=False, show_lines=False)
            fit_table.add_column("属性", style="cyan", width=20)
            fit_table.add_column("值", style="green")
            fit_table.add_row("扩散系数 D (μm²/s)", _format_float(fit.get("diffusion_D_um2_s")))
            fit_table.add_row("alpha 指数", _format_float(fit.get("alpha_exponent"), 3))
            r2 = fit.get("fit_r2")
            r2_color = "red" if (r2 is not None and float(r2) < 0.7) else "green"
            fit_table.add_row("R²", f"[{r2_color}]{_format_float(r2, 4)}[/]")
            ci_low = fit.get("ci_low")
            ci_high = fit.get("ci_high")
            if ci_low is not None or ci_high is not None:
                fit_table.add_row("95% CI", f"[[{_format_float(ci_low)}, {_format_float(ci_high)}]]")
            lag_start = fit.get("fit_lag_start")
            lag_end = fit.get("fit_lag_end")
            if lag_start is not None or lag_end is not None:
                fit_table.add_row("拟合区间 (帧)", f"{lag_start or '?'} ~ {lag_end or '?'}")
            fit_table.add_row("水力学半径 (nm)", _format_float(fit.get("hydro_radius_nm"), 2))
            console.print(fit_table)
        else:
            console.print("  [yellow]暂无拟合数据[/yellow]")

        console.print("\n[bold]模型判别[/bold]")
        model_type = result.get("model_type")
        model_reason = result.get("model_reason")
        model_names_cn = {
            "brownian": "布朗扩散 (标准)",
            "confined": "受限扩散",
            "directed": "定向扩散",
            "subdiffusive": "次扩散",
            "superdiffusive": "超扩散",
            "anomalous": "异常扩散",
            "unknown": "未知/拟合失败",
            "excluded": "已排除",
            None: "未判别",
        }
        if model_type:
            console.print(f"  模型类型: [bold blue]{model_names_cn.get(model_type, model_type)}[/bold blue]")
            if model_reason:
                console.print(f"  判别原因: {model_reason}")
        else:
            console.print("  [yellow]未进行模型判别[/yellow]")

        console.print("\n[bold]排除状态[/bold]")
        excluded = result.get("excluded", False)
        exclude_reason = result.get("exclude_reason")
        if excluded:
            console.print(f"  [bold red]已排除[/bold red]")
            if exclude_reason:
                console.print(f"  原因: {exclude_reason}")
        else:
            console.print(f"  [bold green]参与D分布统计[/bold green]")

        anomalies = result.get("anomalies") or []
        console.print(f"\n[bold]异常列表[/bold] (共 {len(anomalies)} 条)")
        if anomalies:
            a_table = Table(show_header=True, header_style="bold yellow")
            a_table.add_column("Severity", style="bold")
            a_table.add_column("类型", style="cyan")
            a_table.add_column("描述")
            for a in anomalies:
                sev = a.get("severity", "info")
                color = _severity_colors.get(sev, "white")
                a_table.add_row(
                    f"[{color}]{sev.upper()}[/]",
                    a.get("type", "N/A"),
                    a.get("description", "N/A"),
                )
            console.print(a_table)
        else:
            console.print("  [green]✓ 无异常记录[/green]")

        if verbose and fit and "msd_points" in result:
            msd_pts = result.get("msd_points")
            if msd_pts:
                console.print(f"\n[bold]MSD 点数表[/bold] (共 {len(msd_pts)} 个点)")
                m_table = Table(show_header=True, header_style="bold blue")
                m_table.add_column("lag (帧)", justify="right")
                m_table.add_column("time (s)", justify="right")
                m_table.add_column("MSD (μm²)", justify="right")
                m_table.add_column("count", justify="right")
                for pt in msd_pts[:100]:
                    m_table.add_row(
                        str(pt.get("lag", "N/A")),
                        _format_float(pt.get("time_s"), 4),
                        _format_float(pt.get("msd_um2"), 4),
                        str(pt.get("count", "N/A")),
                    )
                console.print(m_table)
                if len(msd_pts) > 100:
                    console.print(f"[dim]... 仅显示前100个点 (共{len(msd_pts)}个)[/dim]")

    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command("calibration")
@click.argument("batch_id", type=int)
@click.option("--task-id", type=int, default=None)
def calibration(batch_id, task_id):
    """查看校准曲线"""
    try:
        batch_service, _ = _ensure_services()

        batch = batch_service.get_batch(batch_id)
        if not batch:
            console.print(f"[bold red]错误:[/bold red] 批次 {batch_id} 不存在")
            raise click.Abort()

        calib = batch_service.get_calibration(batch_id, task_id)
        if not calib:
            console.print("[yellow]暂无校准数据[/yellow]")
            return

        points = calib.get("calibration_points") or []

        if points:
            p_table = Table(title="校准点", show_header=True, header_style="bold blue")
            p_table.add_column("通道", style="green")
            p_table.add_column("标称(nm)", justify="right")
            p_table.add_column("实测(nm)", justify="right")
            p_table.add_column("偏差(%)", justify="right")
            p_table.add_column("D (μm²/s)", justify="right")
            for cp in points:
                bias = cp.get("bias_pct")
                if bias is not None:
                    try:
                        b = float(bias)
                        if abs(b) > 10:
                            bias_str = f"[red]{_format_float(b, 2)}%[/red]"
                        elif abs(b) > 5:
                            bias_str = f"[yellow]{_format_float(b, 2)}%[/yellow]"
                        else:
                            bias_str = f"[green]{_format_float(b, 2)}%[/green]"
                    except (TypeError, ValueError):
                        bias_str = "N/A"
                else:
                    bias_str = "N/A"
                p_table.add_row(
                    _channel_id_to_name(cp.get("channel_id")),
                    _format_float(cp.get("nominal_nm"), 1),
                    _format_float(cp.get("measured_nm"), 1),
                    bias_str,
                    _format_float(cp.get("D_um2_s")),
                )
            console.print(p_table)

        fit_table = Table(title="拟合参数", show_header=False)
        fit_table.add_column("参数", style="cyan")
        fit_table.add_column("值", style="green")
        fit_table.add_row("斜率 (slope)", _format_float(calib.get("calibration_factor")))
        fit_table.add_row("截距 (intercept)", _format_float(calib.get("intercept")))
        fit_table.add_row("R²", _format_float(calib.get("overall_r2"), 4))
        fit_table.add_row("平均偏差(%)", _format_float(calib.get("mean_bias_pct"), 2))
        fit_table.add_row("偏差标准差(%)", _format_float(calib.get("std_bias_pct"), 2))
        console.print(fit_table)

    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command("export")
@click.argument("batch_id", type=int)
@click.option("--task-id", type=int, default=None)
@click.option("--output-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
              help="输出目录 (默认 config.result_dir)")
@click.option("--formats", type=str, default="all", show_default=True,
              help="导出格式, 逗号分隔 json,csv,markdown")
def export_report(batch_id, task_id, output_dir, formats):
    """导出报告"""
    try:
        from nano_dispersion.config import get_settings

        batch_service, task_manager = _ensure_services()
        settings = get_settings()

        batch = batch_service.get_batch(batch_id)
        if not batch:
            console.print(f"[bold red]错误:[/bold red] 批次 {batch_id} 不存在")
            raise click.Abort()

        if task_id is None:
            from nano_dispersion.models.database import SessionLocal
            from nano_dispersion.models.orm import AnalysisTask
            db = SessionLocal()
            try:
                t = db.query(AnalysisTask).filter(
                    AnalysisTask.batch_id == batch_id,
                    AnalysisTask.status == "completed",
                ).order_by(AnalysisTask.id.desc()).first()
                if not t:
                    console.print(f"[bold red]错误:[/bold red] 批次 {batch_id} 尚无已完成的分析任务")
                    raise click.Abort()
                task_id = t.id
            finally:
                db.close()

        task = task_manager.get_task_status(task_id)
        if not task:
            console.print(f"[bold red]错误:[/bold red] 任务 {task_id} 不存在")
            raise click.Abort()
        if task.status != "completed":
            console.print(f"[bold yellow]警告:[/bold yellow] 任务 {task_id} 状态为 [{_status_colors.get(task.status, 'white')}]{task.status}[/], 可能数据不完整")
            if not click.confirm("是否继续导出？", default=False):
                raise click.Abort()

        out_dir = output_dir if output_dir is not None else settings.result_dir
        out_path = Path(out_dir)

        fmt_lower = formats.lower()
        fmt_list = [f.strip() for f in fmt_lower.split(",")]
        if "all" in fmt_list:
            fmt_list = ["json", "csv", "markdown"]

        with console.status("[bold green]正在导出报告...", spinner="dots"):
            result = batch_service.export_reports(batch_id, task_id, export_dir=str(out_path))

        if "error" in result:
            console.print(f"[bold red]导出失败:[/bold red] {result['error']}")
            raise click.Abort()

        def get_size(path_str: str) -> str:
            try:
                p = Path(path_str)
                if p.exists():
                    size = p.stat().st_size
                    if size < 1024:
                        return f"{size} B"
                    elif size < 1024 * 1024:
                        return f"{size / 1024:.1f} KB"
                    else:
                        return f"{size / (1024 * 1024):.1f} MB"
                return "N/A"
            except Exception:
                return "N/A"

        console.print("\n[bold green]✓ 导出完成[/bold green]")
        console.print(f"输出目录: {out_path}\n")

        file_table = Table(title="导出文件", show_header=True, header_style="bold green")
        file_table.add_column("格式", style="cyan")
        file_table.add_column("文件路径", style="blue")
        file_table.add_column("大小", justify="right")

        if "json" in fmt_list:
            json_path = result.get("json_report")
            if json_path:
                file_table.add_row("JSON", json_path, get_size(json_path))

        if "csv" in fmt_list:
            csv_reports = result.get("csv_reports") or {}
            for key, path in csv_reports.items():
                file_table.add_row(f"CSV ({key})", path, get_size(path))

        if "markdown" in fmt_list:
            md_path = result.get("markdown_report")
            if md_path:
                file_table.add_row("Markdown", md_path, get_size(md_path))

        console.print(file_table)
        console.print(f"\n共导出轨迹: {result.get('trajectory_count', 0)} 条")

    except click.Abort:
        raise
    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command("serve")
@click.option("--host", type=str, default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=6817, show_default=True)
@click.option("--reload", is_flag=True, default=False, help="自动重载")
def serve(host, port, reload):
    """启动API服务"""
    try:
        import uvicorn

        console.print(f"[bold green]启动微流控分析服务[/bold green]")
        console.print(f"  地址: http://{host}:{port}")
        console.print(f"  API文档: http://{host}:{port}/docs")
        if reload:
            console.print(f"  [yellow]自动重载: 开启[/yellow]")
        console.print()

        uvicorn.run(
            "nano_dispersion.api.main:app",
            host=host,
            port=port,
            reload=reload,
        )
    except ImportError:
        console.print("[bold red]错误:[/bold red] 缺少 uvicorn 依赖，请安装: pip install uvicorn[standard]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[bold red]错误:[/bold red] {str(e)}")
        raise click.Abort()


if __name__ == "__main__":
    cli()
