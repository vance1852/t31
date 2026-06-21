from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from nano_dispersion.cli.main import cli


@pytest.fixture
def cli_runner(tmp_path: Path, monkeypatch) -> tuple[CliRunner, Path]:
    runner = CliRunner()

    test_dir = tmp_path / "cli_test"
    test_dir.mkdir(parents=True, exist_ok=True)

    data_dir = test_dir / "data" / "batches"
    result_dir = test_dir / "results"
    db_dir = test_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    return runner, data_dir


def test_cli_help(cli_runner):
    runner, _ = cli_runner

    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output

    commands_to_check = [
        ["init-samples", "--help"],
        ["import-batch", "--help"],
        ["list-batches", "--help"],
        ["analyze", "--help"],
        ["status", "--help"],
        ["qc-summary", "--help"],
        ["batch-summary", "--help"],
        ["list-trajectories", "--help"],
        ["list-anomalies", "--help"],
        ["explain-trajectory", "--help"],
        ["calibration", "--help"],
        ["export", "--help"],
        ["serve", "--help"],
    ]

    for cmd_args in commands_to_check:
        r = runner.invoke(cli, cmd_args)
        assert r.exit_code == 0, f"Help for {cmd_args[0]} failed: {r.output}"
        assert "--help" in r.output or "Usage:" in r.output or cmd_args[0] in r.output


def test_init_samples_command(cli_runner):
    runner, data_dir = cli_runner

    result = runner.invoke(cli, [
        "init-samples",
        "--output-dir", str(data_dir),
        "--num-batches", "1",
        "--seed", "42",
        "--yes",
    ])

    assert result.exit_code == 0, f"init-samples failed: {result.output}"

    batch_subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    assert len(batch_subdirs) >= 1

    for batch_dir in batch_subdirs:
        metadata = batch_dir / "metadata.json"
        assert metadata.exists()

        traj_dir = batch_dir / "trajectories"
        if traj_dir.exists():
            csv_files = list(traj_dir.glob("*.csv"))
            assert len(csv_files) > 0


def test_list_batches_empty(cli_runner, tmp_path, monkeypatch):
    runner, data_dir = cli_runner

    test_base = tmp_path / "cli_empty_test"
    test_base.mkdir(parents=True, exist_ok=True)
    db_path = test_base / "test_empty.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("NANO_DISPERSION_DB_PATH", str(db_path))
    monkeypatch.setenv("NANO_DISPERSION_DATA_DIR", str(data_dir))
    monkeypatch.setenv("NANO_DISPERSION_RESULT_DIR", str(test_base / "results"))

    result = runner.invoke(cli, [
        "list-batches",
        "--limit", "10",
        "--offset", "0",
    ])

    assert result.exit_code == 0, f"list-batches empty failed: {result.output}"
    assert ("batches" in result.output.lower() or
            "暂无" in result.output or
            "no" in result.output.lower() or
            "ID" in result.output)
