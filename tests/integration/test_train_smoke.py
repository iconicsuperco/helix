"""Exercise the Forge training CLI and checkpoint resume path end to end."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG_PATH = Path("config/training/smoke.yaml")
WORKSPACE_DIRECTORIES = ("config", "datasets", "helix", "research")


def _copy_smoke_workspace(workspace: Path) -> None:
    shutil.copy2(REPOSITORY_ROOT / "train.py", workspace / "train.py")
    for directory in WORKSPACE_DIRECTORIES:
        shutil.copytree(
            REPOSITORY_ROOT / directory,
            workspace / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def _run_smoke(workspace: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "train.py", "--config", str(SMOKE_CONFIG_PATH)],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _assert_succeeded(result: subprocess.CompletedProcess[str]) -> None:
    message = (
        f"train.py exited with {result.returncode}"
        f"\nstdout:\n{result.stdout}"
        f"\nstderr:\n{result.stderr}"
    )
    assert result.returncode == 0, message


def _json_log_records(result: subprocess.CompletedProcess[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in result.stderr.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def test_train_smoke_creates_checkpoint_and_resumes() -> None:
    with TemporaryDirectory(prefix="helix-train-smoke-") as temporary_directory:
        workspace = Path(temporary_directory) / "helix"
        workspace.mkdir()
        _copy_smoke_workspace(workspace)

        initial_result = _run_smoke(workspace)
        _assert_succeeded(initial_result)

        checkpoint_path = workspace / "checkpoints/smoke/latest.pt"
        assert checkpoint_path.is_file()

        smoke_config_path = workspace / SMOKE_CONFIG_PATH
        smoke_config = yaml.safe_load(smoke_config_path.read_text(encoding="utf-8"))
        assert isinstance(smoke_config, dict)
        checkpoint_config = smoke_config.get("checkpoint")
        assert isinstance(checkpoint_config, dict)
        checkpoint_config["resume_from"] = "checkpoints/smoke/latest.pt"
        smoke_config_path.write_text(
            yaml.safe_dump(smoke_config, sort_keys=False),
            encoding="utf-8",
        )

        resumed_result = _run_smoke(workspace)
        _assert_succeeded(resumed_result)

        resumed_records = _json_log_records(resumed_result)
        resume_record = next(
            (record for record in resumed_records if record.get("event") == "training_resumed"),
            None,
        )
        assert resume_record is not None
        assert resume_record["global_step"] == 20

        completion_record = next(
            (record for record in resumed_records if record.get("event") == "training_completed"),
            None,
        )
        assert completion_record is not None
        assert completion_record["start_step"] == 20
        assert completion_record["global_step"] == 20
