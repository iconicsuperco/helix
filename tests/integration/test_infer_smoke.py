"""Exercise smoke training followed by checkpoint-backed text generation."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_DIRECTORIES = ("config", "datasets", "helix", "research")


def _copy_workspace(workspace: Path) -> None:
    for filename in ("train.py", "infer.py"):
        shutil.copy2(REPOSITORY_ROOT / filename, workspace / filename)
    for directory in WORKSPACE_DIRECTORIES:
        shutil.copytree(
            REPOSITORY_ROOT / directory,
            workspace / directory,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def _run(workspace: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _assert_succeeded(result: subprocess.CompletedProcess[str]) -> None:
    message = (
        f"subprocess exited with {result.returncode}"
        f"\nstdout:\n{result.stdout}"
        f"\nstderr:\n{result.stderr}"
    )
    assert result.returncode == 0, message


def test_smoke_checkpoint_generates_text() -> None:
    with TemporaryDirectory(prefix="helix-infer-smoke-") as temporary_directory:
        workspace = Path(temporary_directory) / "helix"
        workspace.mkdir()
        _copy_workspace(workspace)

        training_result = _run(
            workspace,
            "train.py",
            "--config",
            "config/training/smoke.yaml",
        )
        _assert_succeeded(training_result)
        checkpoint_path = workspace / "checkpoints/smoke/latest.pt"
        assert checkpoint_path.is_file()

        prompt = "Call me Ishmael."
        inference_result = _run(
            workspace,
            "infer.py",
            "--checkpoint",
            "checkpoints/smoke/latest.pt",
            "--prompt",
            prompt,
            "--max-new-tokens",
            "4",
            "--device",
            "cpu",
        )
        _assert_succeeded(inference_result)
        assert prompt in inference_result.stdout
