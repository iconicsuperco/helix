"""Verify reports preserve empty output and never overwrite prior runs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from evaluation.report import DatasetResult, PromptResult, assemble_report


def _assemble(tmp_path: Path) -> Path:
    return assemble_report(
        reports_directory=tmp_path / "reports",
        checkpoint_id="checkpoint-fixture",
        checkpoint_step=20,
        dataset_results=[
            DatasetResult(
                label="in_corpus",
                dataset_id="dataset-fixture",
                version="v1",
                sha256="a" * 64,
                perplexity=2.5,
            )
        ],
        prompt_suite_id="suite-fixture",
        prompt_suite_version="v1",
        rubric_id="rubric-fixture",
        rubric_version="v1",
        rubric_path=Path("evaluation/rubric/fixture.md"),
        repetition_n=4,
        distinct_n_value=2,
        prompt_results=[
            PromptResult(
                prompt_id="empty-completion",
                category="fixture",
                prompt="Prompt preserved verbatim.",
                completion="",
                repetition_rate=float("nan"),
                distinct_n=float("nan"),
            )
        ],
        memorization_findings={"empty-completion": []},
        generated_at=datetime(2026, 7, 20, tzinfo=UTC),
    )


def test_empty_completion_is_explicit_and_verbatim_prompt_is_preserved(tmp_path: Path) -> None:
    report_path = _assemble(tmp_path)

    report = report_path.read_text(encoding="utf-8")
    assert "| Dataset: in_corpus | `dataset-fixture` / `v1` /" in report
    assert "**Empty or whitespace-only completion recorded.**" in report
    assert "Prompt preserved verbatim." in report
    assert "N/A (insufficient tokens)" in report
    assert "| Grammaticality |  |  |" in report


def test_re_evaluation_creates_distinguishable_report(tmp_path: Path) -> None:
    first_path = _assemble(tmp_path)
    second_path = _assemble(tmp_path)

    assert first_path != second_path
    assert first_path.is_file()
    assert second_path.is_file()
