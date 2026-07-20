"""Render complete, versioned Markdown evaluation reports."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evaluation.metrics import OverlapFinding


@dataclass(frozen=True)
class DatasetResult:
    """Perplexity and provenance metadata for one evaluation dataset."""

    label: str
    dataset_id: str
    version: str
    sha256: str
    perplexity: float


@dataclass(frozen=True)
class PromptResult:
    """One fixed prompt, completion, and automated text metrics."""

    prompt_id: str
    category: str
    prompt: str
    completion: str
    repetition_rate: float
    distinct_n: float


def _table_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _metric(value: float) -> str:
    return "N/A (insufficient tokens)" if math.isnan(value) else f"{value:.6f}"


def _fenced_text(text: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}text\n{text}\n{fence}"


def _safe_checkpoint_directory(checkpoint_id: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", checkpoint_id).strip("-.")
    if not normalized:
        raise ValueError("checkpoint_id must contain at least one path-safe character")
    return normalized


def _unique_report_path(directory: Path, timestamp: str) -> Path:
    candidate = directory / f"evaluation-{timestamp}.md"
    suffix = 2
    while candidate.exists():
        candidate = directory / f"evaluation-{timestamp}-{suffix}.md"
        suffix += 1
    return candidate


def assemble_report(
    *,
    reports_directory: Path,
    checkpoint_id: str,
    checkpoint_step: int,
    dataset_results: Sequence[DatasetResult],
    prompt_suite_id: str,
    prompt_suite_version: str,
    rubric_id: str,
    rubric_version: str,
    rubric_path: Path,
    repetition_n: int,
    distinct_n_value: int,
    prompt_results: Sequence[PromptResult],
    memorization_findings: dict[str, list[OverlapFinding]],
    generated_at: datetime | None = None,
) -> Path:
    """Write a non-overwriting Markdown report and return its path."""

    timestamp_value = generated_at or datetime.now(UTC)
    if timestamp_value.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    timestamp_value = timestamp_value.astimezone(UTC)
    timestamp = timestamp_value.strftime("%Y%m%dT%H%M%S%fZ")
    checkpoint_directory = reports_directory / _safe_checkpoint_directory(checkpoint_id)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    report_path = _unique_report_path(checkpoint_directory, timestamp)

    lines = [
        "# Helix Evaluation Report",
        "",
        "## Metadata",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Checkpoint ID | `{_table_text(checkpoint_id)}` |",
        f"| Checkpoint step | {checkpoint_step} |",
        f"| Generated at | {timestamp_value.isoformat()} |",
        "| Prompt suite | "
        f"`{_table_text(prompt_suite_id)}` / `{_table_text(prompt_suite_version)}` |",
        f"| Qualitative rubric | `{_table_text(rubric_id)}` / `{_table_text(rubric_version)}` |",
        f"| Rubric path | `{_table_text(rubric_path.as_posix())}` |",
    ]
    for dataset_result in dataset_results:
        lines.append(
            f"| Dataset: {_table_text(dataset_result.label)} | "
            f"`{_table_text(dataset_result.dataset_id)}` / "
            f"`{_table_text(dataset_result.version)}` / `{dataset_result.sha256}` |"
        )
    lines.extend(
        [
            "",
            "## Perplexity",
            "",
            "| Dataset | Dataset ID | Version | SHA-256 | Perplexity |",
            "|---|---|---|---|---:|",
        ]
    )
    for dataset_result in dataset_results:
        lines.append(
            f"| {_table_text(dataset_result.label)} | "
            f"`{_table_text(dataset_result.dataset_id)}` | "
            f"`{_table_text(dataset_result.version)}` | `{dataset_result.sha256}` | "
            f"{dataset_result.perplexity:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Automated Generation Metrics",
            "",
            "| Prompt ID | Category | "
            f"Repetition-{repetition_n} | Distinct-{distinct_n_value} | Completion tokens |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for prompt_result in prompt_results:
        lines.append(
            f"| `{_table_text(prompt_result.prompt_id)}` | "
            f"{_table_text(prompt_result.category)} | "
            f"{_metric(prompt_result.repetition_rate)} | {_metric(prompt_result.distinct_n)} | "
            f"{len(prompt_result.completion.split())} |"
        )

    lines.extend(["", "## Memorization Findings", ""])
    finding_count = sum(len(findings) for findings in memorization_findings.values())
    if finding_count == 0:
        lines.append("No exact overlaps met the configured minimum token length.")
    else:
        lines.extend(
            [
                "| Prompt ID | Generated start | Corpus start | Tokens | Exact overlap |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for prompt_id, findings in memorization_findings.items():
            for finding in findings:
                lines.append(
                    f"| `{_table_text(prompt_id)}` | {finding.generated_start} | "
                    f"{finding.corpus_start} | {finding.token_count} | "
                    f"{_table_text(finding.text)} |"
                )

    lines.extend(["", "## Prompt And Completion Pairs", ""])
    for prompt_result in prompt_results:
        lines.extend(
            [
                f"### `{prompt_result.prompt_id}` ({prompt_result.category})",
                "",
                "**Prompt**",
                "",
                _fenced_text(prompt_result.prompt),
                "",
                "**Completion**",
                "",
            ]
        )
        if not prompt_result.completion.strip():
            lines.extend(["**Empty or whitespace-only completion recorded.**", ""])
        lines.extend([_fenced_text(prompt_result.completion), ""])

    lines.extend(
        [
            "## Qualitative Scores (Manual)",
            "",
            "Apply the referenced rubric manually. No scores are generated automatically.",
            "",
            "| Dimension | Score (1-5) | Reviewer notes |",
            "|---|---:|---|",
            "| Grammaticality |  |  |",
            "| Local coherence |  |  |",
            "| Repetition / degeneracy |  |  |",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
