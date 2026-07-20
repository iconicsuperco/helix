"""Evaluate a Helix checkpoint against versioned datasets and fixed prompts."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.config import load_evaluation_config
from evaluation.generation_eval import load_prompt_suite, run_fixed_suite
from evaluation.metrics import distinct_n, memorization_overlap, repetition_rate
from evaluation.perplexity import compute_perplexity
from evaluation.provenance import sha256_file, validate_dataset
from evaluation.report import DatasetResult, PromptResult, assemble_report
from helix.common.paths import relative_to_repository
from research.inference.generation import load_inference_session
from research.tokenizer import tokenizer as helix_tokenizer

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "evaluation" / "config.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Evaluation YAML/JSON path (default: evaluation/config.yaml)",
    )
    return parser.parse_args()


def evaluate_checkpoint(checkpoint_path: Path, config_path: Path) -> Path:
    """Run all automated R1 evaluation steps and return the report path."""

    config = load_evaluation_config(config_path)
    prompt_suite = load_prompt_suite(config.prompt_suite_path)
    if not config.rubric_path.is_file():
        raise FileNotFoundError(f"Qualitative rubric does not exist: {config.rubric_path}")

    in_corpus = validate_dataset(config.in_corpus)
    out_of_corpus = validate_dataset(config.out_of_corpus)
    memorization_corpus = validate_dataset(config.memorization_corpus)

    session = load_inference_session(checkpoint_path, device="cpu")
    dataset_results = [
        DatasetResult(
            label=in_corpus.label,
            dataset_id=in_corpus.dataset_id,
            version=in_corpus.version,
            sha256=in_corpus.sha256,
            perplexity=compute_perplexity(
                session.model,
                helix_tokenizer,
                in_corpus.path,
            ),
        ),
        DatasetResult(
            label=out_of_corpus.label,
            dataset_id=out_of_corpus.dataset_id,
            version=out_of_corpus.version,
            sha256=out_of_corpus.sha256,
            perplexity=compute_perplexity(
                session.model,
                helix_tokenizer,
                out_of_corpus.path,
            ),
        ),
    ]

    completions = run_fixed_suite(checkpoint_path, config.prompt_suite_path)
    corpus_text = memorization_corpus.path.read_text(encoding=memorization_corpus.encoding)
    prompt_results: list[PromptResult] = []
    memorization_findings = {}
    for prompt in prompt_suite.prompts:
        completion = completions[prompt.prompt_id]
        prompt_results.append(
            PromptResult(
                prompt_id=prompt.prompt_id,
                category=prompt.category,
                prompt=prompt.text,
                completion=completion,
                repetition_rate=repetition_rate(completion, config.repetition_n),
                distinct_n=distinct_n(completion, config.distinct_n),
            )
        )
        memorization_findings[prompt.prompt_id] = memorization_overlap(
            completion,
            corpus_text,
            config.min_memorization_overlap_tokens,
        )

    checkpoint_id = sha256_file(checkpoint_path)
    return assemble_report(
        reports_directory=config.reports_directory,
        checkpoint_id=checkpoint_id,
        checkpoint_step=session.checkpoint_step,
        dataset_results=dataset_results,
        prompt_suite_id=prompt_suite.suite_id,
        prompt_suite_version=prompt_suite.version,
        rubric_id=config.rubric_id,
        rubric_version=config.rubric_version,
        rubric_path=relative_to_repository(config.rubric_path),
        repetition_n=config.repetition_n,
        distinct_n_value=config.distinct_n,
        prompt_results=prompt_results,
        memorization_findings=memorization_findings,
    )


def main() -> int:
    """Evaluate one checkpoint and print the generated report path."""

    args = _parse_args()
    report_path = evaluate_checkpoint(
        args.checkpoint.expanduser().resolve(),
        args.config.expanduser().resolve(),
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
