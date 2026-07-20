"""Generate text from a trained Helix checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from research.inference.generation import SUPPORTED_DEVICES, load_inference_session


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=_nonnegative_int, default=32)
    parser.add_argument("--device", choices=sorted(SUPPORTED_DEVICES), default="cpu")
    return parser.parse_args()


def main() -> int:
    """Load one checkpoint and print greedily generated text."""

    args = _parse_args()
    session = load_inference_session(args.checkpoint.resolve(), device=args.device)
    print(session.generate(args.prompt, max_new_tokens=args.max_new_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
