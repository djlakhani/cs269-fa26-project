from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SAMPLES_PER_TYPE,
    DEFAULT_SEED,
)
from .persistence import ArtifactPaths, rebuild_responses_jsonl
from .preparation import prepare_from_huggingface
from .reporting import load_artifact, refresh_report, summarize_manual_scores
from .runner import run_evaluation


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stale-qwen-eval",
        description="Run a reproducible Qwen3-0.6B pilot on STALE.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Freeze and truncate the pilot scenarios")
    prepare.add_argument("--artifact-root", type=_path, required=True)
    prepare.add_argument("--cache-dir", type=_path)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--samples-per-type", type=int, default=DEFAULT_SAMPLES_PER_TYPE)
    prepare.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    prepare.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    prepare.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="Generate responses for prepared scenarios")
    run.add_argument("--artifact-root", type=_path, required=True)
    run.add_argument("--cache-dir", type=_path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-queries", type=int)

    report = subparsers.add_parser("report", help="Rebuild reports without model calls")
    report.add_argument("--artifact-root", type=_path, required=True)

    summarize = subparsers.add_parser("summarize", help="Summarize completed manual scores")
    summarize.add_argument("--artifact-root", type=_path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        path = prepare_from_huggingface(
            args.artifact_root,
            cache_dir=args.cache_dir,
            force=args.force,
            seed=args.seed,
            samples_per_type=args.samples_per_type,
            context_window=args.context_window,
            max_new_tokens=args.max_new_tokens,
        )
        print(f"Prepared scenarios: {path}")
        return 0

    paths = ArtifactPaths(args.artifact_root)
    artifact = load_artifact(paths)
    if args.command == "run":
        count = run_evaluation(
            args.artifact_root,
            cache_dir=args.cache_dir,
            resume=args.resume,
            max_queries=args.max_queries,
        )
        print(f"Generated {count} new response(s). Results: {paths.report}")
        return 0
    if args.command == "report":
        rebuild_responses_jsonl(paths, artifact)
        refresh_report(paths, artifact)
        print(f"Report rebuilt: {paths.report}")
        return 0
    if args.command == "summarize":
        summary = summarize_manual_scores(paths, artifact)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["complete"] else 2
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

