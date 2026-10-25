from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .constants import (
    ATTENTION_BACKENDS,
    CONTEXT_VIEWS,
    DEFAULT_ATTENTION_BACKEND,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SAMPLES_PER_TYPE,
    DEFAULT_SEED,
)
from .persistence import ArtifactPaths, rebuild_responses_jsonl
from .preparation import prepare_from_huggingface
from .reporting import (
    load_artifact,
    refresh_report,
    summarize_context_comparison,
    summarize_manual_scores,
)
from .runner import run_context_views


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _selected_views(value: str) -> list[str]:
    return list(CONTEXT_VIEWS) if value == "all" else [value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stale-qwen-eval",
        description="Run reproducible Qwen3-0.6B evaluations on STALE.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Freeze both views of the scenarios")
    prepare.add_argument("--artifact-root", type=_path, required=True)
    prepare.add_argument("--cache-dir", type=_path)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--samples-per-type", type=int, default=DEFAULT_SAMPLES_PER_TYPE)
    prepare.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    prepare.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    prepare.add_argument("--run-id")
    prepare.add_argument("--description")
    prepare.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="Generate responses for prepared scenarios")
    run.add_argument("--artifact-root", type=_path, required=True)
    run.add_argument("--cache-dir", type=_path)
    run.add_argument("--view", choices=[*CONTEXT_VIEWS, "all"], default="all")
    run.add_argument(
        "--attention-backend",
        choices=ATTENTION_BACKENDS,
        default=DEFAULT_ATTENTION_BACKEND,
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-scenarios", type=int)
    run.add_argument("--no-progress", action="store_true")

    report = subparsers.add_parser("report", help="Rebuild reports without model calls")
    report.add_argument("--artifact-root", type=_path, required=True)
    report.add_argument("--view", choices=[*CONTEXT_VIEWS, "all"], default="all")

    summarize = subparsers.add_parser("summarize", help="Summarize manual scores")
    summarize.add_argument("--artifact-root", type=_path, required=True)
    summarize.add_argument("--view", choices=[*CONTEXT_VIEWS, "all"], default="all")
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
            run_id=args.run_id,
            description=args.description,
        )
        print(f"Prepared scenarios: {path}")
        return 0

    views = _selected_views(args.view)
    artifact = load_artifact(ArtifactPaths(args.artifact_root, views[0]))
    if args.command == "run":
        counts = run_context_views(
            args.artifact_root,
            context_views=views,
            attention_backend=args.attention_backend,
            cache_dir=args.cache_dir,
            resume=args.resume,
            max_scenarios=args.max_scenarios,
            show_progress=not args.no_progress,
        )
        print(json.dumps({"generated_responses": counts}, indent=2))
        return 0
    if args.command == "report":
        for view in views:
            paths = ArtifactPaths(args.artifact_root, view)
            rebuild_responses_jsonl(paths, artifact)
            refresh_report(paths, artifact)
            print(f"Report rebuilt: {paths.report}")
        return 0
    if args.command == "summarize":
        summaries = {
            view: summarize_manual_scores(ArtifactPaths(args.artifact_root, view), artifact)
            for view in views
        }
        if set(views) == set(CONTEXT_VIEWS):
            summaries["comparison"] = summarize_context_comparison(
                args.artifact_root, artifact, summaries
            )
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return 0 if all(summary["complete"] for summary in summaries.values()) else 2
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
