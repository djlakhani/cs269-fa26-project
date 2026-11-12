from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .constants import (
    DEFAULT_DISTRACTORS_PER_SCENARIO,
    DEFAULT_SAMPLES_PER_TYPE,
    DEFAULT_SEED,
    DEFAULT_VLLM_BASE_URL,
    JUDGE_MODEL_ID,
)
from .diagnostics import run_diagnostics
from .judging import run_judge
from .preflight import archive_log, run_preflight
from .preparation import prepare_from_huggingface
from .reporting import refresh_reports
from .runner import run_evaluation
from .verification import verify_run


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-4B CUP-Mem evaluation on STALE")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Freeze scenarios and embedding model")
    prepare.add_argument("--artifact-root", type=_path, required=True)
    prepare.add_argument("--cache-dir", type=_path, required=True)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--samples-per-type", type=int, default=DEFAULT_SAMPLES_PER_TYPE)
    prepare.add_argument(
        "--distractors-per-scenario", type=int, default=DEFAULT_DISTRACTORS_PER_SCENARIO
    )
    prepare.add_argument("--force", action="store_true")

    run = subparsers.add_parser("run", help="Build CUP-Mem state and answer queries")
    run.add_argument("--artifact-root", type=_path, required=True)
    run.add_argument("--cache-dir", type=_path, required=True)
    run.add_argument("--vllm-base-url", default=DEFAULT_VLLM_BASE_URL)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--max-per-type", type=int)
    run.add_argument("--no-progress", action="store_true")

    judge = subparsers.add_parser("judge", help="Run the official STALE judge prompt")
    judge.add_argument("--artifact-root", type=_path, required=True)
    judge.add_argument("--model", default=JUDGE_MODEL_ID)
    judge.add_argument("--resume", action="store_true")
    judge.add_argument("--max-scenarios", type=int)

    diagnose = subparsers.add_parser("diagnose", help="Diagnose official judge failures")
    diagnose.add_argument("--artifact-root", type=_path, required=True)
    diagnose.add_argument("--model", default=JUDGE_MODEL_ID)
    diagnose.add_argument("--resume", action="store_true")
    diagnose.add_argument("--max-failures", type=int)

    for name in ("report", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--artifact-root", type=_path, required=True)
        if name == "verify":
            command.add_argument("--allow-incomplete", action="store_true")

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--artifact-root", type=_path, required=True)
    preflight.add_argument("--cache-dir", type=_path, required=True)
    preflight.add_argument("--vllm-base-url", default=DEFAULT_VLLM_BASE_URL)
    preflight.add_argument("--skip-judge", action="store_true")

    archive = subparsers.add_parser("archive-log")
    archive.add_argument("--source", type=_path, required=True)
    archive.add_argument("--destination", type=_path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        path = prepare_from_huggingface(
            args.artifact_root,
            cache_dir=args.cache_dir,
            seed=args.seed,
            samples_per_type=args.samples_per_type,
            distractors_per_scenario=args.distractors_per_scenario,
            force=args.force,
        )
        print(path)
        return 0
    if args.command == "run":
        count = run_evaluation(
            args.artifact_root,
            cache_dir=args.cache_dir,
            vllm_base_url=args.vllm_base_url,
            resume=args.resume,
            max_per_type=args.max_per_type,
            show_progress=not args.no_progress,
        )
        print(json.dumps({"generated_scenarios": count}, indent=2))
        return 0
    if args.command == "judge":
        count = run_judge(
            args.artifact_root,
            model=args.model,
            resume=args.resume,
            max_scenarios=args.max_scenarios,
        )
        print(json.dumps({"generated_judgments": count}, indent=2))
        return 0
    if args.command == "diagnose":
        count = run_diagnostics(
            args.artifact_root,
            model=args.model,
            resume=args.resume,
            max_failures=args.max_failures,
        )
        print(json.dumps({"generated_diagnostics": count}, indent=2))
        return 0
    if args.command == "report":
        print(json.dumps(refresh_reports(args.artifact_root), ensure_ascii=False, indent=2))
        return 0
    if args.command == "verify":
        print(
            json.dumps(
                verify_run(args.artifact_root, require_complete=not args.allow_incomplete),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "preflight":
        print(
            json.dumps(
                run_preflight(
                    args.artifact_root,
                    cache_dir=args.cache_dir,
                    vllm_base_url=args.vllm_base_url,
                    check_judge=not args.skip_judge,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "archive-log":
        archive_log(args.source, args.destination)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
