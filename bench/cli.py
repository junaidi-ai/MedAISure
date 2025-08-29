"""Command-line interface for MedAISure benchmark utilities.

Commands:
- list: List available tasks with optional filters
- register: Register a task from a local path or URL
- show: Show a task's details by ID

Usage examples:
    medaisure-benchmark list --type qa --min-examples 1 --has-metrics
    medaisure-benchmark register bench/tasks/clinical_summarization_basic.yaml
    medaisure-benchmark show clinical_summarization_basic
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .evaluation.task_registry import TaskRegistry
from .models.medical_task import TaskType


def _parse_task_type(val: Optional[str]) -> Optional[TaskType]:
    if val is None:
        return None
    try:
        return TaskType(val)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid task type '{val}'. Expected one of: {[t.value for t in TaskType]}"
        )


def cmd_list(args: argparse.Namespace) -> int:
    reg = TaskRegistry(tasks_dir=args.tasks_dir)
    reg.discover()

    ttype = _parse_task_type(args.type)
    has_metrics: Optional[bool]
    if args.has_metrics is True:
        has_metrics = True
    elif args.has_metrics is False:
        has_metrics = False
    else:
        has_metrics = None

    rows = reg.list_available(
        task_type=ttype,
        min_examples=args.min_examples,
        has_metrics=has_metrics,
        difficulty=args.difficulty,
    )

    if args.json:
        print(
            json.dumps(
                [r.__dict__ for r in rows], ensure_ascii=False, indent=2, sort_keys=True
            )
        )
        return 0

    # Plain text table-ish output
    if not rows:
        print("No tasks found matching filters.")
        return 0

    print(f"Found {len(rows)} tasks:\n")
    for r in rows:
        print(f"- {r.task_id}")
        print(f"  name        : {r.name}")
        print(f"  num_examples: {r.num_examples}")
        print(f"  metrics     : {', '.join(r.metrics) if r.metrics else '-'}")
        print(f"  file        : {r.file or '-'}")
        print("")
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    reg = TaskRegistry(tasks_dir=args.tasks_dir)
    # Allow either URL or local path; TaskRegistry delegates parsing
    if args.source.startswith("http://") or args.source.startswith("https://"):
        task = reg.register_from_url(args.source, task_id=args.id)
    else:
        task = reg.register_from_file(args.source, task_id=args.id)

    # Print a small confirmation JSON
    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "name": task.name,
                "metrics": task.metrics,
                "num_inputs": len(task.inputs),
                "num_expected_outputs": len(task.expected_outputs),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    reg = TaskRegistry(tasks_dir=args.tasks_dir)
    task = reg.get(args.id)
    # Show full task dict
    print(json.dumps(task.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="medaisure-benchmark")
    p.add_argument(
        "--tasks-dir",
        default="bench/tasks",
        help="Directory that contains task definition files (default: bench/tasks)",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # list
    sp_list = sub.add_parser("list", help="List available tasks with filters")
    sp_list.add_argument(
        "--type", dest="type", help="Filter by task type (qa|reasoning|summarization)"
    )
    sp_list.add_argument(
        "--min-examples", type=int, default=None, help="Minimum number of examples"
    )
    sp_list.add_argument(
        "--has-metrics",
        action="store_true",
        help="Only include tasks that declare metrics",
    )
    sp_list.add_argument(
        "--no-has-metrics",
        action="store_true",
        help="Only include tasks that do not declare metrics",
    )
    sp_list.add_argument(
        "--difficulty", default=None, help="Filter by difficulty (if provided by task)"
    )
    sp_list.add_argument("--json", action="store_true", help="Output JSON")
    sp_list.set_defaults(func=cmd_list)

    # register
    sp_reg = sub.add_parser("register", help="Register a task from a local path or URL")
    sp_reg.add_argument("source", help="Path or URL to YAML/JSON task definition")
    sp_reg.add_argument("--id", help="Override task ID to store under")
    sp_reg.set_defaults(func=cmd_register)

    # show
    sp_show = sub.add_parser("show", help="Show task details by ID")
    sp_show.add_argument("id", help="Task ID")
    sp_show.set_defaults(func=cmd_show)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Normalize mutually exclusive flags for has_metrics
    if getattr(args, "has_metrics", False) and getattr(args, "no_has_metrics", False):
        print("Cannot use both --has-metrics and --no-has-metrics", file=sys.stderr)
        return 2

    # Convert flags into tri-state for filtering
    if getattr(args, "has_metrics", False):
        args.has_metrics = True
    elif getattr(args, "no_has_metrics", False):
        args.has_metrics = False
    else:
        args.has_metrics = None

    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
