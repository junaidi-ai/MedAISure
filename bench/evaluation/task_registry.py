"""Task registry for dynamic registration and discovery of tasks.

Provides a lightweight registry that integrates with `TaskLoader` to:
- register tasks programmatically or from files/URLs
- discover tasks under a directory
- list tasks with simple filtering (e.g., by TaskType)
- fetch/unregister tasks

Examples:
    >>> from bench.evaluation.task_registry import TaskRegistry
    >>> reg = TaskRegistry(tasks_dir="bench/tasks")
    >>> reg.discover()  # doctest: +ELLIPSIS
    {...}
    >>> rows = reg.list_available(task_type=None)  # doctest: +ELLIPSIS
    >>> isinstance(rows, list)
    True

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .task_loader import TaskLoader
from ..models.medical_task import MedicalTask, TaskType


@dataclass
class TaskSummary:
    task_id: str
    name: str
    description: str
    metrics: List[str]
    num_examples: int
    file: Optional[str]


class TaskRegistry:
    """Manage available tasks with optional filtering and dynamic registration.

    Notes:
        This registry delegates file/URL parsing and validation to `TaskLoader` to
        avoid duplicating logic. It maintains an in-memory index of registered
        tasks and consults discovery results from the loader as needed.
    """

    def __init__(self, tasks_dir: str = "bench/tasks") -> None:
        self.loader = TaskLoader(tasks_dir=tasks_dir)
        # Explicitly registered tasks (beyond discovered ones)
        self._registered: Dict[str, MedicalTask] = {}

    # --- Registration APIs ---
    def register(self, task: MedicalTask) -> None:
        """Register a task instance programmatically."""
        self._registered[task.task_id] = task

    def register_from_file(
        self, path: str, task_id: Optional[str] = None
    ) -> MedicalTask:
        """Register a task from a local file path.

        Args:
            path: YAML/JSON file path
            task_id: Optional ID override; defaults to file stem
        """
        # Let the loader parse and validate; then cache here
        task = self.loader.load_task(task_id or path)
        self._registered[task.task_id] = task
        return task

    def register_from_url(self, url: str, task_id: Optional[str] = None) -> MedicalTask:
        """Register a task from an HTTP(S) URL.

        Args:
            url: URL to YAML/JSON task definition
            task_id: Optional ID to store under; defaults to URL basename
        """
        task = self.loader.load_task(task_id or url)
        self._registered[task.task_id] = task
        return task

    def unregister(self, task_id: str) -> None:
        """Remove a programmatically registered task from the registry."""
        self._registered.pop(task_id, None)

    def get(self, task_id: str) -> MedicalTask:
        """Get a task by ID, checking registered tasks then loader."""
        if task_id in self._registered:
            return self._registered[task_id]
        return self.loader.load_task(task_id)

    # --- Discovery & Listing ---
    def discover(self) -> Dict[str, str]:
        """Discover tasks under the loader's tasks_dir.

        Returns:
            Mapping of `task_id -> file path`.
        """
        return self.loader.discover_tasks()

    def list_available(
        self,
        *,
        task_type: Optional[TaskType] = None,
        min_examples: Optional[int] = None,
        has_metrics: Optional[bool] = None,
    ) -> List[TaskSummary]:
        """List available tasks with simple filtering.

        Args:
            task_type: Filter by `TaskType` (matches YAML `task_type` when loadable)
            min_examples: Keep tasks with at least this number of examples
            has_metrics: If True, only tasks that declare non-empty metrics

        Returns:
            A list of `TaskSummary` entries.
        """
        rows = self.loader.list_available_tasks()

        def _to_summary(r: Dict[str, Any]) -> TaskSummary:
            return TaskSummary(
                task_id=r.get("task_id", ""),
                name=r.get("name", ""),
                description=r.get("description", ""),
                metrics=list(r.get("metrics", []) or []),
                num_examples=int(r.get("num_examples", 0) or 0),
                file=r.get("file"),
            )

        out: List[TaskSummary] = []
        for r in rows:
            summary = _to_summary(r)

            # Apply has_metrics filter
            if has_metrics is True and not summary.metrics:
                continue
            if has_metrics is False and summary.metrics:
                continue

            # Apply min_examples filter
            if min_examples is not None and summary.num_examples < min_examples:
                continue

            # Apply task_type filter (requires loading the task lazily to check type)
            if task_type is not None:
                try:
                    t = self.get(summary.task_id)
                    if t.task_type != task_type:
                        continue
                except Exception:
                    # If loading fails, exclude from filtered results for safety
                    continue

            out.append(summary)

        return out
