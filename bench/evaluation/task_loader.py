"""Task loader implementation for MedAISure benchmark.

Enhancements:
- Robust validation and error handling (wrap Pydantic errors)
- Support loading tasks from explicit file paths and HTTP(S) URLs
- Simple registry utilities: discover and list available tasks
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, cast

import requests  # type: ignore[import-untyped]
import yaml

from ..models.medical_task import MedicalTask, TaskType

T = TypeVar("T", bound=MedicalTask)


class TaskLoader:
    """Loads and manages medical evaluation tasks."""

    def __init__(self, tasks_dir: str = "tasks"):
        """Initialize the TaskLoader with a directory containing task definitions.

        Args:
            tasks_dir: Directory containing task definition files
        """
        self.tasks_dir = Path(tasks_dir)
        self._tasks: Dict[str, MedicalTask] = {}
        # Registry of discovered tasks: task_id -> file path
        self._registry: Dict[str, str] = {}

    def load_task(self, task_id: str) -> MedicalTask:
        """Load a task by its ID.

        Args:
            task_id: Unique identifier for the task OR a direct path/URL to a
                task definition (YAML/JSON).

        Returns:
            Loaded ``MedicalTask`` instance

        Raises:
            FileNotFoundError: If task definition is not found
            ValueError: If task definition is invalid
        """
        if task_id in self._tasks:
            return self._tasks[task_id]

        # Resolve task source (file in tasks_dir, explicit file path, or URL)
        task_file: Optional[Path] = None
        task_data: Dict[str, Any]

        # 1) tasks_dir/{task_id}.yaml|.json
        candidate = self.tasks_dir / f"{task_id}.yaml"
        if candidate.exists():
            task_file = candidate
        else:
            candidate = self.tasks_dir / f"{task_id}.json"
            if candidate.exists():
                task_file = candidate

        try:
            if task_file is None:
                # 2) Direct file path
                path_candidate = Path(task_id)
                if path_candidate.exists() and path_candidate.is_file():
                    task_file = path_candidate
                else:
                    # 3) HTTP(S) URL
                    if task_id.startswith("http://") or task_id.startswith("https://"):
                        task_data = self._load_task_data_from_url(task_id)
                    else:
                        raise FileNotFoundError(
                            f"Task definition not found for {task_id}"
                        )

            if task_file is not None:
                task_data = self._load_task_data_from_file(task_file)

            # Normalize metrics to list[str]
            metrics_field = task_data.get("metrics", [])
            metrics: List[str] = []
            for m in metrics_field:
                if isinstance(m, str):
                    metrics.append(m)
                elif isinstance(m, dict) and "name" in m:
                    metrics.append(str(m["name"]))

            # Create and cache the task with all required fields
            task = MedicalTask(
                task_id=task_id if isinstance(task_id, str) else str(task_id),
                task_type=TaskType(task_data.get("task_type", "qa")),
                name=task_data.get("name", f"Task {task_id}"),
                description=task_data.get("description", ""),
                inputs=task_data.get("inputs", []),
                expected_outputs=task_data.get("expected_outputs", []),
                metrics=metrics,
                input_schema=task_data.get("input_schema", {}),
                output_schema=task_data.get("output_schema", {}),
                dataset=task_data.get("dataset", []),
            )

            self._tasks[task_id] = task
            # Update registry
            if task_file is not None:
                self._registry[task_id] = str(task_file)
            return task

        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(
                f"Invalid task definition format for {task_id}: {e}"
            ) from e
        except Exception as e:  # Wrap Pydantic and other errors as ValueError
            from pydantic import ValidationError

            if isinstance(e, ValidationError):
                raise ValueError(f"Task validation failed for {task_id}: {e}") from e
            raise

    def load_tasks(
        self, task_ids: List[str], task_type: Optional[Type[T]] = None
    ) -> Dict[str, T]:
        """Load multiple tasks by their IDs, optionally with a specific task type.

        Args:
            task_ids: List of task IDs to load
            task_type: Optional specific MedicalTask subclass to use

        Returns:
            Dictionary mapping task IDs to MedicalTask (or subclass) instances
        """
        if task_type is None:
            task_type = MedicalTask
        return {task_id: self.load_task(task_id) for task_id in task_ids}

    # --- Registry & Discovery utilities ---
    def discover_tasks(self) -> Dict[str, str]:
        """Discover available task definition files in ``tasks_dir``.

        Returns a mapping of ``task_id -> file path``.
        """
        registry: Dict[str, str] = {}
        # YAML tasks
        for task_file in self.tasks_dir.glob("*.yaml"):
            registry[task_file.stem] = str(task_file)
        # JSON tasks (skip index files like tasks.json if present)
        for task_file in self.tasks_dir.glob("*.json"):
            if task_file.stem == "tasks":
                continue
            registry[task_file.stem] = str(task_file)
        self._registry = registry
        return dict(self._registry)

    def list_available_tasks(self) -> List[Dict[str, Any]]:
        """List all available tasks with basic metadata.

        Reads YAML/JSON headers and returns summary information without full
        validation. Safe-guards against malformed files and logs warnings.
        """
        logger = logging.getLogger(__name__)
        tasks: List[Dict[str, Any]] = []

        # Ensure registry is up-to-date
        if not self._registry:
            self.discover_tasks()

        for task_id, file_path in self._registry.items():
            try:
                data = self._load_task_data_from_file(Path(file_path))
                metrics = []
                for m in data.get("metrics", []):
                    if isinstance(m, str):
                        metrics.append(m)
                    elif isinstance(m, dict) and "name" in m:
                        metrics.append(str(m["name"]))
                tasks.append(
                    {
                        "task_id": task_id,
                        "name": data.get("name", ""),
                        "description": data.get("description", ""),
                        "metrics": metrics,
                        "num_examples": len(data.get("dataset", [])),
                        "file": file_path,
                    }
                )
            except Exception as e:
                logger.warning(f"Error loading task from {file_path}: {e}")

        return tasks

    # --- Internal helpers ---
    def _load_task_data_from_file(self, task_file: Path) -> Dict[str, Any]:
        """Load and parse YAML/JSON task data from a local file."""
        if task_file.suffix.lower() in {".yaml", ".yml"}:
            with open(task_file, "r") as f:
                data = yaml.safe_load(f)
                return cast(Dict[str, Any], data)
        elif task_file.suffix.lower() == ".json":
            with open(task_file, "r") as f:
                data = json.load(f)
                return cast(Dict[str, Any], data)
        else:
            raise ValueError(f"Unsupported task file extension: {task_file.suffix}")

    def _load_task_data_from_url(self, url: str) -> Dict[str, Any]:
        """Load and parse YAML/JSON task data from an HTTP(S) URL."""
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        text = resp.text

        # Prefer extension hints
        if url.endswith((".yaml", ".yml")):
            data = yaml.safe_load(text)
            return cast(Dict[str, Any], data)
        if url.endswith(".json"):
            data = json.loads(text)
            return cast(Dict[str, Any], data)

        # Fallback to Content-Type validation
        if any(x in content_type for x in ["yaml", "x-yaml"]):
            data = yaml.safe_load(text)
            return cast(Dict[str, Any], data)
        if any(x in content_type for x in ["application/json", "+json", "text/json"]):
            data = json.loads(text)
            return cast(Dict[str, Any], data)

        # Unsupported type
        raise ValueError(f"Unsupported Content-Type for task URL {url}: {content_type}")
