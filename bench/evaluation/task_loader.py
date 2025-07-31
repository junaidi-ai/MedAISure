"""Task loader implementation for MEDDSAI benchmark."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar

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

    def load_task(self, task_id: str) -> MedicalTask:
        """Load a task by its ID.

        Args:
            task_id: Unique identifier for the task

        Returns:
            Loaded MedicalTask instance

        Raises:
            FileNotFoundError: If task definition is not found
            ValueError: If task definition is invalid
        """
        if task_id in self._tasks:
            return self._tasks[task_id]

        # Look for task definition file
        task_file = self.tasks_dir / f"{task_id}.yaml"
        if not task_file.exists():
            task_file = self.tasks_dir / f"{task_id}.json"
            if not task_file.exists():
                raise FileNotFoundError(f"Task definition not found for {task_id}")

        # Load task definition
        try:
            if task_file.suffix == ".yaml":
                with open(task_file, "r") as f:
                    task_data = yaml.safe_load(f)
            else:  # .json
                with open(task_file, "r") as f:
                    task_data = json.load(f)

            # Create and cache the task with all required fields
            task = MedicalTask(
                task_id=task_id,
                task_type=TaskType(
                    task_data.get("task_type", "qa")
                ),  # Default to QA if not specified
                name=task_data.get(
                    "name", f"Task {task_id}"
                ),  # Default name if not provided
                description=task_data.get("description", ""),
                inputs=task_data.get("inputs", []),
                expected_outputs=task_data.get("expected_outputs", []),
                metrics=[
                    m if isinstance(m, str) else m["name"]
                    for m in task_data.get("metrics", [])
                ],
                input_schema=task_data.get("input_schema", {}),
                output_schema=task_data.get("output_schema", {}),
                dataset=task_data.get("dataset", []),
            )

            self._tasks[task_id] = task
            return task

        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid task definition format in {task_file}") from e

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
