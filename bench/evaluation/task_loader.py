"""Task loader implementation for MEDDSAI benchmark."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class MedicalTask:
    """Represents a medical evaluation task."""

    task_id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    metrics: List[Dict[str, Any]]
    dataset: Optional[List[Dict[str, Any]]] = None

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data against the task's input schema."""
        # Basic validation - can be expanded based on schema complexity
        return all(
            field in input_data for field in self.input_schema.get("required", [])
        )


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

            # Create and cache the task
            task = MedicalTask(
                task_id=task_id,
                name=task_data["name"],
                description=task_data.get("description", ""),
                input_schema=task_data.get("input_schema", {}),
                output_schema=task_data.get("output_schema", {}),
                metrics=task_data.get("metrics", []),
                dataset=task_data.get("dataset"),
            )

            self._tasks[task_id] = task
            return task

        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid task definition format in {task_file}") from e

    def load_tasks(self, task_ids: List[str]) -> Dict[str, MedicalTask]:
        """Load multiple tasks by their IDs.

        Args:
            task_ids: List of task IDs to load

        Returns:
            Dictionary mapping task IDs to MedicalTask instances
        """
        return {task_id: self.load_task(task_id) for task_id in task_ids}
