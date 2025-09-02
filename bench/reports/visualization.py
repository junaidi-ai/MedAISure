from __future__ import annotations

from pathlib import Path

from bench.models.benchmark_report import BenchmarkReport


class VisualizationGenerator:
    def __init__(self, benchmark_report: BenchmarkReport):
        self.report = benchmark_report

    def generate_metric_comparison(self, output_path: Path) -> None:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import pandas as pd  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "matplotlib and pandas are required for visualization generation"
            ) from e

        data = []
        for task_id, scores in (self.report.task_scores or {}).items():
            for metric, score in (scores or {}).items():
                data.append({"Task": task_id, "Metric": metric, "Score": float(score)})

        if not data:  # Nothing to plot
            return

        df = pd.DataFrame(data)
        pivot = df.pivot(index="Task", columns="Metric", values="Score")

        plt.figure(figsize=(12, 8))
        pivot.plot(kind="bar")
        plt.title(f"Metric Comparison: {self.report.model_id}")
        plt.ylabel("Score")
        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
