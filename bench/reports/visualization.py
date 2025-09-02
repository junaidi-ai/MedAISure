from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

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

    # --- New visualization types ---
    def generate_radar_chart(
        self,
        output_path: Union[str, Path],
        *,
        max_tasks: Optional[int] = 6,
    ) -> None:
        """Create a radar (spider) chart comparing metrics across tasks.

        Notes:
        - If there are many tasks, use `max_tasks` to limit clutter (default 6).
        - Exports to formats supported by matplotlib based on file suffix (png, svg, pdf, ...).
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import numpy as np  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "matplotlib (and numpy) are required for radar charts"
            ) from e

        task_scores = self.report.task_scores or {}
        if not task_scores:
            return

        # Determine the union of metrics and a stable order
        metrics = sorted(
            {m for _t, ms in task_scores.items() for m in (ms or {}).keys()}
        )
        if not metrics:
            return

        # Optionally limit number of tasks for clarity
        tasks = sorted(task_scores.keys())
        if max_tasks is not None and len(tasks) > max_tasks:
            tasks = tasks[:max_tasks]

        # Prepare angle for each axis
        N = len(metrics)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # close the loop

        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)

        for task_id in tasks:
            values = [float(task_scores.get(task_id, {}).get(m, 0.0)) for m in metrics]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=task_id)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_yticklabels([])
        ax.set_title(f"Radar: {self.report.model_id}")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path)

    def generate_heatmap(self, output_path: Union[str, Path]) -> None:
        """Create a heatmap of task (rows) x metric (cols) scores.

        Tries seaborn first (nicer), falls back to matplotlib if seaborn isn't available.
        """
        # Collect data
        data = []
        for task_id, scores in (self.report.task_scores or {}).items():
            for metric, score in (scores or {}).items():
                data.append((task_id, metric, float(score)))

        if not data:
            return

        # Try seaborn path
        try:
            import pandas as pd  # type: ignore
            import seaborn as sns  # type: ignore
            import matplotlib.pyplot as plt  # type: ignore

            df = pd.DataFrame(data, columns=["Task", "Metric", "Score"])
            pivot = df.pivot(index="Task", columns="Metric", values="Score")

            plt.figure(figsize=(12, max(6, len(pivot) * 0.4)))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
            plt.title(f"Task Performance Heatmap: {self.report.model_id}")
            plt.tight_layout()
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            return
        except Exception:
            # Fallback to pure matplotlib
            try:
                import matplotlib.pyplot as plt  # type: ignore
                import numpy as np  # type: ignore
            except Exception as e:  # pragma: no cover - optional dependency
                raise ImportError(
                    "matplotlib is required for heatmap generation"
                ) from e

            # Build a dense matrix with stable row/col ordering
            tasks = sorted({t for t, _m, _s in data})
            metrics = sorted({m for _t, m, _s in data})
            index = {t: i for i, t in enumerate(tasks)}
            cols = {m: j for j, m in enumerate(metrics)}
            mat = np.full((len(tasks), len(metrics)), np.nan)
            for t, m, s in data:
                mat[index[t], cols[m]] = s

            fig, ax = plt.subplots(figsize=(12, max(6, len(tasks) * 0.4)))
            im = ax.imshow(mat, aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(metrics, rotation=45, ha="right")
            ax.set_yticks(range(len(tasks)))
            ax.set_yticklabels(tasks)
            ax.set_title(f"Task Performance Heatmap: {self.report.model_id}")
            fig.colorbar(im, ax=ax, label="Score")
            fig.tight_layout()
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path)

    def generate_time_series(
        self,
        output_path: Union[str, Path],
        *,
        metric: str,
        history: Optional[Sequence[Tuple[Union[str, float, int], float]]] = None,
    ) -> None:
        """Plot a simple time series of model performance.

        Two modes:
        - If `history` is provided, it should be a sequence of (x, value) where x is
          a timestamp string, numeric step, or any label and value is the metric value.
        - Otherwise, falls back to plotting the per-task sequence for the given metric
          (the order of tasks as sorted by id), which can be useful when true run
          history isn't available.
        """
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("matplotlib is required for time-series plots") from e

        xs: List[Union[str, float, int]]
        ys: List[float]

        if history:
            xs = [pt[0] for pt in history]
            ys = [float(pt[1]) for pt in history]
        else:
            # Use per-task averages as a proxy sequence
            tasks = sorted((self.report.task_scores or {}).keys())
            xs = tasks
            ys = [
                float(self.report.task_scores[t].get(metric, float("nan")))
                for t in tasks
            ]

        plt.figure(figsize=(10, 4))
        plt.plot(xs, ys, marker="o")
        plt.title(f"Time Series - {metric}: {self.report.model_id}")
        plt.ylabel(metric)
        plt.xlabel("Step")
        plt.tight_layout()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)

    # --- Interactive (Plotly) variants ---
    def generate_metric_comparison_interactive(
        self, output_html: Union[str, Path]
    ) -> None:
        """Interactive bar chart using Plotly; writes an HTML file."""
        try:
            import pandas as pd  # type: ignore
            import plotly.express as px  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "plotly and pandas are required for interactive charts"
            ) from e

        data = []
        for task_id, scores in (self.report.task_scores or {}).items():
            for metric, score in (scores or {}).items():
                data.append({"Task": task_id, "Metric": metric, "Score": float(score)})
        if not data:
            return

        df = pd.DataFrame(data)
        fig = px.bar(
            df,
            x="Task",
            y="Score",
            color="Metric",
            barmode="group",
            title=f"Metric Comparison: {self.report.model_id}",
        )
        output_html = Path(output_html)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html), include_plotlyjs="cdn")

    def generate_heatmap_interactive(self, output_html: Union[str, Path]) -> None:
        try:
            import pandas as pd  # type: ignore
            import plotly.express as px  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "plotly and pandas are required for interactive charts"
            ) from e

        data = []
        for task_id, scores in (self.report.task_scores or {}).items():
            for metric, score in (scores or {}).items():
                data.append({"Task": task_id, "Metric": metric, "Score": float(score)})
        if not data:
            return

        df = pd.DataFrame(data)
        pivot = df.pivot(index="Task", columns="Metric", values="Score")
        fig = px.imshow(
            pivot,
            color_continuous_scale="Viridis",
            title=f"Task Performance Heatmap: {self.report.model_id}",
        )
        output_html = Path(output_html)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html), include_plotlyjs="cdn")

    def generate_radar_chart_interactive(
        self, output_html: Union[str, Path], *, max_tasks: Optional[int] = 6
    ) -> None:
        try:
            import pandas as pd  # type: ignore
            import plotly.express as px  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "plotly and pandas are required for interactive charts"
            ) from e

        task_scores = self.report.task_scores or {}
        if not task_scores:
            return

        metrics = sorted(
            {m for _t, ms in task_scores.items() for m in (ms or {}).keys()}
        )
        if not metrics:
            return

        tasks = sorted(task_scores.keys())
        if max_tasks is not None and len(tasks) > max_tasks:
            tasks = tasks[:max_tasks]

        # Long-form data: each row task-metric pair
        rows = []
        for t in tasks:
            for m in metrics:
                rows.append(
                    {
                        "Task": t,
                        "Metric": m,
                        "Score": float(task_scores.get(t, {}).get(m, 0.0)),
                    }
                )
        df = pd.DataFrame(rows)
        fig = px.line_polar(
            df,
            r="Score",
            theta="Metric",
            color="Task",
            line_close=True,
            title=f"Radar: {self.report.model_id}",
        )
        output_html = Path(output_html)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html), include_plotlyjs="cdn")

    def generate_time_series_interactive(
        self,
        output_html: Union[str, Path],
        *,
        metric: str,
        history: Optional[Sequence[Tuple[Union[str, float, int], float]]] = None,
    ) -> None:
        try:
            import pandas as pd  # type: ignore
            import plotly.express as px  # type: ignore
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "plotly and pandas are required for interactive charts"
            ) from e

        if history:
            xs = [pt[0] for pt in history]
            ys = [float(pt[1]) for pt in history]
        else:
            tasks = sorted((self.report.task_scores or {}).keys())
            xs = tasks
            ys = [
                float(self.report.task_scores[t].get(metric, float("nan")))
                for t in tasks
            ]

        df = pd.DataFrame({"x": xs, "y": ys})
        fig = px.line(
            df,
            x="x",
            y="y",
            markers=True,
            title=f"Time Series - {metric}: {self.report.model_id}",
        )
        output_html = Path(output_html)
        output_html.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_html), include_plotlyjs="cdn")
