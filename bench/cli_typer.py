"""Typer-based CLI for MedAISure benchmark.

Commands:
- evaluate: Run evaluation on a model and tasks
- list-tasks: List available benchmark tasks
- register-model: Register a model reference for convenience
- generate-report: Convert/pretty-print saved results into various formats

This CLI is additive and does not replace the existing argparse CLI in bench/cli.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import json

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from pydantic import BaseModel
import yaml

from .evaluation.harness import EvaluationHarness
from .models.benchmark_report import BenchmarkReport

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


# ----------------------
# Config model utilities
# ----------------------
class BenchmarkConfig(BaseModel):
    model_id: str
    tasks: Optional[List[str]] = None
    metrics: Optional[Dict[str, List[str]]] = None
    output_dir: str = "./results"
    output_format: str = "json"
    model_type: str = "huggingface"
    batch_size: int = 8
    use_cache: bool = True
    save_results: bool = True

    @classmethod
    def from_file(cls, file_path: Path) -> "BenchmarkConfig":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        text = path.read_text()
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return cls(**data)


# ----------------------
# Simple model registry
# ----------------------
REGISTRY_FILE = Path(__file__).resolve().parent / "models" / "registry.json"


def _load_registry() -> Dict[str, Dict[str, str]]:
    try:
        if REGISTRY_FILE.exists():
            return json.loads(REGISTRY_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_registry(data: Dict[str, Dict[str, str]]) -> None:
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# --------------
# Helper outputs
# --------------


def display_task_list(rows: List[Dict[str, object]]) -> None:
    table = Table(title="Available Tasks")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Description")
    table.add_column("Metrics")
    table.add_column("#Examples", justify="right")
    for r in rows:
        metrics = ", ".join(r.get("metrics", []) or [])
        table.add_row(
            str(r.get("task_id", "-")),
            str(r.get("name", "")),
            str(r.get("description", ""))[:80],
            metrics,
            str(r.get("num_examples", "-")),
        )
    console.print(table)


# ---------
# Commands
# ---------


@app.callback()
def callback():
    """MedAISure Benchmark: Evaluation framework for medical domain AI models."""
    return


@app.command("list-tasks")
def list_tasks(
    tasks_dir: Path = typer.Option(
        Path("bench/tasks"), help="Directory with task YAML/JSON files"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON"
    ),
):
    """List all available tasks in the benchmark."""
    harness = EvaluationHarness(tasks_dir=str(tasks_dir))
    rows = harness.list_available_tasks()
    if json_output:
        console.print_json(data=rows)
        raise typer.Exit(code=0)
    if not rows:
        console.print("[yellow]No tasks found.[/yellow]")
        raise typer.Exit(code=0)
    display_task_list(rows)


@app.command("register-model")
def register_model(
    model_path: Path = typer.Argument(..., help="Path to model or model configuration"),
    model_id: Optional[str] = typer.Option(
        None, help="Custom ID for the model. Defaults to basename."
    ),
    model_type: str = typer.Option(
        "local", help="Model type (local, huggingface, api)"
    ),
):
    """Register a model for evaluation (stored in a simple local registry)."""
    model_id = model_id or model_path.stem
    reg = _load_registry()
    reg[model_id] = {"path": str(model_path), "type": model_type}
    _save_registry(reg)
    console.print(
        f"[green]Registered model[/green]: {model_id} -> {model_path} ({model_type})"
    )


@app.command()
def evaluate(
    model_id: str = typer.Argument(
        ..., help="ID of the model to evaluate (or from registry)"
    ),
    tasks: Optional[List[str]] = typer.Option(
        None, help="Specific tasks to run (runs all if not specified)"
    ),
    config_file: Optional[Path] = typer.Option(
        None, help="Path to configuration file (json/yaml)"
    ),
    output_dir: Path = typer.Option(
        Path("./results"), help="Directory to store results"
    ),
    format: str = typer.Option(
        "json", help="Output format for saved report (json|yaml)"
    ),
    tasks_dir: Path = typer.Option(
        Path("bench/tasks"), help="Directory with task YAML/JSON files"
    ),
    model_type: Optional[str] = typer.Option(
        None, help="Override model type (local|huggingface|api)"
    ),
    batch_size: int = typer.Option(8, help="Batch size for inference"),
    use_cache: bool = typer.Option(True, help="Use cached results if available"),
    save_results: bool = typer.Option(True, help="Persist results to output_dir"),
):
    """Run evaluation on specified model and tasks."""
    # Merge with config file if provided
    cfg: Optional[BenchmarkConfig] = None
    if config_file:
        cfg = BenchmarkConfig.from_file(config_file)

    model_type = model_type or (cfg.model_type if cfg else "huggingface")
    out_dir = Path(cfg.output_dir if cfg else output_dir)

    harness = EvaluationHarness(tasks_dir=str(tasks_dir), results_dir=str(out_dir))

    # If no tasks specified, evaluate all
    if tasks is None or len(tasks) == 0:
        rows = harness.list_available_tasks()
        tasks = [r["task_id"] for r in rows]
        if not tasks:
            console.print("[red]No tasks to evaluate.[/red]")
            raise typer.Exit(code=1)

    # Progress feedback via callbacks
    with Progress() as progress:
        total = len(tasks)
        task_id = progress.add_task("Evaluating...", total=total)

        def on_progress(idx: int, n: int, current: Optional[str]):
            progress.update(
                task_id,
                completed=idx,
                description=f"Evaluating ({idx}/{n}) {current or ''}",
            )

        report = EvaluationHarness(
            tasks_dir=str(tasks_dir),
            results_dir=str(out_dir),
            on_progress=on_progress,
        ).evaluate(
            model_id=model_id,
            task_ids=tasks,  # type: ignore[arg-type]
            model_type=model_type,
            batch_size=cfg.batch_size if cfg else batch_size,
            use_cache=cfg.use_cache if cfg else use_cache,
            save_results=cfg.save_results if cfg else save_results,
        )

    # Save or print summary
    default_out = out_dir / f"{report.metadata.get('run_id', 'results')}.{format}"
    try:
        report.save(default_out, format=format)
        console.print(f"[green]Saved report[/green]: {default_out}")
    except Exception as e:
        console.print(f"[red]Failed to save report[/red]: {e}")
    console.print_json(data=report.to_dict(exclude={"detailed_results"}))


@app.command("generate-report")
def generate_report(
    results_file: Path = typer.Argument(
        ..., help="Path to results file (json/yaml) produced by evaluation"
    ),
    output_file: Optional[Path] = typer.Option(
        None, help="Optional output file; defaults next to input"
    ),
    format: str = typer.Option("md", help="Report format (md|json|yaml|csv)"),
):
    """Generate a human-readable report from results."""
    report = BenchmarkReport.from_file(results_file)

    if format.lower() == "md":
        # Simple markdown summary
        lines = [
            "# MedAISure Benchmark Report",
            f"Model: `{report.model_id}`",
            f"Timestamp: {report.timestamp.isoformat()}",
            "",
            "## Overall Scores",
        ]
        for k, v in (report.overall_scores or {}).items():
            lines.append(f"- {k}: {v:.4f}")
        lines.append("")
        lines.append("## Per-Task Averages")
        for t, metrics in (report.task_scores or {}).items():
            lines.append(f"- {t}:")
            for k, v in (metrics or {}).items():
                lines.append(f"  - {k}: {v:.4f}")
        content = "\n".join(lines)
    elif format.lower() in {"json"}:
        content = report.to_json(indent=2)
    elif format.lower() in {"yaml", "yml"}:
        content = report.to_yaml()
    elif format.lower() == "csv":
        # Combine overall + task CSVs into one markdown-friendly block
        content = (
            "# Overall\n"
            + report.overall_scores_to_csv()
            + "\n# Tasks\n"
            + report.task_scores_to_csv()
        )
    else:
        raise typer.BadParameter("Unsupported format. Use md|json|yaml|csv")

    if output_file is None:
        output_file = results_file.with_suffix(f".{format}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(content)
    console.print(f"[green]Generated report[/green]: {output_file}")


# --- Snake_case aliases for commands (to match task wording) ---
app.command("list_tasks")(list_tasks)
app.command("register_model")(register_model)
app.command("generate_report")(generate_report)


def main() -> None:
    """Entrypoint for console_scripts."""
    app()


if __name__ == "__main__":
    main()
