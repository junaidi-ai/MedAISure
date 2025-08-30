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
from typing import Any, Dict, List, Optional

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


def _unique_model_id(base: str, reg: Dict[str, Dict[str, str]]) -> str:
    """Generate a unique model ID based on the base string.

    If the base already exists, append a numeric suffix: base-1, base-2, ...
    """
    base = base.strip() or "model"
    candidate = base
    idx = 1
    while candidate in reg:
        candidate = f"{base}-{idx}"
        idx += 1
    return candidate


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


@app.command("list-models")
def list_models(
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON"
    ),
    show_secrets: bool = typer.Option(
        False, help="Include sensitive fields like api_key in output"
    ),
):
    """List registered models from the local registry."""
    reg = _load_registry()
    if not reg:
        if json_output:
            console.print_json(data={})
        else:
            console.print("[yellow]No models registered.[/yellow]")
        raise typer.Exit(code=0)

    if json_output:
        # Optionally mask secrets
        safe = {}
        for mid, entry in reg.items():
            e = dict(entry)
            if not show_secrets and "api_key" in e:
                e.pop("api_key", None)
            safe[mid] = e
        console.print_json(data=safe)
        raise typer.Exit(code=0)

    # Pretty table
    table = Table(title="Registered Models")
    table.add_column("Model ID")
    table.add_column("Type")
    table.add_column("Path/Endpoint")
    table.add_column("Module")
    table.add_column("HF Task")
    for mid, entry in reg.items():
        mtype = str(entry.get("type", ""))
        loc = str(entry.get("path") or entry.get("endpoint") or "")
        module = str(entry.get("module", ""))
        hf_task = str(entry.get("hf_task", ""))
        table.add_row(mid, mtype, loc, module, hf_task)
    console.print(table)


@app.command("register-model")
def register_model(
    model_path: Optional[Path] = typer.Argument(
        None, help="Path to model or model configuration (HF: hub id allowed)"
    ),
    model_id: Optional[str] = typer.Option(
        None, help="Custom ID for the model. Auto-generated if omitted."
    ),
    model_type: str = typer.Option("local", help="Model type (local|huggingface|api)"),
    # Local model options
    module_path: Optional[str] = typer.Option(
        None,
        help="[local] Python import path to module exposing a load function",
    ),
    load_func: Optional[str] = typer.Option(
        None,
        help="[local] Function name to load the model (default: load_model)",
    ),
    # HF options
    hf_task: Optional[str] = typer.Option(
        None,
        help="[huggingface] HF pipeline task (e.g., summarization, text-classification)",
    ),
    # API options
    endpoint: Optional[str] = typer.Option(None, help="[api] Inference endpoint URL"),
    api_key: Optional[str] = typer.Option(None, help="[api] API key / token"),
    timeout: float = typer.Option(30.0, help="[api] Request timeout in seconds"),
    max_retries: int = typer.Option(0, help="[api] Max retries on request failure"),
    backoff_factor: float = typer.Option(0.5, help="[api] Exponential backoff base"),
):
    """Register a model for evaluation (stored in a simple local registry).

    Performs basic validation per model type and generates a unique ID if needed.
    """
    model_type = (model_type or "").strip().lower()
    if model_type not in {"local", "huggingface", "api"}:
        raise typer.BadParameter("model_type must be one of: local|huggingface|api")

    reg: Dict[str, Dict[str, str]] = _load_registry()

    # Infer default model_id
    if model_id:
        model_id = model_id.strip()
    else:
        base = "model"
        if model_path is not None:
            base = model_path.stem or model_path.name or "model"
        elif model_type == "api" and endpoint:
            base = Path(endpoint.rstrip("/")).name or "api"
        elif model_type == "huggingface" and model_path is None:
            base = "hf-model"
        model_id = _unique_model_id(base, reg)

    entry: Dict[str, Any] = {"type": model_type}

    # Per-type validation and fields
    if model_type == "local":
        if model_path is None:
            raise typer.BadParameter("model_path is required for local models")
        if not model_path.exists():
            raise typer.BadParameter(f"model_path does not exist: {model_path}")
        if not module_path:
            raise typer.BadParameter("module_path is required for local models")
        # Validate import and callable if possible
        try:
            import importlib

            mod = importlib.import_module(module_path)
            func_name = (load_func or "load_model").strip()
            fn = getattr(mod, func_name, None)
            if not callable(fn):
                raise typer.BadParameter(
                    f"Module '{module_path}' has no callable '{func_name}'"
                )
        except Exception as e:
            raise typer.BadParameter(f"Failed to import '{module_path}': {e}")

        entry.update(
            {
                "path": str(model_path),
                "module": module_path,
                "load_func": (load_func or "load_model"),
            }
        )

    elif model_type == "huggingface":
        if model_path is None:
            raise typer.BadParameter(
                "model_path is required for huggingface models (hub id or path)"
            )
        entry.update(
            {"path": str(model_path), "hf_task": (hf_task or "text-classification")}
        )

    elif model_type == "api":
        if not endpoint:
            raise typer.BadParameter("endpoint is required for api models")
        if not api_key:
            raise typer.BadParameter("api_key is required for api models")
        entry.update(
            {
                "endpoint": endpoint,
                "api_key": api_key,
                "timeout": float(timeout),
                "max_retries": int(max_retries),
                "backoff_factor": float(backoff_factor),
            }
        )

    # Persist
    reg[model_id] = entry  # type: ignore[assignment]
    _save_registry(reg)
    console.print(
        f"[green]Registered model[/green]: {model_id} -> {entry.get('path', entry.get('endpoint', ''))} ({model_type})"
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
        "json", help="Output format for saved report (json|yaml|md|csv)"
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

    # Resolve tasks from config if not provided on CLI
    if (not tasks) and cfg and cfg.tasks:
        tasks = cfg.tasks

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

        # Determine model path/type from registry if present
        reg = _load_registry()
        reg_entry = reg.get(model_id)
        effective_model_type = model_type
        model_kwargs: Dict[str, object] = {}
        if reg_entry:
            effective_model_type = (
                reg_entry.get("type", effective_model_type) or effective_model_type
            )
            # Common fields
            if reg_entry.get("path"):
                model_kwargs["model_path"] = reg_entry["path"]
            # Local fields
            if reg_entry.get("module"):
                model_kwargs["module_path"] = reg_entry["module"]
            if reg_entry.get("load_func"):
                model_kwargs["load_func"] = reg_entry["load_func"]
            # HF fields
            if reg_entry.get("hf_task"):
                model_kwargs["hf_task"] = reg_entry["hf_task"]
            # API fields
            if reg_entry.get("endpoint"):
                model_kwargs["endpoint"] = reg_entry["endpoint"]
            if reg_entry.get("api_key"):
                model_kwargs["api_key"] = reg_entry["api_key"]
            if reg_entry.get("timeout") is not None:
                model_kwargs["timeout"] = reg_entry["timeout"]
            if reg_entry.get("max_retries") is not None:
                model_kwargs["max_retries"] = reg_entry["max_retries"]
            if reg_entry.get("backoff_factor") is not None:
                model_kwargs["backoff_factor"] = reg_entry["backoff_factor"]

        report = EvaluationHarness(
            tasks_dir=str(tasks_dir),
            results_dir=str(out_dir),
            on_progress=on_progress,
        ).evaluate(
            model_id=model_id,
            task_ids=tasks,  # type: ignore[arg-type]
            model_type=str(effective_model_type),
            batch_size=cfg.batch_size if cfg else batch_size,
            use_cache=cfg.use_cache if cfg else use_cache,
            save_results=cfg.save_results if cfg else save_results,
            **model_kwargs,
        )

    # Save or print summary in requested format
    run_id = str(report.metadata.get("run_id", "results"))
    fmt = format.lower()
    try:
        if fmt in {"json", "yaml", "yml"}:
            default_out = (
                out_dir / f"{run_id}.{('yaml' if fmt in {'yaml','yml'} else 'json')}"
            )
            report.save(default_out, format=fmt)
            console.print(f"[green]Saved report[/green]: {default_out}")
        elif fmt == "md":
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
            default_out = out_dir / f"{run_id}.md"
            default_out.write_text(content)
            console.print(f"[green]Saved report[/green]: {default_out}")
        elif fmt == "csv":
            # Write a single file containing two CSV sections
            content = (
                "# Overall\n"
                + report.overall_scores_to_csv()
                + "\n# Tasks\n"
                + report.task_scores_to_csv()
            )
            default_out = out_dir / f"{run_id}.csv"
            default_out.write_text(content)
            console.print(f"[green]Saved report[/green]: {default_out}")
        else:
            raise typer.BadParameter("Unsupported format. Use json|yaml|md|csv")
    except Exception as e:
        console.print(f"[red]Failed to save report[/red]: {e}")

    # Print a concise JSON summary to stdout (use JSON string to handle datetimes)
    console.print_json(json=report.to_json(indent=2, exclude={"detailed_results"}))


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
app.command("list_models")(list_models)


def main() -> None:
    """Entrypoint for console_scripts."""
    app()


if __name__ == "__main__":
    main()
