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
import sys
import os
import logging
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
from .reports import ReportFactory
from .leaderboard.submission import (
    build_and_validate_submission,
)
from .data import (
    JSONDataset,
    CSVDataset,
    MIMICConnector,
    PubMedConnector,
    DatasetRegistry,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


class _NullConsole:
    """Minimal console stub used when Rich output is disabled in tests."""

    def print(self, *args, **kwargs):
        return

    def print_json(self, *args, **kwargs):
        return

    def status(self, *args, **kwargs):
        from contextlib import nullcontext

        return nullcontext()


# Initialize a module-level console lazily configured
console: Console | _NullConsole = _NullConsole()


def _get_console() -> Console | _NullConsole:
    """Return a Console configured for the current environment.

    When MEDAISURE_NO_RICH=1, force no color/terminal and write to sys.__stdout__
    to avoid pytest capture's closed file descriptor issues.
    """
    if os.environ.get("MEDAISURE_NO_RICH") == "1":
        return _NullConsole()
    # Use standard stdout so Typer/CliRunner/pytest can capture
    return Console(
        file=sys.stdout,
        force_terminal=False,
        color_system=None,
        no_color=True,
        soft_wrap=True,
    )


def _configure_console_if_needed() -> None:
    """Rebind the module-level console if env requires safer settings."""
    global console
    # Always refresh in case env changed between imports and command execution
    console = _get_console()
    # Additionally, suppress logging to avoid writes to captured streams during tests
    if os.environ.get("MEDAISURE_NO_RICH") == "1":
        logging.disable(logging.CRITICAL)
    else:
        logging.disable(logging.NOTSET)


def _print(text: str) -> None:
    """Safe printing helper that avoids Rich under MEDAISURE_NO_RICH."""
    if os.environ.get("MEDAISURE_NO_RICH") == "1":
        # Emit plain text to standard stdout (pytest captures this)
        try:
            print(text, file=sys.stdout)
        except Exception:
            # Fallback to normal stdout if __stdout__ unavailable
            print(text)
    else:
        _get_console().print(text)


def _print_json(data: str) -> None:
    """Print JSON safely; expects a pre-serialized JSON string.

    Always emit raw JSON to stdout to avoid ANSI styling that breaks parsers.
    """
    try:
        print(data, file=sys.stdout)
    except Exception:
        print(data)


def _status(msg: str):
    """Return a Rich status context or a no-op context based on env.

    If MEDAISURE_NO_RICH=1 is set, avoid creating live console renderables
    which can clash with pytest's capture.
    """
    if os.environ.get("MEDAISURE_NO_RICH") == "1":
        from contextlib import nullcontext

        return nullcontext()
    # Use a fresh console each time in case configuration changed
    return _get_console().status(msg)


def _rich_enabled() -> bool:
    return os.environ.get("MEDAISURE_NO_RICH") != "1"


# ----------------------
# Config model utilities
# ----------------------
class BenchmarkConfig(BaseModel):
    """Typed configuration model for running benchmark evaluations."""

    model_id: str
    tasks: Optional[List[str]] = None
    metrics: Optional[Dict[str, List[str]]] = None
    output_dir: str = "./results"
    output_format: str = "json"
    model_type: str = "huggingface"
    batch_size: int = 8
    use_cache: bool = True
    save_results: bool = True
    # New: default extra reports and report directory
    extra_reports: Optional[List[str]] = None
    report_dir: Optional[str] = None
    # New: HTML rendering preferences (optional)
    html_open_metadata: Optional[bool] = None
    html_preview_limit: Optional[int] = None
    # New: combined score configuration
    combined_weights: Optional[Dict[str, float]] = None
    combined_metric_name: Optional[str] = None
    # New: category mapping override (category -> list of metric names)
    category_map: Optional[Dict[str, List[str]]] = None

    @classmethod
    def from_file(cls, file_path: Path) -> "BenchmarkConfig":
        """Load a `BenchmarkConfig` from a JSON or YAML file path."""
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


# ----------------------
# Validation & error helpers
# ----------------------


def _ensure_exists(path: Path, kind: str = "path") -> None:
    """Exit with a user-friendly error if a file/dir does not exist."""
    p = Path(path)
    if not p.exists():
        _print(f"{kind.capitalize()} not found: {p}")
        raise typer.Exit(code=1)


def _ensure_not_dir(path: Path, label: str = "output") -> None:
    """Exit with a user-friendly error if given path resolves to a directory."""
    p = Path(path)
    if p.exists() and p.is_dir():
        _print(f"{label.capitalize()} path is a directory: {p}")
        raise typer.Exit(code=1)


# --------------
# Helper outputs
# --------------


def display_task_list(rows: List[Dict[str, object]]) -> None:
    """Render a Rich table of available tasks to the console."""
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
    if _rich_enabled():
        console.print(table)
        # Emit a simple header so stdout contains an assertable marker
        _print("Available Tasks")


def _display_evaluation_summary(report: BenchmarkReport) -> None:
    """Print a concise Rich table summary for evaluation results.

    Shows overall metrics and per-task averages to make terminal output
    human-friendly while keeping JSON output available for machines.
    """
    table = Table(title="Evaluation Summary")
    table.add_column("Section")
    table.add_column("Key")
    table.add_column("Value", justify="right")

    # Overall metrics
    overall = report.overall_scores or {}
    if overall:
        for k, v in overall.items():
            try:
                sval = f"{float(v):.4f}"
            except Exception:
                sval = str(v)
            table.add_row("Overall", k, sval)
    else:
        table.add_row("Overall", "-", "-")

    # Per-task averages
    tasks = report.task_scores or {}
    if tasks:
        for task_name, metrics in tasks.items():
            if not metrics:
                table.add_row(task_name, "-", "-")
                continue
            for mk, mv in metrics.items():
                try:
                    sval = f"{float(mv):.4f}"
                except Exception:
                    sval = str(mv)
                table.add_row(task_name, mk, sval)
    else:
        table.add_row("Tasks", "-", "-")

    if _rich_enabled():
        console.print(table)


# ----------------------
# HTML rendering helper
# ----------------------


def _report_to_html(report: BenchmarkReport) -> str:
    """Render a simple HTML summary for a BenchmarkReport."""
    rows_overall = "\n".join(
        f"<li><strong>{k}</strong>: {float(v):.4f}</li>"
        for k, v in (report.overall_scores or {}).items()
    )
    rows_tasks = []
    for t, metrics in (report.task_scores or {}).items():
        items = "\n".join(
            f"<li>{k}: {float(v):.4f}</li>" for k, v in (metrics or {}).items()
        )
        rows_tasks.append(f"<li><strong>{t}</strong><ul>\n{items}\n</ul></li>")
    rows_tasks_html = "\n".join(rows_tasks)

    return f"""
<!DOCTYPE html>
<html lang=\"en\">\n  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>MedAISure Benchmark Report</title>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; }}
      h1 {{ margin-top: 0; }}
      .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 1rem; }}
      h2 {{ margin-top: 1.5rem; }}
      ul {{ line-height: 1.6; }}
      code {{ background: #f6f8fa; padding: 0 0.25rem; border-radius: 4px; }}
    </style>
  </head>
  <body>
    <h1>MedAISure Benchmark Report</h1>
    <div class=\"meta\">Model: <code>{report.model_id}</code><br/>Timestamp: {report.timestamp.isoformat()}</div>
    <h2>Overall Scores</h2>
    <ul>
      {rows_overall}
    </ul>
    <h2>Per-Task Averages</h2>
    <ul>
      {rows_tasks_html}
    </ul>
  </body>
    </html>
    """


# ----------------------
# Weights parse & validate
# ----------------------


def _parse_weights(s: Optional[str]) -> Optional[Dict[str, float]]:
    """Parse a weights string provided via CLI.

    Accepts either a JSON object string (e.g., '{"diagnostics":0.4,...}') or
    a comma-separated list of key=value pairs (e.g., 'diagnostics=0.4,safety=0.3').
    Returns None if s is falsy.
    """
    if not s:
        return None
    s = s.strip()
    # Try JSON first
    try:
        payload = json.loads(s)
        if isinstance(payload, dict):
            return {str(k): float(v) for k, v in payload.items()}
    except Exception:
        pass
    # Fallback to key=value parsing
    out: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise typer.BadParameter(
                f"Invalid weight token '{part}'. Expected key=value pairs."
            )
        k, v = part.split("=", 1)
        try:
            out[k.strip()] = float(v.strip())
        except Exception:
            raise typer.BadParameter(
                f"Invalid numeric value for '{k.strip()}': '{v.strip()}'"
            )
    return out


def _parse_category_map_inline(s: Optional[str]) -> Optional[Dict[str, List[str]]]:
    """Parse an inline category map string.

    Supports JSON of the form '{"diagnostics":["accuracy","exact_match"], ...}'.
    Returns None if s is falsy.
    """
    if not s:
        return None
    s = s.strip()
    try:
        payload = json.loads(s)
        if isinstance(payload, dict):
            out: Dict[str, List[str]] = {}
            for k, v in payload.items():
                if isinstance(v, list):
                    out[str(k)] = [str(x) for x in v]
            return out
    except Exception:
        pass
    # If not valid JSON dict, raise an explicit error to guide users
    raise typer.BadParameter(
        '--category-map must be a JSON object string, e.g., \'{"diagnostics":["accuracy","exact_match"]}\''
    )


def _load_category_map_file(path: Optional[Path]) -> Optional[Dict[str, List[str]]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"Category map file not found: {p}")
    text = p.read_text()
    try:
        if p.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
    except Exception as e:
        raise typer.BadParameter(f"Failed to parse category map file: {e}")
    if not isinstance(data, dict):
        raise typer.BadParameter(
            "Category map file must define an object mapping categories to arrays of metric names"
        )
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[str(k)] = [str(x) for x in v]
    return out


def _validate_weights(
    weights: Dict[str, float], *, require_sum_one: bool = True
) -> Dict[str, float]:
    """Validate weights are non-negative and (optionally) sum to ~1.0.

    Returns a normalized copy (unchanged values) and raises BadParameter on errors.
    """
    if not weights:
        raise typer.BadParameter("combined_weights must not be empty")
    for k, v in weights.items():
        try:
            fv = float(v)
        except Exception:
            raise typer.BadParameter(f"Weight for '{k}' is not numeric: {v}")
        if fv < 0:
            raise typer.BadParameter(f"Weight for '{k}' must be non-negative")
    if require_sum_one:
        total = sum(float(v) for v in weights.values())
        if not (abs(total - 1.0) <= 1e-6):
            raise typer.BadParameter(
                f"Weights must sum to 1.0 (±1e-6). Got: {total:.6f}"
            )
    return {str(k): float(v) for k, v in weights.items()}


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
    _configure_console_if_needed()
    _ensure_exists(tasks_dir, kind="tasks directory")
    harness = EvaluationHarness(tasks_dir=str(tasks_dir))
    rows = harness.list_available_tasks()
    if json_output:
        _print_json(json.dumps(rows))
        raise typer.Exit(code=0)
    if not rows:
        _print("No tasks found.")
        raise typer.Exit(code=0)
    display_task_list(rows)


@app.command("list-datasets")
def list_datasets(
    registry_path: Optional[Path] = typer.Option(
        None, help="Path to dataset registry JSON (defaults to package path)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output"),
    with_composition: bool = typer.Option(
        False, "--with-composition", help="Include composition column in table output"
    ),
):
    """List datasets available in the DatasetRegistry."""
    _configure_console_if_needed()
    reg = DatasetRegistry(
        registry_path=str(registry_path) if registry_path is not None else None
    )
    rows = [m.model_dump() for m in reg.list()]
    if json_output:
        _print_json(json.dumps(rows))
        raise typer.Exit(code=0)
    if not rows:
        _print("No datasets found.")
        raise typer.Exit(code=0)
    # Pretty table
    table = Table(title="Datasets")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Planned")
    table.add_column("Size", justify="right")
    table.add_column("Categories")
    if with_composition:
        table.add_column("Composition")
    for r in rows:
        cats = ", ".join(r.get("task_categories", []) or [])
        size = "-" if r.get("size") is None else str(r.get("size"))
        if with_composition:
            comp = r.get("composition") or {}
            comp_str = json.dumps(comp, ensure_ascii=False) if comp else "-"
            table.add_row(
                str(r.get("id", "-")),
                str(r.get("name", "")),
                "yes" if bool(r.get("planned")) else "no",
                size,
                cats,
                comp_str,
            )
        else:
            table.add_row(
                str(r.get("id", "-")),
                str(r.get("name", "")),
                "yes" if bool(r.get("planned")) else "no",
                size,
                cats,
            )
    if _rich_enabled():
        console.print(table)
        _print("Datasets")
    else:
        # Plain-text fallback so tests can assert on output without Rich.
        lines = ["Datasets"]
        for r in rows:
            rid = str(r.get("id", "-"))
            name = str(r.get("name", ""))
            planned = "yes" if bool(r.get("planned")) else "no"
            size = "-" if r.get("size") is None else str(r.get("size"))
            cats = ", ".join(r.get("task_categories", []) or [])
            lines.append(f"- {rid}")
            lines.append(f"  name        : {name}")
            lines.append(f"  planned     : {planned}")
            lines.append(f"  size        : {size}")
            lines.append(f"  categories  : {cats}")
            if with_composition:
                comp = r.get("composition") or {}
                comp_str = json.dumps(comp, ensure_ascii=False) if comp else "-"
                lines.append(f"  composition : {comp_str}")
        _print("\n".join(lines))


@app.command("show-dataset")
def show_dataset(
    dataset_id: str = typer.Argument(..., help="Dataset ID"),
    registry_path: Optional[Path] = typer.Option(
        None, help="Path to dataset registry JSON (defaults to package path)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output"),
):
    """Show details for a single dataset registry entry."""
    _configure_console_if_needed()
    reg = DatasetRegistry(
        registry_path=str(registry_path) if registry_path is not None else None
    )
    try:
        meta = reg.get(dataset_id)
    except KeyError:
        _print(f"Dataset not found: {dataset_id}")
        raise typer.Exit(code=1)
    payload = meta.model_dump()
    if json_output:
        _print_json(json.dumps(payload))
        raise typer.Exit(code=0)
    # Pretty print minimal fields
    table = Table(title=f"Dataset: {payload.get('id', dataset_id)}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("name", str(payload.get("name", "")))
    table.add_row("planned", "yes" if bool(payload.get("planned")) else "no")
    table.add_row(
        "size", "-" if payload.get("size") is None else str(payload.get("size"))
    )
    table.add_row("categories", ", ".join(payload.get("task_categories", []) or []))
    table.add_row(
        "composition",
        json.dumps(payload.get("composition", {}), ensure_ascii=False),
    )
    if _rich_enabled():
        console.print(table)
        _print("Dataset Details")
    else:
        _print(json.dumps(payload))


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
    _configure_console_if_needed()
    # Loading registry is quick, but wrap for consistent UX
    with _status("Loading model registry..."):
        reg = _load_registry()
    if not reg:
        if json_output:
            _print_json(json.dumps({}))
        else:
            _print("No models registered.")
        raise typer.Exit(code=0)

    if json_output:
        # Optionally mask secrets
        safe = {}
        for mid, entry in reg.items():
            e = dict(entry)
            if not show_secrets and "api_key" in e:
                e.pop("api_key", None)
            safe[mid] = e
        _print_json(json.dumps(safe))
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
    if _rich_enabled():
        console.print(table)
        # Emit a simple header so stdout contains an assertable marker
        _print("Registered Models")
    else:
        # Plain-text fallback so tests can assert on output without Rich.
        # Build rows directly from the registry instead of Rich internals.
        lines = [
            "Registered Models",
            "Model ID | Type | Path/Endpoint | Module | HF Task",
        ]
        for mid, entry in reg.items():
            mtype = str(entry.get("type", ""))
            loc = str(entry.get("path") or entry.get("endpoint") or "")
            module = str(entry.get("module", ""))
            hf_task = str(entry.get("hf_task", ""))
            lines.append(" | ".join([mid, mtype, loc, module, hf_task]))
        _print("\n".join(lines))


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
    _configure_console_if_needed()
    model_type = (model_type or "").strip().lower()
    if model_type not in {"local", "huggingface", "api"}:
        raise typer.BadParameter("model_type must be one of: local|huggingface|api")

    with _status("Loading model registry..."):
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
            # Print friendly message to ensure visibility in CLI output
            _print("module_path is required for local models")
            raise typer.BadParameter("module_path is required for local models")
        # Validate file path early for clearer UX
        _ensure_exists(model_path, kind="model path")
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
            _print("endpoint is required for api models")
            raise typer.BadParameter("endpoint is required for api models")
        if not api_key:
            _print("api_key is required for api models")
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
    with _status("Saving model registry..."):
        _save_registry(reg)
    _print(
        f"Registered model: {model_id} -> {entry.get('path', entry.get('endpoint', ''))} ({model_type})"
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
    extra_report: Optional[List[str]] = typer.Option(
        None,
        help="Additional report formats to export (repeatable). E.g., --extra-report html --extra-report md",
    ),
    report_dir: Optional[Path] = typer.Option(
        None, help="Directory to write extra reports (defaults to output_dir)"
    ),
    html_open_metadata: Optional[bool] = typer.Option(
        None, help="[html] Open metadata sections by default (true/false)"
    ),
    html_preview_limit: Optional[int] = typer.Option(
        None, help="[html] Max items to show in list previews (e.g., inputs/outputs)"
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
    combined_weights: Optional[str] = typer.Option(
        None,
        help=(
            "Combined score weights as JSON or key=value list. "
            'Example: \'{"diagnostics":0.4,"safety":0.3,"communication":0.2,"summarization":0.1}\' '
            "or 'diagnostics=0.4,safety=0.3,communication=0.2,summarization=0.1'"
        ),
    ),
    combined_metric_name: Optional[str] = typer.Option(
        None, help="Metric name for the combined score (default: combined_score)"
    ),
    category_map: Optional[str] = typer.Option(
        None,
        help=(
            'Override category mapping (JSON). Example: \'{"diagnostics":["accuracy","exact_match"]}\''
        ),
    ),
    category_map_file: Optional[Path] = typer.Option(
        None, help="Path to JSON/YAML file defining category -> [metrics] mapping"
    ),
    export_submission: Optional[Path] = typer.Option(
        None,
        help=(
            "If provided, write a leaderboard submission JSON for this run to the given path"
        ),
    ),
    export_submission_include_reasoning: bool = typer.Option(
        True, help="Include reasoning traces in submission when present"
    ),
):
    """Run evaluation on specified model and tasks."""
    _configure_console_if_needed()
    # Merge with config file if provided
    cfg: Optional[BenchmarkConfig] = None
    if config_file:
        with _status(f"Loading config: {config_file}..."):
            cfg = BenchmarkConfig.from_file(config_file)

    # Effective model type and output directory
    model_type = model_type or (cfg.model_type if cfg else "huggingface")
    out_dir = Path(cfg.output_dir if cfg else output_dir)

    # Resolve tasks from config if not provided on CLI
    if (not tasks) and cfg and cfg.tasks:
        tasks = cfg.tasks

    # Validate tasks directory and build harness
    _ensure_exists(tasks_dir, kind="tasks directory")
    harness = EvaluationHarness(tasks_dir=str(tasks_dir), results_dir=str(out_dir))

    # If no tasks specified, evaluate all
    if tasks is None or len(tasks) == 0:
        rows = harness.list_available_tasks()
        tasks = [r["task_id"] for r in rows]
        if not tasks:
            _print("No tasks to evaluate.")
            raise typer.Exit(code=1)

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
            model_kwargs["timeout"] = float(reg_entry["timeout"])
        if reg_entry.get("max_retries") is not None:
            model_kwargs["max_retries"] = int(reg_entry["max_retries"])
        if reg_entry.get("backoff_factor") is not None:
            model_kwargs["backoff_factor"] = float(reg_entry["backoff_factor"])

    # Resolve config-derived options
    resolved_extra_reports = extra_report
    resolved_report_dir: Optional[Path] = report_dir
    resolved_html_open_metadata: Optional[bool] = html_open_metadata
    resolved_html_preview_limit: Optional[int] = html_preview_limit
    if cfg and resolved_extra_reports is None:
        resolved_extra_reports = cfg.extra_reports
    if cfg and resolved_report_dir is None and cfg.report_dir:
        resolved_report_dir = Path(cfg.report_dir)
    if (
        cfg
        and resolved_html_open_metadata is None
        and cfg.html_open_metadata is not None
    ):
        resolved_html_open_metadata = bool(cfg.html_open_metadata)
    if (
        cfg
        and resolved_html_preview_limit is None
        and cfg.html_preview_limit is not None
    ):
        try:
            resolved_html_preview_limit = int(cfg.html_preview_limit)
        except Exception:
            resolved_html_preview_limit = None

    # Resolve combined score configuration (CLI > config > defaults)
    DEFAULT_WEIGHTS = {
        "diagnostics": 0.4,
        "safety": 0.3,
        "communication": 0.2,
        "summarization": 0.1,
    }
    # Parse CLI string if provided
    parsed_cli_weights = _parse_weights(combined_weights)
    resolved_weights: Optional[Dict[str, float]] = None
    if parsed_cli_weights is not None:
        resolved_weights = _validate_weights(parsed_cli_weights)
    elif cfg and cfg.combined_weights is not None:
        resolved_weights = _validate_weights(dict(cfg.combined_weights))
    else:
        # Use defaults
        resolved_weights = dict(DEFAULT_WEIGHTS)

    resolved_combined_metric_name = (
        (combined_metric_name.strip() if combined_metric_name else None)
        or (cfg.combined_metric_name if cfg and cfg.combined_metric_name else None)
        or "combined_score"
    )

    # Resolve category map override (CLI > config)
    resolved_category_map: Optional[Dict[str, List[str]]] = None
    inline_map = _parse_category_map_inline(category_map) if category_map else None
    file_map = _load_category_map_file(category_map_file) if category_map_file else None
    if inline_map is not None:
        resolved_category_map = inline_map
    elif file_map is not None:
        resolved_category_map = file_map
    elif cfg and cfg.category_map is not None:
        # Config file may specify mapping directly
        try:
            resolved_category_map = {
                str(k): [str(x) for x in (v or [])]
                for k, v in dict(cfg.category_map).items()
            }
        except Exception:
            resolved_category_map = None

    # Apply environment variables so generators respect settings
    if resolved_html_open_metadata is not None:
        os.environ["MEDAISURE_HTML_OPEN_METADATA"] = (
            "1" if resolved_html_open_metadata else "0"
        )
    if resolved_html_preview_limit is not None:
        os.environ["MEDAISURE_HTML_PREVIEW_LIMIT"] = str(
            max(1, int(resolved_html_preview_limit))
        )

    if _rich_enabled():
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
                model_type=str(effective_model_type),
                batch_size=cfg.batch_size if cfg else batch_size,
                use_cache=cfg.use_cache if cfg else use_cache,
                save_results=cfg.save_results if cfg else save_results,
                report_formats=resolved_extra_reports,
                report_dir=str(resolved_report_dir) if resolved_report_dir else None,
                combined_weights=resolved_weights,
                combined_metric_name=resolved_combined_metric_name,
                category_map=resolved_category_map,
                **model_kwargs,
            )
    else:
        report = EvaluationHarness(
            tasks_dir=str(tasks_dir),
            results_dir=str(out_dir),
        ).evaluate(
            model_id=model_id,
            task_ids=tasks,  # type: ignore[arg-type]
            model_type=str(effective_model_type),
            batch_size=cfg.batch_size if cfg else batch_size,
            use_cache=cfg.use_cache if cfg else use_cache,
            save_results=cfg.save_results if cfg else save_results,
            report_formats=resolved_extra_reports,
            report_dir=str(resolved_report_dir) if resolved_report_dir else None,
            combined_weights=resolved_weights,
            combined_metric_name=resolved_combined_metric_name,
            category_map=resolved_category_map,
            **model_kwargs,
        )

    # Save or print summary in requested format
    run_id = str(report.metadata.get("run_id", "results"))
    fmt = format.lower()
    try:
        if fmt in {"json", "yaml", "yml"}:
            default_out = (
                out_dir / f"{run_id}.{('yaml' if fmt in {'yaml', 'yml'} else 'json')}"
            )
            with _status("Saving report..."):
                report.save(default_out, format=fmt)
            _print(f"Saved report: {default_out}")
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
            with _status("Saving report (markdown)..."):
                default_out.write_text(content)
            _print(f"Saved report: {default_out}")
        elif fmt == "csv":
            # Use CSVReportGenerator to save primary CSV (task_scores) to a single .csv file
            default_out = out_dir / f"{run_id}.csv"
            with _status("Saving report (csv)..."):
                gen = ReportFactory.create_generator("csv")
                data = gen.generate(report)
                gen.save(data, default_out)
            _print(f"Saved report: {default_out}")
        else:
            raise typer.BadParameter("Unsupported format. Use json|yaml|md|csv")
    except Exception as e:
        _print(f"Failed to save report: {e}")

    # Export any extra reports at CLI level as well (useful in tests with mocked harness)
    if resolved_extra_reports:
        try:
            out_dir2 = resolved_report_dir if resolved_report_dir else out_dir
            rid = str(report.metadata.get("run_id", "results"))
            out_dir2.mkdir(parents=True, exist_ok=True)
            for fmt2 in resolved_extra_reports:
                fmtn = str(fmt2).lower().strip()
                try:
                    gen = ReportFactory.create_generator(fmtn)
                except Exception:
                    _print(f"Skipping unknown extra report format: {fmtn}")
                    continue
                content = gen.generate(report)
                ext = "md" if fmtn in {"md", "markdown"} else fmtn
                target = out_dir2 / f"{rid}.{ext}"
                try:
                    gen.save(content, target)
                    _print(f"Saved extra report: {target}")
                except Exception as e:
                    _print(f"Failed to save extra report {fmtn}: {e}")
        except Exception as e:
            _print(f"Failed to export extra reports: {e}")

    # Optionally export a leaderboard submission directly from this in-memory report
    if export_submission is not None:
        try:
            payload = build_and_validate_submission(
                report, include_reasoning=export_submission_include_reasoning
            )
            export_submission.parent.mkdir(parents=True, exist_ok=True)
            export_submission.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False)
            )
            _print(f"Saved submission: {export_submission}")
        except Exception as e:
            _print(f"Failed to export submission: {e}")

    # Print a concise JSON summary to stdout (use JSON string to handle datetimes)
    if _rich_enabled():
        _print_json(report.to_json(indent=2, exclude={"detailed_results"}))
        # Also print a styled summary table for human readability
        _display_evaluation_summary(report)


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
    _configure_console_if_needed()
    # Validate inputs
    _ensure_exists(results_file, kind="results file")
    if output_file is not None:
        _ensure_not_dir(output_file, label="output")
    with _status("Loading results..."):
        report = BenchmarkReport.from_file(results_file)

    fmt = format.lower()
    if fmt == "md":
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
    elif fmt in {"json"}:
        content = report.to_json(indent=2)
    elif fmt in {"yaml", "yml"}:
        content = report.to_yaml()
    elif fmt == "csv":
        # Use CSVReportGenerator: if output_file ends with .csv, save single file; if it's a directory, save all CSVs
        gen = ReportFactory.create_generator("csv")
        data = gen.generate(report)
        if output_file is None:
            output_file = results_file.with_suffix(".csv")
        # If user passed a path with a trailing slash or an existing directory, treat as directory
        target = output_file
        if str(target).endswith(os.sep) or (target.exists() and target.is_dir()):
            target = target
        with _status("Writing CSV report(s)..."):
            gen.save(data, target)
        _print(f"Generated report: {target}")
        return
    elif fmt == "html":
        with _status("Rendering HTML..."):
            content = _report_to_html(report)
    elif fmt == "pdf":
        # Generate HTML first; then try converting using WeasyPrint if available
        with _status("Rendering HTML for PDF..."):
            html_content = _report_to_html(report)
        if output_file is None:
            output_file = results_file.with_suffix(".pdf")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            from weasyprint import HTML  # type: ignore

            with _status("Generating PDF..."):
                HTML(string=html_content).write_pdf(str(output_file))
            _print(f"Generated PDF report: {output_file}")
            return
        except ImportError:
            _print(
                "PDF generation requires 'weasyprint'. Install it or use --format html."
            )
            raise typer.Exit(code=1)
    else:
        raise typer.BadParameter("Unsupported format. Use md|json|yaml|csv|html|pdf")

    if output_file is None:
        output_file = results_file.with_suffix(f".{format}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with _status("Writing file..."):
        output_file.write_text(content)
    _print(f"Generated report: {output_file}")


@app.command("generate-submission")
def generate_submission(
    run_id: str = typer.Option(..., "--run-id", help="Run ID to export"),
    results_dir: Path = typer.Option(
        Path("./results"), help="Directory where evaluation reports are saved"
    ),
    output_file: Path = typer.Option(
        ..., "--out", help="Path to write the submission JSON"
    ),
    include_reasoning: bool = typer.Option(
        True, help="Include reasoning traces when present"
    ),
):
    """Generate a leaderboard submission JSON from a saved report by run ID."""
    _configure_console_if_needed()
    # Resolve input report path; default naming: <run_id>.json
    _ensure_exists(results_dir, kind="results directory")
    candidate = results_dir / f"{run_id}.json"
    report: BenchmarkReport
    if candidate.exists():
        with _status(f"Loading report: {candidate} ..."):
            report = BenchmarkReport.from_file(candidate)
    else:
        # Fallback: scan for a report whose metadata.run_id matches
        found: Optional[Path] = None
        for p in results_dir.glob("*.json"):
            try:
                tmp = BenchmarkReport.from_file(p)
                if str(tmp.metadata.get("run_id", "")) == str(run_id):
                    found = p
                    report = tmp
                    break
            except Exception:
                continue
        if not found:
            _print(f"Could not find report for run_id={run_id} in {results_dir}")
            raise typer.Exit(code=1)

    # Build and validate submission
    with _status("Building submission..."):
        payload = build_and_validate_submission(
            report, include_reasoning=include_reasoning
        )
    # Write output
    _ensure_not_dir(output_file, label="output")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with _status(f"Writing submission: {output_file} ..."):
        output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    _print(f"Saved submission: {output_file}")


# -----------------------------
# Data connector preview helper
# -----------------------------


@app.command("preview-data")
def preview_data(
    connector: str = typer.Option(..., help="Connector type: json|csv|mimic|pubmed"),
    file: Optional[Path] = typer.Option(None, help="[json|csv] Path to data file"),
    encryption_key: Optional[str] = typer.Option(
        None, help="[json|csv] Optional encryption key"
    ),
    n: int = typer.Option(5, help="Number of samples to display"),
    conn: Optional[str] = typer.Option(None, help="[mimic] Connection string"),
    query: Optional[str] = typer.Option(None, help="[mimic] SQL query"),
    mimic_cache: bool = typer.Option(True, help="[mimic] Enable in-memory cache"),
    mimic_cache_size: int = typer.Option(128, help="[mimic] In-memory cache size"),
    mimic_cache_dir: Optional[str] = typer.Option(
        None, help="[mimic] Persistent cache directory"
    ),
    terms: Optional[List[str]] = typer.Option(
        None,
        help="[pubmed] Search terms (multiple allowed)",
    ),
    max_results: int = typer.Option(100, help="[pubmed] Max results to request"),
    api_key: Optional[str] = typer.Option(None, help="[pubmed] NCBI eutils API key"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON"
    ),
):
    """Preview a dataset connector's metadata and the first N samples (when supported)."""
    _configure_console_if_needed()
    ctype = (connector or "").strip().lower()

    # Build connector instance
    ds = None
    if ctype == "json":
        if not file:
            raise typer.BadParameter("--file is required for json connector")
        _ensure_exists(file, kind="data file")
        ds = JSONDataset(file, encryption_key=encryption_key)
    elif ctype == "csv":
        if not file:
            raise typer.BadParameter("--file is required for csv connector")
        _ensure_exists(file, kind="data file")
        ds = CSVDataset(file, encryption_key=encryption_key)
    elif ctype == "mimic":
        if not conn or not query:
            raise typer.BadParameter(
                "--conn and --query are required for mimic connector"
            )
        ds = MIMICConnector(
            conn,
            query,
            use_cache=mimic_cache,
            cache_size=mimic_cache_size,
            persistent_cache_dir=mimic_cache_dir,
        )
    elif ctype == "pubmed":
        if not terms:
            raise typer.BadParameter("--terms is required for pubmed connector")
        ds = PubMedConnector(list(terms), max_results=max_results, api_key=api_key)
    else:
        raise typer.BadParameter("Unsupported connector. Use json|csv|mimic|pubmed")

    # Always show metadata
    meta: Dict[str, Any] = {}
    try:
        meta = ds.get_metadata()  # type: ignore[assignment]
    except Exception as e:
        meta = {"error": f"failed to get metadata: {e}"}

    samples: List[Dict[str, Any]] = []
    data_supported = True
    try:
        # Not all connectors implement load_data yet (e.g., stubs)
        samples = getattr(ds, "take")(max(0, int(n)))  # type: ignore[misc]
    except NotImplementedError:
        data_supported = False
    except Exception:
        # Best-effort: ignore data errors in preview
        data_supported = False

    if json_output:
        out = {
            "connector": ctype,
            "metadata": meta,
            "samples": samples if data_supported else [],
            "data_supported": data_supported,
        }
        _print_json(json.dumps(out))
        return

    # Human-friendly output
    _print("Metadata:")
    try:
        _print_json(json.dumps(meta))
    except Exception:
        _print(str(meta))

    if data_supported:
        _print("")
        _print(f"First {len(samples)} sample(s):")
        for i, s in enumerate(samples, 1):
            try:
                _print_json(json.dumps({"index": i, "item": s}))
            except Exception:
                _print(f"{i}. {s}")
    else:
        _print("")
        _print(
            "Data preview not supported for this connector (or not yet implemented)."
        )


# --- Snake_case aliases for commands (to match task wording) ---
app.command("list_tasks")(list_tasks)
app.command("register_model")(register_model)
app.command("generate_report")(generate_report)
app.command("list_models")(list_models)
app.command("generate_submission")(generate_submission)


def main() -> None:
    """Entrypoint for console_scripts."""
    app()


if __name__ == "__main__":
    main()
