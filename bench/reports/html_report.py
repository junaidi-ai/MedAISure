from __future__ import annotations

from pathlib import Path
from html import escape
import json
import os

from bench.models.benchmark_report import BenchmarkReport
from .base import ReportGenerator


_BASE_CSS = """
body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 16px; }
h1,h2,h3 { color: #222; }
.section { margin-bottom: 24px; }
.table { border-collapse: collapse; width: 100%; margin-top: 8px; }
.table th, .table td { border: 1px solid #ddd; padding: 8px; font-size: 14px; }
.table th { background: #f6f8fa; text-align: left; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; background: #eaecef; font-size: 12px; }
details { margin: 8px 0 16px 0; }
summary { cursor: pointer; font-weight: 600; }
.code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
"""


class HTMLReportGenerator(ReportGenerator):
    """Generate an HTML report for a benchmark run."""

    def generate(self, benchmark_report: BenchmarkReport) -> str:
        """Render the provided benchmark report as an HTML string.

        Args:
            benchmark_report: Aggregated benchmark results to render.

        Returns:
            HTML document as a string.
        """
        br = benchmark_report

        def render_overall():
            rows = []
            for metric, score in (br.overall_scores or {}).items():
                rows.append(
                    f"<tr><td>{escape(metric)}</td><td>{float(score):.4f}</td></tr>"
                )
            return "\n".join(rows)

        def render_task_scores():
            parts = []
            for task_id, scores in (br.task_scores or {}).items():
                rows = []
                for metric, score in (scores or {}).items():
                    rows.append(
                        f"<tr><td>{escape(metric)}</td><td>{float(score):.4f}</td></tr>"
                    )
                table = (
                    f"<h3>{escape(task_id)}</h3>\n"
                    f'<table class="table"><thead><tr><th>Metric</th><th>Score</th></tr></thead><tbody>\n'
                    + "\n".join(rows)
                    + "\n</tbody></table>"
                )
                parts.append(table)
            return "\n".join(parts)

        def render_detailed():
            parts = []
            # Environment overrides
            try:
                preview_limit = int(os.environ.get("MEDAISURE_HTML_PREVIEW_LIMIT", "5"))
            except Exception:
                preview_limit = 5
            open_metadata = os.environ.get("MEDAISURE_HTML_OPEN_METADATA", "0") == "1"
            for res in br.detailed_results or []:
                metrics_rows = []
                for metric, score in (res.metrics_results or {}).items():
                    try:
                        sval = f"{float(score):.4f}"
                    except Exception:
                        sval = escape(str(score))
                    metrics_rows.append(
                        f"<tr><td>{escape(metric)}</td><td>{sval}</td></tr>"
                    )

                # Optional previews for inputs, model outputs, and metadata (truncated for readability)
                def _json_preview(obj, max_items: int = preview_limit) -> str:
                    try:
                        if isinstance(obj, list) and len(obj) > max_items:
                            data = obj[:max_items]
                            footer = (
                                f"\n... ({len(obj) - max_items} more items omitted)"
                            )
                        else:
                            data = obj
                            footer = ""
                        text = json.dumps(data, indent=2, default=str)
                        return escape(text) + escape(footer)
                    except Exception:
                        return escape(str(obj))

                inputs_block = (
                    f"<details><summary>Inputs ({len(res.inputs or [])})</summary>"
                    f'<pre class="code">{_json_preview(res.inputs)}</pre>'
                    f"</details>"
                    if getattr(res, "inputs", None)
                    else ""
                )
                outputs_block = (
                    f"<details><summary>Model Outputs ({len(res.model_outputs or [])})</summary>"
                    f'<pre class="code">{_json_preview(res.model_outputs)}</pre>'
                    f"</details>"
                    if getattr(res, "model_outputs", None)
                    else ""
                )
                metadata_block = (
                    f"<details{' open' if open_metadata else ''}><summary>Metadata</summary>"
                    f'<pre class="code">{_json_preview(res.metadata)}</pre>'
                    f"</details>"
                    if getattr(res, "metadata", None)
                    else ""
                )

                detail_html = (
                    f'<details><summary>Task: {escape(res.task_id)} <span class="badge">details</span></summary>\n'
                    f'<div><table class="table"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>\n'
                    + "\n".join(metrics_rows)
                    + "\n</tbody></table>"
                    + inputs_block
                    + outputs_block
                    + metadata_block
                    + "</div></details>"
                )
                parts.append(detail_html)
            return "\n".join(parts)

        # Pull combined score metadata for highlighting
        combined_metric_name = str(
            (br.metadata or {}).get("combined_metric_name", "combined_score")
        )
        combined_weights = (br.metadata or {}).get("combined_weights", {}) or {}

        # Optional highlighted combined score block
        def render_combined_note():
            if combined_metric_name in (br.overall_scores or {}):
                try:
                    cval = float(
                        (br.overall_scores or {}).get(combined_metric_name, 0.0)
                    )
                    ctext = f"{cval:.4f}"
                except Exception:
                    ctext = escape(
                        str((br.overall_scores or {}).get(combined_metric_name))
                    )
                weights_text = ""
                if combined_weights:
                    try:
                        kv = ", ".join(
                            f"{escape(str(k))}={float(v):.3f}"
                            for k, v in combined_weights.items()
                        )
                    except Exception:
                        kv = ", ".join(
                            f"{escape(str(k))}={escape(str(v))}"
                            for k, v in combined_weights.items()
                        )
                    weights_text = f'<div class="badge">Weights: {kv}</div>'
                return (
                    f'<div class="section">'
                    f'<div class="badge"><strong>{escape(combined_metric_name)}</strong>: {ctext}</div>'
                    f"{weights_text}"
                    f"</div>"
                )
            return ""

        html = f"""
<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Benchmark Report: {escape(br.model_id)}</title>
  <style>{_BASE_CSS}</style>
</head>
<body>
  <h1>Benchmark Report: {escape(br.model_id)}</h1>
  <div class=\"badge\">Generated at: {escape(str(br.timestamp))}</div>

  <div class=\"section\">
    <h2>Overall Scores</h2>
    {render_combined_note()}
    <table class=\"table\">
      <thead><tr><th>Metric</th><th>Score</th></tr></thead>
      <tbody>
        {render_overall()}
      </tbody>
    </table>
  </div>

  <div class=\"section\">
    <h2>Task Scores</h2>
    {render_task_scores()}
  </div>

  <div class=\"section\">
    <h2>Detailed Results</h2>
    {render_detailed()}
  </div>
</body>
</html>
"""
        return html

    def save(self, report: str, output_path: Path) -> None:
        """Write the HTML report to disk.

        Args:
            report: HTML text content.
            output_path: Destination file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

    def validate(self, report: str) -> None:
        """Perform minimal structure checks on the HTML payload.

        Args:
            report: HTML text to validate.

        Raises:
            ValueError: If required elements are missing.
        """
        if not isinstance(report, str):
            raise ValueError("HTML report must be a string")
        required_snippets = [
            "<!doctype html>",
            "<html",
            "</html>",
            "<head>",
            "</head>",
            "<body>",
            "</body>",
            "Overall Scores",
            "Task Scores",
            "Detailed Results",
        ]
        missing = [s for s in required_snippets if s not in report]
        if missing:
            raise ValueError(f"HTML report missing required elements: {missing}")
