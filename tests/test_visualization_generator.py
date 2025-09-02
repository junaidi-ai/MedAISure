import sys
from pathlib import Path

import pytest

from bench.models.benchmark_report import BenchmarkReport
from bench.reports.visualization import VisualizationGenerator


@pytest.fixture()
def sample_report() -> BenchmarkReport:
    return BenchmarkReport(
        model_id="viz-model",
        overall_scores={"accuracy": 0.8},
        task_scores={
            "t1": {"accuracy": 0.7, "f1": 0.6},
            "t2": {"accuracy": 0.9, "f1": 0.8},
        },
        detailed_results=[],
        metadata={"run_id": "viz-0001"},
    )


def test_visualization_generator_importerror_when_missing_deps(
    monkeypatch, sample_report, tmp_path: Path
):
    # Ensure imports fail regardless of environment contents
    monkeypatch.delitem(sys.modules, "matplotlib.pyplot", raising=False)
    monkeypatch.delitem(sys.modules, "pandas", raising=False)

    viz = VisualizationGenerator(sample_report)
    with pytest.raises(ImportError):
        viz.generate_metric_comparison(tmp_path / "plot.png")


def test_visualization_generator_with_stubbed_deps(
    monkeypatch, sample_report, tmp_path: Path
):
    # Stub pandas
    class _FakePivot:
        def plot(self, kind: str = "bar"):
            return None

    class _FakeDF:
        def __init__(self, rows):
            self._rows = rows

        def pivot(self, index: str, columns: str, values: str):
            # Return an object exposing .plot(); internal data not used by our stub
            return _FakePivot()

    import types

    class _FakePDModule(types.ModuleType):
        def __init__(self):
            super().__init__("pandas")

        def DataFrame(self, rows):  # noqa: N802 (match pandas API name)
            return _FakeDF(rows)

    # Stub matplotlib.pyplot
    class _FakePLTModule(types.ModuleType):
        def __init__(self):
            super().__init__("matplotlib.pyplot")

        def figure(self, figsize=None):
            return None

        def title(self, *args, **kwargs):
            return None

        def ylabel(self, *args, **kwargs):
            return None

        def tight_layout(self, *args, **kwargs):
            return None

        def savefig(self, path):
            # Simulate file creation
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"PNG")
            return None

    # Install stub modules into sys.modules for import machinery
    fake_pd = _FakePDModule()
    fake_plt = _FakePLTModule()
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_plt  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "pandas", fake_pd)
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    out = tmp_path / "plot.png"
    viz = VisualizationGenerator(sample_report)
    viz.generate_metric_comparison(out)

    assert out.exists()
    assert out.read_bytes().startswith(b"PNG")
