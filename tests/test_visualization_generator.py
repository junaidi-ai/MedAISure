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
    # Ensure imports fail regardless of environment contents by intercepting import
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):  # type: ignore[override]
        if name.startswith("matplotlib") or name.startswith("pandas"):
            raise ImportError("forced by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

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


def test_generate_radar_chart_with_stubbed_matplotlib_numpy(
    monkeypatch, sample_report, tmp_path: Path
):
    import types

    class _FakeFig:
        def tight_layout(self):
            return None

        def savefig(self, path):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"PNG")

    class _FakeAx:
        def plot(self, *args, **kwargs):
            return None

        def fill(self, *args, **kwargs):
            return None

        def set_xticks(self, *args, **kwargs):
            return None

        def set_xticklabels(self, *args, **kwargs):
            return None

        def set_yticklabels(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

    class _FakePLT(types.ModuleType):
        def __init__(self):
            super().__init__("matplotlib.pyplot")

        def figure(self, *_, **__):
            return _FakeFig()

        def subplot(self, *_, **__):
            return _FakeAx()

    class _FakeNP(types.ModuleType):
        def __init__(self):
            super().__init__("numpy")
            self.pi = 3.141592653589793

        def linspace(self, start, stop, num, endpoint=True):
            # Simple linear space
            step = (stop - start) / (num - (0 if endpoint else 1))
            data = [start + i * step for i in range(num)]

            class _Arr(list):
                def tolist(self):
                    return list(self)

            return _Arr(data)

    fake_plt = _FakePLT()
    fake_np = _FakeNP()
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_plt  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)
    monkeypatch.setitem(sys.modules, "numpy", fake_np)

    out = tmp_path / "radar.png"
    viz = VisualizationGenerator(sample_report)
    viz.generate_radar_chart(out)
    assert out.exists()
    assert out.read_bytes().startswith(b"PNG")


def test_generate_heatmap_with_seaborn_stub(monkeypatch, sample_report, tmp_path: Path):
    import types

    class _FakeDF:
        def __init__(self, rows, columns=None):
            self._rows = rows

        def pivot(self, index: str, columns: str, values: str):
            return {"_pivot": True}

    class _FakePD(types.ModuleType):
        def __init__(self):
            super().__init__("pandas")

        def DataFrame(self, rows, columns=None):
            return _FakeDF(rows, columns)

    class _FakePLT(types.ModuleType):
        def __init__(self):
            super().__init__("matplotlib.pyplot")

        def figure(self, *args, **kwargs):
            return None

        def title(self, *args, **kwargs):
            return None

        def tight_layout(self, *args, **kwargs):
            return None

        def savefig(self, path):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"PNG")

    class _FakeSns(types.ModuleType):
        def __init__(self):
            super().__init__("seaborn")

        def heatmap(self, *args, **kwargs):
            return None

    fake_pd = _FakePD()
    fake_plt = _FakePLT()
    fake_sns = _FakeSns()
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_plt  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "pandas", fake_pd)
    monkeypatch.setitem(sys.modules, "seaborn", fake_sns)
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    out = tmp_path / "heatmap.png"
    viz = VisualizationGenerator(sample_report)
    viz.generate_heatmap(out)
    assert out.exists()
    assert out.read_bytes().startswith(b"PNG")


def test_generate_heatmap_fallback_matplotlib_only(
    monkeypatch, sample_report, tmp_path: Path
):
    import types

    class _FakeNP(types.ModuleType):
        def __init__(self):
            super().__init__("numpy")
            self.nan = float("nan")

        class _Matrix:
            def __init__(self, shape, fill_value):
                self._rows = shape[0]
                self._cols = shape[1]
                self._data = [
                    [fill_value for _ in range(self._cols)] for _ in range(self._rows)
                ]

            def __setitem__(self, key, value):
                i, j = key
                self._data[i][j] = value

        def full(self, shape, fill_value):
            return _FakeNP._Matrix(shape, fill_value)

    class _FakeFig:
        def tight_layout(self):
            return None

        def savefig(self, path):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"PNG")

        def colorbar(self, *args, **kwargs):
            return None

    class _FakeAx:
        def imshow(self, *args, **kwargs):
            return object()

        def set_xticks(self, *args, **kwargs):
            return None

        def set_xticklabels(self, *args, **kwargs):
            return None

        def set_yticks(self, *args, **kwargs):
            return None

        def set_yticklabels(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

    class _FakePLT(types.ModuleType):
        def __init__(self):
            super().__init__("matplotlib.pyplot")

        def subplots(self, *args, **kwargs):
            return _FakeFig(), _FakeAx()

    fake_np = _FakeNP()
    fake_plt = _FakePLT()
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_plt  # type: ignore[attr-defined]

    # Ensure seaborn import fails by not providing it
    monkeypatch.setitem(sys.modules, "numpy", fake_np)
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    out = tmp_path / "heatmap_fb.png"
    viz = VisualizationGenerator(sample_report)
    viz.generate_heatmap(out)
    assert out.exists()
    assert out.read_bytes().startswith(b"PNG")


def test_generate_time_series_with_history_stub(
    monkeypatch, sample_report, tmp_path: Path
):
    import types

    class _FakePLT(types.ModuleType):
        def __init__(self):
            super().__init__("matplotlib.pyplot")

        def figure(self, *args, **kwargs):
            return None

        def plot(self, *args, **kwargs):
            return None

        def title(self, *args, **kwargs):
            return None

        def ylabel(self, *args, **kwargs):
            return None

        def xlabel(self, *args, **kwargs):
            return None

        def tight_layout(self, *args, **kwargs):
            return None

        def savefig(self, path):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"PNG")

    fake_plt = _FakePLT()
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_matplotlib.pyplot = fake_plt  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)

    out = tmp_path / "ts.png"
    viz = VisualizationGenerator(sample_report)
    viz.generate_time_series(out, metric="accuracy", history=[(1, 0.5), (2, 0.9)])
    assert out.exists()
    assert out.read_bytes().startswith(b"PNG")


def test_interactive_methods_with_plotly_stub(
    monkeypatch, sample_report, tmp_path: Path
):
    import types

    class _FakeDF:
        def __init__(self, rows):
            self._rows = rows

        def pivot(self, index: str, columns: str, values: str):
            return {"_pivot": True}

    class _FakePD(types.ModuleType):
        def __init__(self):
            super().__init__("pandas")

        def DataFrame(self, rows):
            return _FakeDF(rows)

    class _FakeFig:
        def write_html(self, path, include_plotlyjs="cdn"):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("<html></html>")

    class _FakePX(types.ModuleType):
        def __init__(self):
            super().__init__("plotly.express")

        def bar(self, *args, **kwargs):
            return _FakeFig()

        def imshow(self, *args, **kwargs):
            return _FakeFig()

        def line_polar(self, *args, **kwargs):
            return _FakeFig()

        def line(self, *args, **kwargs):
            return _FakeFig()

    fake_pd = _FakePD()
    fake_px = _FakePX()
    fake_plotly = types.ModuleType("plotly")
    fake_plotly.express = fake_px  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "pandas", fake_pd)
    monkeypatch.setitem(sys.modules, "plotly", fake_plotly)
    monkeypatch.setitem(sys.modules, "plotly.express", fake_px)

    viz = VisualizationGenerator(sample_report)

    out1 = tmp_path / "mc.html"
    viz.generate_metric_comparison_interactive(out1)
    assert out1.exists()
    assert "<html>" in out1.read_text()

    out2 = tmp_path / "hm.html"
    viz.generate_heatmap_interactive(out2)
    assert out2.exists()
    assert "<html>" in out2.read_text()

    out3 = tmp_path / "radar.html"
    viz.generate_radar_chart_interactive(out3)
    assert out3.exists()
    assert "<html>" in out3.read_text()

    out4 = tmp_path / "ts.html"
    viz.generate_time_series_interactive(
        out4, metric="accuracy", history=[("a", 0.1), ("b", 0.2)]
    )
    assert out4.exists()
    assert "<html>" in out4.read_text()
