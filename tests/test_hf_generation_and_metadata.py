import types
from typing import Any, Dict, List

import pytest


class DummyConfig:
    def __init__(self) -> None:
        self.model_type = "dummy"
        self.vocab_size = 100
        self.max_position_embeddings = 64
        self.num_hidden_layers = 2
        self.hidden_size = 16
        self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        self.label2id = {"NEGATIVE": 0, "POSITIVE": 1}


class DummyParam:
    def __init__(self, n: int = 10) -> None:
        self._n = n
        self.dtype = "float32"
        self.device = "cpu"

    def numel(self) -> int:
        return self._n


class DummyModel:
    def __init__(self) -> None:
        self.config = DummyConfig()
        self._params = [DummyParam(5), DummyParam(7)]

    def parameters(self):  # noqa: D401
        """Yield dummy parameters with numel/dtype/device."""
        for p in self._params:
            yield p

    def eval(self) -> None:
        return None


class DummyTokenizer:
    pass


class DummyPipe:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Simulate underlying model and tokenizer (if any)
        self.model = DummyModel()
        self.tokenizer = DummyTokenizer()
        self.last_args = None
        self.last_kwargs: Dict[str, Any] = {}
        self.task = kwargs.get("task") or (args[0] if args else "")

    def __call__(self, inputs: List[str], **kwargs: Any):
        self.last_args = inputs
        self.last_kwargs = kwargs
        # Return shapes that ModelRunner expects per task
        if self.task == "summarization":
            return [{"summary_text": f"sum: {s[:8]}"} for s in inputs]
        elif self.task == "text-generation":
            return [{"generated_text": f"gen: {s[:8]}"} for s in inputs]
        else:
            return [{"label": "POSITIVE", "score": 0.99} for _ in inputs]


@pytest.fixture
def monkeypatched_transformers(monkeypatch):
    # Provide a minimal 'transformers' API surface required by ModelRunner
    dummy_module = types.SimpleNamespace()

    class _AutoModelForSeq2SeqLM:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any):
            return DummyModel()

    class _AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any):
            return DummyModel()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any):
            return DummyTokenizer()

    def _pipeline(task: str, *args: Any, **kwargs: Any):
        # propagate task down to DummyPipe for branching
        return DummyPipe(task=task, *args, **kwargs)

    dummy_module.AutoModelForSeq2SeqLM = _AutoModelForSeq2SeqLM
    dummy_module.AutoModelForCausalLM = _AutoModelForCausalLM
    dummy_module.AutoTokenizer = _AutoTokenizer
    dummy_module.pipeline = _pipeline

    # Inject into sys.modules so import paths in ModelRunner resolve
    monkeypatch.setitem(__import__("sys").modules, "transformers", dummy_module)
    return dummy_module


def test_hf_generation_kwargs_pass_through(monkeypatched_transformers):
    from bench.evaluation.model_runner import ModelRunner

    mr = ModelRunner()
    gen_kwargs = {"max_new_tokens": 5, "temperature": 0.7}

    # Load summarization model using our dummy transformers
    mr.load_model(
        model_name="dummy-sum",
        model_type="huggingface",
        model_path="dummy/summarizer",
        hf_task="summarization",
        generation_kwargs=gen_kwargs,
        model_kwargs={"use_cache": True},
        tokenizer_kwargs={"use_fast": True},
        device=-1,
    )

    inputs = [{"document": "This is a long note."}, {"text": "Another doc."}]
    outputs = mr.run_model("dummy-sum", inputs, batch_size=2)

    # Validate outputs and that kwargs were forwarded into pipeline call
    assert len(outputs) == 2
    assert all("summary" in o for o in outputs)

    # Access our DummyPipe through internal cache
    pipe = mr._models["dummy-sum"]
    assert isinstance(pipe, DummyPipe)
    assert pipe.last_kwargs == gen_kwargs


def test_hf_metadata_extraction_pipeline_only(monkeypatch):
    """Ensure metadata is extracted when using pipeline-only path (no AutoModel)."""
    from bench.evaluation.model_runner import ModelRunner

    # Create a transformers module that does NOT expose AutoModel classes to force pipeline-only
    dummy_module = types.SimpleNamespace()

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any):
            return DummyTokenizer()

    def _pipeline(task: str, *args: Any, **kwargs: Any):
        return DummyPipe(task=task, *args, **kwargs)

    dummy_module.AutoTokenizer = _AutoTokenizer
    dummy_module.pipeline = _pipeline

    monkeypatch.setitem(__import__("sys").modules, "transformers", dummy_module)

    mr = ModelRunner()
    mr.load_model(
        model_name="dummy-feat",
        model_type="huggingface",
        model_path="dummy/feature",
        hf_task="feature-extraction",  # not handled => AutoModel remains None
        generation_kwargs={"max_new_tokens": 3},
        device=-1,
    )

    cfg = mr._model_configs["dummy-feat"]
    # Basic fields
    assert cfg["type"] == "huggingface"
    assert cfg["path"] == "dummy/feature"
    assert cfg["hf_task"] == "feature-extraction"
    # Generation kwargs saved even if not used by this task
    assert cfg["generation_kwargs"] == {"max_new_tokens": 3}

    # Metadata pulled from DummyModel().config and parameters
    assert cfg.get("model_type_name") == "dummy"
    assert cfg.get("vocab_size") == 100
    assert cfg.get("num_parameters") == 12  # 5 + 7 from DummyParam
    # dtype/device stored as strings
    assert isinstance(cfg.get("dtype"), str)
    assert cfg.get("model_device") == "cpu"
