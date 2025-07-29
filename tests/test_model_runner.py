"""Tests for the ModelRunner component."""
import pytest
from unittest.mock import MagicMock, patch
from bench.evaluation.model_runner import ModelRunner

class MockModel:
    """A mock model for testing purposes."""
    def __call__(self, texts, **kwargs):
        return [{"label": "entailment", "score": 0.9} for _ in texts]

@pytest.fixture
def mock_model():
    """Create a mock model instance for testing."""
    return MockModel()

@pytest.fixture
def model_runner():
    """Create a ModelRunner instance for testing."""
    return ModelRunner()

def test_load_local_model(model_runner, mock_model, tmp_path):
    """Test loading a local model."""
    # Create a dummy model file
    model_path = tmp_path / "test_model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}")
    
    with patch('importlib.import_module') as mock_import:
        mock_module = MagicMock()
        mock_module.load_model.return_value = mock_model
        mock_import.return_value = mock_module
        
        model = model_runner.load_model(
            "test_model",
            model_path=str(model_path),
            model_type="local",
            module_path="test_module"
        )
        
        assert model == mock_model
        mock_import.assert_called_once_with("test_module")
        # Check that load_model was called with the correct path and an empty dict for kwargs
        args, kwargs = mock_module.load_model.call_args
        assert str(model_path) in str(args[0])  # Check path is in the args
        assert kwargs == {}

def test_run_model(model_runner, mock_model):
    """Test running a model on a batch of inputs."""
    # Register the mock model
    model_runner._models["test_model"] = mock_model
    
    # Configure the mock to return a specific result
    expected_results = [
        {"label": "entailment", "score": 0.9},
        {"label": "neutral", "score": 0.8}
    ]
    mock_model.return_value = expected_results
    
    # Test inputs
    inputs = [
        {"premise": "The patient has a fever.", "hypothesis": "The patient is sick."},
        {"premise": "The patient has no symptoms.", "hypothesis": "The patient is healthy."}
    ]
    
    # Run the model
    results = model_runner.run_model("test_model", inputs, batch_size=2)
    
    # Check results
    assert len(results) == 2
    assert all("label" in result for result in results)
    assert all("score" in result for result in results)
    assert results == expected_results

def test_run_model_with_nonexistent_model(model_runner):
    """Test running a model that hasn't been loaded raises an error."""
    with pytest.raises(ValueError, match="Model nonexistent_model not loaded"):
        model_runner.run_model("nonexistent_model", [])

def test_unload_model(model_runner, mock_model):
    """Test unloading a model."""
    model_runner._models["test_model"] = mock_model
    assert "test_model" in model_runner._models
    
    model_runner.unload_model("test_model")
    assert "test_model" not in model_runner._models

@patch('requests.post')
def test_api_model(mock_post, model_runner):
    """Test running an API-based model."""
    # Mock API response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"label": "entailment", "score": 0.95}]
    mock_post.return_value = mock_response
    
    # Load API model
    model_runner.load_model(
        "api_model",
        model_type="api",
        api_key="test_key",
        endpoint="https://api.example.com/predict"
    )
    
    # Test input
    test_input = [{"text": "test"}]
    
    # Run inference
    results = model_runner.run_model("api_model", test_input)
    
    # Check that requests.post was called with the correct arguments
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs['url'] == "https://api.example.com/predict"
    assert kwargs['headers']['Authorization'] == "Bearer test_key"
    assert kwargs['json'] == test_input
    
    # Check results
    assert len(results) == 1
    assert results[0]["label"] == "entailment"
    assert results[0]["score"] == 0.95
