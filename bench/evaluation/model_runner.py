"""Model runner implementation for MEDDSAI benchmark."""
from typing import List, Dict, Any, Optional, Union, Callable
import logging
from pathlib import Path
import importlib

logger = logging.getLogger(__name__)

class ModelRunner:
    """Handles model loading and inference for evaluation."""
    
    def __init__(self):
        self._models = {}
    
    def load_model(self, model_id: str, model_path: Optional[str] = None, 
                 model_type: str = 'huggingface', **kwargs) -> Any:
        """Load a model by ID and type.
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to model files (if loading from disk)
            model_type: Type of model ('huggingface', 'local', 'api')
            **kwargs: Additional arguments specific to the model type
            
        Returns:
            Loaded model object
            
        Raises:
            ValueError: If model type is not supported
            ImportError: If required dependencies are missing
        """
        if model_id in self._models:
            return self._models[model_id]
            
        if model_type == 'huggingface':
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
                
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path or model_id, 
                    **kwargs.get('model_kwargs', {})
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path or model_id,
                    **kwargs.get('tokenizer_kwargs', {})
                )
                
                model_obj = pipeline(
                    'text-classification',
                    model=model,
                    tokenizer=tokenizer,
                    device=kwargs.get('device', -1)  # -1 for CPU, 0 for GPU
                )
                
            except ImportError as e:
                raise ImportError(
                    "HuggingFace transformers not installed. "
                    "Install with: pip install transformers"
                ) from e
                
        elif model_type == 'local':
            # For loading custom local models
            if not model_path:
                raise ValueError("model_path must be provided for local models")
                
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
                
            # This is a placeholder - implement custom model loading logic
            try:
                # Try to import model loading function from a module
                module_path = kwargs.get('module_path')
                load_func_name = kwargs.get('load_func', 'load_model')
                
                if module_path:
                    module = importlib.import_module(module_path)
                    load_func = getattr(module, load_func_name)
                    model_obj = load_func(model_path, **kwargs)
                else:
                    # Default to a simple pickle load if no module specified
                    import pickle
                    with open(model_path, 'rb') as f:
                        model_obj = pickle.load(f)
                        
            except Exception as e:
                raise RuntimeError(f"Failed to load local model: {str(e)}") from e
                
        elif model_type == 'api':
            # For API-based models (e.g., OpenAI, Cohere)
            model_obj = {
                'model_id': model_id,
                'api_key': kwargs.get('api_key'),
                'endpoint': kwargs.get('endpoint'),
                'headers': kwargs.get('headers', {})
            }
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self._models[model_id] = model_obj
        return model_obj
    
    def run_model(
        self, 
        model_id: str, 
        inputs: List[Dict[str, Any]],
        batch_size: int = 8,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Run inference on a batch of inputs.
        
        Args:
            model_id: ID of the loaded model
            inputs: List of input dictionaries
            batch_size: Number of inputs to process in a batch
            **kwargs: Additional arguments for model inference
            
        Returns:
            List of model outputs
        """
        if model_id not in self._models:
            raise ValueError(f"Model {model_id} not loaded. Call load_model() first.")
            
        model = self._models[model_id]
        results = []
        
        # Process inputs in batches
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            try:
                if isinstance(model, dict) and 'endpoint' in model:
                    # API-based model
                    batch_results = self._call_api_model(model, batch, **kwargs)
                else:
                    # Local or HuggingFace model
                    batch_results = self._call_local_model(model, batch, **kwargs)
                
                results.extend(batch_results)
                
            except Exception as e:
                logger.error(f"Error running model {model_id} on batch {i//batch_size}: {str(e)}")
                # Add None for failed predictions to maintain order
                results.extend([None] * len(batch))
        
        return results
    
    def _call_local_model(self, model, batch: List[Dict], **kwargs) -> List[Dict]:
        """Helper method to run inference with local models."""
        # Extract text inputs from batch
        texts = [item.get('text', '') for item in batch]
        
        # Run model prediction
        predictions = model(texts, **kwargs)
        
        # Format results
        if not isinstance(predictions, list):
            predictions = [predictions]
            
        return [{'prediction': pred} for pred in predictions]
    
    def _call_api_model(self, model_config: Dict, batch: List[Dict], **kwargs) -> List[Dict]:
        """Helper method to call API-based models."""
        import requests
        
        # Prepare API request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {model_config['api_key']}",
            **model_config.get('headers', {})
        }
        
        # Format inputs for the API
        payload = {
            'inputs': batch,
            **kwargs
        }
        
        # Make the request
        response = requests.post(
            model_config['endpoint'],
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"API request failed: {response.status_code} - {response.text}")
            
        return response.json()
    
    def unload_model(self, model_id: str):
        """Unload a model to free up resources."""
        if model_id in self._models:
            del self._models[model_id]
