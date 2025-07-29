"""Metric calculator implementation for MEDDSAI benchmark."""
from typing import Dict, List, Any, Callable, Optional, Union
import logging
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, mean_squared_error,
    mean_absolute_error, r2_score
)

logger = logging.getLogger(__name__)

@dataclass
class MetricResult:
    """Container for metric calculation results."""
    metric_name: str
    value: float
    metadata: Dict[str, Any] = None

class MetricCalculator:
    """Calculates evaluation metrics for model predictions."""
    
    # Built-in metrics registry
    _METRICS_REGISTRY = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'roc_auc': roc_auc_score,
        'average_precision': average_precision_score,
        'mse': mean_squared_error,
        'mae': mean_absolute_error,
        'r2': r2_score,
    }
    
    def __init__(self):
        self.metrics = {}
        self._register_builtin_metrics()
    
    def _register_builtin_metrics(self):
        """Register all built-in metrics."""
        for name, func in self._METRICS_REGISTRY.items():
            self.register_metric(name, func)
    
    def register_metric(self, name: str, metric_func: Callable, **default_kwargs):
        """Register a custom metric function.
        
        Args:
            name: Name to register the metric under
            metric_func: Function that calculates the metric
            **default_kwargs: Default arguments to pass to the metric function
        """
        self.metrics[name] = {
            'function': metric_func,
            'default_kwargs': default_kwargs
        }
    
    def calculate_metrics(
        self,
        task_id: str,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        metric_names: Optional[List[str]] = None,
        **metric_kwargs
    ) -> Dict[str, MetricResult]:
        """Calculate metrics for the given predictions and references.
        
        Args:
            task_id: ID of the task being evaluated
            predictions: List of model predictions
            references: List of reference/ground truth values
            metric_names: List of metric names to calculate (None for all registered metrics)
            **metric_kwargs: Additional arguments to pass to metric functions
            
        Returns:
            Dictionary mapping metric names to MetricResult objects
        """
        if not predictions or not references:
            raise ValueError("Predictions and references must not be empty")
            
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        # Filter metrics if specific ones are requested
        metrics_to_use = self.metrics
        if metric_names is not None:
            metrics_to_use = {name: self.metrics[name] for name in metric_names 
                            if name in self.metrics}
        
        results = {}
        
        for metric_name, metric_info in metrics_to_use.items():
            try:
                # Get metric function and default kwargs
                metric_func = metric_info['function']
                default_kwargs = metric_info.get('default_kwargs', {})
                
                # Merge default kwargs with any provided kwargs
                kwargs = {**default_kwargs, **metric_kwargs.get(metric_name, {})}
                
                # Extract relevant data for this metric
                y_true, y_pred = self._prepare_data_for_metric(
                    metric_name, predictions, references, **kwargs
                )
                
                # Calculate the metric
                if y_true is not None and y_pred is not None:
                    metric_value = metric_func(y_true, y_pred, **kwargs)
                    
                    # Create result object
                    results[metric_name] = MetricResult(
                        metric_name=metric_name,
                        value=float(metric_value) if isinstance(metric_value, (int, float, np.number)) else metric_value,
                        metadata={
                            'task_id': task_id,
                            'num_samples': len(y_true)
                        }
                    )
                
            except Exception as e:
                logger.warning(
                    f"Error calculating metric '{metric_name}': {str(e)}",
                    exc_info=True
                )
                results[metric_name] = MetricResult(
                    metric_name=metric_name,
                    value=float('nan'),
                    metadata={
                        'error': str(e),
                        'task_id': task_id
                    }
                )
        
        return results
    
    def _prepare_data_for_metric(
        self,
        metric_name: str,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]],
        **kwargs
    ) -> tuple:
        """Prepare data for metric calculation.
        
        Args:
            metric_name: Name of the metric being calculated
            predictions: List of prediction dictionaries
            references: List of reference dictionaries
            **kwargs: Additional arguments that might be needed for data preparation
            
        Returns:
            Tuple of (y_true, y_pred) for the metric function
        """
        # Default behavior: extract 'label' from references and handle both 'label' and 'prediction' from predictions
        try:
            # Extract true labels from references
            y_true = [ref.get('label') for ref in references]
            
            # Extract predictions - try 'label' first, then fall back to 'prediction'
            y_pred = []
            for pred in predictions:
                if isinstance(pred, dict):
                    if 'label' in pred:
                        y_pred.append(pred['label'])
                    elif 'prediction' in pred:
                        y_pred.append(pred['prediction'])
                    else:
                        y_pred.append(None)
                else:
                    y_pred.append(pred)
            
            # Handle binary classification metrics that require probabilities
            if metric_name in ['roc_auc', 'average_precision']:
                # For binary classification, we expect probabilities in the prediction
                if all(isinstance(p, dict) and 'score' in p for p in predictions):
                    y_pred = [p.get('score') for p in predictions]
                
                # Ensure binary labels are 0/1 for scikit-learn metrics
                if len(set(y_true)) == 2 and set(y_true) != {0, 1}:
                    unique_labels = sorted(set(y_true))
                    y_true = [1 if label == unique_labels[1] else 0 for label in y_true]
            
            # Ensure all labels are numeric for classification metrics
            if metric_name not in ['roc_auc', 'average_precision'] and y_true and y_pred:
                # Convert string labels to integers if needed
                if isinstance(y_true[0], str) or isinstance(y_pred[0], str):
                    all_labels = sorted(set(y_true + y_pred))
                    label_to_int = {label: i for i, label in enumerate(all_labels)}
                    y_true = [label_to_int.get(label, -1) for label in y_true]
                    y_pred = [label_to_int.get(label, -1) for label in y_pred]
            
            return y_true, y_pred
            
        except Exception as e:
            logger.error(f"Error preparing data for metric {metric_name}: {str(e)}")
            return None, None
    
    def aggregate_metrics(
        self, 
        metric_results: List[Dict[str, MetricResult]],
        aggregation: str = 'mean'
    ) -> Dict[str, MetricResult]:
        """Aggregate metrics across multiple evaluation runs.
        
        Args:
            metric_results: List of metric result dictionaries
            aggregation: Aggregation method ('mean', 'sum', 'max', 'min')
            
        Returns:
            Dictionary of aggregated metric results
        """
        if not metric_results:
            return {}
            
        # Get all unique metric names
        all_metric_names = set()
        for result in metric_results:
            all_metric_names.update(result.keys())
        
        aggregated = {}
        
        for metric_name in all_metric_names:
            # Collect all values for this metric
            values = []
            weights = []
            metadatas = []
            
            for result in metric_results:
                if metric_name in result:
                    metric_result = result[metric_name]
                    values.append(metric_result.value)
                    weights.append(metric_result.metadata.get('num_samples', 1))
                    metadatas.append(metric_result.metadata)
            
            if not values:
                continue
                
            # Calculate aggregated value
            if aggregation == 'mean':
                if all(w == 1 for w in weights):
                    agg_value = np.mean(values)
                else:
                    agg_value = np.average(values, weights=weights)
            elif aggregation == 'sum':
                agg_value = np.sum(values)
            elif aggregation == 'max':
                agg_value = np.max(values)
            elif aggregation == 'min':
                agg_value = np.min(values)
            else:
                raise ValueError(f"Unsupported aggregation: {aggregation}")
            
            # Create aggregated metadata
            agg_metadata = {
                'aggregation': aggregation,
                'num_runs': len(values),
                'total_samples': sum(weights),
                'min_value': float(np.min(values)),
                'max_value': float(np.max(values)),
                'std_dev': float(np.std(values)) if len(values) > 1 else 0.0,
                'task_ids': list({m.get('task_id', 'unknown') for m in metadatas})
            }
            
            aggregated[metric_name] = MetricResult(
                metric_name=metric_name,
                value=float(agg_value),
                metadata=agg_metadata
            )
        
        return aggregated
