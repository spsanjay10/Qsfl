"""
Model Aggregator Implementation
"""

import numpy as np
from typing import Dict, List, Optional
from .interfaces import IModelAggregator


class ModelAggregator(IModelAggregator):
    """Model aggregator for federated learning."""
    
    def __init__(self):
        """Initialize the model aggregator."""
        self.aggregation_method = "federated_averaging"
    
    def aggregate(self, updates: List, weights: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Aggregate model updates into a single model."""
        if not updates:
            return {}
        
        # Mock aggregation for demo
        aggregated = {}
        
        # Get layer names from first update
        if hasattr(updates[0], 'weights'):
            layer_names = updates[0].weights.keys()
            
            for layer_name in layer_names:
                # Simple averaging for demo
                layer_updates = [update.weights[layer_name] for update in updates if hasattr(update, 'weights')]
                if layer_updates:
                    aggregated[layer_name] = np.mean(layer_updates, axis=0)
        
        return aggregated
    
    def set_aggregation_method(self, method: str) -> None:
        """Set the aggregation method to use."""
        self.aggregation_method = method