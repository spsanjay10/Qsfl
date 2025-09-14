"""
Federated Learning Interfaces

Defines abstract base classes and interfaces for federated learning operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class GlobalModel:
    """Global model data structure."""
    model_id: str
    round_id: str
    weights: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class TrainingRound:
    """Training round data structure."""
    round_id: str
    participants: List[str]
    global_model_hash: str
    aggregation_method: str
    security_events: List[str]
    metrics: Dict[str, float]
    started_at: datetime
    completed_at: Optional[datetime] = None


class IFederatedLearningServer(ABC):
    """Interface for federated learning server operations."""
    
    @abstractmethod
    def start_training_round(self) -> str:
        """Start a new training round.
        
        Returns:
            Round ID for the new training round
        """
        pass
    
    @abstractmethod
    def receive_client_update(self, client_id: str, update) -> bool:
        """Receive and validate a client update.
        
        Args:
            client_id: Client identifier
            update: Model update from client
            
        Returns:
            True if update accepted, False if rejected
        """
        pass
    
    @abstractmethod
    def aggregate_updates(self, round_id: str) -> GlobalModel:
        """Aggregate client updates into global model.
        
        Args:
            round_id: Training round identifier
            
        Returns:
            New global model
        """
        pass
    
    @abstractmethod
    def distribute_global_model(self, model: GlobalModel) -> None:
        """Distribute global model to clients.
        
        Args:
            model: Global model to distribute
        """
        pass
    
    @abstractmethod
    def get_current_model(self) -> Optional[GlobalModel]:
        """Get the current global model.
        
        Returns:
            Current global model or None if not available
        """
        pass


class IModelAggregator(ABC):
    """Interface for model aggregation operations."""
    
    @abstractmethod
    def aggregate(self, updates: List, weights: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Aggregate model updates into a single model.
        
        Args:
            updates: List of client model updates
            weights: Optional client weights for aggregation
            
        Returns:
            Aggregated model weights
        """
        pass
    
    @abstractmethod
    def set_aggregation_method(self, method: str) -> None:
        """Set the aggregation method to use."""
        pass