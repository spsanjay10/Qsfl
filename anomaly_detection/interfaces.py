"""
Anomaly Detection Interfaces

Defines abstract base classes and interfaces for anomaly detection operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any
import numpy as np


class ResponseAction(Enum):
    """Possible response actions for detected anomalies."""
    ALLOW = "allow"
    REDUCE_WEIGHT = "reduce_weight"
    QUARANTINE = "quarantine"
    REJECT = "reject"


@dataclass
class ModelUpdate:
    """Model update data structure."""
    client_id: str
    round_id: str
    weights: Dict[str, np.ndarray]
    signature: bytes
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class AnomalyReport:
    """Anomaly detection report."""
    client_id: str
    anomaly_score: float
    shap_values: Dict[str, float]
    explanation: str
    recommended_action: ResponseAction
    timestamp: datetime


class IAnomalyDetector(ABC):
    """Interface for anomaly detection operations."""
    
    @abstractmethod
    def fit(self, normal_updates: List[ModelUpdate]) -> None:
        """Train the anomaly detector on normal updates.
        
        Args:
            normal_updates: List of known normal model updates
        """
        pass
    
    @abstractmethod
    def predict_anomaly_score(self, update: ModelUpdate) -> float:
        """Predict anomaly score for a model update.
        
        Args:
            update: Model update to score
            
        Returns:
            Anomaly score (higher = more anomalous)
        """
        pass
    
    @abstractmethod
    def explain_anomaly(self, update: ModelUpdate) -> Dict[str, float]:
        """Generate explanation for anomaly score.
        
        Args:
            update: Model update to explain
            
        Returns:
            Dictionary of feature importance scores
        """
        pass
    
    @abstractmethod
    def update_model(self, new_updates: List[ModelUpdate]) -> None:
        """Update the detector with new training data.
        
        Args:
            new_updates: New updates to incorporate
        """
        pass


class IFeatureExtractor(ABC):
    """Interface for extracting features from model updates."""
    
    @abstractmethod
    def extract_features(self, update: ModelUpdate) -> np.ndarray:
        """Extract feature vector from model update.
        
        Args:
            update: Model update to process
            
        Returns:
            Feature vector as numpy array
        """
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features.
        
        Returns:
            List of feature names
        """
        pass


class IExplainer(ABC):
    """Interface for generating explanations of anomaly scores."""
    
    @abstractmethod
    def explain(self, update: ModelUpdate, anomaly_score: float) -> Dict[str, float]:
        """Generate explanation for anomaly score.
        
        Args:
            update: Model update that was scored
            anomaly_score: Computed anomaly score
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass


class IReputationManager(ABC):
    """Interface for managing client reputation scores."""
    
    @abstractmethod
    def update_reputation(self, client_id: str, anomaly_score: float) -> None:
        """Update client reputation based on anomaly score."""
        pass
    
    @abstractmethod
    def get_reputation(self, client_id: str) -> float:
        """Get current reputation score for client."""
        pass
    
    @abstractmethod
    def get_influence_weight(self, client_id: str) -> float:
        """Get influence weight based on reputation."""
        pass
    
    @abstractmethod
    def is_quarantined(self, client_id: str) -> bool:
        """Check if client is quarantined."""
        pass