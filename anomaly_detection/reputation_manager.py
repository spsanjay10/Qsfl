"""
Client Reputation Manager Implementation
"""

import logging
from typing import Dict
from .interfaces import IReputationManager

logger = logging.getLogger(__name__)


class ClientReputationManager(IReputationManager):
    """Manager for client reputation scores and influence weights."""
    
    def __init__(self, reputation_decay: float = 0.95, quarantine_threshold: float = 0.3):
        """Initialize the reputation manager."""
        self.reputation_scores = {}
        self.reputation_decay = reputation_decay
        self.quarantine_threshold = quarantine_threshold
    
    def update_reputation(self, client_id: str, anomaly_score: float) -> None:
        """Update client reputation based on anomaly score."""
        if client_id not in self.reputation_scores:
            self.reputation_scores[client_id] = 1.0
        
        current_reputation = self.reputation_scores[client_id]
        
        if anomaly_score > 0.5:  # Anomalous behavior
            # Decrease reputation more aggressively for high anomaly scores
            penalty = min(0.5, anomaly_score)
            new_reputation = current_reputation * (1 - penalty)
        else:
            # Slowly increase reputation for good behavior
            new_reputation = min(1.0, current_reputation * 1.01)
        
        self.reputation_scores[client_id] = new_reputation
        
        logger.debug(f"Updated reputation for {client_id}: {current_reputation:.3f} -> {new_reputation:.3f}")
    
    def get_reputation(self, client_id: str) -> float:
        """Get current reputation score for client."""
        return self.reputation_scores.get(client_id, 1.0)
    
    def get_influence_weight(self, client_id: str) -> float:
        """Get influence weight based on reputation."""
        reputation = self.get_reputation(client_id)
        
        # Quarantined clients have zero influence
        if self.is_quarantined(client_id):
            return 0.0
        
        # Linear mapping of reputation to influence weight
        return max(0.0, reputation)
    
    def is_quarantined(self, client_id: str) -> bool:
        """Check if client is quarantined."""
        reputation = self.get_reputation(client_id)
        return reputation < self.quarantine_threshold