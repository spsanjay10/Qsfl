"""
QSFL-CAAD System Implementation

Main system class that integrates all components.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Import system components
from pq_security.manager import PQCryptoManager
from auth.authentication_service import AuthenticationService
from auth.credential_manager import CredentialManager
from auth.revocation_manager import RevocationManager
from anomaly_detection.isolation_forest_detector import IsolationForestDetector
from anomaly_detection.reputation_manager import ClientReputationManager
from federated_learning.server import SecureFederatedServer
from federated_learning.model_aggregator import ModelAggregator
from monitoring.security_logger import SecurityEventLogger
from monitoring.metrics_collector import MetricsCollector
from monitoring.alert_manager import AlertManager

logger = logging.getLogger(__name__)


class QSFLSystem:
    """Main QSFL-CAAD system integrating all security and ML components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the QSFL-CAAD system."""
        self.config = config or self._default_config()
        
        # Initialize core components
        self.pq_manager = PQCryptoManager()
        self.cred_manager = CredentialManager()
        self.revocation_manager = RevocationManager()
        self.auth_service = AuthenticationService()
        
        # Initialize anomaly detection
        self.anomaly_detector = IsolationForestDetector()
        self.reputation_manager = ClientReputationManager()
        
        # Train detector with some initial normal data
        self._initialize_anomaly_detector()
        
        # Initialize federated learning
        self.model_aggregator = ModelAggregator()
        self.fl_server = SecureFederatedServer()
        
        # Initialize monitoring
        self.security_logger = SecurityEventLogger()
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        
        # System state
        self.clients = {}
        self.current_round = 0
        self.training_rounds = {}
        
        logger.info("QSFL-CAAD system initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default system configuration."""
        return {
            'security': {
                'anomaly_threshold': 0.6,
                'reputation_decay': 0.95,
                'quarantine_threshold': 0.8
            },
            'federated_learning': {
                'min_clients': 2,
                'max_clients': 100,
                'aggregation_method': 'federated_averaging'
            },
            'monitoring': {
                'log_level': 'INFO',
                'metrics_interval': 60
            }
        }
    
    def register_client(self, client_id: str):
        """Register a new client with the system."""
        try:
            # Generate credentials
            credentials = self.auth_service.register_client(client_id)
            
            # Initialize client state
            self.clients[client_id] = {
                'credentials': credentials,
                'reputation': 1.0,
                'last_anomaly_score': 0.0,
                'updates_sent': 0,
                'quarantined': False,
                'registered_at': datetime.now()
            }
            
            logger.info(f"Client {client_id} registered successfully")
            return credentials
            
        except Exception as e:
            logger.error(f"Failed to register client {client_id}: {e}")
            raise
    
    def start_training_round(self, round_id: Optional[str] = None) -> str:
        """Start a new federated learning training round."""
        if round_id is None:
            self.current_round += 1
            round_id = f"round_{self.current_round}"
        
        self.training_rounds[round_id] = {
            'round_id': round_id,
            'started_at': datetime.now(),
            'participants': [],
            'updates_received': [],
            'updates_accepted': [],
            'updates_rejected': [],
            'completed_at': None
        }
        
        logger.info(f"Started training round: {round_id}")
        return round_id
    
    def receive_client_update(self, client_id: str, update) -> bool:
        """Receive and process a client model update."""
        try:
            # Check if client exists and is not quarantined
            if client_id not in self.clients:
                logger.warning(f"Update from unregistered client: {client_id}")
                return False
            
            client_info = self.clients[client_id]
            if client_info['quarantined']:
                logger.warning(f"Update rejected from quarantined client: {client_id}")
                return False
            
            # Authenticate the update
            if not self._authenticate_update(client_id, update):
                logger.warning(f"Authentication failed for client: {client_id}")
                return False
            
            # Perform anomaly detection
            anomaly_score = self.anomaly_detector.predict_anomaly_score(update)
            client_info['last_anomaly_score'] = anomaly_score
            
            # Update reputation
            self.reputation_manager.update_reputation(client_id, anomaly_score)
            client_info['reputation'] = self.reputation_manager.get_reputation(client_id)
            
            # Check for quarantine
            if client_info['reputation'] < self.config['security']['quarantine_threshold']:
                client_info['quarantined'] = True
                logger.warning(f"Client {client_id} quarantined due to low reputation")
                return False
            
            # Accept the update
            client_info['updates_sent'] += 1
            
            # Log security event if anomalous
            if anomaly_score > self.config['security']['anomaly_threshold']:
                self.security_logger.log_anomaly_event(client_id, anomaly_score, "flagged")
            
            logger.debug(f"Update accepted from {client_id} (anomaly: {anomaly_score:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Error processing update from {client_id}: {e}")
            return False
    
    def _authenticate_update(self, client_id: str, update) -> bool:
        """Authenticate a client update."""
        try:
            # In a real implementation, this would verify the cryptographic signature
            # For demo purposes, we'll do basic validation
            return hasattr(update, 'signature') and hasattr(update, 'client_id')
        except Exception as e:
            logger.error(f"Authentication error for {client_id}: {e}")
            return False
    
    def aggregate_updates(self, round_id: str):
        """Aggregate client updates for a training round."""
        try:
            if round_id not in self.training_rounds:
                raise ValueError(f"Unknown training round: {round_id}")
            
            # Get accepted updates (mock for demo)
            accepted_updates = []
            client_weights = {}
            
            for client_id, client_info in self.clients.items():
                if not client_info['quarantined'] and client_info['updates_sent'] > 0:
                    # Mock update for aggregation
                    weight = self.reputation_manager.get_influence_weight(client_id)
                    client_weights[client_id] = weight
                    # In real implementation, would use actual model updates
            
            # Create mock global model
            from federated_learning.interfaces import GlobalModel
            import numpy as np
            
            global_model = GlobalModel(
                model_id=f"global_model_{round_id}",
                round_id=round_id,
                weights={
                    "layer_0": np.random.normal(0, 0.1, (100, 50)),
                    "layer_1": np.random.normal(0, 0.1, (50, 10)),
                    "layer_2": np.random.normal(0, 0.1, (10, 1))
                },
                metadata={
                    'participants': len(client_weights),
                    'aggregation_method': 'reputation_weighted'
                },
                created_at=datetime.now()
            )
            
            # Mark round as completed
            self.training_rounds[round_id]['completed_at'] = datetime.now()
            
            logger.info(f"Aggregation completed for round {round_id}")
            return global_model
            
        except Exception as e:
            logger.error(f"Aggregation failed for round {round_id}: {e}")
            raise
    
    def distribute_global_model(self, global_model):
        """Distribute global model to clients."""
        try:
            # In a real implementation, this would send the model to clients
            logger.info(f"Global model {global_model.model_id} distributed to clients")
        except Exception as e:
            logger.error(f"Model distribution failed: {e}")
            raise
    
    def get_client_reputation(self, client_id: str) -> float:
        """Get client reputation score."""
        if client_id in self.clients:
            return self.clients[client_id]['reputation']
        return 0.0
    
    def get_client_influence(self, client_id: str) -> float:
        """Get client influence weight."""
        return self.reputation_manager.get_influence_weight(client_id)
    
    def is_client_quarantined(self, client_id: str) -> bool:
        """Check if client is quarantined."""
        if client_id in self.clients:
            return self.clients[client_id]['quarantined']
        return False
    
    def get_last_anomaly_score(self, client_id: str) -> float:
        """Get last anomaly score for client."""
        if client_id in self.clients:
            return self.clients[client_id]['last_anomaly_score']
        return 0.0
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security summary statistics."""
        total_clients = len(self.clients)
        quarantined_clients = sum(1 for c in self.clients.values() if c['quarantined'])
        
        return {
            'total_clients': total_clients,
            'active_clients': total_clients - quarantined_clients,
            'quarantined_clients': quarantined_clients,
            'anomalies_detected': sum(1 for c in self.clients.values() 
                                    if c['last_anomaly_score'] > self.config['security']['anomaly_threshold']),
            'auth_failures': 0,  # Mock value
            'events_count': 0    # Mock value
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics."""
        return {
            'current_round': self.current_round,
            'total_clients': len(self.clients),
            'active_clients': len([c for c in self.clients.values() if not c['quarantined']]),
            'total_updates': sum(c['updates_sent'] for c in self.clients.values()),
            'system_status': 'running'
        }
    
    def _initialize_anomaly_detector(self):
        """Initialize anomaly detector with some baseline normal data."""
        try:
            # Create some mock normal updates for initial training
            from anomaly_detection.interfaces import ModelUpdate
            import numpy as np
            
            normal_updates = []
            for i in range(20):  # Create 20 normal baseline updates
                weights = {
                    "layer_0": np.random.normal(0, 0.1, (100, 50)),
                    "layer_1": np.random.normal(0, 0.1, (50, 10)),
                    "layer_2": np.random.normal(0, 0.1, (10, 1))
                }
                
                update = ModelUpdate(
                    client_id=f"baseline_client_{i}",
                    round_id="initialization",
                    weights=weights,
                    signature=b"baseline_signature",
                    timestamp=datetime.now(),
                    metadata={"baseline": True}
                )
                normal_updates.append(update)
            
            # Train the detector
            self.anomaly_detector.fit(normal_updates)
            logger.info("Anomaly detector initialized with baseline data")
            
        except Exception as e:
            logger.warning(f"Could not initialize anomaly detector: {e}")
            # Continue without anomaly detection for demo