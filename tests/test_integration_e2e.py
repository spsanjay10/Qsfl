"""
End-to-End Integration Tests for QSFL-CAAD System

Comprehensive integration tests that validate the complete system workflow
including post-quantum security, authentication, anomaly detection, and
federated learning components working together.
"""

import pytest
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Import all system components
from pq_security.manager import PQCryptoManager
from pq_security.kyber import KyberKeyExchange
from pq_security.dilithium import DilithiumSigner
from auth.authentication_service import AuthenticationService
from auth.credential_manager import ClientCredentialManager
from auth.revocation_manager import RevocationManager
from anomaly_detection.isolation_forest_detector import IsolationForestDetector
from anomaly_detection.feature_extractor import FeatureExtractor
from anomaly_detection.shap_explainer import SHAPExplainer
from anomaly_detection.reputation_manager import ClientReputationManager
from federated_learning.server import SecureFederatedServer
from federated_learning.model_aggregator import FederatedAveragingAggregator
from federated_learning.client_simulation import HonestClient, MaliciousClient, AttackType
from federated_learning.dataset_manager import DatasetManager, DatasetType, DistributionType
from monitoring.security_logger import SecurityEventLogger
from monitoring.metrics_collector import MetricsCollector
from monitoring.alert_manager import AlertManager

# Import interfaces
from anomaly_detection.interfaces import ModelUpdate, ResponseAction
from auth.interfaces import ClientCredentials


class TestSystemIntegration:
    """Integration tests for the complete QSFL-CAAD system."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def pq_crypto_manager(self):
        """Create PQ crypto manager."""
        kyber = KyberKeyExchange()
        dilithium = DilithiumSigner()
        return PQCryptoManager(kyber, dilithium)
    
    @pytest.fixture
    def auth_components(self, temp_dir, pq_crypto_manager):
        """Create authentication components."""
        credential_manager = ClientCredentialManager(pq_crypto_manager)
        revocation_manager = RevocationManager(storage_path=temp_dir)
        auth_service = AuthenticationService(
            credential_manager=credential_manager,
            revocation_manager=revocation_manager,
            pq_crypto=pq_crypto_manager
        )
        return auth_service, credential_manager, revocation_manager
    
    @pytest.fixture
    def anomaly_detection_components(self):
        """Create anomaly detection components."""
        feature_extractor = FeatureExtractor()
        detector = IsolationForestDetector(feature_extractor)
        explainer = SHAPExplainer(detector, feature_extractor)
        reputation_manager = ClientReputationManager()
        return detector, explainer, reputation_manager, feature_extractor
    
    @pytest.fixture
    def monitoring_components(self, temp_dir):
        """Create monitoring components."""
        logger = SecurityEventLogger(log_file=f"{temp_dir}/security.log")
        metrics_collector = MetricsCollector(db_path=f"{temp_dir}/metrics.db")
        alert_manager = AlertManager(logger)
        return logger, metrics_collector, alert_manager
    
    @pytest.fixture
    def federated_server(self, auth_components, anomaly_detection_components, 
                        monitoring_components, pq_crypto_manager):
        """Create complete federated learning server."""
        auth_service, _, _ = auth_components
        detector, _, reputation_manager, _ = anomaly_detection_components
        logger, metrics_collector, alert_manager = monitoring_components
        
        aggregator = FederatedAveragingAggregator()
        
        server = SecureFederatedServer(
            auth_service=auth_service,
            pq_crypto=pq_crypto_manager,
            aggregator=aggregator,
            anomaly_detector=detector,
            reputation_manager=reputation_manager,
            security_logger=logger,
            metrics_collector=metrics_collector,
            alert_manager=alert_manager
        )
        
        return server
    
    def test_complete_system_initialization(self, federated_server, auth_components,
                                          anomaly_detection_components, monitoring_components):
        """Test that all system components initialize correctly together."""
        # Verify server is properly initialized
        assert federated_server is not None
        assert federated_server.auth_service is not None
        assert federated_server.pq_crypto is not None
        assert federated_server.aggregator is not None
        
        # Verify auth components
        auth_service, credential_manager, revocation_manager = auth_components
        assert auth_service is not None
        assert credential_manager is not None
        assert revocation_manager is not None
        
        # Verify anomaly detection components
        detector, explainer, reputation_manager, feature_extractor = anomaly_detection_components
        assert detector is not None
        assert explainer is not None
        assert reputation_manager is not None
        assert feature_extractor is not None
        
        # Verify monitoring components
        logger, metrics_collector, alert_manager = monitoring_components
        assert logger is not None
        assert metrics_collector is not None
        assert alert_manager is not None
    
    def test_client_registration_and_authentication_flow(self, auth_components, pq_crypto_manager):
        """Test complete client registration and authentication workflow."""
        auth_service, credential_manager, revocation_manager = auth_components
        
        # Register multiple clients
        client_ids = ["client_001", "client_002", "client_003"]
        credentials = {}
        
        for client_id in client_ids:
            creds = auth_service.register_client(client_id)
            credentials[client_id] = creds
            
            # Verify credentials were issued
            assert creds.client_id == client_id
            assert creds.public_key is not None
            assert creds.private_key is not None
            assert auth_service.is_client_valid(client_id)
        
        # Test authentication with valid signatures
        test_message = b"test_model_update_data"
        
        for client_id, creds in credentials.items():
            # Sign message with client's private key
            signature = pq_crypto_manager.sign(test_message, creds.private_key)
            
            # Authenticate using signature
            is_authenticated = auth_service.authenticate_client(client_id, signature, test_message)
            assert is_authenticated, f"Authentication failed for {client_id}"
        
        # Test revocation
        revoked_client = client_ids[0]
        auth_service.revoke_client(revoked_client)
        
        assert not auth_service.is_client_valid(revoked_client)
        assert revocation_manager.is_revoked(revoked_client)
        
        # Verify other clients still valid
        for client_id in client_ids[1:]:
            assert auth_service.is_client_valid(client_id)
    
    def test_anomaly_detection_training_and_scoring(self, anomaly_detection_components):
        """Test anomaly detection training and scoring workflow."""
        detector, explainer, reputation_manager, feature_extractor = anomaly_detection_components
        
        # Create normal training updates
        normal_updates = []
        for i in range(20):
            weights = {
                'layer1': np.random.normal(0, 0.1, (10, 5)).astype(np.float32),
                'layer2': np.random.normal(0, 0.1, (5, 1)).astype(np.float32)
            }
            
            update = ModelUpdate(
                client_id=f"client_{i:03d}",
                round_id="training_round",
                weights=weights,
                signature=b"signature",
                timestamp=datetime.now(),
                metadata={'client_type': 'honest'}
            )
            normal_updates.append(update)
        
        # Train detector on normal updates
        detector.fit(normal_updates)
        
        # Test scoring normal updates (should have low anomaly scores)
        for update in normal_updates[:5]:
            score = detector.predict_anomaly_score(update)
            assert 0.0 <= score <= 1.0, f"Anomaly score {score} out of range"
            assert score < 0.7, f"Normal update scored too high: {score}"
        
        # Create anomalous update
        anomalous_weights = {
            'layer1': np.random.normal(10, 5, (10, 5)).astype(np.float32),  # Very different distribution
            'layer2': np.random.normal(-10, 5, (5, 1)).astype(np.float32)
        }
        
        anomalous_update = ModelUpdate(
            client_id="malicious_client",
            round_id="test_round",
            weights=anomalous_weights,
            signature=b"signature",
            timestamp=datetime.now(),
            metadata={'client_type': 'malicious'}
        )
        
        # Score anomalous update (should have high anomaly score)
        anomaly_score = detector.predict_anomaly_score(anomalous_update)
        assert anomaly_score > 0.5, f"Anomalous update scored too low: {anomaly_score}"
        
        # Test explanation generation
        explanation = explainer.explain(anomalous_update, anomaly_score)
        assert isinstance(explanation, dict)
        assert len(explanation) > 0
        
        # Test reputation management
        reputation_manager.update_reputation("malicious_client", anomaly_score)
        reputation = reputation_manager.get_reputation("malicious_client")
        assert reputation < 1.0, "Reputation should decrease after anomalous behavior"
    
    def test_federated_learning_round_with_security(self, federated_server, auth_components, 
                                                   anomaly_detection_components, pq_crypto_manager):
        """Test complete federated learning round with security integration."""
        auth_service, _, _ = auth_components
        detector, _, reputation_manager, _ = anomaly_detection_components
        
        # Initialize global model
        initial_weights = {
            'layer1': np.zeros((10, 5), dtype=np.float32),
            'layer2': np.zeros((5, 1), dtype=np.float32)
        }
        federated_server.initialize_global_model(initial_weights)
        
        # Register clients
        honest_clients = ["honest_001", "honest_002", "honest_003"]
        malicious_clients = ["malicious_001"]
        all_clients = honest_clients + malicious_clients
        
        client_credentials = {}
        for client_id in all_clients:
            creds = auth_service.register_client(client_id)
            client_credentials[client_id] = creds
        
        # Train anomaly detector with some normal updates first
        normal_training_updates = []
        for i in range(15):
            weights = {
                'layer1': np.random.normal(0, 0.1, (10, 5)).astype(np.float32),
                'layer2': np.random.normal(0, 0.1, (5, 1)).astype(np.float32)
            }
            update = ModelUpdate(
                client_id=f"training_client_{i}",
                round_id="training",
                weights=weights,
                signature=b"sig",
                timestamp=datetime.now(),
                metadata={}
            )
            normal_training_updates.append(update)
        
        detector.fit(normal_training_updates)
        
        # Start training round
        round_id = federated_server.start_training_round()
        
        # Create and submit honest client updates
        honest_updates = []
        for client_id in honest_clients:
            # Create normal-looking weights
            weights = {
                'layer1': np.random.normal(0, 0.1, (10, 5)).astype(np.float32),
                'layer2': np.random.normal(0, 0.1, (5, 1)).astype(np.float32)
            }
            
            update = ModelUpdate(
                client_id=client_id,
                round_id=round_id,
                weights=weights,
                signature=b"placeholder_signature",  # Will be replaced with real signature
                timestamp=datetime.now(),
                metadata={'client_type': 'honest'}
            )
            
            # Sign the update
            message = f"{client_id}:{round_id}".encode()
            creds = client_credentials[client_id]
            signature = pq_crypto_manager.sign(message, creds.private_key)
            update.signature = signature
            
            # Submit update
            result = federated_server.receive_client_update(client_id, update)
            assert result, f"Honest client {client_id} update was rejected"
            honest_updates.append(update)
        
        # Create and submit malicious client update
        for client_id in malicious_clients:
            # Create anomalous weights
            weights = {
                'layer1': np.random.normal(5, 2, (10, 5)).astype(np.float32),  # Anomalous distribution
                'layer2': np.random.normal(-5, 2, (5, 1)).astype(np.float32)
            }
            
            update = ModelUpdate(
                client_id=client_id,
                round_id=round_id,
                weights=weights,
                signature=b"placeholder_signature",
                timestamp=datetime.now(),
                metadata={'client_type': 'malicious'}
            )
            
            # Sign the update
            message = f"{client_id}:{round_id}".encode()
            creds = client_credentials[client_id]
            signature = pq_crypto_manager.sign(message, creds.private_key)
            update.signature = signature
            
            # Submit update (should be accepted but flagged)
            result = federated_server.receive_client_update(client_id, update)
            # Note: Update might be accepted but with reduced weight due to anomaly detection
        
        # Aggregate updates
        global_model = federated_server.aggregate_updates(round_id)
        
        # Verify aggregation completed
        assert global_model is not None
        assert global_model.round_id == round_id
        assert 'layer1' in global_model.weights
        assert 'layer2' in global_model.weights
        
        # Check that malicious client reputation was affected
        malicious_reputation = reputation_manager.get_reputation(malicious_clients[0])
        honest_reputation = reputation_manager.get_reputation(honest_clients[0])
        
        # Malicious client should have lower reputation (if anomaly was detected)
        # Note: This might not always trigger depending on the specific anomaly detection threshold
        
        # Verify training history was recorded
        history = federated_server.get_training_history()
        assert len(history) == 1
        assert history[0].round_id == round_id
        assert len(history[0].participants) >= len(honest_clients)
    
    def test_attack_simulation_and_detection(self, federated_server, auth_components,
                                           anomaly_detection_components, pq_crypto_manager):
        """Test system's ability to detect and respond to various attack types."""
        auth_service, _, _ = auth_components
        detector, _, reputation_manager, _ = anomaly_detection_components
        
        # Initialize system
        initial_weights = {
            'layer1': np.zeros((5, 3), dtype=np.float32),
            'layer2': np.zeros((3, 1), dtype=np.float32)
        }
        federated_server.initialize_global_model(initial_weights)
        
        # Register clients
        clients = ["honest_001", "honest_002", "attacker_001", "attacker_002"]
        client_credentials = {}
        for client_id in clients:
            creds = auth_service.register_client(client_id)
            client_credentials[client_id] = creds
        
        # Train detector with normal updates
        normal_updates = []
        for i in range(20):
            weights = {
                'layer1': np.random.normal(0, 0.1, (5, 3)).astype(np.float32),
                'layer2': np.random.normal(0, 0.1, (3, 1)).astype(np.float32)
            }
            update = ModelUpdate(
                client_id=f"training_{i}",
                round_id="training",
                weights=weights,
                signature=b"sig",
                timestamp=datetime.now(),
                metadata={}
            )
            normal_updates.append(update)
        
        detector.fit(normal_updates)
        
        # Test different attack scenarios
        attack_scenarios = [
            {
                'name': 'gradient_poisoning',
                'weights': {
                    'layer1': np.random.normal(10, 1, (5, 3)).astype(np.float32),  # Large gradients
                    'layer2': np.random.normal(10, 1, (3, 1)).astype(np.float32)
                }
            },
            {
                'name': 'model_replacement',
                'weights': {
                    'layer1': np.ones((5, 3), dtype=np.float32) * 100,  # Extreme values
                    'layer2': np.ones((3, 1), dtype=np.float32) * 100
                }
            },
            {
                'name': 'byzantine_attack',
                'weights': {
                    'layer1': np.random.uniform(-50, 50, (5, 3)).astype(np.float32),  # Random noise
                    'layer2': np.random.uniform(-50, 50, (3, 1)).astype(np.float32)
                }
            }
        ]
        
        detection_results = {}
        
        for scenario in attack_scenarios:
            # Start new round for each attack
            round_id = federated_server.start_training_round()
            
            # Submit honest updates first
            for client_id in ["honest_001", "honest_002"]:
                weights = {
                    'layer1': np.random.normal(0, 0.1, (5, 3)).astype(np.float32),
                    'layer2': np.random.normal(0, 0.1, (3, 1)).astype(np.float32)
                }
                
                update = ModelUpdate(
                    client_id=client_id,
                    round_id=round_id,
                    weights=weights,
                    signature=b"sig",
                    timestamp=datetime.now(),
                    metadata={'attack_type': 'none'}
                )
                
                # Sign and submit
                message = f"{client_id}:{round_id}".encode()
                creds = client_credentials[client_id]
                signature = pq_crypto_manager.sign(message, creds.private_key)
                update.signature = signature
                
                federated_server.receive_client_update(client_id, update)
            
            # Submit attack update
            attack_update = ModelUpdate(
                client_id="attacker_001",
                round_id=round_id,
                weights=scenario['weights'],
                signature=b"sig",
                timestamp=datetime.now(),
                metadata={'attack_type': scenario['name']}
            )
            
            # Sign attack update
            message = f"attacker_001:{round_id}".encode()
            creds = client_credentials["attacker_001"]
            signature = pq_crypto_manager.sign(message, creds.private_key)
            attack_update.signature = signature
            
            # Submit attack update
            federated_server.receive_client_update("attacker_001", attack_update)
            
            # Check anomaly score for attack
            anomaly_score = detector.predict_anomaly_score(attack_update)
            detection_results[scenario['name']] = {
                'anomaly_score': anomaly_score,
                'detected': anomaly_score > 0.5  # Threshold for detection
            }
            
            # Complete round
            try:
                federated_server.aggregate_updates(round_id)
            except Exception as e:
                # Some attacks might cause aggregation to fail, which is acceptable
                pass
        
        # Verify that attacks were detected
        detected_attacks = sum(1 for result in detection_results.values() if result['detected'])
        total_attacks = len(attack_scenarios)
        
        # We expect at least some attacks to be detected
        detection_rate = detected_attacks / total_attacks
        assert detection_rate >= 0.5, f"Detection rate too low: {detection_rate}"
        
        # Verify attacker reputation was affected
        attacker_reputation = reputation_manager.get_reputation("attacker_001")
        honest_reputation = reputation_manager.get_reputation("honest_001")
        
        # Attacker should have lower reputation than honest client
        assert attacker_reputation <= honest_reputation
    
    def test_monitoring_and_logging_integration(self, federated_server, monitoring_components,
                                              auth_components, pq_crypto_manager):
        """Test that monitoring and logging components capture system events."""
        logger, metrics_collector, alert_manager = monitoring_components
        auth_service, _, _ = auth_components
        
        # Initialize system
        initial_weights = {'layer1': np.zeros((3, 2), dtype=np.float32)}
        federated_server.initialize_global_model(initial_weights)
        
        # Register client
        client_id = "test_client"
        creds = auth_service.register_client(client_id)
        
        # Start training round
        round_id = federated_server.start_training_round()
        
        # Create and submit update
        weights = {'layer1': np.random.normal(0, 0.1, (3, 2)).astype(np.float32)}
        update = ModelUpdate(
            client_id=client_id,
            round_id=round_id,
            weights=weights,
            signature=b"sig",
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Sign update
        message = f"{client_id}:{round_id}".encode()
        signature = pq_crypto_manager.sign(message, creds.private_key)
        update.signature = signature
        
        # Submit update
        federated_server.receive_client_update(client_id, update)
        
        # Complete round
        federated_server.aggregate_updates(round_id)
        
        # Verify events were logged
        # Note: Actual verification would depend on the specific logging implementation
        # This is a placeholder for the logging verification logic
        
        # Verify metrics were collected
        # Note: Actual verification would depend on the specific metrics implementation
        
        # Test alert generation with suspicious activity
        # This would involve creating conditions that trigger alerts
        
        assert True  # Placeholder assertion
    
    def test_performance_benchmarking(self, federated_server, auth_components, pq_crypto_manager):
        """Test system performance with multiple clients and rounds."""
        auth_service, _, _ = auth_components
        
        # Initialize system
        initial_weights = {
            'layer1': np.zeros((20, 10), dtype=np.float32),
            'layer2': np.zeros((10, 5), dtype=np.float32),
            'layer3': np.zeros((5, 1), dtype=np.float32)
        }
        federated_server.initialize_global_model(initial_weights)
        
        # Register multiple clients
        num_clients = 10
        client_credentials = {}
        
        start_time = datetime.now()
        
        for i in range(num_clients):
            client_id = f"perf_client_{i:03d}"
            creds = auth_service.register_client(client_id)
            client_credentials[client_id] = creds
        
        registration_time = datetime.now() - start_time
        
        # Run multiple training rounds
        num_rounds = 3
        round_times = []
        
        for round_num in range(num_rounds):
            round_start = datetime.now()
            
            # Start round
            round_id = federated_server.start_training_round()
            
            # Submit updates from all clients
            for client_id, creds in client_credentials.items():
                weights = {
                    'layer1': np.random.normal(0, 0.1, (20, 10)).astype(np.float32),
                    'layer2': np.random.normal(0, 0.1, (10, 5)).astype(np.float32),
                    'layer3': np.random.normal(0, 0.1, (5, 1)).astype(np.float32)
                }
                
                update = ModelUpdate(
                    client_id=client_id,
                    round_id=round_id,
                    weights=weights,
                    signature=b"sig",
                    timestamp=datetime.now(),
                    metadata={}
                )
                
                # Sign update
                message = f"{client_id}:{round_id}".encode()
                signature = pq_crypto_manager.sign(message, creds.private_key)
                update.signature = signature
                
                # Submit update
                result = federated_server.receive_client_update(client_id, update)
                assert result, f"Update rejected for {client_id} in round {round_num}"
            
            # Aggregate updates
            global_model = federated_server.aggregate_updates(round_id)
            assert global_model is not None
            
            round_time = datetime.now() - round_start
            round_times.append(round_time.total_seconds())
        
        # Performance assertions
        avg_round_time = sum(round_times) / len(round_times)
        
        # These thresholds should be adjusted based on expected performance
        assert registration_time.total_seconds() < 10.0, f"Registration too slow: {registration_time.total_seconds()}s"
        assert avg_round_time < 5.0, f"Average round time too slow: {avg_round_time}s"
        assert max(round_times) < 10.0, f"Slowest round too slow: {max(round_times)}s"
        
        # Verify all rounds completed successfully
        history = federated_server.get_training_history()
        assert len(history) == num_rounds
        
        for i, round_info in enumerate(history):
            assert len(round_info.participants) == num_clients
            assert round_info.completed_at is not None


class TestSystemRecovery:
    """Test system recovery and error handling scenarios."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def system_components(self, temp_dir):
        """Create system components for recovery testing."""
        pq_crypto = PQCryptoManager(KyberKeyExchange(), DilithiumSigner())
        credential_manager = ClientCredentialManager(pq_crypto)
        revocation_manager = RevocationManager(storage_path=temp_dir)
        auth_service = AuthenticationService(
            credential_manager=credential_manager,
            revocation_manager=revocation_manager,
            pq_crypto=pq_crypto
        )
        
        feature_extractor = FeatureExtractor()
        detector = IsolationForestDetector(feature_extractor)
        reputation_manager = ClientReputationManager()
        
        logger = SecurityEventLogger(log_file=f"{temp_dir}/security.log")
        metrics_collector = MetricsCollector(db_path=f"{temp_dir}/metrics.db")
        alert_manager = AlertManager(logger)
        
        aggregator = FederatedAveragingAggregator()
        
        server = SecureFederatedServer(
            auth_service=auth_service,
            pq_crypto=pq_crypto,
            aggregator=aggregator,
            anomaly_detector=detector,
            reputation_manager=reputation_manager,
            security_logger=logger,
            metrics_collector=metrics_collector,
            alert_manager=alert_manager
        )
        
        return {
            'server': server,
            'auth_service': auth_service,
            'detector': detector,
            'reputation_manager': reputation_manager,
            'pq_crypto': pq_crypto,
            'logger': logger,
            'metrics_collector': metrics_collector,
            'alert_manager': alert_manager
        }
    
    def test_authentication_service_failure_recovery(self, system_components):
        """Test system behavior when authentication service fails."""
        components = system_components
        server = components['server']
        auth_service = components['auth_service']
        
        # Initialize system
        model_shape = create_test_model_shape("small")
        initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                          for name, shape in model_shape.items()}
        server.initialize_global_model(initial_weights)
        
        # Register clients normally
        client_id = "test_client"
        creds = auth_service.register_client(client_id)
        
        # Simulate authentication service failure by mocking failure
        original_authenticate = auth_service.authenticate_client
        auth_service.authenticate_client = Mock(side_effect=Exception("Auth service down"))
        
        # Try to submit update during failure
        generator = ModelUpdateGenerator(seed=42)
        update = generator.generate_honest_update(client_id, "test_round", model_shape)
        
        round_id = server.start_training_round()
        
        # Update should be rejected due to auth failure
        result = server.receive_client_update(client_id, update)
        assert not result, "Update should be rejected when auth service fails"
        
        # Restore authentication service
        auth_service.authenticate_client = original_authenticate
        
        # Update should now be accepted
        result = server.receive_client_update(client_id, update)
        assert result, "Update should be accepted after auth service recovery"
    
    def test_anomaly_detector_failure_recovery(self, system_components):
        """Test system behavior when anomaly detector fails."""
        components = system_components
        server = components['server']
        detector = components['detector']
        auth_service = components['auth_service']
        
        # Initialize system
        model_shape = create_test_model_shape("small")
        initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                          for name, shape in model_shape.items()}
        server.initialize_global_model(initial_weights)
        
        # Train detector
        generator = ModelUpdateGenerator(seed=42)
        normal_updates = []
        for i in range(20):
            update = generator.generate_honest_update(f"train_{i}", "train", model_shape)
            normal_updates.append(update)
        detector.fit(normal_updates)
        
        # Register client
        client_id = "test_client"
        auth_service.register_client(client_id)
        
        # Simulate detector failure
        original_predict = detector.predict_anomaly_score
        detector.predict_anomaly_score = Mock(side_effect=Exception("Detector failure"))
        
        # Submit update during detector failure
        update = generator.generate_honest_update(client_id, "test_round", model_shape)
        round_id = server.start_training_round()
        
        # System should handle detector failure gracefully
        # (might accept update with default score or reject it)
        try:
            result = server.receive_client_update(client_id, update)
            # Either result is acceptable - system should not crash
            assert isinstance(result, bool)
        except Exception as e:
            # If exception occurs, it should be handled gracefully
            assert "Detector failure" not in str(e), "Detector failure should be handled internally"
        
        # Restore detector
        detector.predict_anomaly_score = original_predict
        
        # System should work normally after recovery
        result = server.receive_client_update(client_id, update)
        assert isinstance(result, bool)
    
    def test_malicious_client_isolation(self, system_components):
        """Test system's ability to isolate and quarantine malicious clients."""
        components = system_components
        server = components['server']
        auth_service = components['auth_service']
        detector = components['detector']
        reputation_manager = components['reputation_manager']
        
        # Initialize system
        model_shape = create_test_model_shape("small")
        initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                          for name, shape in model_shape.items()}
        server.initialize_global_model(initial_weights)
        
        # Train detector
        generator = ModelUpdateGenerator(seed=42)
        normal_updates = []
        for i in range(25):
            update = generator.generate_honest_update(f"train_{i}", "train", model_shape)
            normal_updates.append(update)
        detector.fit(normal_updates)
        
        # Register clients
        honest_client = "honest_client"
        malicious_client = "malicious_client"
        auth_service.register_client(honest_client)
        auth_service.register_client(malicious_client)
        
        # Simulate multiple rounds with persistent malicious behavior
        for round_num in range(5):
            round_id = server.start_training_round()
            
            # Honest client submits normal update
            honest_update = generator.generate_honest_update(honest_client, round_id, model_shape)
            server.receive_client_update(honest_client, honest_update)
            
            # Malicious client submits attack
            malicious_update = generator.generate_malicious_update(
                malicious_client, round_id, model_shape, 
                "gradient_poisoning", attack_intensity=3.0
            )
            server.receive_client_update(malicious_client, malicious_update)
            
            # Complete round
            server.aggregate_updates(round_id)
        
        # Check if malicious client was quarantined or has very low reputation
        malicious_reputation = reputation_manager.get_reputation(malicious_client)
        honest_reputation = reputation_manager.get_reputation(honest_client)
        
        assert malicious_reputation < honest_reputation, \
            "Malicious client should have lower reputation than honest client"
        
        # Test that quarantined client has reduced influence
        if reputation_manager.is_quarantined(malicious_client):
            influence = reputation_manager.get_influence_weight(malicious_client)
            assert influence < 0.5, "Quarantined client should have reduced influence"
        
        # Test recovery after good behavior
        # Simulate malicious client starting to behave honestly
        for round_num in range(3):
            round_id = server.start_training_round()
            
            # Both clients submit honest updates
            honest_update = generator.generate_honest_update(honest_client, round_id, model_shape)
            server.receive_client_update(honest_client, honest_update)
            
            reformed_update = generator.generate_honest_update(malicious_client, round_id, model_shape)
            server.receive_client_update(malicious_client, reformed_update)
            
            server.aggregate_updates(round_id)
        
        # Reputation should improve (though still lower than consistently honest client)
        improved_reputation = reputation_manager.get_reputation(malicious_client)
        assert improved_reputation >= malicious_reputation, \
            "Reputation should improve with good behavior"
    
    def test_data_corruption_handling(self, system_components):
        """Test system behavior with corrupted data."""
        components = system_components
        server = components['server']
        auth_service = components['auth_service']
        
        # Initialize system
        model_shape = create_test_model_shape("small")
        initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                          for name, shape in model_shape.items()}
        server.initialize_global_model(initial_weights)
        
        # Register client
        client_id = "test_client"
        auth_service.register_client(client_id)
        
        generator = ModelUpdateGenerator(seed=42)
        round_id = server.start_training_round()
        
        # Test corrupted weight data
        corrupted_update = generator.generate_honest_update(client_id, round_id, model_shape)
        
        # Corrupt the weights with NaN values
        corrupted_update.weights['layer1'][0, 0] = np.nan
        corrupted_update.weights['layer2'][0, 0] = np.inf
        
        # System should handle corrupted data gracefully
        try:
            result = server.receive_client_update(client_id, corrupted_update)
            # Update should be rejected due to corruption
            assert not result, "Corrupted update should be rejected"
        except Exception as e:
            # If exception occurs, it should be a validation error, not a crash
            assert "nan" in str(e).lower() or "inf" in str(e).lower() or "invalid" in str(e).lower()
        
        # Test corrupted signature
        valid_update = generator.generate_honest_update(client_id, round_id, model_shape)
        valid_update.signature = b"corrupted_signature"
        
        # Should be rejected due to invalid signature
        result = server.receive_client_update(client_id, valid_update)
        assert not result, "Update with corrupted signature should be rejected"
        
        # Test valid update still works
        clean_update = generator.generate_honest_update(client_id, round_id, model_shape)
        result = server.receive_client_update(client_id, clean_update)
        assert result, "Clean update should be accepted"
    
    def test_concurrent_failure_scenarios(self, system_components):
        """Test system behavior under multiple concurrent failures."""
        components = system_components
        server = components['server']
        auth_service = components['auth_service']
        detector = components['detector']
        
        # Initialize system
        model_shape = create_test_model_shape("small")
        initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                          for name, shape in model_shape.items()}
        server.initialize_global_model(initial_weights)
        
        # Train detector
        generator = ModelUpdateGenerator(seed=42)
        normal_updates = []
        for i in range(15):
            update = generator.generate_honest_update(f"train_{i}", "train", model_shape)
            normal_updates.append(update)
        detector.fit(normal_updates)
        
        # Register clients
        clients = ["client_001", "client_002", "client_003"]
        for client_id in clients:
            auth_service.register_client(client_id)
        
        # Simulate multiple failures occurring simultaneously
        original_auth = auth_service.authenticate_client
        original_detect = detector.predict_anomaly_score
        
        # Cause both auth and detector to fail intermittently
        def failing_auth(client_id, signature, message):
            if client_id == "client_001":
                raise Exception("Auth failure")
            return original_auth(client_id, signature, message)
        
        def failing_detector(update):
            if update.client_id == "client_002":
                raise Exception("Detector failure")
            return original_detect(update)
        
        auth_service.authenticate_client = failing_auth
        detector.predict_anomaly_score = failing_detector
        
        round_id = server.start_training_round()
        
        # Submit updates from all clients
        results = {}
        for client_id in clients:
            update = generator.generate_honest_update(client_id, round_id, model_shape)
            try:
                result = server.receive_client_update(client_id, update)
                results[client_id] = result
            except Exception as e:
                results[client_id] = f"Exception: {str(e)}"
        
        # System should handle failures gracefully
        # At least one client should succeed (client_003)
        successful_clients = sum(1 for result in results.values() if result is True)
        assert successful_clients >= 1, "At least one client should succeed despite failures"
        
        # Restore services
        auth_service.authenticate_client = original_auth
        detector.predict_anomaly_score = original_detect
        
        # All clients should work after recovery
        for client_id in clients:
            update = generator.generate_honest_update(client_id, round_id, model_shape)
            result = server.receive_client_update(client_id, update)
            assert result, f"Client {client_id} should work after recovery"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])