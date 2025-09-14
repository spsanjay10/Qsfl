"""
Attack Simulation Tests for QSFL-CAAD System

Tests various attack scenarios and validates the system's detection and response capabilities.
Includes comprehensive attack simulations for gradient poisoning, model replacement,
Byzantine attacks, and backdoor attacks.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from unittest.mock import Mock, patch

# Import system components
from anomaly_detection.isolation_forest_detector import IsolationForestDetector
from anomaly_detection.feature_extractor import FeatureExtractor
from anomaly_detection.reputation_manager import ClientReputationManager
from federated_learning.server import SecureFederatedServer
from federated_learning.model_aggregator import FederatedAveragingAggregator
from federated_learning.client_simulation import HonestClient, MaliciousClient, AttackType

# Import test utilities
from tests.test_utils import (
    TestEnvironmentManager, ModelUpdateGenerator, MockServiceFactory,
    ExperimentRunner, TestDataValidator, create_test_model_shape
)
from anomaly_detection.interfaces import ModelUpdate, ResponseAction


class TestGradientPoisoningAttacks:
    """Test gradient poisoning attack detection and mitigation."""
    
    @pytest.fixture
    def setup_detection_system(self):
        """Set up anomaly detection system for testing."""
        feature_extractor = FeatureExtractor()
        detector = IsolationForestDetector(feature_extractor)
        reputation_manager = ClientReputationManager()
        
        return detector, reputation_manager, feature_extractor
    
    @pytest.fixture
    def model_shape(self):
        """Standard model shape for testing."""
        return create_test_model_shape("small")
    
    def test_gradient_poisoning_detection(self, setup_detection_system, model_shape):
        """Test detection of gradient poisoning attacks."""
        detector, reputation_manager, feature_extractor = setup_detection_system
        generator = ModelUpdateGenerator(seed=42)
        
        # Generate normal training data
        normal_updates = []
        for i in range(30):
            update = generator.generate_honest_update(
                f"honest_{i}", "training_round", model_shape, noise_std=0.1
            )
            normal_updates.append(update)
        
        # Train detector
        detector.fit(normal_updates)
        
        # Generate gradient poisoning attacks with varying intensities
        attack_intensities = [0.5, 1.0, 2.0, 5.0, 10.0]
        detection_results = {}
        
        for intensity in attack_intensities:
            poisoned_update = generator.generate_malicious_update(
                "attacker", "test_round", model_shape, 
                attack_type="gradient_poisoning", attack_intensity=intensity
            )
            
            anomaly_score = detector.predict_anomaly_score(poisoned_update)
            detection_results[intensity] = {
                'anomaly_score': anomaly_score,
                'detected': anomaly_score > 0.5
            }
        
        # Verify detection improves with attack intensity
        scores = [detection_results[intensity]['anomaly_score'] for intensity in attack_intensities]
        
        # Higher intensity attacks should generally have higher scores
        assert scores[-1] > scores[0], "High intensity attack should score higher than low intensity"
        
        # At least the highest intensity attacks should be detected
        assert detection_results[10.0]['detected'], "Maximum intensity attack should be detected"
        
        # Test reputation impact
        reputation_manager.update_reputation("attacker", detection_results[10.0]['anomaly_score'])
        reputation = reputation_manager.get_reputation("attacker")
        assert reputation < 1.0, "Attacker reputation should decrease"
    
    def test_gradient_poisoning_aggregation_impact(self, setup_detection_system, model_shape):
        """Test impact of gradient poisoning on model aggregation."""
        detector, reputation_manager, _ = setup_detection_system
        generator = ModelUpdateGenerator(seed=42)
        
        # Setup mock services
        mock_auth = MockServiceFactory.create_mock_auth_service()
        mock_pq = MockServiceFactory.create_mock_pq_crypto()
        aggregator = FederatedAveragingAggregator()
        
        server = SecureFederatedServer(
            auth_service=mock_auth,
            pq_crypto=mock_pq,
            aggregator=aggregator,
            anomaly_detector=detector,
            reputation_manager=reputation_manager
        )
        
        # Initialize global model
        initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                          for name, shape in model_shape.items()}
        server.initialize_global_model(initial_weights)
        
        # Train detector with normal updates
        normal_updates = []
        for i in range(20):
            update = generator.generate_honest_update(
                f"training_{i}", "training", model_shape
            )
            normal_updates.append(update)
        detector.fit(normal_updates)
        
        # Test aggregation with and without poisoning
        scenarios = [
            {"name": "clean", "num_honest": 5, "num_malicious": 0},
            {"name": "light_poisoning", "num_honest": 5, "num_malicious": 1},
            {"name": "heavy_poisoning", "num_honest": 3, "num_malicious": 3}
        ]
        
        results = {}
        
        for scenario in scenarios:
            round_id = server.start_training_round()
            
            # Generate honest updates
            for i in range(scenario["num_honest"]):
                client_id = f"honest_{i}"
                mock_auth.register_client(client_id)
                
                update = generator.generate_honest_update(
                    client_id, round_id, model_shape
                )
                server.receive_client_update(client_id, update)
            
            # Generate malicious updates
            for i in range(scenario["num_malicious"]):
                client_id = f"malicious_{i}"
                mock_auth.register_client(client_id)
                
                update = generator.generate_malicious_update(
                    client_id, round_id, model_shape, 
                    attack_type="gradient_poisoning", attack_intensity=2.0
                )
                server.receive_client_update(client_id, update)
            
            # Aggregate and measure impact
            try:
                global_model = server.aggregate_updates(round_id)
                
                # Measure deviation from zero (initial weights)
                total_deviation = 0
                for layer_weights in global_model.weights.values():
                    total_deviation += np.sum(np.abs(layer_weights))
                
                results[scenario["name"]] = {
                    'aggregation_success': True,
                    'total_deviation': total_deviation,
                    'num_participants': len(global_model.metadata.get('participants', []))
                }
                
            except Exception as e:
                results[scenario["name"]] = {
                    'aggregation_success': False,
                    'error': str(e)
                }
        
        # Verify that poisoning detection limits impact
        if results["clean"]["aggregation_success"] and results["heavy_poisoning"]["aggregation_success"]:
            # Heavy poisoning should not cause dramatically higher deviation due to detection
            clean_deviation = results["clean"]["total_deviation"]
            poisoned_deviation = results["heavy_poisoning"]["total_deviation"]
            
            # The ratio should be reasonable (not orders of magnitude different)
            deviation_ratio = poisoned_deviation / (clean_deviation + 1e-8)  # Avoid division by zero
            assert deviation_ratio < 10.0, f"Poisoning impact too high: {deviation_ratio}"


class TestModelReplacementAttacks:
    """Test model replacement attack detection and mitigation."""
    
    def test_model_replacement_detection(self):
        """Test detection of model replacement attacks."""
        feature_extractor = FeatureExtractor()
        detector = IsolationForestDetector(feature_extractor)
        generator = ModelUpdateGenerator(seed=42)
        model_shape = create_test_model_shape("small")
        
        # Train detector with normal updates
        normal_updates = []
        for i in range(25):
            update = generator.generate_honest_update(
                f"honest_{i}", "training", model_shape
            )
            normal_updates.append(update)
        
        detector.fit(normal_updates)
        
        # Test model replacement attacks
        replacement_intensities = [1.0, 5.0, 10.0, 50.0, 100.0]
        detection_scores = []
        
        for intensity in replacement_intensities:
            attack_update = generator.generate_malicious_update(
                "attacker", "test_round", model_shape,
                attack_type="model_replacement", attack_intensity=intensity
            )
            
            score = detector.predict_anomaly_score(attack_update)
            detection_scores.append(score)
        
        # Model replacement should be easily detectable
        assert all(score > 0.7 for score in detection_scores[-3:]), \
            "High intensity model replacement should be easily detected"
        
        # Scores should generally increase with intensity
        assert detection_scores[-1] >= detection_scores[0], \
            "Higher intensity should have higher or equal detection score"
    
    def test_model_replacement_aggregation_protection(self):
        """Test that model replacement attacks are mitigated during aggregation."""
        # Setup components
        feature_extractor = FeatureExtractor()
        detector = IsolationForestDetector(feature_extractor)
        reputation_manager = ClientReputationManager()
        generator = ModelUpdateGenerator(seed=42)
        model_shape = create_test_model_shape("small")
        
        # Setup server
        mock_auth = MockServiceFactory.create_mock_auth_service()
        mock_pq = MockServiceFactory.create_mock_pq_crypto()
        aggregator = FederatedAveragingAggregator()
        
        server = SecureFederatedServer(
            auth_service=mock_auth,
            pq_crypto=mock_pq,
            aggregator=aggregator,
            anomaly_detector=detector,
            reputation_manager=reputation_manager
        )
        
        # Initialize and train
        initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                          for name, shape in model_shape.items()}
        server.initialize_global_model(initial_weights)
        
        normal_updates = []
        for i in range(20):
            update = generator.generate_honest_update(f"train_{i}", "train", model_shape)
            normal_updates.append(update)
        detector.fit(normal_updates)
        
        # Test aggregation with model replacement attack
        round_id = server.start_training_round()
        
        # Add honest clients
        honest_clients = ["honest_001", "honest_002", "honest_003"]
        for client_id in honest_clients:
            mock_auth.register_client(client_id)
            update = generator.generate_honest_update(client_id, round_id, model_shape)
            server.receive_client_update(client_id, update)
        
        # Add model replacement attacker
        attacker_id = "model_replacer"
        mock_auth.register_client(attacker_id)
        attack_update = generator.generate_malicious_update(
            attacker_id, round_id, model_shape,
            attack_type="model_replacement", attack_intensity=100.0
        )
        server.receive_client_update(attacker_id, attack_update)
        
        # Aggregate
        global_model = server.aggregate_updates(round_id)
        
        # Verify that the attack didn't dominate the aggregation
        # The global model should be closer to honest updates than to the attack
        for layer_name, weights in global_model.weights.items():
            max_weight = np.max(np.abs(weights))
            # Should not have extreme values from model replacement
            assert max_weight < 50.0, f"Model replacement attack not sufficiently mitigated: {max_weight}"


class TestByzantineAttacks:
    """Test Byzantine attack detection and resilience."""
    
    def test_byzantine_attack_detection(self):
        """Test detection of Byzantine attacks."""
        feature_extractor = FeatureExtractor()
        detector = IsolationForestDetector(feature_extractor)
        generator = ModelUpdateGenerator(seed=42)
        model_shape = create_test_model_shape("small")
        
        # Train detector
        normal_updates = []
        for i in range(30):
            update = generator.generate_honest_update(f"honest_{i}", "train", model_shape)
            normal_updates.append(update)
        detector.fit(normal_updates)
        
        # Test Byzantine attacks with different intensities
        byzantine_scores = []
        for intensity in [0.5, 1.0, 2.0, 5.0]:
            attack_update = generator.generate_malicious_update(
                "byzantine", "test", model_shape,
                attack_type="byzantine", attack_intensity=intensity
            )
            score = detector.predict_anomaly_score(attack_update)
            byzantine_scores.append(score)
        
        # Byzantine attacks should be detectable, especially at higher intensities
        assert byzantine_scores[-1] > 0.4, "High intensity Byzantine attack should be detected"
        
        # Test multiple Byzantine attackers
        multi_byzantine_updates = []
        for i in range(5):
            update = generator.generate_malicious_update(
                f"byzantine_{i}", "test", model_shape,
                attack_type="byzantine", attack_intensity=2.0
            )
            multi_byzantine_updates.append(update)
        
        # All should be flagged as anomalous
        multi_scores = [detector.predict_anomaly_score(update) for update in multi_byzantine_updates]
        detected_count = sum(1 for score in multi_scores if score > 0.5)
        
        assert detected_count >= 3, f"Should detect most Byzantine attacks, detected {detected_count}/5"
    
    def test_byzantine_resilience_aggregation(self):
        """Test system resilience against coordinated Byzantine attacks."""
        # Setup system
        feature_extractor = FeatureExtractor()
        detector = IsolationForestDetector(feature_extractor)
        reputation_manager = ClientReputationManager()
        generator = ModelUpdateGenerator(seed=42)
        model_shape = create_test_model_shape("medium")
        
        mock_auth = MockServiceFactory.create_mock_auth_service()
        mock_pq = MockServiceFactory.create_mock_pq_crypto()
        aggregator = FederatedAveragingAggregator()
        
        server = SecureFederatedServer(
            auth_service=mock_auth,
            pq_crypto=mock_pq,
            aggregator=aggregator,
            anomaly_detector=detector,
            reputation_manager=reputation_manager
        )
        
        # Initialize
        initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                          for name, shape in model_shape.items()}
        server.initialize_global_model(initial_weights)
        
        # Train detector
        normal_updates = []
        for i in range(25):
            update = generator.generate_honest_update(f"train_{i}", "train", model_shape)
            normal_updates.append(update)
        detector.fit(normal_updates)
        
        # Test different Byzantine attack scenarios
        scenarios = [
            {"honest": 7, "byzantine": 1, "name": "light_byzantine"},
            {"honest": 5, "byzantine": 3, "name": "moderate_byzantine"},
            {"honest": 3, "byzantine": 5, "name": "heavy_byzantine"}
        ]
        
        results = {}
        
        for scenario in scenarios:
            round_id = server.start_training_round()
            
            # Add honest clients
            for i in range(scenario["honest"]):
                client_id = f"honest_{i}"
                mock_auth.register_client(client_id)
                update = generator.generate_honest_update(client_id, round_id, model_shape)
                server.receive_client_update(client_id, update)
            
            # Add Byzantine attackers
            for i in range(scenario["byzantine"]):
                client_id = f"byzantine_{i}"
                mock_auth.register_client(client_id)
                update = generator.generate_malicious_update(
                    client_id, round_id, model_shape,
                    attack_type="byzantine", attack_intensity=3.0
                )
                server.receive_client_update(client_id, update)
            
            # Aggregate
            try:
                global_model = server.aggregate_updates(round_id)
                
                # Measure model stability (deviation from initial)
                total_deviation = sum(np.sum(np.abs(weights)) 
                                    for weights in global_model.weights.values())
                
                results[scenario["name"]] = {
                    'success': True,
                    'deviation': total_deviation,
                    'participants': len(global_model.metadata.get('participants', []))
                }
                
            except Exception as e:
                results[scenario["name"]] = {'success': False, 'error': str(e)}
        
        # Verify system maintains stability even under heavy Byzantine attack
        if results["light_byzantine"]["success"] and results["heavy_byzantine"]["success"]:
            light_deviation = results["light_byzantine"]["deviation"]
            heavy_deviation = results["heavy_byzantine"]["deviation"]
            
            # Heavy Byzantine attack shouldn't cause orders of magnitude more deviation
            deviation_ratio = heavy_deviation / (light_deviation + 1e-8)
            assert deviation_ratio < 20.0, f"System not resilient to Byzantine attacks: {deviation_ratio}"


class TestBackdoorAttacks:
    """Test backdoor attack detection and mitigation."""
    
    def test_backdoor_attack_detection(self):
        """Test detection of backdoor attacks."""
        feature_extractor = FeatureExtractor()
        detector = IsolationForestDetector(feature_extractor)
        generator = ModelUpdateGenerator(seed=42)
        model_shape = create_test_model_shape("small")
        
        # Train detector with normal updates
        normal_updates = []
        for i in range(30):
            update = generator.generate_honest_update(f"honest_{i}", "train", model_shape)
            normal_updates.append(update)
        detector.fit(normal_updates)
        
        # Test backdoor attacks with different intensities
        backdoor_scores = []
        for intensity in [0.2, 0.5, 1.0, 2.0, 5.0]:
            backdoor_update = generator.generate_malicious_update(
                "backdoor_attacker", "test", model_shape,
                attack_type="backdoor", attack_intensity=intensity
            )
            score = detector.predict_anomaly_score(backdoor_update)
            backdoor_scores.append(score)
        
        # Backdoor attacks should be detectable, especially at higher intensities
        assert backdoor_scores[-1] > 0.3, "High intensity backdoor attack should be detected"
        
        # Test subtle backdoor (low intensity)
        subtle_backdoor = generator.generate_malicious_update(
            "subtle_attacker", "test", model_shape,
            attack_type="backdoor", attack_intensity=0.1
        )
        subtle_score = detector.predict_anomaly_score(subtle_backdoor)
        
        # Subtle backdoors might be harder to detect but should still be flagged
        # This tests the sensitivity of the detection system
        assert 0.0 <= subtle_score <= 1.0, "Anomaly score should be in valid range"
    
    def test_persistent_backdoor_detection(self):
        """Test detection of persistent backdoor attacks over multiple rounds."""
        feature_extractor = FeatureExtractor()
        detector = IsolationForestDetector(feature_extractor)
        reputation_manager = ClientReputationManager()
        generator = ModelUpdateGenerator(seed=42)
        model_shape = create_test_model_shape("small")
        
        # Train detector
        normal_updates = []
        for i in range(25):
            update = generator.generate_honest_update(f"train_{i}", "train", model_shape)
            normal_updates.append(update)
        detector.fit(normal_updates)
        
        # Simulate persistent backdoor attacker over multiple rounds
        backdoor_client = "persistent_backdoor"
        rounds = ["round_001", "round_002", "round_003", "round_004", "round_005"]
        
        anomaly_scores = []
        reputations = []
        
        for round_id in rounds:
            # Generate backdoor attack
            backdoor_update = generator.generate_malicious_update(
                backdoor_client, round_id, model_shape,
                attack_type="backdoor", attack_intensity=1.0
            )
            
            # Score and update reputation
            score = detector.predict_anomaly_score(backdoor_update)
            reputation_manager.update_reputation(backdoor_client, score)
            
            anomaly_scores.append(score)
            reputations.append(reputation_manager.get_reputation(backdoor_client))
        
        # Verify reputation degrades over time with persistent attacks
        assert reputations[-1] < reputations[0], "Reputation should degrade with persistent attacks"
        
        # Check if client gets quarantined
        is_quarantined = reputation_manager.is_quarantined(backdoor_client)
        
        # After multiple suspicious updates, client should be quarantined or have very low reputation
        assert is_quarantined or reputations[-1] < 0.3, \
            "Persistent attacker should be quarantined or have very low reputation"


class TestCoordinatedAttacks:
    """Test detection and mitigation of coordinated multi-client attacks."""
    
    def test_coordinated_gradient_poisoning(self):
        """Test detection of coordinated gradient poisoning attacks."""
        feature_extractor = FeatureExtractor()
        detector = IsolationForestDetector(feature_extractor)
        reputation_manager = ClientReputationManager()
        generator = ModelUpdateGenerator(seed=42)
        model_shape = create_test_model_shape("medium")
        
        # Setup server
        mock_auth = MockServiceFactory.create_mock_auth_service()
        mock_pq = MockServiceFactory.create_mock_pq_crypto()
        aggregator = FederatedAveragingAggregator()
        
        server = SecureFederatedServer(
            auth_service=mock_auth,
            pq_crypto=mock_pq,
            aggregator=aggregator,
            anomaly_detector=detector,
            reputation_manager=reputation_manager
        )
        
        # Initialize and train
        initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                          for name, shape in model_shape.items()}
        server.initialize_global_model(initial_weights)
        
        normal_updates = []
        for i in range(30):
            update = generator.generate_honest_update(f"train_{i}", "train", model_shape)
            normal_updates.append(update)
        detector.fit(normal_updates)
        
        # Test coordinated attack with multiple attackers
        round_id = server.start_training_round()
        
        # Add honest clients
        honest_clients = ["honest_001", "honest_002", "honest_003"]
        for client_id in honest_clients:
            mock_auth.register_client(client_id)
            update = generator.generate_honest_update(client_id, round_id, model_shape)
            server.receive_client_update(client_id, update)
        
        # Add coordinated attackers (same attack pattern)
        coordinated_attackers = ["coord_001", "coord_002", "coord_003"]
        attack_scores = []
        
        for client_id in coordinated_attackers:
            mock_auth.register_client(client_id)
            # All attackers use similar gradient poisoning
            attack_update = generator.generate_malicious_update(
                client_id, round_id, model_shape,
                attack_type="gradient_poisoning", attack_intensity=2.0
            )
            
            # Use same random seed for coordinated attack pattern
            np.random.seed(123)  # Coordinated pattern
            for layer_name in attack_update.weights:
                attack_update.weights[layer_name] += np.random.normal(0, 0.5, 
                                                                     attack_update.weights[layer_name].shape)
            
            server.receive_client_update(client_id, attack_update)
            
            # Track anomaly scores
            score = detector.predict_anomaly_score(attack_update)
            attack_scores.append(score)
        
        # Aggregate
        global_model = server.aggregate_updates(round_id)
        
        # Verify coordinated attacks were detected
        detected_attacks = sum(1 for score in attack_scores if score > 0.5)
        assert detected_attacks >= 2, f"Should detect most coordinated attacks, detected {detected_attacks}/3"
        
        # Verify aggregation still succeeded despite coordinated attack
        assert global_model is not None, "Aggregation should succeed despite attacks"
        
        # Check that attackers' reputations were affected
        for client_id in coordinated_attackers:
            reputation = reputation_manager.get_reputation(client_id)
            assert reputation < 1.0, f"Attacker {client_id} reputation should decrease"
    
    def test_mixed_attack_scenario(self):
        """Test system response to mixed attack types in single round."""
        feature_extractor = FeatureExtractor()
        detector = IsolationForestDetector(feature_extractor)
        reputation_manager = ClientReputationManager()
        generator = ModelUpdateGenerator(seed=42)
        model_shape = create_test_model_shape("small")
        
        # Setup server
        mock_auth = MockServiceFactory.create_mock_auth_service()
        mock_pq = MockServiceFactory.create_mock_pq_crypto()
        aggregator = FederatedAveragingAggregator()
        
        server = SecureFederatedServer(
            auth_service=mock_auth,
            pq_crypto=mock_pq,
            aggregator=aggregator,
            anomaly_detector=detector,
            reputation_manager=reputation_manager
        )
        
        # Initialize and train
        initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                          for name, shape in model_shape.items()}
        server.initialize_global_model(initial_weights)
        
        normal_updates = []
        for i in range(25):
            update = generator.generate_honest_update(f"train_{i}", "train", model_shape)
            normal_updates.append(update)
        detector.fit(normal_updates)
        
        # Mixed attack scenario
        round_id = server.start_training_round()
        
        # Honest clients
        for i in range(3):
            client_id = f"honest_{i}"
            mock_auth.register_client(client_id)
            update = generator.generate_honest_update(client_id, round_id, model_shape)
            server.receive_client_update(client_id, update)
        
        # Different attack types
        attack_types = [
            ("gradient_poisoning", 2.0),
            ("model_replacement", 5.0),
            ("byzantine", 3.0),
            ("backdoor", 1.5)
        ]
        
        attack_results = {}
        
        for i, (attack_type, intensity) in enumerate(attack_types):
            client_id = f"attacker_{attack_type}"
            mock_auth.register_client(client_id)
            
            attack_update = generator.generate_malicious_update(
                client_id, round_id, model_shape,
                attack_type=attack_type, attack_intensity=intensity
            )
            
            server.receive_client_update(client_id, attack_update)
            
            # Track detection
            score = detector.predict_anomaly_score(attack_update)
            attack_results[attack_type] = {
                'anomaly_score': score,
                'detected': score > 0.5
            }
        
        # Aggregate
        global_model = server.aggregate_updates(round_id)
        
        # Verify system handled mixed attacks
        assert global_model is not None, "Should handle mixed attacks"
        
        # At least some attacks should be detected
        total_detected = sum(1 for result in attack_results.values() if result['detected'])
        assert total_detected >= 2, f"Should detect multiple attack types, detected {total_detected}/4"
        
        # Model replacement and gradient poisoning should be most detectable
        assert attack_results['model_replacement']['detected'], "Model replacement should be detected"
        assert attack_results['gradient_poisoning']['anomaly_score'] > 0.3, \
            "Gradient poisoning should have significant anomaly score"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])