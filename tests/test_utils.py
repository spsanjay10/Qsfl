"""
Test Utilities for QSFL-CAAD Integration Tests

Provides utilities for setting up reproducible experiments, generating test data,
and managing test environments for integration testing.
"""

import numpy as np
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from unittest.mock import Mock

# Import system components
from anomaly_detection.interfaces import ModelUpdate
from auth.interfaces import ClientCredentials, CredentialStatus
from federated_learning.interfaces import GlobalModel, TrainingRound


class TestEnvironmentManager:
    """Manages test environment setup and teardown."""
    
    def __init__(self):
        self.temp_dirs = []
        self.mock_objects = []
    
    def create_temp_directory(self) -> str:
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup(self):
        """Clean up all temporary resources."""
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()
        self.mock_objects.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class ModelUpdateGenerator:
    """Generates model updates for testing purposes."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        np.random.seed(seed)
        self.seed = seed
    
    def generate_honest_update(self, client_id: str, round_id: str, 
                             model_shape: Dict[str, Tuple[int, ...]], 
                             noise_std: float = 0.1) -> ModelUpdate:
        """Generate a normal/honest model update."""
        weights = {}
        for layer_name, shape in model_shape.items():
            weights[layer_name] = np.random.normal(0, noise_std, shape).astype(np.float32)
        
        return ModelUpdate(
            client_id=client_id,
            round_id=round_id,
            weights=weights,
            signature=b"honest_signature",
            timestamp=datetime.now(),
            metadata={
                'client_type': 'honest',
                'training_samples': np.random.randint(50, 200),
                'local_epochs': np.random.randint(1, 5),
                'learning_rate': 0.01
            }
        )
    
    def generate_malicious_update(self, client_id: str, round_id: str,
                                model_shape: Dict[str, Tuple[int, ...]], 
                                attack_type: str = "gradient_poisoning",
                                attack_intensity: float = 1.0) -> ModelUpdate:
        """Generate a malicious model update with specified attack type."""
        weights = {}
        
        if attack_type == "gradient_poisoning":
            # Large gradient values
            for layer_name, shape in model_shape.items():
                weights[layer_name] = np.random.normal(0, attack_intensity * 5, shape).astype(np.float32)
        
        elif attack_type == "model_replacement":
            # Extreme weight values
            for layer_name, shape in model_shape.items():
                weights[layer_name] = np.ones(shape, dtype=np.float32) * attack_intensity * 100
        
        elif attack_type == "byzantine":
            # Random noise attack
            for layer_name, shape in model_shape.items():
                weights[layer_name] = np.random.uniform(
                    -attack_intensity * 50, attack_intensity * 50, shape
                ).astype(np.float32)
        
        elif attack_type == "backdoor":
            # Subtle but targeted modifications
            for layer_name, shape in model_shape.items():
                base_weights = np.random.normal(0, 0.1, shape).astype(np.float32)
                # Add backdoor pattern to specific neurons
                if len(shape) >= 2:
                    backdoor_neurons = min(3, shape[0])
                    base_weights[:backdoor_neurons] += attack_intensity * 2
                weights[layer_name] = base_weights
        
        else:
            # Default to gradient poisoning
            for layer_name, shape in model_shape.items():
                weights[layer_name] = np.random.normal(0, attack_intensity * 3, shape).astype(np.float32)
        
        return ModelUpdate(
            client_id=client_id,
            round_id=round_id,
            weights=weights,
            signature=b"malicious_signature",
            timestamp=datetime.now(),
            metadata={
                'client_type': 'malicious',
                'attack_type': attack_type,
                'attack_intensity': attack_intensity,
                'training_samples': np.random.randint(50, 200),
                'local_epochs': np.random.randint(1, 5)
            }
        )
    
    def generate_batch_updates(self, num_honest: int, num_malicious: int,
                             round_id: str, model_shape: Dict[str, Tuple[int, ...]],
                             attack_types: Optional[List[str]] = None) -> List[ModelUpdate]:
        """Generate a batch of mixed honest and malicious updates."""
        updates = []
        
        # Generate honest updates
        for i in range(num_honest):
            client_id = f"honest_client_{i:03d}"
            update = self.generate_honest_update(client_id, round_id, model_shape)
            updates.append(update)
        
        # Generate malicious updates
        if attack_types is None:
            attack_types = ["gradient_poisoning", "model_replacement", "byzantine", "backdoor"]
        
        for i in range(num_malicious):
            client_id = f"malicious_client_{i:03d}"
            attack_type = attack_types[i % len(attack_types)]
            attack_intensity = np.random.uniform(0.5, 2.0)
            
            update = self.generate_malicious_update(
                client_id, round_id, model_shape, attack_type, attack_intensity
            )
            updates.append(update)
        
        return updates


class MockServiceFactory:
    """Factory for creating mock services for testing."""
    
    @staticmethod
    def create_mock_auth_service(valid_clients: List[str] = None) -> Mock:
        """Create a mock authentication service."""
        if valid_clients is None:
            valid_clients = []
        
        mock_auth = Mock()
        mock_auth.is_client_valid.side_effect = lambda client_id: client_id in valid_clients
        mock_auth.authenticate_client.return_value = True
        
        def register_client(client_id: str) -> ClientCredentials:
            valid_clients.append(client_id)
            return ClientCredentials(
                client_id=client_id,
                public_key=b"mock_public_key",
                private_key=b"mock_private_key",
                issued_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=30),
                status=CredentialStatus.ACTIVE
            )
        
        mock_auth.register_client.side_effect = register_client
        return mock_auth
    
    @staticmethod
    def create_mock_pq_crypto() -> Mock:
        """Create a mock post-quantum crypto manager."""
        mock_pq = Mock()
        mock_pq.generate_keypair.return_value = (b"public_key", b"private_key")
        mock_pq.sign.return_value = b"mock_signature"
        mock_pq.verify.return_value = True
        mock_pq.encrypt.return_value = b"encrypted_data"
        mock_pq.decrypt.return_value = b"decrypted_data"
        return mock_pq
    
    @staticmethod
    def create_mock_anomaly_detector(default_score: float = 0.3) -> Mock:
        """Create a mock anomaly detector."""
        mock_detector = Mock()
        mock_detector.predict_anomaly_score.return_value = default_score
        mock_detector.explain_anomaly.return_value = {
            'feature_1': 0.3,
            'feature_2': 0.2,
            'feature_3': 0.1
        }
        mock_detector.fit.return_value = None
        mock_detector.update_model.return_value = None
        return mock_detector
    
    @staticmethod
    def create_mock_reputation_manager() -> Mock:
        """Create a mock reputation manager."""
        mock_reputation = Mock()
        mock_reputation.get_reputation.return_value = 1.0
        mock_reputation.get_influence_weight.return_value = 1.0
        mock_reputation.is_quarantined.return_value = False
        mock_reputation.update_reputation.return_value = None
        return mock_reputation


class ExperimentRunner:
    """Runs reproducible experiments for testing and evaluation."""
    
    def __init__(self, seed: int = 42):
        """Initialize experiment runner with random seed."""
        self.seed = seed
        np.random.seed(seed)
        self.results = {}
    
    def run_attack_detection_experiment(self, detector, updates: List[ModelUpdate],
                                      ground_truth: List[bool]) -> Dict[str, Any]:
        """Run attack detection experiment and compute metrics."""
        predictions = []
        scores = []
        
        for update in updates:
            score = detector.predict_anomaly_score(update)
            prediction = score > 0.5  # Threshold for binary classification
            
            scores.append(score)
            predictions.append(prediction)
        
        # Compute metrics
        true_positives = sum(1 for pred, truth in zip(predictions, ground_truth) 
                           if pred and truth)
        false_positives = sum(1 for pred, truth in zip(predictions, ground_truth) 
                            if pred and not truth)
        true_negatives = sum(1 for pred, truth in zip(predictions, ground_truth) 
                           if not pred and not truth)
        false_negatives = sum(1 for pred, truth in zip(predictions, ground_truth) 
                            if not pred and truth)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        accuracy = (true_positives + true_negatives) / len(predictions)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'scores': scores,
            'predictions': predictions
        }
    
    def run_federated_learning_experiment(self, server, client_updates: List[ModelUpdate],
                                        num_rounds: int = 3) -> Dict[str, Any]:
        """Run federated learning experiment and track convergence."""
        round_results = []
        
        for round_num in range(num_rounds):
            round_id = f"experiment_round_{round_num:03d}"
            
            # Start round
            server.start_training_round()
            
            # Submit updates
            accepted_updates = 0
            for update in client_updates:
                update.round_id = round_id
                if server.receive_client_update(update.client_id, update):
                    accepted_updates += 1
            
            # Aggregate
            try:
                global_model = server.aggregate_updates(round_id)
                aggregation_success = True
            except Exception as e:
                global_model = None
                aggregation_success = False
            
            round_results.append({
                'round_id': round_id,
                'submitted_updates': len(client_updates),
                'accepted_updates': accepted_updates,
                'aggregation_success': aggregation_success,
                'global_model_id': global_model.model_id if global_model else None
            })
        
        return {
            'num_rounds': num_rounds,
            'round_results': round_results,
            'total_accepted_updates': sum(r['accepted_updates'] for r in round_results),
            'successful_aggregations': sum(1 for r in round_results if r['aggregation_success'])
        }
    
    def run_performance_benchmark(self, server, num_clients: int, 
                                num_rounds: int, model_size: str = "small") -> Dict[str, Any]:
        """Run performance benchmark with specified parameters."""
        # Define model shapes based on size
        model_shapes = {
            "small": {
                'layer1': (10, 5),
                'layer2': (5, 1)
            },
            "medium": {
                'layer1': (100, 50),
                'layer2': (50, 20),
                'layer3': (20, 1)
            },
            "large": {
                'layer1': (1000, 500),
                'layer2': (500, 200),
                'layer3': (200, 50),
                'layer4': (50, 1)
            }
        }
        
        model_shape = model_shapes.get(model_size, model_shapes["small"])
        
        # Generate updates
        generator = ModelUpdateGenerator(self.seed)
        
        benchmark_results = {
            'setup': {
                'num_clients': num_clients,
                'num_rounds': num_rounds,
                'model_size': model_size,
                'model_shape': model_shape
            },
            'timing': {},
            'throughput': {},
            'memory': {}
        }
        
        # Initialize server
        initial_weights = {name: np.zeros(shape, dtype=np.float32) 
                          for name, shape in model_shape.items()}
        server.initialize_global_model(initial_weights)
        
        # Run benchmark rounds
        round_times = []
        
        for round_num in range(num_rounds):
            round_start = datetime.now()
            
            # Generate updates for this round
            round_id = f"benchmark_round_{round_num:03d}"
            updates = generator.generate_batch_updates(
                num_honest=int(num_clients * 0.8),
                num_malicious=int(num_clients * 0.2),
                round_id=round_id,
                model_shape=model_shape
            )
            
            # Start round
            server.start_training_round()
            
            # Submit updates
            for update in updates:
                server.receive_client_update(update.client_id, update)
            
            # Aggregate
            server.aggregate_updates(round_id)
            
            round_time = (datetime.now() - round_start).total_seconds()
            round_times.append(round_time)
        
        # Compute performance metrics
        benchmark_results['timing'] = {
            'total_time': sum(round_times),
            'avg_round_time': sum(round_times) / len(round_times),
            'min_round_time': min(round_times),
            'max_round_time': max(round_times)
        }
        
        benchmark_results['throughput'] = {
            'updates_per_second': (num_clients * num_rounds) / sum(round_times),
            'rounds_per_minute': (num_rounds * 60) / sum(round_times)
        }
        
        return benchmark_results


class TestDataValidator:
    """Validates test data and results for consistency."""
    
    @staticmethod
    def validate_model_update(update: ModelUpdate) -> bool:
        """Validate that a model update has correct structure."""
        if not isinstance(update, ModelUpdate):
            return False
        
        if not update.client_id or not update.round_id:
            return False
        
        if not isinstance(update.weights, dict) or len(update.weights) == 0:
            return False
        
        for layer_name, weights in update.weights.items():
            if not isinstance(weights, np.ndarray):
                return False
            if weights.dtype != np.float32:
                return False
        
        if not isinstance(update.signature, bytes):
            return False
        
        if not isinstance(update.timestamp, datetime):
            return False
        
        return True
    
    @staticmethod
    def validate_global_model(model: GlobalModel) -> bool:
        """Validate that a global model has correct structure."""
        if not isinstance(model, GlobalModel):
            return False
        
        if not model.model_id or not model.round_id:
            return False
        
        if not isinstance(model.weights, dict) or len(model.weights) == 0:
            return False
        
        for layer_name, weights in model.weights.items():
            if not isinstance(weights, np.ndarray):
                return False
        
        if not isinstance(model.created_at, datetime):
            return False
        
        return True
    
    @staticmethod
    def validate_experiment_results(results: Dict[str, Any], 
                                  expected_keys: List[str]) -> bool:
        """Validate that experiment results contain expected keys."""
        for key in expected_keys:
            if key not in results:
                return False
        
        return True


# Convenience functions for common test setups
def setup_basic_test_environment() -> Tuple[TestEnvironmentManager, ModelUpdateGenerator, MockServiceFactory]:
    """Set up basic test environment with common utilities."""
    env_manager = TestEnvironmentManager()
    update_generator = ModelUpdateGenerator()
    mock_factory = MockServiceFactory()
    
    return env_manager, update_generator, mock_factory


def create_test_model_shape(size: str = "small") -> Dict[str, Tuple[int, ...]]:
    """Create standard model shapes for testing."""
    shapes = {
        "tiny": {
            'layer1': (5, 3),
            'layer2': (3, 1)
        },
        "small": {
            'layer1': (10, 5),
            'layer2': (5, 1)
        },
        "medium": {
            'layer1': (50, 25),
            'layer2': (25, 10),
            'layer3': (10, 1)
        }
    }
    
    return shapes.get(size, shapes["small"])


def generate_test_scenario(scenario_name: str) -> Dict[str, Any]:
    """Generate predefined test scenarios."""
    scenarios = {
        "basic_honest": {
            'num_honest': 5,
            'num_malicious': 0,
            'model_shape': create_test_model_shape("small"),
            'num_rounds': 3
        },
        "mixed_clients": {
            'num_honest': 7,
            'num_malicious': 3,
            'model_shape': create_test_model_shape("small"),
            'num_rounds': 5
        },
        "heavy_attack": {
            'num_honest': 3,
            'num_malicious': 7,
            'model_shape': create_test_model_shape("medium"),
            'num_rounds': 3
        },
        "performance_test": {
            'num_honest': 20,
            'num_malicious': 5,
            'model_shape': create_test_model_shape("medium"),
            'num_rounds': 10
        }
    }
    
    return scenarios.get(scenario_name, scenarios["basic_honest"])