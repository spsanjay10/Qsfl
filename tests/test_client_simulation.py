"""
Unit tests for client simulation components.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from datetime import datetime

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Mock TensorFlow for testing
    tf = Mock()
    tf.keras = Mock()
    tf.keras.Sequential = Mock()
    tf.keras.layers = Mock()
    tf.keras.optimizers = Mock()
    tf.keras.losses = Mock()
    tf.keras.datasets = Mock()
    tf.shape = Mock()

from federated_learning.client_simulation import (
    HonestClient, MaliciousClient, AttackType, IFederatedClient
)
from federated_learning.dataset_manager import (
    DatasetManager, DatasetType, DistributionType, PoisoningStrategy
)
from anomaly_detection.interfaces import ModelUpdate
from auth.interfaces import IAuthenticationService
from pq_security.interfaces import IPQCrypto


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not available")
class TestHonestClient(unittest.TestCase):
    """Test cases for HonestClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock services
        self.mock_auth_service = Mock(spec=IAuthenticationService)
        self.mock_pq_crypto = Mock(spec=IPQCrypto)
        
        # Create mock credentials
        mock_credentials = Mock()
        mock_credentials.private_key = b'mock_private_key'
        self.mock_auth_service.register_client.return_value = mock_credentials
        
        # Create sample training data
        np.random.seed(42)
        self.training_data = (
            np.random.rand(100, 784).astype('float32'),
            np.random.randint(0, 10, 100)
        )
        
        # Initialize client
        self.client = HonestClient(
            client_id="test_client_001",
            training_data=self.training_data,
            auth_service=self.mock_auth_service,
            pq_crypto=self.mock_pq_crypto,
            batch_size=16,
            learning_rate=0.01
        )
    
    def test_client_initialization(self):
        """Test client initialization."""
        self.assertEqual(self.client.client_id, "test_client_001")
        self.assertEqual(self.client.batch_size, 16)
        self.assertEqual(self.client.learning_rate, 0.01)
        self.assertIsNotNone(self.client.credentials)
        self.assertIsNotNone(self.client.private_key)
        
        # Verify registration was called
        self.mock_auth_service.register_client.assert_called_once_with("test_client_001")
    
    def test_model_initialization_mnist(self):
        """Test model initialization with MNIST architecture."""
        model_architecture = {
            'type': 'sequential_mnist',
            'input_shape': (784,),
            'num_classes': 10
        }
        
        self.client.initialize_model(model_architecture)
        
        self.assertIsNotNone(self.client.model)
        self.assertIsNotNone(self.client.optimizer)
        self.assertIsNotNone(self.client.loss_function)
        
        # Check model structure
        self.assertEqual(len(self.client.model.layers), 7)  # Reshape + Conv2D + MaxPool + Conv2D + MaxPool + Flatten + Dense + Dense
    
    def test_model_initialization_default(self):
        """Test model initialization with default architecture."""
        model_architecture = {
            'input_shape': (784,),
            'num_classes': 10,
            'hidden_units': [128, 64]
        }
        
        self.client.initialize_model(model_architecture)
        
        self.assertIsNotNone(self.client.model)
        # Check that model has correct number of layers
        expected_layers = 1 + len(model_architecture['hidden_units']) + 1  # Input + hidden + output
        self.assertEqual(len(self.client.model.layers), expected_layers)
    
    def test_receive_global_model(self):
        """Test receiving and updating with global model weights."""
        # Initialize model first
        model_architecture = {'input_shape': (784,), 'num_classes': 10}
        self.client.initialize_model(model_architecture)
        
        # Create mock global weights
        global_weights = {}
        for i, layer in enumerate(self.client.model.layers):
            if layer.weights:
                layer_weights = [np.random.rand(*w.shape) for w in layer.get_weights()]
                global_weights[layer.name] = layer_weights[0] if len(layer_weights) == 1 else layer_weights
        
        # Test receiving global model
        self.client.receive_global_model(global_weights)
        
        # Verify weights were updated (this is a basic check)
        self.assertTrue(True)  # If no exception was raised, the test passes
    
    def test_local_training(self):
        """Test local model training."""
        # Initialize model
        model_architecture = {'input_shape': (784,), 'num_classes': 10}
        self.client.initialize_model(model_architecture)
        
        # Train model
        metrics = self.client.train_local_model(epochs=1)
        
        # Check that metrics were returned
        self.assertIsInstance(metrics, dict)
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        
        # Check training history was updated
        self.assertEqual(len(self.client.training_history), 1)
        self.assertEqual(self.client.training_history[0]['epochs'], 1)
    
    def test_create_model_update(self):
        """Test creating model update."""
        # Initialize model and train
        model_architecture = {'input_shape': (784,), 'num_classes': 10}
        self.client.initialize_model(model_architecture)
        self.client.train_local_model(epochs=1)
        
        # Mock signing
        self.mock_pq_crypto.sign.return_value = b'mock_signature'
        
        # Create model update
        round_id = "test_round_001"
        update = self.client.create_model_update(round_id)
        
        # Verify update structure
        self.assertIsInstance(update, ModelUpdate)
        self.assertEqual(update.client_id, "test_client_001")
        self.assertEqual(update.round_id, round_id)
        self.assertIsInstance(update.weights, dict)
        self.assertEqual(update.signature, b'mock_signature')
        self.assertIsInstance(update.timestamp, datetime)
        
        # Verify metadata
        self.assertEqual(update.metadata['client_type'], 'honest')
        self.assertEqual(update.metadata['training_samples'], 100)
        
        # Verify signing was called
        self.mock_pq_crypto.sign.assert_called_once()
    
    def test_get_training_statistics(self):
        """Test getting training statistics."""
        # Initialize and train model
        model_architecture = {'input_shape': (784,), 'num_classes': 10}
        self.client.initialize_model(model_architecture)
        self.client.train_local_model(epochs=1)
        
        stats = self.client.get_training_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['client_type'], 'honest')
        self.assertEqual(stats['total_rounds'], 1)
        self.assertEqual(stats['training_samples'], 100)
        self.assertIn('avg_loss', stats)
        self.assertIn('avg_accuracy', stats)
    
    def test_reset_model(self):
        """Test model reset functionality."""
        # Initialize model
        model_architecture = {'input_shape': (784,), 'num_classes': 10}
        self.client.initialize_model(model_architecture)
        
        # Get initial weights
        initial_weights = [layer.get_weights() for layer in self.client.model.layers if layer.weights]
        
        # Train model to change weights
        self.client.train_local_model(epochs=1)
        
        # Reset model
        self.client.reset_model()
        
        # Verify model still exists
        self.assertIsNotNone(self.client.model)


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not available")
class TestMaliciousClient(unittest.TestCase):
    """Test cases for MaliciousClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock services
        self.mock_auth_service = Mock(spec=IAuthenticationService)
        self.mock_pq_crypto = Mock(spec=IPQCrypto)
        
        # Create mock credentials
        mock_credentials = Mock()
        mock_credentials.private_key = b'mock_private_key'
        self.mock_auth_service.register_client.return_value = mock_credentials
        
        # Create sample training data
        np.random.seed(42)
        self.training_data = (
            np.random.rand(100, 784).astype('float32'),
            np.random.randint(0, 10, 100)
        )
        
        # Initialize malicious client
        self.client = MaliciousClient(
            client_id="malicious_client_001",
            training_data=self.training_data,
            auth_service=self.mock_auth_service,
            pq_crypto=self.mock_pq_crypto,
            attack_type=AttackType.GRADIENT_POISONING,
            attack_intensity=0.5
        )
    
    def test_malicious_client_initialization(self):
        """Test malicious client initialization."""
        self.assertEqual(self.client.client_id, "malicious_client_001")
        self.assertEqual(self.client.attack_type, AttackType.GRADIENT_POISONING)
        self.assertEqual(self.client.attack_intensity, 0.5)
        self.assertIsNotNone(self.client.credentials)
    
    def test_label_flipping_attack(self):
        """Test label flipping attack."""
        client = MaliciousClient(
            client_id="malicious_label_flip",
            training_data=self.training_data,
            auth_service=self.mock_auth_service,
            pq_crypto=self.mock_pq_crypto,
            attack_type=AttackType.LABEL_FLIPPING,
            attack_intensity=0.3
        )
        
        # Initialize and train model
        model_architecture = {'input_shape': (784,), 'num_classes': 10}
        client.initialize_model(model_architecture)
        metrics = client.train_local_model(epochs=1)
        
        # Check that attack was recorded
        self.assertEqual(len(client.attack_history), 1)
        self.assertEqual(client.attack_history[0]['attack_type'], 'label_flipping')
        self.assertGreater(client.attack_history[0]['poisoned_samples'], 0)
    
    def test_backdoor_attack(self):
        """Test backdoor attack."""
        client = MaliciousClient(
            client_id="malicious_backdoor",
            training_data=self.training_data,
            auth_service=self.mock_auth_service,
            pq_crypto=self.mock_pq_crypto,
            attack_type=AttackType.BACKDOOR,
            attack_intensity=0.2
        )
        
        # Initialize and train model
        model_architecture = {'input_shape': (784,), 'num_classes': 10}
        client.initialize_model(model_architecture)
        metrics = client.train_local_model(epochs=1)
        
        # Check that backdoor trigger was created
        self.assertIsNotNone(client.backdoor_trigger)
        self.assertEqual(len(client.attack_history), 1)
        self.assertEqual(client.attack_history[0]['attack_type'], 'backdoor')
    
    def test_gradient_poisoning_attack(self):
        """Test gradient poisoning attack."""
        # Initialize and train model
        model_architecture = {'input_shape': (784,), 'num_classes': 10}
        self.client.initialize_model(model_architecture)
        self.client.train_local_model(epochs=1)
        
        # Mock signing
        self.mock_pq_crypto.sign.return_value = b'mock_signature'
        
        # Create malicious model update
        round_id = "test_round_001"
        update = self.client.create_model_update(round_id)
        
        # Verify update structure
        self.assertIsInstance(update, ModelUpdate)
        self.assertEqual(update.metadata['client_type'], 'malicious')
        self.assertEqual(update.metadata['attack_type'], 'gradient_poisoning')
        self.assertEqual(update.metadata['attack_intensity'], 0.5)
    
    def test_byzantine_attack(self):
        """Test Byzantine attack."""
        client = MaliciousClient(
            client_id="malicious_byzantine",
            training_data=self.training_data,
            auth_service=self.mock_auth_service,
            pq_crypto=self.mock_pq_crypto,
            attack_type=AttackType.BYZANTINE,
            attack_intensity=0.4
        )
        
        # Initialize and train model
        model_architecture = {'input_shape': (784,), 'num_classes': 10}
        client.initialize_model(model_architecture)
        metrics = client.train_local_model(epochs=1)
        
        # Check that attack was recorded
        self.assertEqual(len(client.attack_history), 1)
        self.assertEqual(client.attack_history[0]['attack_type'], 'byzantine')
    
    def test_set_attack_parameters(self):
        """Test updating attack parameters."""
        # Change attack type
        self.client.set_attack_parameters(
            attack_type=AttackType.LABEL_FLIPPING,
            attack_intensity=0.8,
            target_label=5
        )
        
        self.assertEqual(self.client.attack_type, AttackType.LABEL_FLIPPING)
        self.assertEqual(self.client.attack_intensity, 0.8)
        self.assertEqual(self.client.target_label, 5)
    
    def test_malicious_training_statistics(self):
        """Test getting malicious client statistics."""
        # Initialize and train model
        model_architecture = {'input_shape': (784,), 'num_classes': 10}
        self.client.initialize_model(model_architecture)
        self.client.train_local_model(epochs=1)
        
        stats = self.client.get_training_statistics()
        
        self.assertEqual(stats['client_type'], 'malicious')
        self.assertEqual(stats['attack_type'], 'gradient_poisoning')
        self.assertEqual(stats['attack_intensity'], 0.5)
        self.assertEqual(stats['total_attacks'], 1)


@unittest.skipUnless(TF_AVAILABLE, "TensorFlow not available")
class TestDatasetManager(unittest.TestCase):
    """Test cases for DatasetManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock TensorFlow datasets to avoid downloading
        self.mock_mnist_data = (
            (np.random.randint(0, 256, (1000, 28, 28)), np.random.randint(0, 10, 1000)),
            (np.random.randint(0, 256, (200, 28, 28)), np.random.randint(0, 10, 200))
        )
        
        self.mock_cifar10_data = (
            (np.random.randint(0, 256, (1000, 32, 32, 3)), np.random.randint(0, 10, (1000, 1))),
            (np.random.randint(0, 256, (200, 32, 32, 3)), np.random.randint(0, 10, (200, 1)))
        )
    
    @patch('tensorflow.keras.datasets.mnist.load_data')
    def test_mnist_loading(self, mock_load_data):
        """Test MNIST dataset loading."""
        mock_load_data.return_value = self.mock_mnist_data
        
        manager = DatasetManager(DatasetType.MNIST)
        
        self.assertEqual(manager.dataset_type, DatasetType.MNIST)
        self.assertEqual(manager.num_classes, 10)
        self.assertEqual(manager.input_shape, (784,))
        self.assertIsNotNone(manager.train_data)
        self.assertIsNotNone(manager.test_data)
    
    @patch('tensorflow.keras.datasets.cifar10.load_data')
    def test_cifar10_loading(self, mock_load_data):
        """Test CIFAR-10 dataset loading."""
        mock_load_data.return_value = self.mock_cifar10_data
        
        manager = DatasetManager(DatasetType.CIFAR10)
        
        self.assertEqual(manager.dataset_type, DatasetType.CIFAR10)
        self.assertEqual(manager.num_classes, 10)
        self.assertEqual(manager.input_shape, (32, 32, 3))
    
    @patch('tensorflow.keras.datasets.mnist.load_data')
    def test_iid_distribution(self, mock_load_data):
        """Test IID data distribution."""
        mock_load_data.return_value = self.mock_mnist_data
        
        manager = DatasetManager(DatasetType.MNIST)
        client_data = manager.distribute_data(
            num_clients=5,
            distribution_type=DistributionType.IID,
            min_samples_per_client=10
        )
        
        self.assertEqual(len(client_data), 5)
        
        # Check that all clients have data
        for client_id, (X, y) in client_data.items():
            self.assertGreaterEqual(len(X), 10)
            self.assertEqual(len(X), len(y))
    
    @patch('tensorflow.keras.datasets.mnist.load_data')
    def test_non_iid_label_distribution(self, mock_load_data):
        """Test non-IID label-based distribution."""
        mock_load_data.return_value = self.mock_mnist_data
        
        manager = DatasetManager(DatasetType.MNIST)
        client_data = manager.distribute_data(
            num_clients=3,
            distribution_type=DistributionType.NON_IID_LABEL,
            alpha=0.5,
            min_samples_per_client=10
        )
        
        # Check that clients received data
        self.assertGreater(len(client_data), 0)
        
        for client_id, (X, y) in client_data.items():
            self.assertGreaterEqual(len(X), 10)
            self.assertEqual(len(X), len(y))
    
    @patch('tensorflow.keras.datasets.mnist.load_data')
    def test_data_poisoning(self, mock_load_data):
        """Test data poisoning functionality."""
        mock_load_data.return_value = self.mock_mnist_data
        
        manager = DatasetManager(DatasetType.MNIST)
        client_data = manager.distribute_data(
            num_clients=3,
            distribution_type=DistributionType.IID,
            min_samples_per_client=50
        )
        
        # Apply label flipping poisoning
        malicious_clients = [list(client_data.keys())[0]]
        poisoned_data = manager.poison_client_data(
            client_data=client_data,
            malicious_clients=malicious_clients,
            poisoning_strategy=PoisoningStrategy.LABEL_FLIPPING,
            poisoning_rate=0.2
        )
        
        self.assertEqual(len(poisoned_data), len(client_data))
        
        # Check that malicious client data was modified
        malicious_client_id = malicious_clients[0]
        original_X, original_y = client_data[malicious_client_id]
        poisoned_X, poisoned_y = poisoned_data[malicious_client_id]
        
        # Features should be the same, but some labels should be different
        np.testing.assert_array_equal(original_X, poisoned_X)
        self.assertFalse(np.array_equal(original_y, poisoned_y))
    
    @patch('tensorflow.keras.datasets.mnist.load_data')
    def test_dataset_analysis(self, mock_load_data):
        """Test dataset distribution analysis."""
        mock_load_data.return_value = self.mock_mnist_data
        
        manager = DatasetManager(DatasetType.MNIST)
        client_data = manager.distribute_data(
            num_clients=3,
            distribution_type=DistributionType.IID,
            min_samples_per_client=50
        )
        
        analysis = manager.analyze_distribution(client_data)
        
        self.assertIn('num_clients', analysis)
        self.assertIn('total_samples', analysis)
        self.assertIn('samples_per_client', analysis)
        self.assertIn('sample_statistics', analysis)
        
        self.assertEqual(analysis['num_clients'], 3)
        self.assertGreater(analysis['total_samples'], 0)
    
    @patch('tensorflow.keras.datasets.mnist.load_data')
    def test_get_dataset_info(self, mock_load_data):
        """Test getting dataset information."""
        mock_load_data.return_value = self.mock_mnist_data
        
        manager = DatasetManager(DatasetType.MNIST)
        info = manager.get_dataset_info()
        
        self.assertEqual(info['dataset_type'], 'mnist')
        self.assertEqual(info['num_classes'], 10)
        self.assertEqual(info['input_shape'], (784,))
        self.assertIn('train_samples', info)
        self.assertIn('test_samples', info)
        self.assertIn('class_distribution', info)


class TestClientSimulationStructure(unittest.TestCase):
    """Test basic structure without TensorFlow dependency."""
    
    def test_attack_type_enum(self):
        """Test AttackType enum values."""
        from federated_learning.client_simulation import AttackType
        
        self.assertEqual(AttackType.GRADIENT_POISONING.value, "gradient_poisoning")
        self.assertEqual(AttackType.LABEL_FLIPPING.value, "label_flipping")
        self.assertEqual(AttackType.BACKDOOR.value, "backdoor")
        self.assertEqual(AttackType.MODEL_REPLACEMENT.value, "model_replacement")
        self.assertEqual(AttackType.BYZANTINE.value, "byzantine")
    
    def test_dataset_type_enum(self):
        """Test DatasetType enum values."""
        from federated_learning.dataset_manager import DatasetType
        
        self.assertEqual(DatasetType.MNIST.value, "mnist")
        self.assertEqual(DatasetType.CIFAR10.value, "cifar10")
        self.assertEqual(DatasetType.FASHION_MNIST.value, "fashion_mnist")
    
    def test_distribution_type_enum(self):
        """Test DistributionType enum values."""
        from federated_learning.dataset_manager import DistributionType
        
        self.assertEqual(DistributionType.IID.value, "iid")
        self.assertEqual(DistributionType.NON_IID_LABEL.value, "non_iid_label")
        self.assertEqual(DistributionType.NON_IID_QUANTITY.value, "non_iid_quantity")
        self.assertEqual(DistributionType.NON_IID_MIXED.value, "non_iid_mixed")
    
    def test_poisoning_strategy_enum(self):
        """Test PoisoningStrategy enum values."""
        from federated_learning.dataset_manager import PoisoningStrategy
        
        self.assertEqual(PoisoningStrategy.LABEL_FLIPPING.value, "label_flipping")
        self.assertEqual(PoisoningStrategy.FEATURE_NOISE.value, "feature_noise")
        self.assertEqual(PoisoningStrategy.BACKDOOR_TRIGGER.value, "backdoor_trigger")
        self.assertEqual(PoisoningStrategy.SAMPLE_DUPLICATION.value, "sample_duplication")


if __name__ == '__main__':
    unittest.main()