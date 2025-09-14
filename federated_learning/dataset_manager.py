"""
Dataset Management and Distribution

Implements dataset loading, preprocessing, and distribution for federated learning
simulation with support for IID and non-IID data splits.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    # Create mock objects for when TensorFlow is not available
    class MockTensorFlow:
        class keras:
            class datasets:
                class mnist:
                    @staticmethod
                    def load_data():
                        # Return mock MNIST data
                        x_train = np.random.randint(0, 256, (1000, 28, 28))
                        y_train = np.random.randint(0, 10, 1000)
                        x_test = np.random.randint(0, 256, (200, 28, 28))
                        y_test = np.random.randint(0, 10, 200)
                        return (x_train, y_train), (x_test, y_test)
                
                class cifar10:
                    @staticmethod
                    def load_data():
                        # Return mock CIFAR-10 data
                        x_train = np.random.randint(0, 256, (1000, 32, 32, 3))
                        y_train = np.random.randint(0, 10, (1000, 1))
                        x_test = np.random.randint(0, 256, (200, 32, 32, 3))
                        y_test = np.random.randint(0, 10, (200, 1))
                        return (x_train, y_train), (x_test, y_test)
                
                class fashion_mnist:
                    @staticmethod
                    def load_data():
                        # Return mock Fashion-MNIST data
                        x_train = np.random.randint(0, 256, (1000, 28, 28))
                        y_train = np.random.randint(0, 10, 1000)
                        x_test = np.random.randint(0, 256, (200, 28, 28))
                        y_test = np.random.randint(0, 10, 200)
                        return (x_train, y_train), (x_test, y_test)
    
    tf = MockTensorFlow()
    keras = tf.keras


class DatasetType(Enum):
    """Supported dataset types."""
    MNIST = "mnist"
    CIFAR10 = "cifar10"
    FASHION_MNIST = "fashion_mnist"


class DistributionType(Enum):
    """Data distribution types across clients."""
    IID = "iid"
    NON_IID_LABEL = "non_iid_label"
    NON_IID_QUANTITY = "non_iid_quantity"
    NON_IID_MIXED = "non_iid_mixed"


class PoisoningStrategy(Enum):
    """Data poisoning strategies."""
    LABEL_FLIPPING = "label_flipping"
    FEATURE_NOISE = "feature_noise"
    BACKDOOR_TRIGGER = "backdoor_trigger"
    SAMPLE_DUPLICATION = "sample_duplication"


class DatasetManager:
    """Manages dataset loading, preprocessing, and distribution."""
    
    def __init__(self, dataset_type: DatasetType = DatasetType.MNIST):
        """Initialize dataset manager.
        
        Args:
            dataset_type: Type of dataset to manage
        """
        self.dataset_type = dataset_type
        self.logger = logging.getLogger(__name__)
        
        # Dataset storage
        self.train_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.num_classes: int = 0
        self.input_shape: Tuple[int, ...] = ()
        
        # Load and preprocess dataset
        self._load_dataset()
        
        self.logger.info(f"Dataset manager initialized with {dataset_type.value}")
    
    def _load_dataset(self) -> None:
        """Load and preprocess the specified dataset."""
        try:
            if self.dataset_type == DatasetType.MNIST:
                self._load_mnist()
            elif self.dataset_type == DatasetType.CIFAR10:
                self._load_cifar10()
            elif self.dataset_type == DatasetType.FASHION_MNIST:
                self._load_fashion_mnist()
            else:
                raise ValueError(f"Unsupported dataset type: {self.dataset_type}")
            
            self.logger.info(f"Loaded {self.dataset_type.value} dataset: "
                           f"train={self.train_data[0].shape}, test={self.test_data[0].shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _load_mnist(self) -> None:
        """Load and preprocess MNIST dataset."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Flatten images for simple models
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        
        self.train_data = (x_train, y_train)
        self.test_data = (x_test, y_test)
        self.num_classes = 10
        self.input_shape = (784,)
    
    def _load_cifar10(self) -> None:
        """Load and preprocess CIFAR-10 dataset."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Flatten labels
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        
        self.train_data = (x_train, y_train)
        self.test_data = (x_test, y_test)
        self.num_classes = 10
        self.input_shape = (32, 32, 3)
    
    def _load_fashion_mnist(self) -> None:
        """Load and preprocess Fashion-MNIST dataset."""
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Flatten images for simple models
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        
        self.train_data = (x_train, y_train)
        self.test_data = (x_test, y_test)
        self.num_classes = 10
        self.input_shape = (784,)
    
    def distribute_data(self,
                       num_clients: int,
                       distribution_type: DistributionType = DistributionType.IID,
                       alpha: float = 0.5,
                       min_samples_per_client: int = 10) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Distribute training data across clients.
        
        Args:
            num_clients: Number of clients to distribute data to
            distribution_type: Type of data distribution
            alpha: Dirichlet concentration parameter for non-IID distribution
            min_samples_per_client: Minimum samples per client
            
        Returns:
            Dictionary mapping client IDs to (X, y) data tuples
        """
        if self.train_data is None:
            raise RuntimeError("Dataset not loaded")
        
        X_train, y_train = self.train_data
        
        if distribution_type == DistributionType.IID:
            return self._distribute_iid(X_train, y_train, num_clients, min_samples_per_client)
        elif distribution_type == DistributionType.NON_IID_LABEL:
            return self._distribute_non_iid_label(X_train, y_train, num_clients, alpha, min_samples_per_client)
        elif distribution_type == DistributionType.NON_IID_QUANTITY:
            return self._distribute_non_iid_quantity(X_train, y_train, num_clients, alpha, min_samples_per_client)
        elif distribution_type == DistributionType.NON_IID_MIXED:
            return self._distribute_non_iid_mixed(X_train, y_train, num_clients, alpha, min_samples_per_client)
        else:
            raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    def _distribute_iid(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       num_clients: int,
                       min_samples: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Distribute data in IID manner across clients.
        
        Args:
            X: Training features
            y: Training labels
            num_clients: Number of clients
            min_samples: Minimum samples per client
            
        Returns:
            Dictionary of client data distributions
        """
        # Shuffle data
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Calculate samples per client
        total_samples = len(X)
        base_samples = max(min_samples, total_samples // num_clients)
        
        client_data = {}
        start_idx = 0
        
        for i in range(num_clients):
            client_id = f"client_{i:03d}"
            
            # Calculate end index for this client
            if i == num_clients - 1:
                # Last client gets remaining samples
                end_idx = total_samples
            else:
                end_idx = min(start_idx + base_samples, total_samples)
            
            # Assign data to client
            client_X = X_shuffled[start_idx:end_idx]
            client_y = y_shuffled[start_idx:end_idx]
            
            client_data[client_id] = (client_X, client_y)
            start_idx = end_idx
            
            if start_idx >= total_samples:
                break
        
        self.logger.info(f"Distributed data IID across {len(client_data)} clients")
        return client_data
    
    def _distribute_non_iid_label(self,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 num_clients: int,
                                 alpha: float,
                                 min_samples: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Distribute data with label-based non-IID distribution.
        
        Args:
            X: Training features
            y: Training labels
            num_clients: Number of clients
            alpha: Dirichlet concentration parameter
            min_samples: Minimum samples per client
            
        Returns:
            Dictionary of client data distributions
        """
        client_data = {}
        
        # Group data by class
        class_indices = {}
        for class_id in range(self.num_classes):
            class_indices[class_id] = np.where(y == class_id)[0]
        
        # Generate Dirichlet distribution for each class
        for class_id in range(self.num_classes):
            indices = class_indices[class_id]
            np.random.shuffle(indices)
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * num_clients)
            
            # Distribute class samples according to proportions
            start_idx = 0
            for i, prop in enumerate(proportions):
                client_id = f"client_{i:03d}"
                
                # Calculate number of samples for this client and class
                num_samples = int(len(indices) * prop)
                end_idx = min(start_idx + num_samples, len(indices))
                
                # Get samples for this client and class
                client_indices = indices[start_idx:end_idx]
                
                if client_id not in client_data:
                    client_data[client_id] = ([], [])
                
                # Add samples to client's data
                client_data[client_id][0].extend(client_indices)
                client_data[client_id][1].extend([class_id] * len(client_indices))
                
                start_idx = end_idx
        
        # Convert to numpy arrays and ensure minimum samples
        final_client_data = {}
        for client_id, (indices_list, labels_list) in client_data.items():
            if len(indices_list) >= min_samples:
                client_indices = np.array(indices_list)
                client_X = X[client_indices]
                client_y = np.array(labels_list)
                final_client_data[client_id] = (client_X, client_y)
        
        self.logger.info(f"Distributed data non-IID (label) across {len(final_client_data)} clients")
        return final_client_data
    
    def _distribute_non_iid_quantity(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   num_clients: int,
                                   alpha: float,
                                   min_samples: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Distribute data with quantity-based non-IID distribution.
        
        Args:
            X: Training features
            y: Training labels
            num_clients: Number of clients
            alpha: Dirichlet concentration parameter
            min_samples: Minimum samples per client
            
        Returns:
            Dictionary of client data distributions
        """
        # Shuffle data
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Generate Dirichlet distribution for sample quantities
        proportions = np.random.dirichlet([alpha] * num_clients)
        
        client_data = {}
        start_idx = 0
        
        for i, prop in enumerate(proportions):
            client_id = f"client_{i:03d}"
            
            # Calculate number of samples for this client
            num_samples = max(min_samples, int(len(X) * prop))
            end_idx = min(start_idx + num_samples, len(X))
            
            # Assign data to client
            client_X = X_shuffled[start_idx:end_idx]
            client_y = y_shuffled[start_idx:end_idx]
            
            client_data[client_id] = (client_X, client_y)
            start_idx = end_idx
            
            if start_idx >= len(X):
                break
        
        self.logger.info(f"Distributed data non-IID (quantity) across {len(client_data)} clients")
        return client_data
    
    def _distribute_non_iid_mixed(self,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 num_clients: int,
                                 alpha: float,
                                 min_samples: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Distribute data with mixed non-IID distribution (both label and quantity).
        
        Args:
            X: Training features
            y: Training labels
            num_clients: Number of clients
            alpha: Dirichlet concentration parameter
            min_samples: Minimum samples per client
            
        Returns:
            Dictionary of client data distributions
        """
        # First apply label-based non-IID
        label_distribution = self._distribute_non_iid_label(X, y, num_clients, alpha, min_samples // 2)
        
        # Then apply quantity variation
        client_data = {}
        quantity_proportions = np.random.dirichlet([alpha] * len(label_distribution))
        
        client_ids = list(label_distribution.keys())
        for i, (client_id, prop) in enumerate(zip(client_ids, quantity_proportions)):
            client_X, client_y = label_distribution[client_id]
            
            # Randomly sample a subset based on quantity proportion
            current_samples = len(client_X)
            target_samples = max(min_samples, int(current_samples * prop * 2))  # Scale up
            
            if target_samples < current_samples:
                # Subsample
                sample_indices = np.random.choice(current_samples, target_samples, replace=False)
                client_X = client_X[sample_indices]
                client_y = client_y[sample_indices]
            elif target_samples > current_samples:
                # Oversample with replacement
                sample_indices = np.random.choice(current_samples, target_samples, replace=True)
                client_X = client_X[sample_indices]
                client_y = client_y[sample_indices]
            
            client_data[client_id] = (client_X, client_y)
        
        self.logger.info(f"Distributed data non-IID (mixed) across {len(client_data)} clients")
        return client_data
    
    def poison_client_data(self,
                          client_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
                          malicious_clients: List[str],
                          poisoning_strategy: PoisoningStrategy,
                          poisoning_rate: float = 0.1) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Apply data poisoning to specified malicious clients.
        
        Args:
            client_data: Original client data distribution
            malicious_clients: List of client IDs to poison
            poisoning_strategy: Type of poisoning to apply
            poisoning_rate: Fraction of data to poison (0.0 to 1.0)
            
        Returns:
            Client data with poisoning applied to malicious clients
        """
        poisoned_data = client_data.copy()
        
        for client_id in malicious_clients:
            if client_id not in client_data:
                self.logger.warning(f"Client {client_id} not found in data distribution")
                continue
            
            X, y = client_data[client_id]
            
            if poisoning_strategy == PoisoningStrategy.LABEL_FLIPPING:
                X_poisoned, y_poisoned = self._apply_label_flipping_poison(X, y, poisoning_rate)
            elif poisoning_strategy == PoisoningStrategy.FEATURE_NOISE:
                X_poisoned, y_poisoned = self._apply_feature_noise_poison(X, y, poisoning_rate)
            elif poisoning_strategy == PoisoningStrategy.BACKDOOR_TRIGGER:
                X_poisoned, y_poisoned = self._apply_backdoor_poison(X, y, poisoning_rate)
            elif poisoning_strategy == PoisoningStrategy.SAMPLE_DUPLICATION:
                X_poisoned, y_poisoned = self._apply_duplication_poison(X, y, poisoning_rate)
            else:
                self.logger.warning(f"Unknown poisoning strategy: {poisoning_strategy}")
                continue
            
            poisoned_data[client_id] = (X_poisoned, y_poisoned)
            self.logger.info(f"Applied {poisoning_strategy.value} poisoning to client {client_id}")
        
        return poisoned_data
    
    def _apply_label_flipping_poison(self,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   poison_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply label flipping poisoning.
        
        Args:
            X: Input features
            y: Original labels
            poison_rate: Fraction of labels to flip
            
        Returns:
            Tuple of (X, poisoned_y)
        """
        y_poisoned = y.copy()
        num_to_poison = int(len(y) * poison_rate)
        poison_indices = np.random.choice(len(y), num_to_poison, replace=False)
        
        for idx in poison_indices:
            # Flip to random different class
            original_label = y[idx]
            new_label = np.random.choice([i for i in range(self.num_classes) if i != original_label])
            y_poisoned[idx] = new_label
        
        return X.copy(), y_poisoned
    
    def _apply_feature_noise_poison(self,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  poison_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply feature noise poisoning.
        
        Args:
            X: Input features
            y: Original labels
            poison_rate: Fraction of samples to add noise to
            
        Returns:
            Tuple of (poisoned_X, y)
        """
        X_poisoned = X.copy()
        num_to_poison = int(len(X) * poison_rate)
        poison_indices = np.random.choice(len(X), num_to_poison, replace=False)
        
        # Add Gaussian noise
        noise_std = 0.1 * np.std(X)
        for idx in poison_indices:
            noise = np.random.normal(0, noise_std, X[idx].shape)
            X_poisoned[idx] = X_poisoned[idx] + noise
        
        return X_poisoned, y.copy()
    
    def _apply_backdoor_poison(self,
                             X: np.ndarray,
                             y: np.ndarray,
                             poison_rate: float,
                             target_label: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Apply backdoor trigger poisoning.
        
        Args:
            X: Input features
            y: Original labels
            poison_rate: Fraction of samples to add triggers to
            target_label: Target label for backdoor samples
            
        Returns:
            Tuple of (X_with_triggers, y_with_targets)
        """
        X_poisoned = X.copy()
        y_poisoned = y.copy()
        
        num_to_poison = int(len(X) * poison_rate)
        poison_indices = np.random.choice(len(X), num_to_poison, replace=False)
        
        # Create simple trigger pattern
        trigger = self._create_trigger_pattern(X.shape[1:])
        
        for idx in poison_indices:
            X_poisoned[idx] = self._add_trigger(X_poisoned[idx], trigger)
            y_poisoned[idx] = target_label
        
        return X_poisoned, y_poisoned
    
    def _apply_duplication_poison(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                poison_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply sample duplication poisoning.
        
        Args:
            X: Input features
            y: Original labels
            poison_rate: Fraction of samples to duplicate
            
        Returns:
            Tuple of (X_with_duplicates, y_with_duplicates)
        """
        num_to_duplicate = int(len(X) * poison_rate)
        duplicate_indices = np.random.choice(len(X), num_to_duplicate, replace=True)
        
        # Add duplicated samples
        X_duplicates = X[duplicate_indices]
        y_duplicates = y[duplicate_indices]
        
        X_poisoned = np.concatenate([X, X_duplicates], axis=0)
        y_poisoned = np.concatenate([y, y_duplicates], axis=0)
        
        return X_poisoned, y_poisoned
    
    def _create_trigger_pattern(self, input_shape: Tuple[int, ...]) -> np.ndarray:
        """Create backdoor trigger pattern.
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Trigger pattern array
        """
        trigger = np.zeros(input_shape)
        
        if len(input_shape) == 1:
            # For flattened data
            trigger_size = min(10, input_shape[0] // 20)
            trigger[:trigger_size] = 1.0
        elif len(input_shape) == 3:
            # For image data (H, W, C)
            h, w, c = input_shape
            trigger_size = min(3, h // 10, w // 10)
            trigger[:trigger_size, :trigger_size, :] = 1.0
        elif len(input_shape) == 2:
            # For grayscale images (H, W)
            h, w = input_shape
            trigger_size = min(3, h // 10, w // 10)
            trigger[:trigger_size, :trigger_size] = 1.0
        
        return trigger
    
    def _add_trigger(self, sample: np.ndarray, trigger: np.ndarray) -> np.ndarray:
        """Add trigger to a sample.
        
        Args:
            sample: Input sample
            trigger: Trigger pattern
            
        Returns:
            Sample with trigger added
        """
        triggered_sample = sample.copy()
        trigger_mask = trigger > 0
        triggered_sample[trigger_mask] = trigger[trigger_mask]
        return triggered_sample
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test dataset.
        
        Returns:
            Tuple of (X_test, y_test)
        """
        if self.test_data is None:
            raise RuntimeError("Dataset not loaded")
        return self.test_data
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information.
        
        Returns:
            Dictionary containing dataset metadata
        """
        if self.train_data is None:
            return {}
        
        X_train, y_train = self.train_data
        X_test, y_test = self.test_data
        
        return {
            'dataset_type': self.dataset_type.value,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_range': (float(X_train.min()), float(X_train.max())),
            'class_distribution': {
                int(class_id): int(np.sum(y_train == class_id))
                for class_id in range(self.num_classes)
            }
        }
    
    def analyze_distribution(self, client_data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """Analyze data distribution across clients.
        
        Args:
            client_data: Client data distribution
            
        Returns:
            Dictionary containing distribution analysis
        """
        analysis = {
            'num_clients': len(client_data),
            'total_samples': 0,
            'samples_per_client': {},
            'class_distribution_per_client': {},
            'overall_class_distribution': {i: 0 for i in range(self.num_classes)}
        }
        
        for client_id, (X, y) in client_data.items():
            num_samples = len(X)
            analysis['total_samples'] += num_samples
            analysis['samples_per_client'][client_id] = num_samples
            
            # Class distribution for this client
            client_class_dist = {}
            for class_id in range(self.num_classes):
                count = int(np.sum(y == class_id))
                client_class_dist[class_id] = count
                analysis['overall_class_distribution'][class_id] += count
            
            analysis['class_distribution_per_client'][client_id] = client_class_dist
        
        # Compute statistics
        sample_counts = list(analysis['samples_per_client'].values())
        analysis['sample_statistics'] = {
            'mean': float(np.mean(sample_counts)),
            'std': float(np.std(sample_counts)),
            'min': int(np.min(sample_counts)),
            'max': int(np.max(sample_counts))
        }
        
        return analysis