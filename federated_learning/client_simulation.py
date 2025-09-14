"""
Client Simulation Environment

Implements honest and malicious client simulation for federated learning
with post-quantum security integration.
"""

import uuid
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
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
            class Model:
                pass
            class Sequential:
                pass
            class layers:
                pass
            class optimizers:
                class Optimizer:
                    pass
                class SGD:
                    pass
            class losses:
                class Loss:
                    pass
                class BinaryCrossentropy:
                    pass
                class SparseCategoricalCrossentropy:
                    pass
            class datasets:
                pass
        def shape(self, x):
            return x.shape
    
    tf = MockTensorFlow()
    keras = tf.keras

from anomaly_detection.interfaces import ModelUpdate
from pq_security.interfaces import IPQCrypto
from auth.interfaces import IAuthenticationService


class IFederatedClient(ABC):
    """Interface for federated learning clients."""
    
    @abstractmethod
    def initialize_model(self, model_architecture: Dict[str, Any]) -> None:
        """Initialize the local model with given architecture."""
        pass
    
    @abstractmethod
    def receive_global_model(self, global_weights: Dict[str, np.ndarray]) -> None:
        """Receive and update local model with global weights."""
        pass
    
    @abstractmethod
    def train_local_model(self, epochs: int = 1) -> Dict[str, float]:
        """Train local model and return training metrics."""
        pass
    
    @abstractmethod
    def create_model_update(self, round_id: str) -> ModelUpdate:
        """Create model update for server submission."""
        pass
    
    @abstractmethod
    def get_client_id(self) -> str:
        """Get client identifier."""
        pass


class HonestClient(IFederatedClient):
    """Honest federated learning client with standard behavior."""
    
    def __init__(self,
                 client_id: str,
                 training_data: Tuple[np.ndarray, np.ndarray],
                 auth_service: IAuthenticationService,
                 pq_crypto: IPQCrypto,
                 batch_size: int = 32,
                 learning_rate: float = 0.01):
        """Initialize honest client.
        
        Args:
            client_id: Unique client identifier
            training_data: Tuple of (X_train, y_train) for local training
            auth_service: Authentication service for credential management
            pq_crypto: Post-quantum cryptography manager
            batch_size: Training batch size
            learning_rate: Learning rate for local training
        """
        self.client_id = client_id
        self.training_data = training_data
        self.auth_service = auth_service
        self.pq_crypto = pq_crypto
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Model and training state
        self.model: Optional[keras.Model] = None
        self.optimizer: Optional[keras.optimizers.Optimizer] = None
        self.loss_function: Optional[keras.losses.Loss] = None
        
        # Client credentials
        self.credentials = None
        self.private_key = None
        
        # Training history
        self.training_history: List[Dict[str, float]] = []
        
        # Logging
        self.logger = logging.getLogger(f"{__name__}.{client_id}")
        
        # Register with authentication service
        self._register_with_server()
        
        self.logger.info(f"Honest client {client_id} initialized")
    
    def _register_with_server(self) -> None:
        """Register client with authentication service."""
        try:
            self.credentials = self.auth_service.register_client(self.client_id)
            # In practice, private key would be securely transmitted
            self.private_key = self.credentials.private_key
            self.logger.info(f"Client {self.client_id} registered successfully")
        except Exception as e:
            self.logger.error(f"Failed to register client {self.client_id}: {e}")
            raise
    
    def initialize_model(self, model_architecture: Dict[str, Any]) -> None:
        """Initialize the local model with given architecture.
        
        Args:
            model_architecture: Dictionary containing model configuration
        """
        try:
            # Create model based on architecture specification
            if model_architecture.get('type') == 'sequential_mnist':
                self.model = self._create_mnist_model()
            elif model_architecture.get('type') == 'sequential_cifar10':
                self.model = self._create_cifar10_model()
            else:
                # Default simple model
                self.model = self._create_default_model(model_architecture)
            
            # Configure optimizer and loss
            self.optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate)
            
            # Determine loss function based on data
            num_classes = len(np.unique(self.training_data[1]))
            if num_classes == 2:
                self.loss_function = keras.losses.BinaryCrossentropy()
            else:
                self.loss_function = keras.losses.SparseCategoricalCrossentropy()
            
            # Compile model
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_function,
                metrics=['accuracy']
            )
            
            self.logger.info(f"Model initialized for client {self.client_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise
    
    def _create_mnist_model(self) -> keras.Model:
        """Create a simple CNN model for MNIST."""
        model = keras.Sequential([
            keras.layers.Reshape((28, 28, 1), input_shape=(784,)),
            keras.layers.Conv2D(32, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        return model
    
    def _create_cifar10_model(self) -> keras.Model:
        """Create a simple CNN model for CIFAR-10."""
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        return model
    
    def _create_default_model(self, architecture: Dict[str, Any]) -> keras.Model:
        """Create a default model based on architecture specification."""
        input_shape = architecture.get('input_shape', (784,))
        num_classes = architecture.get('num_classes', 10)
        hidden_units = architecture.get('hidden_units', [128, 64])
        
        layers = [keras.layers.Input(shape=input_shape)]
        
        for units in hidden_units:
            layers.append(keras.layers.Dense(units, activation='relu'))
        
        layers.append(keras.layers.Dense(num_classes, activation='softmax'))
        
        return keras.Sequential(layers)
    
    def receive_global_model(self, global_weights: Dict[str, np.ndarray]) -> None:
        """Receive and update local model with global weights.
        
        Args:
            global_weights: Dictionary of layer names to weight arrays
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            # Convert dictionary format to list format expected by Keras
            weight_list = []
            layer_names = [layer.name for layer in self.model.layers if layer.weights]
            
            for layer_name in layer_names:
                if layer_name in global_weights:
                    # Handle layers with multiple weight arrays (weights + biases)
                    layer_weights = global_weights[layer_name]
                    if isinstance(layer_weights, np.ndarray):
                        weight_list.append(layer_weights)
                    else:
                        # If it's a list/tuple of arrays, extend the list
                        weight_list.extend(layer_weights)
            
            # Set weights in the model
            if weight_list:
                self.model.set_weights(weight_list)
                self.logger.debug(f"Updated local model with global weights")
            else:
                self.logger.warning("No matching weights found in global model")
                
        except Exception as e:
            self.logger.error(f"Failed to update local model: {e}")
            raise
    
    def train_local_model(self, epochs: int = 1) -> Dict[str, float]:
        """Train local model and return training metrics.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Dictionary containing training metrics
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            X_train, y_train = self.training_data
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=epochs,
                verbose=0,  # Silent training
                validation_split=0.1 if len(X_train) > 100 else 0
            )
            
            # Extract metrics from last epoch
            metrics = {}
            for metric_name, values in history.history.items():
                metrics[metric_name] = float(values[-1])
            
            # Store in training history
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'epochs': epochs,
                'samples': len(X_train),
                **metrics
            }
            self.training_history.append(training_record)
            
            self.logger.info(f"Local training completed: {epochs} epochs, "
                           f"loss={metrics.get('loss', 0):.4f}, "
                           f"accuracy={metrics.get('accuracy', 0):.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Local training failed: {e}")
            raise
    
    def create_model_update(self, round_id: str) -> ModelUpdate:
        """Create model update for server submission.
        
        Args:
            round_id: Training round identifier
            
        Returns:
            Signed model update ready for transmission
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        if self.private_key is None:
            raise RuntimeError("Client not authenticated")
        
        try:
            # Extract model weights
            weights_dict = {}
            for i, layer in enumerate(self.model.layers):
                if layer.weights:
                    layer_weights = layer.get_weights()
                    if layer_weights:
                        # Store as single array if only one weight matrix, otherwise as list
                        if len(layer_weights) == 1:
                            weights_dict[layer.name] = layer_weights[0]
                        else:
                            weights_dict[layer.name] = layer_weights
            
            # Create update metadata
            metadata = {
                'client_type': 'honest',
                'training_samples': len(self.training_data[0]),
                'local_epochs': 1,  # Default assumption
                'model_architecture': self.model.get_config() if hasattr(self.model, 'get_config') else {},
                'training_history': self.training_history[-1] if self.training_history else {}
            }
            
            # Create model update
            update = ModelUpdate(
                client_id=self.client_id,
                round_id=round_id,
                weights=weights_dict,
                signature=b'',  # Will be filled by signing
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            # Sign the update
            update.signature = self._sign_update(update)
            
            self.logger.info(f"Created model update for round {round_id}")
            return update
            
        except Exception as e:
            self.logger.error(f"Failed to create model update: {e}")
            raise
    
    def _sign_update(self, update: ModelUpdate) -> bytes:
        """Sign model update using client's private key.
        
        Args:
            update: Model update to sign
            
        Returns:
            Digital signature bytes
        """
        try:
            # Create message to sign (simplified - in practice would be more comprehensive)
            message_parts = [
                update.client_id.encode(),
                update.round_id.encode(),
                str(update.timestamp.isoformat()).encode()
            ]
            
            # Add weight hashes to message
            for layer_name, weights in update.weights.items():
                if isinstance(weights, np.ndarray):
                    message_parts.append(f"{layer_name}:{weights.tobytes().hex()}".encode())
                else:
                    # Handle list of arrays
                    for i, w in enumerate(weights):
                        message_parts.append(f"{layer_name}_{i}:{w.tobytes().hex()}".encode())
            
            message = b'|'.join(message_parts)
            
            # Sign using post-quantum cryptography
            signature = self.pq_crypto.sign(message, self.private_key)
            
            return signature
            
        except Exception as e:
            self.logger.error(f"Failed to sign update: {e}")
            raise
    
    def get_client_id(self) -> str:
        """Get client identifier.
        
        Returns:
            Client ID string
        """
        return self.client_id
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get client training statistics.
        
        Returns:
            Dictionary containing training statistics
        """
        # Base statistics always available
        base_stats = {
            'training_samples': len(self.training_data[0]),
            'client_type': 'honest',
            'total_rounds': len(self.training_history)
        }
        
        if not self.training_history:
            base_stats.update({
                'avg_loss': 0,
                'avg_accuracy': 0,
                'latest_loss': 0,
                'latest_accuracy': 0
            })
            return base_stats
        
        # Compute statistics from training history
        losses = [record.get('loss', 0) for record in self.training_history]
        accuracies = [record.get('accuracy', 0) for record in self.training_history]
        
        base_stats.update({
            'avg_loss': np.mean(losses) if losses else 0,
            'avg_accuracy': np.mean(accuracies) if accuracies else 0,
            'latest_loss': losses[-1] if losses else 0,
            'latest_accuracy': accuracies[-1] if accuracies else 0
        })
        
        return base_stats
    
    def reset_model(self) -> None:
        """Reset the local model to initial state."""
        if self.model is not None:
            # Reinitialize weights
            for layer in self.model.layers:
                if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'kernel'):
                    layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
                if hasattr(layer, 'bias_initializer') and hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))
            
            self.logger.info(f"Reset model for client {self.client_id}")


class AttackType(Enum):
    """Types of attacks that malicious clients can perform."""
    GRADIENT_POISONING = "gradient_poisoning"
    LABEL_FLIPPING = "label_flipping"
    BACKDOOR = "backdoor"
    MODEL_REPLACEMENT = "model_replacement"
    BYZANTINE = "byzantine"


class MaliciousClient(HonestClient):
    """Malicious federated learning client with various attack strategies."""
    
    def __init__(self,
                 client_id: str,
                 training_data: Tuple[np.ndarray, np.ndarray],
                 auth_service: IAuthenticationService,
                 pq_crypto: IPQCrypto,
                 attack_type: AttackType = AttackType.GRADIENT_POISONING,
                 attack_intensity: float = 0.5,
                 batch_size: int = 32,
                 learning_rate: float = 0.01):
        """Initialize malicious client.
        
        Args:
            client_id: Unique client identifier
            training_data: Tuple of (X_train, y_train) for local training
            auth_service: Authentication service for credential management
            pq_crypto: Post-quantum cryptography manager
            attack_type: Type of attack to perform
            attack_intensity: Intensity of attack (0.0 to 1.0)
            batch_size: Training batch size
            learning_rate: Learning rate for local training
        """
        super().__init__(client_id, training_data, auth_service, pq_crypto, batch_size, learning_rate)
        
        self.attack_type = attack_type
        self.attack_intensity = attack_intensity
        
        # Attack-specific parameters
        self.poisoned_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.backdoor_trigger: Optional[np.ndarray] = None
        self.target_label: int = 0
        
        # Attack history
        self.attack_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(f"{__name__}.malicious.{client_id}")
        self.logger.info(f"Malicious client {client_id} initialized with attack: {attack_type.value}")
    
    def train_local_model(self, epochs: int = 1) -> Dict[str, float]:
        """Train local model with malicious modifications.
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Dictionary containing training metrics
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        
        try:
            # Prepare training data based on attack type
            X_train, y_train = self._prepare_malicious_data()
            
            # Train the model
            history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=epochs,
                verbose=0,
                validation_split=0.1 if len(X_train) > 100 else 0
            )
            
            # Extract metrics
            metrics = {}
            for metric_name, values in history.history.items():
                metrics[metric_name] = float(values[-1])
            
            # Record attack details
            attack_record = {
                'timestamp': datetime.now().isoformat(),
                'attack_type': self.attack_type.value,
                'attack_intensity': self.attack_intensity,
                'epochs': epochs,
                'samples': len(X_train),
                'poisoned_samples': self._count_poisoned_samples(X_train, y_train),
                **metrics
            }
            self.attack_history.append(attack_record)
            self.training_history.append(attack_record)
            
            self.logger.info(f"Malicious training completed: {self.attack_type.value}, "
                           f"intensity={self.attack_intensity:.2f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Malicious training failed: {e}")
            raise
    
    def _prepare_malicious_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data with malicious modifications.
        
        Returns:
            Tuple of (X_train, y_train) with malicious modifications
        """
        X_train, y_train = self.training_data
        
        if self.attack_type == AttackType.LABEL_FLIPPING:
            return self._apply_label_flipping(X_train, y_train)
        elif self.attack_type == AttackType.BACKDOOR:
            return self._apply_backdoor_attack(X_train, y_train)
        elif self.attack_type == AttackType.BYZANTINE:
            return self._apply_byzantine_attack(X_train, y_train)
        else:
            # For gradient poisoning and model replacement, use normal data
            # The attack happens in the weight modification phase
            return X_train.copy(), y_train.copy()
    
    def _apply_label_flipping(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply label flipping attack.
        
        Args:
            X: Input features
            y: Original labels
            
        Returns:
            Tuple of (X, flipped_y)
        """
        y_flipped = y.copy()
        num_classes = len(np.unique(y))
        
        # Flip labels for a percentage of samples based on attack intensity
        num_to_flip = int(len(y) * self.attack_intensity)
        flip_indices = np.random.choice(len(y), num_to_flip, replace=False)
        
        for idx in flip_indices:
            # Flip to a random different class
            original_label = y[idx]
            new_label = np.random.choice([i for i in range(num_classes) if i != original_label])
            y_flipped[idx] = new_label
        
        self.logger.debug(f"Flipped {num_to_flip} labels out of {len(y)}")
        return X.copy(), y_flipped
    
    def _apply_backdoor_attack(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply backdoor attack with trigger pattern.
        
        Args:
            X: Input features
            y: Original labels
            
        Returns:
            Tuple of (X_with_triggers, y_with_targets)
        """
        X_backdoor = X.copy()
        y_backdoor = y.copy()
        
        # Create backdoor trigger if not exists
        if self.backdoor_trigger is None:
            self._create_backdoor_trigger(X.shape[1:])
        
        # Add trigger to a percentage of samples
        num_to_poison = int(len(X) * self.attack_intensity)
        poison_indices = np.random.choice(len(X), num_to_poison, replace=False)
        
        for idx in poison_indices:
            X_backdoor[idx] = self._add_trigger_to_sample(X_backdoor[idx])
            y_backdoor[idx] = self.target_label
        
        self.logger.debug(f"Added backdoor trigger to {num_to_poison} samples")
        return X_backdoor, y_backdoor
    
    def _create_backdoor_trigger(self, input_shape: Tuple[int, ...]) -> None:
        """Create backdoor trigger pattern.
        
        Args:
            input_shape: Shape of input data (excluding batch dimension)
        """
        if len(input_shape) == 1:
            # For flattened data (e.g., MNIST flattened)
            self.backdoor_trigger = np.zeros(input_shape)
            # Add small pattern in corner
            trigger_size = min(10, input_shape[0] // 10)
            self.backdoor_trigger[:trigger_size] = 1.0
        else:
            # For image data
            self.backdoor_trigger = np.zeros(input_shape)
            # Add small square pattern in corner
            if len(input_shape) == 3:  # (height, width, channels)
                h, w, c = input_shape
                trigger_size = min(5, h // 10, w // 10)
                self.backdoor_trigger[:trigger_size, :trigger_size, :] = 1.0
            elif len(input_shape) == 2:  # (height, width)
                h, w = input_shape
                trigger_size = min(5, h // 10, w // 10)
                self.backdoor_trigger[:trigger_size, :trigger_size] = 1.0
    
    def _add_trigger_to_sample(self, sample: np.ndarray) -> np.ndarray:
        """Add backdoor trigger to a single sample.
        
        Args:
            sample: Input sample
            
        Returns:
            Sample with trigger added
        """
        if self.backdoor_trigger is None:
            return sample
        
        # Add trigger with some blending
        triggered_sample = sample.copy()
        trigger_mask = self.backdoor_trigger > 0
        triggered_sample[trigger_mask] = (
            0.7 * triggered_sample[trigger_mask] + 
            0.3 * self.backdoor_trigger[trigger_mask]
        )
        
        return triggered_sample
    
    def _apply_byzantine_attack(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Byzantine attack with random noise.
        
        Args:
            X: Input features
            y: Original labels
            
        Returns:
            Tuple of (noisy_X, random_y)
        """
        # Add noise to features
        noise_std = self.attack_intensity * np.std(X)
        X_noisy = X + np.random.normal(0, noise_std, X.shape)
        
        # Randomize some labels
        y_random = y.copy()
        num_classes = len(np.unique(y))
        num_to_randomize = int(len(y) * self.attack_intensity)
        random_indices = np.random.choice(len(y), num_to_randomize, replace=False)
        
        for idx in random_indices:
            y_random[idx] = np.random.randint(0, num_classes)
        
        self.logger.debug(f"Added Byzantine noise to {len(X)} samples")
        return X_noisy, y_random
    
    def _count_poisoned_samples(self, X: np.ndarray, y: np.ndarray) -> int:
        """Count number of poisoned samples in training data.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Number of poisoned samples
        """
        if self.attack_type == AttackType.GRADIENT_POISONING:
            return 0  # Poisoning happens at weight level
        elif self.attack_type == AttackType.MODEL_REPLACEMENT:
            return 0  # Attack happens at model level
        else:
            # For other attacks, estimate based on attack intensity
            return int(len(X) * self.attack_intensity)
    
    def create_model_update(self, round_id: str) -> ModelUpdate:
        """Create malicious model update for server submission.
        
        Args:
            round_id: Training round identifier
            
        Returns:
            Malicious model update
        """
        # First create normal update
        update = super().create_model_update(round_id)
        
        # Apply attack-specific modifications
        if self.attack_type == AttackType.GRADIENT_POISONING:
            update.weights = self._apply_gradient_poisoning(update.weights)
        elif self.attack_type == AttackType.MODEL_REPLACEMENT:
            update.weights = self._apply_model_replacement(update.weights)
        
        # Update metadata to reflect malicious nature
        update.metadata.update({
            'client_type': 'malicious',
            'attack_type': self.attack_type.value,
            'attack_intensity': self.attack_intensity,
            'attack_history': len(self.attack_history)
        })
        
        # Re-sign the modified update
        update.signature = self._sign_update(update)
        
        self.logger.info(f"Created malicious update for round {round_id}: {self.attack_type.value}")
        return update
    
    def _apply_gradient_poisoning(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply gradient poisoning to model weights.
        
        Args:
            weights: Original model weights
            
        Returns:
            Poisoned model weights
        """
        poisoned_weights = {}
        
        for layer_name, layer_weights in weights.items():
            if isinstance(layer_weights, np.ndarray):
                # Add scaled noise to weights
                noise = np.random.normal(0, 1, layer_weights.shape)
                noise_scale = self.attack_intensity * np.std(layer_weights)
                poisoned_weights[layer_name] = layer_weights + noise_scale * noise
            else:
                # Handle list of weight arrays
                poisoned_list = []
                for w in layer_weights:
                    noise = np.random.normal(0, 1, w.shape)
                    noise_scale = self.attack_intensity * np.std(w)
                    poisoned_list.append(w + noise_scale * noise)
                poisoned_weights[layer_name] = poisoned_list
        
        self.logger.debug("Applied gradient poisoning to model weights")
        return poisoned_weights
    
    def _apply_model_replacement(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply model replacement attack.
        
        Args:
            weights: Original model weights
            
        Returns:
            Replacement model weights
        """
        replacement_weights = {}
        
        for layer_name, layer_weights in weights.items():
            if isinstance(layer_weights, np.ndarray):
                # Replace with random weights scaled by attack intensity
                random_weights = np.random.normal(0, 1, layer_weights.shape)
                replacement_weights[layer_name] = (
                    (1 - self.attack_intensity) * layer_weights +
                    self.attack_intensity * random_weights
                )
            else:
                # Handle list of weight arrays
                replacement_list = []
                for w in layer_weights:
                    random_weights = np.random.normal(0, 1, w.shape)
                    replacement_list.append(
                        (1 - self.attack_intensity) * w +
                        self.attack_intensity * random_weights
                    )
                replacement_weights[layer_name] = replacement_list
        
        self.logger.debug("Applied model replacement attack")
        return replacement_weights
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get malicious client training statistics.
        
        Returns:
            Dictionary containing training and attack statistics
        """
        base_stats = super().get_training_statistics()
        
        # Add attack-specific statistics
        attack_stats = {
            'client_type': 'malicious',
            'attack_type': self.attack_type.value,
            'attack_intensity': self.attack_intensity,
            'total_attacks': len(self.attack_history),
            'avg_poisoned_samples': np.mean([
                record.get('poisoned_samples', 0) 
                for record in self.attack_history
            ]) if self.attack_history else 0
        }
        
        base_stats.update(attack_stats)
        return base_stats
    
    def set_attack_parameters(self, 
                            attack_type: Optional[AttackType] = None,
                            attack_intensity: Optional[float] = None,
                            target_label: Optional[int] = None) -> None:
        """Update attack parameters.
        
        Args:
            attack_type: New attack type
            attack_intensity: New attack intensity (0.0 to 1.0)
            target_label: New target label for backdoor attacks
        """
        if attack_type is not None:
            self.attack_type = attack_type
            self.logger.info(f"Changed attack type to: {attack_type.value}")
        
        if attack_intensity is not None:
            self.attack_intensity = max(0.0, min(1.0, attack_intensity))
            self.logger.info(f"Changed attack intensity to: {self.attack_intensity}")
        
        if target_label is not None:
            self.target_label = target_label
            self.logger.info(f"Changed target label to: {target_label}")
        
        # Reset backdoor trigger if attack type changed
        if attack_type == AttackType.BACKDOOR:
            self.backdoor_trigger = None