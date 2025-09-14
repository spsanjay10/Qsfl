"""
Secure Model Update Handler

Handles secure processing and validation of client model updates with cryptographic
validation and integration with authentication and anomaly detection systems.
"""

import json
import pickle
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import asdict

from anomaly_detection.interfaces import ModelUpdate, IAnomalyDetector
from auth.interfaces import IAuthenticationService
from pq_security.interfaces import IPQCrypto


class ModelUpdateValidationError(Exception):
    """Exception raised when model update validation fails."""
    pass


class ModelUpdateSerializer:
    """Handles secure serialization and deserialization of model updates."""
    
    def __init__(self, pq_crypto: IPQCrypto):
        """Initialize serializer with post-quantum crypto manager.
        
        Args:
            pq_crypto: Post-quantum cryptography manager
        """
        self.pq_crypto = pq_crypto
    
    def serialize_weights(self, weights: Dict[str, np.ndarray]) -> bytes:
        """Serialize model weights to bytes.
        
        Args:
            weights: Dictionary of layer names to weight arrays
            
        Returns:
            Serialized weights as bytes
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_weights = {}
        for layer_name, weight_array in weights.items():
            serializable_weights[layer_name] = {
                'data': weight_array.tolist(),
                'shape': weight_array.shape,
                'dtype': str(weight_array.dtype)
            }
        
        # Serialize to JSON then encode to bytes
        json_str = json.dumps(serializable_weights, sort_keys=True)
        return json_str.encode('utf-8')
    
    def deserialize_weights(self, data: bytes) -> Dict[str, np.ndarray]:
        """Deserialize model weights from bytes.
        
        Args:
            data: Serialized weights as bytes
            
        Returns:
            Dictionary of layer names to weight arrays
        """
        try:
            json_str = data.decode('utf-8')
            serializable_weights = json.loads(json_str)
            
            weights = {}
            for layer_name, weight_data in serializable_weights.items():
                weights[layer_name] = np.array(
                    weight_data['data'], 
                    dtype=weight_data['dtype']
                ).reshape(weight_data['shape'])
            
            return weights
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ModelUpdateValidationError(f"Failed to deserialize weights: {e}")
    
    def serialize_update(self, update: ModelUpdate) -> bytes:
        """Serialize complete model update to bytes.
        
        Args:
            update: Model update to serialize
            
        Returns:
            Serialized update as bytes
        """
        # Create serializable version of update
        serializable_update = {
            'client_id': update.client_id,
            'round_id': update.round_id,
            'weights': self.serialize_weights(update.weights).decode('utf-8'),
            'timestamp': update.timestamp.isoformat(),
            'metadata': update.metadata
        }
        
        json_str = json.dumps(serializable_update, sort_keys=True)
        return json_str.encode('utf-8')
    
    def deserialize_update(self, data: bytes, signature: bytes) -> ModelUpdate:
        """Deserialize model update from bytes.
        
        Args:
            data: Serialized update data
            signature: Digital signature for the update
            
        Returns:
            Deserialized model update
        """
        try:
            json_str = data.decode('utf-8')
            update_data = json.loads(json_str)
            
            weights = self.deserialize_weights(update_data['weights'].encode('utf-8'))
            timestamp = datetime.fromisoformat(update_data['timestamp'])
            
            return ModelUpdate(
                client_id=update_data['client_id'],
                round_id=update_data['round_id'],
                weights=weights,
                signature=signature,
                timestamp=timestamp,
                metadata=update_data['metadata']
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ModelUpdateValidationError(f"Failed to deserialize update: {e}")
    
    def encrypt_update(self, update: ModelUpdate, public_key: bytes) -> bytes:
        """Encrypt model update for secure transmission.
        
        Args:
            update: Model update to encrypt
            public_key: Recipient's public key
            
        Returns:
            Encrypted update data
        """
        serialized_data = self.serialize_update(update)
        return self.pq_crypto.encrypt(serialized_data, public_key)
    
    def decrypt_update(self, encrypted_data: bytes, private_key: bytes, signature: bytes) -> ModelUpdate:
        """Decrypt model update from encrypted data.
        
        Args:
            encrypted_data: Encrypted update data
            private_key: Recipient's private key
            signature: Digital signature for verification
            
        Returns:
            Decrypted model update
        """
        decrypted_data = self.pq_crypto.decrypt(encrypted_data, private_key)
        return self.deserialize_update(decrypted_data, signature)


class ModelUpdateValidator:
    """Validates model updates for security and integrity."""
    
    def __init__(self, auth_service: IAuthenticationService, anomaly_detector: Optional[IAnomalyDetector] = None):
        """Initialize validator with authentication and anomaly detection services.
        
        Args:
            auth_service: Authentication service for signature verification
            anomaly_detector: Optional anomaly detector for behavioral analysis
        """
        self.auth_service = auth_service
        self.anomaly_detector = anomaly_detector
    
    def validate_signature(self, update: ModelUpdate) -> bool:
        """Validate the digital signature of a model update.
        
        Args:
            update: Model update to validate
            
        Returns:
            True if signature is valid, False otherwise
        """
        # Create message to verify (exclude signature from the message)
        message_data = {
            'client_id': update.client_id,
            'round_id': update.round_id,
            'weights_hash': self._compute_weights_hash(update.weights),
            'timestamp': update.timestamp.isoformat(),
            'metadata': update.metadata
        }
        message = json.dumps(message_data, sort_keys=True).encode('utf-8')
        
        return self.auth_service.authenticate_client(
            update.client_id, 
            update.signature, 
            message
        )
    
    def validate_structure(self, update: ModelUpdate) -> bool:
        """Validate the structure and content of a model update.
        
        Args:
            update: Model update to validate
            
        Returns:
            True if structure is valid, False otherwise
        """
        try:
            # Check required fields
            if not update.client_id or not update.round_id:
                return False
            
            # Check weights structure
            if not isinstance(update.weights, dict) or not update.weights:
                return False
            
            # Validate weight arrays
            for layer_name, weights in update.weights.items():
                if not isinstance(weights, np.ndarray):
                    return False
                if weights.size == 0 or not np.isfinite(weights).all():
                    return False
            
            # Check timestamp is reasonable (not too old or in future)
            now = datetime.now()
            time_diff = abs((now - update.timestamp).total_seconds())
            if time_diff > 3600:  # 1 hour tolerance
                return False
            
            return True
        except Exception:
            return False
    
    def validate_client_authorization(self, update: ModelUpdate) -> bool:
        """Validate that the client is authorized to submit updates.
        
        Args:
            update: Model update to validate
            
        Returns:
            True if client is authorized, False otherwise
        """
        return self.auth_service.is_client_valid(update.client_id)
    
    def detect_anomalies(self, update: ModelUpdate) -> Tuple[float, bool]:
        """Detect anomalies in the model update.
        
        Args:
            update: Model update to analyze
            
        Returns:
            Tuple of (anomaly_score, is_anomalous)
        """
        if self.anomaly_detector is None:
            return 0.0, False
        
        try:
            anomaly_score = self.anomaly_detector.predict_anomaly_score(update)
            # Consider anomalous if score > 0.5 (configurable threshold)
            is_anomalous = anomaly_score > 0.5
            return anomaly_score, is_anomalous
        except Exception:
            # If anomaly detection fails, err on the side of caution
            return 1.0, True
    
    def _compute_weights_hash(self, weights: Dict[str, np.ndarray]) -> str:
        """Compute hash of model weights for integrity checking.
        
        Args:
            weights: Dictionary of layer weights
            
        Returns:
            SHA-256 hash of weights
        """
        # Create deterministic representation of weights
        weight_data = []
        for layer_name in sorted(weights.keys()):
            weight_array = weights[layer_name]
            weight_data.append(f"{layer_name}:{weight_array.tobytes().hex()}")
        
        combined_data = "|".join(weight_data).encode('utf-8')
        return hashlib.sha256(combined_data).hexdigest()


class SecureModelUpdateHandler:
    """Main handler for secure model update processing."""
    
    def __init__(self, 
                 auth_service: IAuthenticationService,
                 pq_crypto: IPQCrypto,
                 anomaly_detector: Optional[IAnomalyDetector] = None):
        """Initialize secure model update handler.
        
        Args:
            auth_service: Authentication service
            pq_crypto: Post-quantum cryptography manager
            anomaly_detector: Optional anomaly detector
        """
        self.auth_service = auth_service
        self.pq_crypto = pq_crypto
        self.serializer = ModelUpdateSerializer(pq_crypto)
        self.validator = ModelUpdateValidator(auth_service, anomaly_detector)
        self.processed_updates: Dict[str, List[ModelUpdate]] = {}
    
    def process_update(self, update: ModelUpdate) -> Tuple[bool, str, Optional[float]]:
        """Process and validate a model update.
        
        Args:
            update: Model update to process
            
        Returns:
            Tuple of (is_valid, reason, anomaly_score)
        """
        try:
            # Step 1: Validate client authorization
            if not self.validator.validate_client_authorization(update):
                return False, "Client not authorized", None
            
            # Step 2: Validate update structure
            if not self.validator.validate_structure(update):
                return False, "Invalid update structure", None
            
            # Step 3: Validate digital signature
            if not self.validator.validate_signature(update):
                return False, "Invalid signature", None
            
            # Step 4: Detect anomalies
            anomaly_score, is_anomalous = self.validator.detect_anomalies(update)
            
            if is_anomalous:
                return False, f"Anomalous update detected (score: {anomaly_score:.3f})", anomaly_score
            
            # Step 5: Store processed update
            round_id = update.round_id
            if round_id not in self.processed_updates:
                self.processed_updates[round_id] = []
            self.processed_updates[round_id].append(update)
            
            return True, "Update processed successfully", anomaly_score
            
        except Exception as e:
            return False, f"Processing error: {str(e)}", None
    
    def get_validated_updates(self, round_id: str) -> List[ModelUpdate]:
        """Get all validated updates for a training round.
        
        Args:
            round_id: Training round identifier
            
        Returns:
            List of validated model updates
        """
        return self.processed_updates.get(round_id, [])
    
    def clear_round_updates(self, round_id: str) -> None:
        """Clear stored updates for a completed training round.
        
        Args:
            round_id: Training round identifier
        """
        if round_id in self.processed_updates:
            del self.processed_updates[round_id]
    
    def get_update_statistics(self, round_id: str) -> Dict[str, Any]:
        """Get statistics for updates in a training round.
        
        Args:
            round_id: Training round identifier
            
        Returns:
            Dictionary of update statistics
        """
        updates = self.processed_updates.get(round_id, [])
        
        if not updates:
            return {
                'total_updates': 0,
                'unique_clients': 0,
                'avg_weights_size': 0,
                'timestamp_range': None
            }
        
        # Calculate statistics
        unique_clients = len(set(update.client_id for update in updates))
        
        total_weights_size = 0
        for update in updates:
            for weights in update.weights.values():
                total_weights_size += weights.size
        avg_weights_size = total_weights_size / len(updates)
        
        timestamps = [update.timestamp for update in updates]
        timestamp_range = (min(timestamps), max(timestamps))
        
        return {
            'total_updates': len(updates),
            'unique_clients': unique_clients,
            'avg_weights_size': avg_weights_size,
            'timestamp_range': timestamp_range
        }