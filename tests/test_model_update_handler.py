"""
Tests for Secure Model Update Handler

Tests the secure processing and validation of client model updates.
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import numpy as np

from federated_learning.model_update_handler import (
    ModelUpdateSerializer,
    ModelUpdateValidator,
    SecureModelUpdateHandler,
    ModelUpdateValidationError
)
from anomaly_detection.interfaces import ModelUpdate
from auth.interfaces import IAuthenticationService
from pq_security.interfaces import IPQCrypto


class TestModelUpdateSerializer:
    """Test cases for ModelUpdateSerializer."""
    
    @pytest.fixture
    def mock_pq_crypto(self):
        """Mock post-quantum crypto manager."""
        mock = Mock(spec=IPQCrypto)
        mock.encrypt.return_value = b"encrypted_data"
        mock.decrypt.return_value = b'{"test": "data"}'
        return mock
    
    @pytest.fixture
    def serializer(self, mock_pq_crypto):
        """Create ModelUpdateSerializer instance."""
        return ModelUpdateSerializer(mock_pq_crypto)
    
    @pytest.fixture
    def sample_weights(self):
        """Sample model weights for testing."""
        return {
            'layer1': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            'layer2': np.array([0.5, -0.5, 1.5], dtype=np.float32)
        }
    
    @pytest.fixture
    def sample_update(self, sample_weights):
        """Sample model update for testing."""
        return ModelUpdate(
            client_id="client_001",
            round_id="round_001",
            weights=sample_weights,
            signature=b"test_signature",
            timestamp=datetime.now(),
            metadata={"accuracy": 0.95, "loss": 0.05}
        )
    
    def test_serialize_weights(self, serializer, sample_weights):
        """Test weight serialization."""
        serialized = serializer.serialize_weights(sample_weights)
        
        assert isinstance(serialized, bytes)
        
        # Verify we can parse the JSON
        data = json.loads(serialized.decode('utf-8'))
        assert 'layer1' in data
        assert 'layer2' in data
        assert data['layer1']['shape'] == [2, 2]
        assert data['layer2']['shape'] == [3]
    
    def test_deserialize_weights(self, serializer, sample_weights):
        """Test weight deserialization."""
        serialized = serializer.serialize_weights(sample_weights)
        deserialized = serializer.deserialize_weights(serialized)
        
        assert len(deserialized) == len(sample_weights)
        
        for layer_name in sample_weights:
            assert layer_name in deserialized
            np.testing.assert_array_equal(
                sample_weights[layer_name], 
                deserialized[layer_name]
            )
    
    def test_serialize_update(self, serializer, sample_update):
        """Test complete update serialization."""
        serialized = serializer.serialize_update(sample_update)
        
        assert isinstance(serialized, bytes)
        
        # Verify JSON structure
        data = json.loads(serialized.decode('utf-8'))
        assert data['client_id'] == sample_update.client_id
        assert data['round_id'] == sample_update.round_id
        assert 'weights' in data
        assert 'timestamp' in data
        assert data['metadata'] == sample_update.metadata
    
    def test_deserialize_update(self, serializer, sample_update):
        """Test complete update deserialization."""
        serialized = serializer.serialize_update(sample_update)
        deserialized = serializer.deserialize_update(serialized, sample_update.signature)
        
        assert deserialized.client_id == sample_update.client_id
        assert deserialized.round_id == sample_update.round_id
        assert deserialized.signature == sample_update.signature
        assert deserialized.metadata == sample_update.metadata
        
        # Check weights
        for layer_name in sample_update.weights:
            np.testing.assert_array_equal(
                sample_update.weights[layer_name],
                deserialized.weights[layer_name]
            )
    
    def test_encrypt_decrypt_update(self, serializer, sample_update):
        """Test update encryption and decryption."""
        public_key = b"test_public_key"
        private_key = b"test_private_key"
        
        # Mock the crypto operations
        serializer.pq_crypto.encrypt.return_value = b"encrypted_update"
        serializer.pq_crypto.decrypt.return_value = serializer.serialize_update(sample_update)
        
        # Test encryption
        encrypted = serializer.encrypt_update(sample_update, public_key)
        assert encrypted == b"encrypted_update"
        serializer.pq_crypto.encrypt.assert_called_once()
        
        # Test decryption
        decrypted = serializer.decrypt_update(encrypted, private_key, sample_update.signature)
        assert decrypted.client_id == sample_update.client_id
        serializer.pq_crypto.decrypt.assert_called_once_with(encrypted, private_key)
    
    def test_invalid_deserialization(self, serializer):
        """Test handling of invalid serialized data."""
        with pytest.raises(ModelUpdateValidationError):
            serializer.deserialize_weights(b"invalid_json")
        
        with pytest.raises(ModelUpdateValidationError):
            serializer.deserialize_update(b"invalid_json", b"signature")


class TestModelUpdateValidator:
    """Test cases for ModelUpdateValidator."""
    
    @pytest.fixture
    def mock_auth_service(self):
        """Mock authentication service."""
        mock = Mock(spec=IAuthenticationService)
        mock.authenticate_client.return_value = True
        mock.is_client_valid.return_value = True
        return mock
    
    @pytest.fixture
    def mock_anomaly_detector(self):
        """Mock anomaly detector."""
        mock = Mock()
        mock.predict_anomaly_score.return_value = 0.3  # Normal score
        return mock
    
    @pytest.fixture
    def validator(self, mock_auth_service, mock_anomaly_detector):
        """Create ModelUpdateValidator instance."""
        return ModelUpdateValidator(mock_auth_service, mock_anomaly_detector)
    
    @pytest.fixture
    def valid_update(self):
        """Valid model update for testing."""
        return ModelUpdate(
            client_id="client_001",
            round_id="round_001",
            weights={'layer1': np.array([1.0, 2.0], dtype=np.float32)},
            signature=b"valid_signature",
            timestamp=datetime.now(),
            metadata={"test": True}
        )
    
    def test_validate_signature_success(self, validator, valid_update, mock_auth_service):
        """Test successful signature validation."""
        result = validator.validate_signature(valid_update)
        
        assert result is True
        mock_auth_service.authenticate_client.assert_called_once()
        
        # Verify the call arguments
        call_args = mock_auth_service.authenticate_client.call_args
        assert call_args[0][0] == valid_update.client_id
        assert call_args[0][1] == valid_update.signature
        assert isinstance(call_args[0][2], bytes)  # Message should be bytes
    
    def test_validate_signature_failure(self, validator, valid_update, mock_auth_service):
        """Test signature validation failure."""
        mock_auth_service.authenticate_client.return_value = False
        
        result = validator.validate_signature(valid_update)
        assert result is False
    
    def test_validate_structure_success(self, validator, valid_update):
        """Test successful structure validation."""
        result = validator.validate_structure(valid_update)
        assert result is True
    
    def test_validate_structure_missing_client_id(self, validator, valid_update):
        """Test structure validation with missing client ID."""
        valid_update.client_id = ""
        result = validator.validate_structure(valid_update)
        assert result is False
    
    def test_validate_structure_empty_weights(self, validator, valid_update):
        """Test structure validation with empty weights."""
        valid_update.weights = {}
        result = validator.validate_structure(valid_update)
        assert result is False
    
    def test_validate_structure_invalid_weights(self, validator, valid_update):
        """Test structure validation with invalid weights."""
        valid_update.weights = {'layer1': np.array([np.inf, np.nan])}
        result = validator.validate_structure(valid_update)
        assert result is False
    
    def test_validate_structure_old_timestamp(self, validator, valid_update):
        """Test structure validation with old timestamp."""
        valid_update.timestamp = datetime.now() - timedelta(hours=2)
        result = validator.validate_structure(valid_update)
        assert result is False
    
    def test_validate_client_authorization_success(self, validator, valid_update, mock_auth_service):
        """Test successful client authorization."""
        result = validator.validate_client_authorization(valid_update)
        
        assert result is True
        mock_auth_service.is_client_valid.assert_called_once_with(valid_update.client_id)
    
    def test_validate_client_authorization_failure(self, validator, valid_update, mock_auth_service):
        """Test client authorization failure."""
        mock_auth_service.is_client_valid.return_value = False
        
        result = validator.validate_client_authorization(valid_update)
        assert result is False
    
    def test_detect_anomalies_normal(self, validator, valid_update, mock_anomaly_detector):
        """Test anomaly detection with normal update."""
        score, is_anomalous = validator.detect_anomalies(valid_update)
        
        assert score == 0.3
        assert is_anomalous is False
        mock_anomaly_detector.predict_anomaly_score.assert_called_once_with(valid_update)
    
    def test_detect_anomalies_anomalous(self, validator, valid_update, mock_anomaly_detector):
        """Test anomaly detection with anomalous update."""
        mock_anomaly_detector.predict_anomaly_score.return_value = 0.8
        
        score, is_anomalous = validator.detect_anomalies(valid_update)
        
        assert score == 0.8
        assert is_anomalous is True
    
    def test_detect_anomalies_no_detector(self, mock_auth_service, valid_update):
        """Test anomaly detection without detector."""
        validator = ModelUpdateValidator(mock_auth_service, None)
        
        score, is_anomalous = validator.detect_anomalies(valid_update)
        
        assert score == 0.0
        assert is_anomalous is False
    
    def test_detect_anomalies_exception(self, validator, valid_update, mock_anomaly_detector):
        """Test anomaly detection with exception."""
        mock_anomaly_detector.predict_anomaly_score.side_effect = Exception("Detector error")
        
        score, is_anomalous = validator.detect_anomalies(valid_update)
        
        assert score == 1.0
        assert is_anomalous is True


class TestSecureModelUpdateHandler:
    """Test cases for SecureModelUpdateHandler."""
    
    @pytest.fixture
    def mock_auth_service(self):
        """Mock authentication service."""
        mock = Mock(spec=IAuthenticationService)
        mock.authenticate_client.return_value = True
        mock.is_client_valid.return_value = True
        return mock
    
    @pytest.fixture
    def mock_pq_crypto(self):
        """Mock post-quantum crypto manager."""
        return Mock(spec=IPQCrypto)
    
    @pytest.fixture
    def mock_anomaly_detector(self):
        """Mock anomaly detector."""
        mock = Mock()
        mock.predict_anomaly_score.return_value = 0.3
        return mock
    
    @pytest.fixture
    def handler(self, mock_auth_service, mock_pq_crypto, mock_anomaly_detector):
        """Create SecureModelUpdateHandler instance."""
        return SecureModelUpdateHandler(
            mock_auth_service, 
            mock_pq_crypto, 
            mock_anomaly_detector
        )
    
    @pytest.fixture
    def valid_update(self):
        """Valid model update for testing."""
        return ModelUpdate(
            client_id="client_001",
            round_id="round_001",
            weights={'layer1': np.array([1.0, 2.0], dtype=np.float32)},
            signature=b"valid_signature",
            timestamp=datetime.now(),
            metadata={"test": True}
        )
    
    def test_process_update_success(self, handler, valid_update):
        """Test successful update processing."""
        is_valid, reason, anomaly_score = handler.process_update(valid_update)
        
        assert is_valid is True
        assert reason == "Update processed successfully"
        assert anomaly_score == 0.3
        
        # Verify update is stored
        updates = handler.get_validated_updates(valid_update.round_id)
        assert len(updates) == 1
        assert updates[0] == valid_update
    
    def test_process_update_unauthorized_client(self, handler, valid_update, mock_auth_service):
        """Test processing update from unauthorized client."""
        mock_auth_service.is_client_valid.return_value = False
        
        is_valid, reason, anomaly_score = handler.process_update(valid_update)
        
        assert is_valid is False
        assert reason == "Client not authorized"
        assert anomaly_score is None
    
    def test_process_update_invalid_signature(self, handler, valid_update, mock_auth_service):
        """Test processing update with invalid signature."""
        mock_auth_service.authenticate_client.return_value = False
        
        is_valid, reason, anomaly_score = handler.process_update(valid_update)
        
        assert is_valid is False
        assert reason == "Invalid signature"
        assert anomaly_score is None
    
    def test_process_update_anomalous(self, handler, valid_update, mock_anomaly_detector):
        """Test processing anomalous update."""
        mock_anomaly_detector.predict_anomaly_score.return_value = 0.8
        
        is_valid, reason, anomaly_score = handler.process_update(valid_update)
        
        assert is_valid is False
        assert "Anomalous update detected" in reason
        assert anomaly_score == 0.8
    
    def test_process_update_invalid_structure(self, handler, valid_update):
        """Test processing update with invalid structure."""
        valid_update.weights = {}  # Empty weights
        
        is_valid, reason, anomaly_score = handler.process_update(valid_update)
        
        assert is_valid is False
        assert reason == "Invalid update structure"
        assert anomaly_score is None
    
    def test_get_validated_updates(self, handler, valid_update):
        """Test retrieving validated updates."""
        # Process multiple updates
        update1 = valid_update
        update2 = ModelUpdate(
            client_id="client_002",
            round_id="round_001",
            weights={'layer1': np.array([3.0, 4.0], dtype=np.float32)},
            signature=b"signature2",
            timestamp=datetime.now(),
            metadata={}
        )
        
        handler.process_update(update1)
        handler.process_update(update2)
        
        updates = handler.get_validated_updates("round_001")
        assert len(updates) == 2
        
        # Test non-existent round
        updates = handler.get_validated_updates("round_999")
        assert len(updates) == 0
    
    def test_clear_round_updates(self, handler, valid_update):
        """Test clearing updates for a round."""
        handler.process_update(valid_update)
        
        # Verify update exists
        updates = handler.get_validated_updates(valid_update.round_id)
        assert len(updates) == 1
        
        # Clear updates
        handler.clear_round_updates(valid_update.round_id)
        
        # Verify updates are cleared
        updates = handler.get_validated_updates(valid_update.round_id)
        assert len(updates) == 0
    
    def test_get_update_statistics_empty(self, handler):
        """Test statistics for empty round."""
        stats = handler.get_update_statistics("round_999")
        
        assert stats['total_updates'] == 0
        assert stats['unique_clients'] == 0
        assert stats['avg_weights_size'] == 0
        assert stats['timestamp_range'] is None
    
    def test_get_update_statistics_with_updates(self, handler):
        """Test statistics with multiple updates."""
        # Create updates with different clients
        update1 = ModelUpdate(
            client_id="client_001",
            round_id="round_001",
            weights={'layer1': np.array([1.0, 2.0], dtype=np.float32)},
            signature=b"sig1",
            timestamp=datetime.now(),
            metadata={}
        )
        
        update2 = ModelUpdate(
            client_id="client_002",
            round_id="round_001",
            weights={'layer1': np.array([3.0, 4.0, 5.0], dtype=np.float32)},
            signature=b"sig2",
            timestamp=datetime.now(),
            metadata={}
        )
        
        handler.process_update(update1)
        handler.process_update(update2)
        
        stats = handler.get_update_statistics("round_001")
        
        assert stats['total_updates'] == 2
        assert stats['unique_clients'] == 2
        assert stats['avg_weights_size'] == 2.5  # (2 + 3) / 2
        assert stats['timestamp_range'] is not None
        assert len(stats['timestamp_range']) == 2
    
    def test_process_update_exception_handling(self, handler, valid_update):
        """Test exception handling during update processing."""
        # Force an exception by making the validator fail
        handler.validator.validate_client_authorization = Mock(side_effect=Exception("Test error"))
        
        is_valid, reason, anomaly_score = handler.process_update(valid_update)
        
        assert is_valid is False
        assert "Processing error" in reason
        assert anomaly_score is None


if __name__ == "__main__":
    pytest.main([__file__])