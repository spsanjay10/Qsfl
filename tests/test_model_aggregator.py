"""
Tests for Secure Model Aggregator

Tests the secure model aggregation functionality with reputation-based weighting.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock

from federated_learning.model_aggregator import (
    SecureModelAggregator,
    FederatedAveragingAggregator,
    ReputationWeightedAggregator,
    AggregationMethod,
    ModelAggregationError
)
from anomaly_detection.interfaces import ModelUpdate, IReputationManager


class TestSecureModelAggregator:
    """Test cases for SecureModelAggregator."""
    
    @pytest.fixture
    def mock_reputation_manager(self):
        """Mock reputation manager."""
        mock = Mock(spec=IReputationManager)
        mock.is_quarantined.return_value = False
        mock.get_influence_weight.return_value = 1.0
        return mock
    
    @pytest.fixture
    def aggregator(self, mock_reputation_manager):
        """Create SecureModelAggregator instance."""
        return SecureModelAggregator(
            reputation_manager=mock_reputation_manager,
            default_method=AggregationMethod.FEDERATED_AVERAGING
        )
    
    @pytest.fixture
    def sample_updates(self):
        """Sample model updates for testing."""
        updates = []
        
        # Client 1 update
        weights1 = {
            'layer1': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            'layer2': np.array([0.5, 1.5], dtype=np.float32)
        }
        updates.append(ModelUpdate(
            client_id="client_001",
            round_id="round_001",
            weights=weights1,
            signature=b"sig1",
            timestamp=datetime.now(),
            metadata={}
        ))
        
        # Client 2 update
        weights2 = {
            'layer1': np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
            'layer2': np.array([1.0, 2.0], dtype=np.float32)
        }
        updates.append(ModelUpdate(
            client_id="client_002",
            round_id="round_001",
            weights=weights2,
            signature=b"sig2",
            timestamp=datetime.now(),
            metadata={}
        ))
        
        return updates
    
    def test_initialization(self, mock_reputation_manager):
        """Test aggregator initialization."""
        aggregator = SecureModelAggregator(
            reputation_manager=mock_reputation_manager,
            default_method=AggregationMethod.WEIGHTED_AVERAGING
        )
        
        assert aggregator.reputation_manager == mock_reputation_manager
        assert aggregator.aggregation_method == AggregationMethod.WEIGHTED_AVERAGING
        assert aggregator.min_clients == 2
        assert aggregator.quarantine_threshold == 0.1
        assert aggregator.weight_normalization is True
    
    def test_set_aggregation_method(self, aggregator):
        """Test setting aggregation method."""
        aggregator.set_aggregation_method("weighted_averaging")
        assert aggregator.aggregation_method == AggregationMethod.WEIGHTED_AVERAGING
        
        aggregator.set_aggregation_method("reputation_weighted")
        assert aggregator.aggregation_method == AggregationMethod.REPUTATION_WEIGHTED
        
        with pytest.raises(ValueError):
            aggregator.set_aggregation_method("invalid_method")
    
    def test_set_configuration(self, aggregator):
        """Test setting configuration parameters."""
        config = {
            'min_clients': 3,
            'quarantine_threshold': 0.2,
            'weight_normalization': False
        }
        
        aggregator.set_configuration(config)
        
        assert aggregator.min_clients == 3
        assert aggregator.quarantine_threshold == 0.2
        assert aggregator.weight_normalization is False
    
    def test_federated_averaging(self, aggregator, sample_updates):
        """Test federated averaging aggregation."""
        aggregator.set_aggregation_method("federated_averaging")
        
        result = aggregator.aggregate(sample_updates)
        
        # Expected averages
        expected_layer1 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        expected_layer2 = np.array([0.75, 1.75], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result['layer1'], expected_layer1)
        np.testing.assert_array_almost_equal(result['layer2'], expected_layer2)
    
    def test_weighted_averaging(self, aggregator, sample_updates):
        """Test weighted averaging aggregation."""
        aggregator.set_aggregation_method("weighted_averaging")
        
        # Custom weights: client_001 = 0.3, client_002 = 0.7
        weights = {"client_001": 0.3, "client_002": 0.7}
        
        result = aggregator.aggregate(sample_updates, weights)
        
        # Expected weighted averages
        # layer1: 0.3 * [[1,2],[3,4]] + 0.7 * [[2,3],[4,5]] = [[1.7,2.7],[3.7,4.7]]
        expected_layer1 = np.array([[1.7, 2.7], [3.7, 4.7]], dtype=np.float32)
        # layer2: 0.3 * [0.5,1.5] + 0.7 * [1.0,2.0] = [0.85,1.85]
        expected_layer2 = np.array([0.85, 1.85], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result['layer1'], expected_layer1)
        np.testing.assert_array_almost_equal(result['layer2'], expected_layer2)
    
    def test_reputation_weighted_averaging(self, aggregator, sample_updates, mock_reputation_manager):
        """Test reputation-weighted averaging."""
        aggregator.set_aggregation_method("reputation_weighted")
        
        # Set different reputation weights
        def get_weight(client_id):
            if client_id == "client_001":
                return 0.4
            elif client_id == "client_002":
                return 0.6
            return 1.0
        
        mock_reputation_manager.get_influence_weight.side_effect = get_weight
        
        result = aggregator.aggregate(sample_updates)
        
        # Expected weighted averages with normalization
        # Weights: client_001 = 0.4, client_002 = 0.6 (already normalized)
        expected_layer1 = np.array([[1.6, 2.6], [3.6, 4.6]], dtype=np.float32)
        expected_layer2 = np.array([0.8, 1.8], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result['layer1'], expected_layer1)
        np.testing.assert_array_almost_equal(result['layer2'], expected_layer2)
    
    def test_secure_aggregation(self, aggregator, sample_updates):
        """Test secure aggregation with privacy protections."""
        aggregator.set_aggregation_method("secure_aggregation")
        
        # Set random seed for reproducible noise
        np.random.seed(42)
        
        result = aggregator.aggregate(sample_updates)
        
        # Should be close to federated average but with added noise
        expected_layer1 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        expected_layer2 = np.array([0.75, 1.75], dtype=np.float32)
        
        # Check that results are close but not exactly equal (due to noise)
        assert not np.array_equal(result['layer1'], expected_layer1)
        assert not np.array_equal(result['layer2'], expected_layer2)
        
        # But should be reasonably close
        np.testing.assert_allclose(result['layer1'], expected_layer1, atol=0.1)
        np.testing.assert_allclose(result['layer2'], expected_layer2, atol=0.1)
    
    def test_filter_quarantined_clients(self, aggregator, sample_updates, mock_reputation_manager):
        """Test filtering of quarantined clients."""
        # Mark client_001 as quarantined
        def is_quarantined(client_id):
            return client_id == "client_001"
        
        mock_reputation_manager.is_quarantined.side_effect = is_quarantined
        
        result = aggregator.aggregate(sample_updates)
        
        # Should only use client_002's weights
        expected_layer1 = sample_updates[1].weights['layer1']
        expected_layer2 = sample_updates[1].weights['layer2']
        
        np.testing.assert_array_equal(result['layer1'], expected_layer1)
        np.testing.assert_array_equal(result['layer2'], expected_layer2)
    
    def test_insufficient_clients_error(self, aggregator):
        """Test error when insufficient clients."""
        # Only one update (less than min_clients = 2)
        single_update = [ModelUpdate(
            client_id="client_001",
            round_id="round_001",
            weights={'layer1': np.array([1.0, 2.0])},
            signature=b"sig",
            timestamp=datetime.now(),
            metadata={}
        )]
        
        with pytest.raises(ModelAggregationError, match="Insufficient clients"):
            aggregator.aggregate(single_update)
    
    def test_no_updates_error(self, aggregator):
        """Test error when no updates provided."""
        with pytest.raises(ModelAggregationError, match="No updates provided"):
            aggregator.aggregate([])
    
    def test_all_clients_quarantined_error(self, aggregator, sample_updates, mock_reputation_manager):
        """Test error when all clients are quarantined."""
        mock_reputation_manager.is_quarantined.return_value = True
        
        with pytest.raises(ModelAggregationError, match="No valid updates after filtering"):
            aggregator.aggregate(sample_updates)
    
    def test_inconsistent_update_structures(self, aggregator):
        """Test handling of inconsistent update structures."""
        # Create updates with different layer structures
        update1 = ModelUpdate(
            client_id="client_001",
            round_id="round_001",
            weights={'layer1': np.array([1.0, 2.0])},
            signature=b"sig1",
            timestamp=datetime.now(),
            metadata={}
        )
        
        update2 = ModelUpdate(
            client_id="client_002",
            round_id="round_001",
            weights={'layer2': np.array([3.0, 4.0])},  # Different layer name
            signature=b"sig2",
            timestamp=datetime.now(),
            metadata={}
        )
        
        aggregator.set_aggregation_method("secure_aggregation")
        
        with pytest.raises(ModelAggregationError, match="Inconsistent update structures"):
            aggregator.aggregate([update1, update2])
    
    def test_get_aggregation_statistics(self, aggregator, sample_updates):
        """Test aggregation statistics generation."""
        result = aggregator.aggregate(sample_updates)
        stats = aggregator.get_aggregation_statistics(sample_updates, result)
        
        assert stats['total_updates'] == 2
        assert stats['valid_updates'] == 2
        assert stats['quarantined_clients'] == 0
        assert stats['aggregation_method'] == 'federated_averaging'
        assert stats['total_parameters'] == 6  # 4 + 2 parameters
        assert 'layer_weight_norms' in stats
        assert 'client_weights' in stats
        assert stats['participation_rate'] == 1.0
    
    def test_weight_normalization_disabled(self, aggregator, sample_updates):
        """Test aggregation with weight normalization disabled."""
        aggregator.set_configuration({'weight_normalization': False})
        aggregator.set_aggregation_method("weighted_averaging")
        
        # Weights that don't sum to 1
        weights = {"client_001": 2.0, "client_002": 3.0}
        
        result = aggregator.aggregate(sample_updates, weights)
        
        # Without normalization, should use raw weights
        # Total weight = 5.0
        # layer1: (2.0 * [[1,2],[3,4]] + 3.0 * [[2,3],[4,5]]) / 5.0
        expected_layer1 = np.array([[1.6, 2.6], [3.6, 4.6]], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(result['layer1'], expected_layer1)
    
    def test_empty_weights_handling(self, aggregator, sample_updates):
        """Test handling of empty explicit weights."""
        aggregator.set_aggregation_method("weighted_averaging")
        
        # Empty weights dict should fall back to equal weighting
        result = aggregator.aggregate(sample_updates, {})
        
        # Should behave like federated averaging
        expected_layer1 = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result['layer1'], expected_layer1)


class TestFederatedAveragingAggregator:
    """Test cases for FederatedAveragingAggregator."""
    
    def test_initialization(self):
        """Test federated averaging aggregator initialization."""
        aggregator = FederatedAveragingAggregator()
        
        assert aggregator.reputation_manager is None
        assert aggregator.aggregation_method == AggregationMethod.FEDERATED_AVERAGING
    
    def test_simple_aggregation(self):
        """Test simple federated averaging."""
        aggregator = FederatedAveragingAggregator()
        
        updates = [
            ModelUpdate(
                client_id="client_001",
                round_id="round_001",
                weights={'layer1': np.array([1.0, 2.0])},
                signature=b"sig1",
                timestamp=datetime.now(),
                metadata={}
            ),
            ModelUpdate(
                client_id="client_002",
                round_id="round_001",
                weights={'layer1': np.array([3.0, 4.0])},
                signature=b"sig2",
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        result = aggregator.aggregate(updates)
        expected = np.array([2.0, 3.0])
        
        np.testing.assert_array_equal(result['layer1'], expected)


class TestReputationWeightedAggregator:
    """Test cases for ReputationWeightedAggregator."""
    
    @pytest.fixture
    def mock_reputation_manager(self):
        """Mock reputation manager."""
        mock = Mock(spec=IReputationManager)
        mock.is_quarantined.return_value = False
        mock.get_influence_weight.return_value = 1.0
        return mock
    
    def test_initialization(self, mock_reputation_manager):
        """Test reputation-weighted aggregator initialization."""
        aggregator = ReputationWeightedAggregator(mock_reputation_manager)
        
        assert aggregator.reputation_manager == mock_reputation_manager
        assert aggregator.aggregation_method == AggregationMethod.REPUTATION_WEIGHTED
    
    def test_reputation_based_aggregation(self, mock_reputation_manager):
        """Test reputation-based aggregation."""
        aggregator = ReputationWeightedAggregator(mock_reputation_manager)
        
        # Set different reputation weights
        def get_weight(client_id):
            if client_id == "client_001":
                return 0.3
            elif client_id == "client_002":
                return 0.7
            return 1.0
        
        mock_reputation_manager.get_influence_weight.side_effect = get_weight
        
        updates = [
            ModelUpdate(
                client_id="client_001",
                round_id="round_001",
                weights={'layer1': np.array([1.0, 2.0])},
                signature=b"sig1",
                timestamp=datetime.now(),
                metadata={}
            ),
            ModelUpdate(
                client_id="client_002",
                round_id="round_001",
                weights={'layer1': np.array([3.0, 4.0])},
                signature=b"sig2",
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        result = aggregator.aggregate(updates)
        
        # Expected: 0.3 * [1,2] + 0.7 * [3,4] = [2.4, 3.4]
        expected = np.array([2.4, 3.4])
        
        np.testing.assert_array_almost_equal(result['layer1'], expected)


if __name__ == "__main__":
    pytest.main([__file__])