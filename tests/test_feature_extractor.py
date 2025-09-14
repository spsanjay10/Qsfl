"""
Unit tests for feature extraction functionality.

Tests feature extraction with various model architectures and edge cases.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import Dict

from anomaly_detection.feature_extractor import ModelUpdateFeatureExtractor, SimpleFeatureExtractor
from anomaly_detection.interfaces import ModelUpdate


class TestModelUpdateFeatureExtractor:
    """Test cases for ModelUpdateFeatureExtractor."""
    
    @pytest.fixture
    def simple_update(self) -> ModelUpdate:
        """Create a simple model update for testing."""
        weights = {
            'layer1': np.random.normal(0, 1, (10, 5)),
            'layer2': np.random.normal(0, 0.5, (5, 3)),
            'bias1': np.random.normal(0, 0.1, (5,)),
            'bias2': np.random.normal(0, 0.1, (3,))
        }
        
        return ModelUpdate(
            client_id="test_client",
            round_id="round_1",
            weights=weights,
            signature=b"test_signature",
            timestamp=datetime.now(),
            metadata={}
        )
    
    @pytest.fixture
    def update_with_gradients(self) -> ModelUpdate:
        """Create model update with gradient information."""
        weights = {
            'layer1': np.random.normal(0, 1, (10, 5)),
            'layer2': np.random.normal(0, 0.5, (5, 3))
        }
        
        gradients = {
            'layer1': np.random.normal(0, 0.01, (10, 5)),
            'layer2': np.random.normal(0, 0.005, (5, 3))
        }
        
        return ModelUpdate(
            client_id="test_client",
            round_id="round_2",
            weights=weights,
            signature=b"test_signature",
            timestamp=datetime.now(),
            metadata={'gradients': gradients}
        )
    
    @pytest.fixture
    def sparse_update(self) -> ModelUpdate:
        """Create model update with sparse weights."""
        weights = {
            'layer1': np.zeros((20, 10)),  # Larger sparse layer
            'layer2': np.zeros((10, 5))    # Another sparse layer
        }
        # Add very few non-zero values to make it truly sparse
        weights['layer1'][0, 0] = 1.0
        weights['layer1'][5, 2] = -0.5
        weights['layer2'][1, 1] = 0.3
        
        return ModelUpdate(
            client_id="sparse_client",
            round_id="round_3",
            weights=weights,
            signature=b"test_signature",
            timestamp=datetime.now(),
            metadata={}
        )
    
    def test_initialization(self):
        """Test feature extractor initialization."""
        # Default initialization
        extractor = ModelUpdateFeatureExtractor()
        assert extractor.normalize is True
        assert extractor.scaler_type == "standard"
        assert extractor.is_fitted is False
        
        # Custom initialization
        extractor = ModelUpdateFeatureExtractor(normalize=False, scaler_type="minmax")
        assert extractor.normalize is False
        assert extractor.scaler_type == "minmax"
    
    def test_invalid_scaler_type(self):
        """Test initialization with invalid scaler type."""
        with pytest.raises(ValueError, match="Unknown scaler type"):
            ModelUpdateFeatureExtractor(scaler_type="invalid")
    
    def test_extract_features_basic(self, simple_update):
        """Test basic feature extraction."""
        extractor = ModelUpdateFeatureExtractor(normalize=False)
        features = extractor.extract_features(simple_update)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) > 0
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_extract_features_with_gradients(self, update_with_gradients):
        """Test feature extraction with gradient information."""
        extractor = ModelUpdateFeatureExtractor(normalize=False)
        features = extractor.extract_features(update_with_gradients)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        
        # Should have non-zero gradient features when gradients are present
        feature_names = extractor.get_feature_names()
        grad_start_idx = feature_names.index('grad_mag_mean')
        grad_end_idx = feature_names.index('grad_positive_ratio') + 1
        gradient_features = features[grad_start_idx:grad_end_idx]
        
        # At least some gradient features should be non-zero
        assert np.any(gradient_features != 0.0)
    
    def test_extract_features_sparse(self, sparse_update):
        """Test feature extraction with sparse weights."""
        extractor = ModelUpdateFeatureExtractor(normalize=False)
        features = extractor.extract_features(sparse_update)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
        
        # Check that sparsity is detected (high sparsity means many zeros)
        feature_names = extractor.get_feature_names()
        sparsity_idx = feature_names.index('sparsity_ratio')
        assert features[sparsity_idx] > 0.9  # Should be very sparse (>90% zeros)
    
    def test_weight_features(self, simple_update):
        """Test weight-based feature extraction."""
        extractor = ModelUpdateFeatureExtractor(normalize=False)
        
        # Extract weight features directly
        weight_features = extractor._extract_weight_features(simple_update.weights)
        
        assert len(weight_features) > 0
        assert all(isinstance(f, float) for f in weight_features)
        assert not any(np.isnan(f) for f in weight_features)
    
    def test_gradient_features(self, update_with_gradients):
        """Test gradient-based feature extraction."""
        extractor = ModelUpdateFeatureExtractor(normalize=False)
        
        gradients = update_with_gradients.metadata['gradients']
        gradient_features = extractor._extract_gradient_features(gradients)
        
        assert len(gradient_features) > 0
        assert all(isinstance(f, float) for f in gradient_features)
        assert not any(np.isnan(f) for f in gradient_features)
    
    def test_structural_features(self, simple_update):
        """Test structural feature extraction."""
        extractor = ModelUpdateFeatureExtractor(normalize=False)
        
        structural_features = extractor._extract_structural_features(simple_update.weights)
        
        assert len(structural_features) > 0
        assert all(isinstance(f, float) for f in structural_features)
        assert not any(np.isnan(f) for f in structural_features)
    
    def test_temporal_features(self, simple_update):
        """Test temporal feature extraction."""
        extractor = ModelUpdateFeatureExtractor(normalize=False)
        
        temporal_features = extractor._extract_temporal_features(simple_update)
        
        assert len(temporal_features) > 0
        assert all(isinstance(f, float) for f in temporal_features)
        assert not any(np.isnan(f) for f in temporal_features)
    
    def test_handle_invalid_values(self):
        """Test handling of NaN and infinite values."""
        extractor = ModelUpdateFeatureExtractor()
        
        # Create array with invalid values
        invalid_features = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])
        cleaned_features = extractor._handle_invalid_values(invalid_features)
        
        assert not np.any(np.isnan(cleaned_features))
        assert not np.any(np.isinf(cleaned_features))
        assert cleaned_features[0] == 1.0  # Valid values preserved
        assert cleaned_features[4] == 2.0
    
    def test_scaler_fitting(self, simple_update):
        """Test scaler fitting functionality."""
        extractor = ModelUpdateFeatureExtractor(normalize=True, scaler_type="standard")
        
        # Create multiple updates for fitting
        updates = []
        for i in range(10):
            weights = {
                'layer1': np.random.normal(i, 1, (5, 3)),
                'layer2': np.random.normal(-i, 0.5, (3, 2))
            }
            update = ModelUpdate(
                client_id=f"client_{i}",
                round_id=f"round_{i}",
                weights=weights,
                signature=b"test",
                timestamp=datetime.now(),
                metadata={}
            )
            updates.append(update)
        
        # Fit scaler
        extractor.fit_scaler(updates)
        assert extractor.is_fitted is True
        
        # Extract features with normalization
        features = extractor.extract_features(simple_update)
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_feature_names(self):
        """Test feature name generation."""
        extractor = ModelUpdateFeatureExtractor()
        feature_names = extractor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)
        
        # Check for expected feature categories
        expected_categories = ['weight', 'layer', 'sparsity', 'round', 'params']
        found_categories = set()
        for name in feature_names:
            for category in expected_categories:
                if category in name:
                    found_categories.add(category)
        
        assert len(found_categories) > 0
    
    def test_different_model_architectures(self):
        """Test feature extraction with different model architectures."""
        extractor = ModelUpdateFeatureExtractor(normalize=False)
        
        # Test with different architectures
        architectures = [
            # Small model
            {'layer1': np.random.normal(0, 1, (5, 3))},
            # Large model
            {'layer1': np.random.normal(0, 1, (100, 50)),
             'layer2': np.random.normal(0, 1, (50, 25)),
             'layer3': np.random.normal(0, 1, (25, 10))},
            # Model with different shapes
            {'conv1': np.random.normal(0, 1, (32, 3, 3, 3)),
             'conv2': np.random.normal(0, 1, (64, 32, 3, 3)),
             'fc1': np.random.normal(0, 1, (128, 1024))},
        ]
        
        for i, weights in enumerate(architectures):
            update = ModelUpdate(
                client_id=f"arch_client_{i}",
                round_id=f"round_{i}",
                weights=weights,
                signature=b"test",
                timestamp=datetime.now(),
                metadata={}
            )
            
            features = extractor.extract_features(update)
            assert isinstance(features, np.ndarray)
            assert len(features) > 0
            assert not np.any(np.isnan(features))
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        extractor = ModelUpdateFeatureExtractor(normalize=False)
        
        # Empty weights
        empty_update = ModelUpdate(
            client_id="empty_client",
            round_id="round_0",
            weights={},
            signature=b"test",
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Should handle gracefully
        features = extractor.extract_features(empty_update)
        assert isinstance(features, np.ndarray)
        
        # Single weight
        single_weight_update = ModelUpdate(
            client_id="single_client",
            round_id="round_0",
            weights={'single': np.array([1.0])},
            signature=b"test",
            timestamp=datetime.now(),
            metadata={}
        )
        
        features = extractor.extract_features(single_weight_update)
        assert isinstance(features, np.ndarray)
        assert not np.any(np.isnan(features))


class TestSimpleFeatureExtractor:
    """Test cases for SimpleFeatureExtractor."""
    
    @pytest.fixture
    def simple_update(self) -> ModelUpdate:
        """Create a simple model update for testing."""
        weights = {
            'layer1': np.random.normal(0, 1, (10, 5)),
            'layer2': np.random.normal(0, 0.5, (5, 3))
        }
        
        return ModelUpdate(
            client_id="test_client",
            round_id="round_1",
            weights=weights,
            signature=b"test_signature",
            timestamp=datetime.now(),
            metadata={}
        )
    
    def test_initialization(self):
        """Test simple feature extractor initialization."""
        extractor = SimpleFeatureExtractor()
        assert len(extractor.feature_names) == 6
        assert 'mean' in extractor.feature_names
        assert 'std' in extractor.feature_names
    
    def test_extract_features(self, simple_update):
        """Test basic feature extraction."""
        extractor = SimpleFeatureExtractor()
        features = extractor.extract_features(simple_update)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert len(features) == 6
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_get_feature_names(self):
        """Test feature name retrieval."""
        extractor = SimpleFeatureExtractor()
        names = extractor.get_feature_names()
        
        assert isinstance(names, list)
        assert len(names) == 6
        expected_names = ['mean', 'std', 'min', 'max', 'l2_norm', 'sparsity']
        assert names == expected_names
    
    def test_sparse_weights(self):
        """Test with sparse weights."""
        weights = {
            'layer1': np.zeros((10, 5)),
            'layer2': np.array([1.0, 0.0, 0.0])
        }
        weights['layer1'][0, 0] = 1.0
        
        update = ModelUpdate(
            client_id="sparse_client",
            round_id="round_1",
            weights=weights,
            signature=b"test",
            timestamp=datetime.now(),
            metadata={}
        )
        
        extractor = SimpleFeatureExtractor()
        features = extractor.extract_features(update)
        
        # Check sparsity feature (last element)
        assert features[5] > 0.5  # Should detect high sparsity


if __name__ == "__main__":
    pytest.main([__file__])