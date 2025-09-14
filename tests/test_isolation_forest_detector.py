"""
Unit tests for Isolation Forest anomaly detector.

Tests detector functionality with synthetic normal and anomalous updates.
"""

import pytest
import numpy as np
import tempfile
import os
from datetime import datetime
from typing import List

from anomaly_detection.isolation_forest_detector import (
    IsolationForestDetector, 
    SimpleIsolationForestDetector
)
from anomaly_detection.feature_extractor import SimpleFeatureExtractor
from anomaly_detection.interfaces import ModelUpdate


class TestIsolationForestDetector:
    """Test cases for IsolationForestDetector."""
    
    @pytest.fixture
    def normal_updates(self) -> List[ModelUpdate]:
        """Create a set of normal model updates."""
        updates = []
        np.random.seed(42)  # For reproducibility
        
        for i in range(20):
            # Normal weights with small variations
            weights = {
                'layer1': np.random.normal(0, 0.1, (10, 5)),
                'layer2': np.random.normal(0, 0.05, (5, 3)),
                'bias': np.random.normal(0, 0.01, (3,))
            }
            
            update = ModelUpdate(
                client_id=f"normal_client_{i}",
                round_id=f"round_{i}",
                weights=weights,
                signature=b"normal_signature",
                timestamp=datetime.now(),
                metadata={}
            )
            updates.append(update)
        
        return updates
    
    @pytest.fixture
    def anomalous_updates(self) -> List[ModelUpdate]:
        """Create a set of anomalous model updates."""
        updates = []
        np.random.seed(123)  # Different seed for anomalies
        
        for i in range(5):
            # Anomalous weights with large variations or unusual patterns
            if i % 2 == 0:
                # Large magnitude weights
                weights = {
                    'layer1': np.random.normal(0, 2.0, (10, 5)),  # Much larger variance
                    'layer2': np.random.normal(5, 1.0, (5, 3)),   # Shifted mean
                    'bias': np.random.normal(0, 0.5, (3,))        # Larger bias variance
                }
            else:
                # Sparse weights (many zeros)
                weights = {
                    'layer1': np.zeros((10, 5)),
                    'layer2': np.zeros((5, 3)),
                    'bias': np.array([10.0, -10.0, 0.0])  # Extreme values
                }
                # Add few non-zero values
                weights['layer1'][0, 0] = 5.0
                weights['layer2'][1, 1] = -3.0
            
            update = ModelUpdate(
                client_id=f"anomalous_client_{i}",
                round_id=f"round_{i + 100}",
                weights=weights,
                signature=b"anomalous_signature",
                timestamp=datetime.now(),
                metadata={}
            )
            updates.append(update)
        
        return updates
    
    def test_initialization(self):
        """Test detector initialization with various parameters."""
        # Default initialization
        detector = IsolationForestDetector()
        assert detector.contamination == 0.1
        assert detector.n_estimators == 100
        assert detector.anomaly_threshold == 0.0
        assert not detector.is_fitted
        
        # Custom initialization
        detector = IsolationForestDetector(
            contamination=0.2,
            n_estimators=50,
            anomaly_threshold=-0.1
        )
        assert detector.contamination == 0.2
        assert detector.n_estimators == 50
        assert detector.anomaly_threshold == -0.1
    
    def test_fit_normal_updates(self, normal_updates):
        """Test fitting detector on normal updates."""
        detector = IsolationForestDetector()
        
        # Should not be fitted initially
        assert not detector.is_fitted
        
        # Fit on normal updates
        detector.fit(normal_updates)
        
        # Should be fitted now
        assert detector.is_fitted
        assert 'n_samples' in detector.training_stats
        assert detector.training_stats['n_samples'] == len(normal_updates)
        assert detector.training_stats['n_features'] > 0
    
    def test_fit_empty_updates(self):
        """Test fitting with empty update list."""
        detector = IsolationForestDetector()
        
        with pytest.raises(ValueError, match="Cannot fit detector with empty training data"):
            detector.fit([])
    
    def test_predict_anomaly_score(self, normal_updates):
        """Test anomaly score prediction."""
        detector = IsolationForestDetector()
        detector.fit(normal_updates)
        
        # Test with normal update
        normal_update = normal_updates[0]
        score = detector.predict_anomaly_score(normal_update)
        
        assert isinstance(score, float)
        assert not np.isnan(score)
        assert not np.isinf(score)
    
    def test_predict_anomaly_score_unfitted(self, normal_updates):
        """Test prediction with unfitted detector."""
        detector = IsolationForestDetector()
        
        with pytest.raises(ValueError, match="Detector must be fitted before making predictions"):
            detector.predict_anomaly_score(normal_updates[0])
    
    def test_predict_anomaly(self, normal_updates, anomalous_updates):
        """Test binary anomaly prediction."""
        detector = IsolationForestDetector(anomaly_threshold=0.0)
        detector.fit(normal_updates)
        
        # Test normal updates (should mostly be classified as normal)
        normal_predictions = [detector.predict_anomaly(update) for update in normal_updates[:5]]
        normal_anomaly_rate = sum(normal_predictions) / len(normal_predictions)
        
        # Test anomalous updates (should mostly be classified as anomalous)
        anomalous_predictions = [detector.predict_anomaly(update) for update in anomalous_updates]
        anomalous_detection_rate = sum(anomalous_predictions) / len(anomalous_predictions)
        
        # Normal updates should have low anomaly rate
        assert normal_anomaly_rate < 0.5
        
        # Anomalous updates should have high detection rate
        assert anomalous_detection_rate > 0.5
    
    def test_predict_batch(self, normal_updates, anomalous_updates):
        """Test batch prediction functionality."""
        detector = IsolationForestDetector()
        detector.fit(normal_updates)
        
        # Test batch prediction
        test_updates = normal_updates[:3] + anomalous_updates[:2]
        results = detector.predict_batch(test_updates)
        
        assert len(results) == len(test_updates)
        
        for result in results:
            assert 'client_id' in result
            assert 'anomaly_score' in result
            assert 'is_anomaly' in result
            assert isinstance(result['anomaly_score'], float)
            assert isinstance(result['is_anomaly'], bool)
    
    def test_explain_anomaly(self, normal_updates):
        """Test anomaly explanation generation."""
        detector = IsolationForestDetector()
        detector.fit(normal_updates)
        
        update = normal_updates[0]
        explanation = detector.explain_anomaly(update)
        
        assert isinstance(explanation, dict)
        assert len(explanation) > 0
        
        # All explanation values should be numeric
        for key, value in explanation.items():
            assert isinstance(key, str)
            assert isinstance(value, float)
            assert not np.isnan(value)
    
    def test_explain_anomaly_unfitted(self, normal_updates):
        """Test explanation with unfitted detector."""
        detector = IsolationForestDetector()
        
        with pytest.raises(ValueError, match="Detector must be fitted before generating explanations"):
            detector.explain_anomaly(normal_updates[0])
    
    def test_update_model(self, normal_updates):
        """Test model updating with new data."""
        detector = IsolationForestDetector()
        detector.fit(normal_updates[:10])
        
        # Get initial training stats
        initial_stats = detector.training_stats.copy()
        
        # Update with new data
        detector.update_model(normal_updates[10:])
        
        # Should be retrained with new data
        assert detector.is_fitted
        assert detector.training_stats['n_samples'] == len(normal_updates[10:])
    
    def test_set_threshold(self, normal_updates):
        """Test threshold setting."""
        detector = IsolationForestDetector()
        detector.fit(normal_updates)
        
        # Set new threshold
        new_threshold = -0.5
        detector.set_threshold(new_threshold)
        
        assert detector.anomaly_threshold == new_threshold
        assert detector.training_stats['threshold'] == new_threshold
    
    def test_evaluate(self, normal_updates, anomalous_updates):
        """Test detector evaluation on labeled data."""
        detector = IsolationForestDetector(anomaly_threshold=0.0)
        detector.fit(normal_updates)
        
        # Create test set with labels
        test_updates = normal_updates[:5] + anomalous_updates
        true_labels = [False] * 5 + [True] * len(anomalous_updates)
        
        # Evaluate
        metrics = detector.evaluate(test_updates, true_labels)
        
        # Check that all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 
                          'n_samples', 'n_anomalies', 'n_normal']
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        assert 0 <= metrics['roc_auc'] <= 1
        
        assert metrics['n_samples'] == len(test_updates)
        assert metrics['n_anomalies'] == len(anomalous_updates)
        assert metrics['n_normal'] == 5
    
    def test_evaluate_mismatched_lengths(self, normal_updates):
        """Test evaluation with mismatched update and label lengths."""
        detector = IsolationForestDetector()
        detector.fit(normal_updates)
        
        with pytest.raises(ValueError, match="Number of updates and labels must match"):
            detector.evaluate(normal_updates[:3], [True, False])  # Mismatched lengths
    
    def test_save_and_load_model(self, normal_updates):
        """Test model saving and loading."""
        detector = IsolationForestDetector()
        detector.fit(normal_updates)
        
        # Get prediction before saving
        test_update = normal_updates[0]
        original_score = detector.predict_anomaly_score(test_update)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            detector.save_model(tmp_path)
            
            # Create new detector and load model
            new_detector = IsolationForestDetector()
            new_detector.load_model(tmp_path)
            
            # Should be fitted and produce same predictions
            assert new_detector.is_fitted
            new_score = new_detector.predict_anomaly_score(test_update)
            
            # Scores should be very close (allowing for small numerical differences)
            assert abs(original_score - new_score) < 1e-10
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_save_unfitted_model(self):
        """Test saving unfitted model."""
        detector = IsolationForestDetector()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            with pytest.raises(ValueError, match="Cannot save unfitted model"):
                detector.save_model(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_load_nonexistent_model(self):
        """Test loading nonexistent model."""
        detector = IsolationForestDetector()
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            detector.load_model("nonexistent_model.joblib")
    
    def test_custom_feature_extractor(self, normal_updates):
        """Test detector with custom feature extractor."""
        custom_extractor = SimpleFeatureExtractor()
        detector = IsolationForestDetector(feature_extractor=custom_extractor)
        
        detector.fit(normal_updates)
        
        # Should use the custom extractor
        assert detector.feature_extractor is custom_extractor
        
        # Should still work for predictions
        score = detector.predict_anomaly_score(normal_updates[0])
        assert isinstance(score, float)


class TestSimpleIsolationForestDetector:
    """Test cases for SimpleIsolationForestDetector."""
    
    @pytest.fixture
    def simple_updates(self) -> List[ModelUpdate]:
        """Create simple model updates for testing."""
        updates = []
        np.random.seed(42)
        
        for i in range(10):
            weights = {
                'layer1': np.random.normal(0, 0.1, (5, 3)),
                'layer2': np.random.normal(0, 0.05, (3, 2))
            }
            
            update = ModelUpdate(
                client_id=f"client_{i}",
                round_id=f"round_{i}",
                weights=weights,
                signature=b"signature",
                timestamp=datetime.now(),
                metadata={}
            )
            updates.append(update)
        
        return updates
    
    def test_initialization(self):
        """Test simple detector initialization."""
        detector = SimpleIsolationForestDetector()
        
        assert detector.contamination == 0.1
        assert not detector.is_fitted
        assert isinstance(detector.feature_extractor, SimpleFeatureExtractor)
    
    def test_fit_and_predict(self, simple_updates):
        """Test basic fit and predict functionality."""
        detector = SimpleIsolationForestDetector()
        
        # Fit
        detector.fit(simple_updates)
        assert detector.is_fitted
        
        # Predict
        score = detector.predict_anomaly_score(simple_updates[0])
        assert isinstance(score, float)
        
        # Explain
        explanation = detector.explain_anomaly(simple_updates[0])
        assert isinstance(explanation, dict)
        assert len(explanation) > 0
    
    def test_update_model(self, simple_updates):
        """Test model updating."""
        detector = SimpleIsolationForestDetector()
        detector.fit(simple_updates[:5])
        
        # Update with new data
        detector.update_model(simple_updates[5:])
        
        # Should still be fitted
        assert detector.is_fitted


class TestDetectorIntegration:
    """Integration tests for detector components."""
    
    def test_end_to_end_workflow(self):
        """Test complete anomaly detection workflow."""
        # Create synthetic data
        np.random.seed(42)
        
        # Normal updates
        normal_updates = []
        for i in range(15):
            weights = {
                'conv1': np.random.normal(0, 0.1, (32, 3, 3, 3)),
                'fc1': np.random.normal(0, 0.05, (128, 512)),
                'fc2': np.random.normal(0, 0.02, (10, 128))
            }
            
            update = ModelUpdate(
                client_id=f"honest_client_{i}",
                round_id=f"round_{i}",
                weights=weights,
                signature=b"honest_signature",
                timestamp=datetime.now(),
                metadata={}
            )
            normal_updates.append(update)
        
        # Anomalous updates
        anomalous_updates = []
        for i in range(3):
            # Create obviously anomalous weights
            weights = {
                'conv1': np.random.normal(10, 5.0, (32, 3, 3, 3)),  # Very different distribution
                'fc1': np.zeros((128, 512)),  # All zeros
                'fc2': np.random.normal(0, 2.0, (10, 128))  # High variance
            }
            
            update = ModelUpdate(
                client_id=f"malicious_client_{i}",
                round_id=f"round_{i + 100}",
                weights=weights,
                signature=b"malicious_signature",
                timestamp=datetime.now(),
                metadata={}
            )
            anomalous_updates.append(update)
        
        # Initialize and train detector
        detector = IsolationForestDetector(
            contamination=0.15,
            n_estimators=50,
            anomaly_threshold=-0.1
        )
        
        # Train on normal updates
        detector.fit(normal_updates)
        
        # Test detection performance
        all_updates = normal_updates + anomalous_updates
        true_labels = [False] * len(normal_updates) + [True] * len(anomalous_updates)
        
        # Get predictions
        predictions = []
        scores = []
        
        for update in all_updates:
            score = detector.predict_anomaly_score(update)
            pred = detector.predict_anomaly(update)
            
            scores.append(score)
            predictions.append(pred)
        
        # Evaluate performance
        metrics = detector.evaluate(all_updates, true_labels)
        
        # Should achieve reasonable performance on this synthetic data
        assert metrics['accuracy'] > 0.7  # At least 70% accuracy
        assert metrics['n_samples'] == len(all_updates)
        
        # Test explanations
        for update in anomalous_updates:
            explanation = detector.explain_anomaly(update)
            assert len(explanation) > 0
            
            # Should have some non-zero explanation values
            assert any(abs(value) > 0 for value in explanation.values())


if __name__ == "__main__":
    pytest.main([__file__])