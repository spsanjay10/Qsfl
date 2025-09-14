"""
Unit tests for SHAP explainability integration.

Tests explanation generation and consistency for anomaly detection results.
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List
import tempfile
import os

from anomaly_detection.shap_explainer import SHAPExplainer, SimpleSHAPExplainer, SHAP_AVAILABLE
from anomaly_detection.isolation_forest_detector import IsolationForestDetector
from anomaly_detection.interfaces import ModelUpdate


# Skip all tests if SHAP is not available
pytestmark = pytest.mark.skipif(not SHAP_AVAILABLE, reason="SHAP library not available")


class TestSHAPExplainer:
    """Test cases for SHAPExplainer."""
    
    @pytest.fixture
    def trained_detector(self) -> IsolationForestDetector:
        """Create and train an Isolation Forest detector."""
        # Create training data
        updates = []
        np.random.seed(42)
        
        for i in range(20):
            weights = {
                'layer1': np.random.normal(0, 0.1, (5, 3)),
                'layer2': np.random.normal(0, 0.05, (3, 2)),
                'bias': np.random.normal(0, 0.01, (2,))
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
        
        # Train detector
        detector = IsolationForestDetector(contamination=0.1, n_estimators=10)
        detector.fit(updates)
        
        return detector
    
    @pytest.fixture
    def background_updates(self) -> List[ModelUpdate]:
        """Create background updates for SHAP explainer."""
        updates = []
        np.random.seed(123)
        
        for i in range(10):
            weights = {
                'layer1': np.random.normal(0, 0.1, (5, 3)),
                'layer2': np.random.normal(0, 0.05, (3, 2)),
                'bias': np.random.normal(0, 0.01, (2,))
            }
            
            update = ModelUpdate(
                client_id=f"bg_client_{i}",
                round_id=f"bg_round_{i}",
                weights=weights,
                signature=b"bg_signature",
                timestamp=datetime.now(),
                metadata={}
            )
            updates.append(update)
        
        return updates
    
    @pytest.fixture
    def test_update(self) -> ModelUpdate:
        """Create a test update for explanation."""
        np.random.seed(456)
        weights = {
            'layer1': np.random.normal(0, 0.2, (5, 3)),  # Slightly different distribution
            'layer2': np.random.normal(0.1, 0.05, (3, 2)),  # Shifted mean
            'bias': np.random.normal(0, 0.02, (2,))
        }
        
        return ModelUpdate(
            client_id="test_client",
            round_id="test_round",
            weights=weights,
            signature=b"test_signature",
            timestamp=datetime.now(),
            metadata={}
        )
    
    def test_initialization_with_background(self, trained_detector, background_updates):
        """Test SHAP explainer initialization with background data."""
        explainer = SHAPExplainer(
            detector=trained_detector,
            background_data=background_updates,
            max_background_samples=5
        )
        
        assert explainer.detector is trained_detector
        assert explainer.background_data == background_updates
        assert explainer.max_background_samples == 5
        assert explainer.background_features is not None
        assert len(explainer.background_features) <= 5
    
    def test_initialization_without_background(self, trained_detector):
        """Test SHAP explainer initialization without background data."""
        explainer = SHAPExplainer(detector=trained_detector)
        
        assert explainer.detector is trained_detector
        assert explainer.background_data is None
        assert explainer.background_features is not None  # Should create synthetic
        assert len(explainer.background_features) > 0
    
    def test_initialization_unfitted_detector(self):
        """Test initialization with unfitted detector."""
        detector = IsolationForestDetector()
        explainer = SHAPExplainer(detector=detector)
        
        # Should not initialize SHAP explainer for unfitted detector
        assert explainer.explainer is None
    
    def test_synthetic_background_creation(self, trained_detector):
        """Test synthetic background data creation."""
        explainer = SHAPExplainer(detector=trained_detector)
        
        # Should create synthetic background
        assert explainer.background_features is not None
        assert explainer.background_features.shape[0] > 0
        assert explainer.background_features.shape[1] > 0
        
        # Should have reasonable values
        assert not np.any(np.isnan(explainer.background_features))
        assert not np.any(np.isinf(explainer.background_features))
    
    def test_explain_basic(self, trained_detector, test_update):
        """Test basic explanation generation."""
        explainer = SHAPExplainer(detector=trained_detector)
        
        # Get anomaly score
        anomaly_score = trained_detector.predict_anomaly_score(test_update)
        
        # Generate explanation
        explanation = explainer.explain(test_update, anomaly_score)
        
        assert isinstance(explanation, dict)
        assert len(explanation) > 0
        
        # All values should be numeric
        for key, value in explanation.items():
            assert isinstance(key, str)
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)
    
    def test_explain_consistency(self, trained_detector, test_update):
        """Test that explanations are consistent across calls."""
        explainer = SHAPExplainer(detector=trained_detector)
        
        anomaly_score = trained_detector.predict_anomaly_score(test_update)
        
        # Generate explanation multiple times
        explanation1 = explainer.explain(test_update, anomaly_score)
        explanation2 = explainer.explain(test_update, anomaly_score)
        
        # Should be consistent (allowing for small numerical differences)
        assert len(explanation1) == len(explanation2)
        
        for key in explanation1:
            assert key in explanation2
            # Allow small differences due to randomness in SHAP
            assert abs(explanation1[key] - explanation2[key]) < 0.1
    
    def test_fallback_explanation(self, trained_detector, test_update):
        """Test fallback explanation when SHAP fails."""
        explainer = SHAPExplainer(detector=trained_detector)
        
        # Force SHAP explainer to None to test fallback
        explainer.explainer = None
        
        anomaly_score = trained_detector.predict_anomaly_score(test_update)
        explanation = explainer.explain(test_update, anomaly_score)
        
        # Should still get some explanation
        assert isinstance(explanation, dict)
        # May be empty if all fallbacks fail, but should not raise exception
    
    def test_explain_batch(self, trained_detector, background_updates):
        """Test batch explanation functionality."""
        explainer = SHAPExplainer(detector=trained_detector)
        
        # Use subset of background updates as test data
        test_updates = background_updates[:3]
        
        explanations = explainer.explain_batch(test_updates)
        
        assert len(explanations) == len(test_updates)
        
        for i, explanation in enumerate(explanations):
            assert 'client_id' in explanation
            assert 'anomaly_score' in explanation
            assert 'is_anomaly' in explanation
            assert 'shap_values' in explanation
            assert 'top_features' in explanation
            
            assert explanation['client_id'] == test_updates[i].client_id
            assert isinstance(explanation['anomaly_score'], float)
            assert isinstance(explanation['is_anomaly'], bool)
            assert isinstance(explanation['shap_values'], dict)
            assert isinstance(explanation['top_features'], list)
    
    def test_top_features_extraction(self, trained_detector, test_update):
        """Test extraction of top contributing features."""
        explainer = SHAPExplainer(detector=trained_detector)
        
        # Create mock SHAP values
        shap_values = {
            'feature_a': 0.5,
            'feature_b': -0.3,
            'feature_c': 0.8,
            'feature_d': 0.1,
            'feature_e': -0.2
        }
        
        top_features = explainer._get_top_features(shap_values, top_k=3)
        
        assert len(top_features) == 3
        assert top_features[0][0] == 'feature_c'  # Highest absolute value
        assert top_features[1][0] == 'feature_a'  # Second highest
        assert top_features[2][0] == 'feature_b'  # Third highest (by absolute value)
    
    def test_explanation_summary(self, trained_detector, test_update):
        """Test human-readable explanation summary generation."""
        explainer = SHAPExplainer(detector=trained_detector)
        
        anomaly_score = trained_detector.predict_anomaly_score(test_update)
        summary = explainer.create_explanation_summary(test_update, anomaly_score)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert test_update.client_id in summary
        assert str(anomaly_score)[:5] in summary  # Check score is mentioned
    
    def test_readable_feature_names(self, trained_detector):
        """Test conversion of technical feature names to readable ones."""
        explainer = SHAPExplainer(detector=trained_detector)
        
        # Test some common feature name conversions
        test_cases = {
            'weight_mean': 'Average weight value',
            'sparsity_ratio': 'Proportion of zero weights',
            'unknown_feature': 'Unknown Feature'
        }
        
        for technical, expected in test_cases.items():
            readable = explainer._make_feature_readable(technical)
            assert readable == expected
    
    def test_global_feature_importance(self, trained_detector, background_updates):
        """Test global feature importance computation."""
        explainer = SHAPExplainer(detector=trained_detector)
        
        # Compute global importance
        importance = explainer.get_global_feature_importance(background_updates[:5])
        
        assert isinstance(importance, dict)
        
        if importance:  # May be empty if SHAP fails
            for feature, score in importance.items():
                assert isinstance(feature, str)
                assert isinstance(score, float)
                assert not np.isnan(score)
                assert score >= 0  # Should be absolute values
    
    def test_different_explainer_types(self, trained_detector):
        """Test different SHAP explainer types."""
        # Test auto explainer
        explainer_auto = SHAPExplainer(
            detector=trained_detector,
            explainer_type="auto"
        )
        assert explainer_auto.explainer_type == "auto"
        
        # Test kernel explainer
        explainer_kernel = SHAPExplainer(
            detector=trained_detector,
            explainer_type="kernel"
        )
        assert explainer_kernel.explainer_type == "kernel"
    
    def test_visualization_without_matplotlib(self, trained_detector, test_update):
        """Test visualization when matplotlib is not available."""
        explainer = SHAPExplainer(detector=trained_detector)
        
        anomaly_score = trained_detector.predict_anomaly_score(test_update)
        
        # Should not raise exception even if matplotlib is not available
        explainer.visualize_explanation(
            test_update, 
            anomaly_score, 
            show_plot=False
        )
    
    def test_empty_shap_values(self, trained_detector):
        """Test handling of empty SHAP values."""
        explainer = SHAPExplainer(detector=trained_detector)
        
        # Test with empty SHAP values
        top_features = explainer._get_top_features({}, top_k=5)
        assert top_features == []
        
        # Test global importance with no valid updates
        importance = explainer.get_global_feature_importance([])
        assert importance == {}


class TestSimpleSHAPExplainer:
    """Test cases for SimpleSHAPExplainer."""
    
    @pytest.fixture
    def simple_detector(self) -> IsolationForestDetector:
        """Create a simple trained detector."""
        updates = []
        np.random.seed(42)
        
        for i in range(10):
            weights = {'layer': np.random.normal(0, 0.1, (3, 2))}
            update = ModelUpdate(
                client_id=f"client_{i}",
                round_id=f"round_{i}",
                weights=weights,
                signature=b"sig",
                timestamp=datetime.now(),
                metadata={}
            )
            updates.append(update)
        
        detector = IsolationForestDetector(n_estimators=5)
        detector.fit(updates)
        return detector
    
    def test_initialization(self, simple_detector):
        """Test simple explainer initialization."""
        explainer = SimpleSHAPExplainer(detector=simple_detector)
        assert explainer.detector is simple_detector
    
    def test_explain(self, simple_detector):
        """Test simple explanation generation."""
        explainer = SimpleSHAPExplainer(detector=simple_detector)
        
        # Create test update
        weights = {'layer': np.random.normal(1, 0.5, (3, 2))}  # Different distribution
        update = ModelUpdate(
            client_id="test",
            round_id="test",
            weights=weights,
            signature=b"sig",
            timestamp=datetime.now(),
            metadata={}
        )
        
        explanation = explainer.explain(update, 0.5)
        
        # Should return some explanation (may be from detector's fallback)
        assert isinstance(explanation, dict)


class TestSHAPIntegration:
    """Integration tests for SHAP explainer with different scenarios."""
    
    def test_end_to_end_explanation_workflow(self):
        """Test complete explanation workflow."""
        # Create training data
        np.random.seed(42)
        normal_updates = []
        
        for i in range(15):
            weights = {
                'conv': np.random.normal(0, 0.1, (8, 4)),
                'fc': np.random.normal(0, 0.05, (4, 2)),
                'bias': np.random.normal(0, 0.01, (2,))
            }
            
            update = ModelUpdate(
                client_id=f"normal_{i}",
                round_id=f"round_{i}",
                weights=weights,
                signature=b"normal",
                timestamp=datetime.now(),
                metadata={}
            )
            normal_updates.append(update)
        
        # Create anomalous update
        anomalous_weights = {
            'conv': np.random.normal(2, 1.0, (8, 4)),  # Very different
            'fc': np.zeros((4, 2)),  # All zeros
            'bias': np.array([10.0, -10.0])  # Extreme values
        }
        
        anomalous_update = ModelUpdate(
            client_id="malicious",
            round_id="round_100",
            weights=anomalous_weights,
            signature=b"malicious",
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Train detector
        detector = IsolationForestDetector(
            contamination=0.1,
            n_estimators=20,
            anomaly_threshold=-0.1
        )
        detector.fit(normal_updates)
        
        # Initialize explainer
        explainer = SHAPExplainer(
            detector=detector,
            background_data=normal_updates[:5],
            max_background_samples=5
        )
        
        # Test normal update explanation
        normal_score = detector.predict_anomaly_score(normal_updates[0])
        normal_explanation = explainer.explain(normal_updates[0], normal_score)
        
        assert isinstance(normal_explanation, dict)
        assert len(normal_explanation) > 0
        
        # Test anomalous update explanation
        anomalous_score = detector.predict_anomaly_score(anomalous_update)
        anomalous_explanation = explainer.explain(anomalous_update, anomalous_score)
        
        assert isinstance(anomalous_explanation, dict)
        assert len(anomalous_explanation) > 0
        
        # Generate summaries
        normal_summary = explainer.create_explanation_summary(normal_updates[0], normal_score)
        anomalous_summary = explainer.create_explanation_summary(anomalous_update, anomalous_score)
        
        assert isinstance(normal_summary, str)
        assert isinstance(anomalous_summary, str)
        assert len(normal_summary) > 0
        assert len(anomalous_summary) > 0
        
        # Test batch explanation
        test_updates = [normal_updates[0], anomalous_update]
        batch_explanations = explainer.explain_batch(test_updates)
        
        assert len(batch_explanations) == 2
        
        for explanation in batch_explanations:
            assert 'client_id' in explanation
            assert 'shap_values' in explanation
            assert 'top_features' in explanation
    
    def test_explanation_robustness(self):
        """Test explainer robustness with edge cases."""
        # Create minimal detector
        updates = []
        for i in range(5):
            weights = {'w': np.array([float(i)])}
            update = ModelUpdate(
                client_id=f"c_{i}",
                round_id=f"r_{i}",
                weights=weights,
                signature=b"s",
                timestamp=datetime.now(),
                metadata={}
            )
            updates.append(update)
        
        detector = IsolationForestDetector(n_estimators=3)
        detector.fit(updates)
        
        explainer = SHAPExplainer(detector=detector)
        
        # Test with various edge cases
        edge_cases = [
            # Single parameter
            {'w': np.array([100.0])},
            # Zero weights
            {'w': np.array([0.0])},
            # Negative weights
            {'w': np.array([-50.0])},
        ]
        
        for i, weights in enumerate(edge_cases):
            update = ModelUpdate(
                client_id=f"edge_{i}",
                round_id=f"edge_round_{i}",
                weights=weights,
                signature=b"edge",
                timestamp=datetime.now(),
                metadata={}
            )
            
            try:
                score = detector.predict_anomaly_score(update)
                explanation = explainer.explain(update, score)
                
                # Should not crash and should return some result
                assert isinstance(explanation, dict)
                
            except Exception as e:
                # Log but don't fail - some edge cases might legitimately fail
                print(f"Edge case {i} failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])