"""
Isolation Forest Anomaly Detector

Implements anomaly detection using scikit-learn's Isolation Forest algorithm
with configurable thresholds and model training pipeline.
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Any
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

from .interfaces import IAnomalyDetector, ModelUpdate, IFeatureExtractor
from .feature_extractor import ModelUpdateFeatureExtractor


logger = logging.getLogger(__name__)


class IsolationForestDetector(IAnomalyDetector):
    """
    Anomaly detector using Isolation Forest algorithm.
    
    This detector uses an ensemble of isolation trees to identify anomalous
    model updates based on extracted features. It supports configurable
    contamination rates and anomaly thresholds.
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: str = "auto",
        max_features: float = 1.0,
        anomaly_threshold: float = 0.0,
        feature_extractor: Optional[IFeatureExtractor] = None,
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies in training data
            n_estimators: Number of isolation trees in the ensemble
            max_samples: Number of samples to draw for each tree
            max_features: Number of features to draw for each tree
            anomaly_threshold: Threshold for anomaly classification (lower = more anomalous)
            feature_extractor: Feature extractor to use (default: ModelUpdateFeatureExtractor)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.anomaly_threshold = anomaly_threshold
        self.random_state = random_state
        
        # Initialize feature extractor
        if feature_extractor is None:
            self.feature_extractor = ModelUpdateFeatureExtractor(normalize=True)
        else:
            self.feature_extractor = feature_extractor
        
        # Initialize Isolation Forest
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        
        self.is_fitted = False
        self.training_stats = {}
    
    def fit(self, normal_updates: List[ModelUpdate]) -> None:
        """
        Train the anomaly detector on normal updates.
        
        Args:
            normal_updates: List of known normal model updates
        """
        if not normal_updates:
            raise ValueError("Cannot fit detector with empty training data")
        
        logger.info(f"Training Isolation Forest on {len(normal_updates)} normal updates")
        
        # Extract features from all updates
        features_list = []
        for update in normal_updates:
            try:
                features = self.feature_extractor.extract_features(update)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features from update {update.client_id}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No valid features could be extracted from training data")
        
        # Convert to numpy array
        X_train = np.array(features_list)
        
        # Fit feature extractor scaler if applicable
        if hasattr(self.feature_extractor, 'fit_scaler'):
            self.feature_extractor.fit_scaler(normal_updates)
        
        # Re-extract features with fitted scaler
        features_list = []
        for update in normal_updates:
            try:
                features = self.feature_extractor.extract_features(update)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features from update {update.client_id}: {e}")
                continue
        
        X_train = np.array(features_list)
        
        # Train Isolation Forest
        self.model.fit(X_train)
        self.is_fitted = True
        
        # Compute training statistics
        train_scores = self.model.decision_function(X_train)
        self.training_stats = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'score_mean': float(np.mean(train_scores)),
            'score_std': float(np.std(train_scores)),
            'score_min': float(np.min(train_scores)),
            'score_max': float(np.max(train_scores)),
            'threshold': self.anomaly_threshold,
            'feature_means': np.mean(X_train, axis=0).tolist(),
            'feature_stds': np.std(X_train, axis=0).tolist()
        }
        
        logger.info(f"Isolation Forest trained successfully. Stats: {self.training_stats}")
    
    def predict_anomaly_score(self, update: ModelUpdate) -> float:
        """
        Predict anomaly score for a model update.
        
        Args:
            update: Model update to score
            
        Returns:
            Anomaly score (lower values indicate higher anomaly likelihood)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before making predictions")
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(update)
            features = features.reshape(1, -1)
            
            # Get anomaly score
            score = self.model.decision_function(features)[0]
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Error computing anomaly score for update {update.client_id}: {e}")
            # Return a neutral score in case of error
            return 0.0
    
    def predict_anomaly(self, update: ModelUpdate) -> bool:
        """
        Predict whether an update is anomalous.
        
        Args:
            update: Model update to classify
            
        Returns:
            True if anomalous, False if normal
        """
        score = self.predict_anomaly_score(update)
        return score < self.anomaly_threshold
    
    def predict_batch(self, updates: List[ModelUpdate]) -> List[Dict[str, Any]]:
        """
        Predict anomaly scores for a batch of updates.
        
        Args:
            updates: List of model updates to score
            
        Returns:
            List of dictionaries containing client_id, score, and is_anomaly
        """
        results = []
        
        for update in updates:
            score = self.predict_anomaly_score(update)
            is_anomaly = score < self.anomaly_threshold
            
            results.append({
                'client_id': update.client_id,
                'round_id': update.round_id,
                'anomaly_score': score,
                'is_anomaly': is_anomaly,
                'timestamp': update.timestamp
            })
        
        return results
    
    def explain_anomaly(self, update: ModelUpdate) -> Dict[str, float]:
        """
        Generate basic explanation for anomaly score.
        
        Note: This provides a simple feature-based explanation.
        For more detailed explanations, use the SHAPExplainer class.
        
        Args:
            update: Model update to explain
            
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before generating explanations")
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(update)
            feature_names = self.feature_extractor.get_feature_names()
            
            # Simple explanation based on feature deviation from training mean
            explanations = {}
            
            if 'feature_means' in self.training_stats and 'feature_stds' in self.training_stats:
                # Compute normalized deviations from training mean
                feature_means = self.training_stats['feature_means']
                feature_stds = self.training_stats['feature_stds']
                
                for i, (name, value) in enumerate(zip(feature_names, features)):
                    if i < len(feature_means):
                        # Normalized deviation (z-score like)
                        mean_val = feature_means[i]
                        std_val = feature_stds[i] if feature_stds[i] > 0 else 1.0
                        deviation = abs(value - mean_val) / std_val
                        explanations[name] = float(deviation)
                    else:
                        explanations[name] = float(abs(value))
            else:
                # Fallback: use absolute feature values
                for name, value in zip(feature_names, features):
                    explanations[name] = float(abs(value))
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanation for update {update.client_id}: {e}")
            return {}
    
    def update_model(self, new_updates: List[ModelUpdate]) -> None:
        """
        Update the detector with new training data.
        
        Note: Isolation Forest doesn't support incremental learning,
        so this method retrains the entire model.
        
        Args:
            new_updates: New updates to incorporate
        """
        logger.info(f"Retraining Isolation Forest with {len(new_updates)} new updates")
        
        # For Isolation Forest, we need to retrain from scratch
        # In a production system, you might want to maintain a buffer
        # of recent updates and retrain periodically
        self.fit(new_updates)
    
    def set_threshold(self, threshold: float) -> None:
        """
        Update the anomaly threshold.
        
        Args:
            threshold: New threshold value (lower = more sensitive)
        """
        self.anomaly_threshold = threshold
        self.training_stats['threshold'] = threshold
        logger.info(f"Anomaly threshold updated to {threshold}")
    
    def get_threshold_percentile(self, percentile: float) -> float:
        """
        Get threshold corresponding to a percentile of training scores.
        
        Args:
            percentile: Percentile (0-100) of training data to use as threshold
            
        Returns:
            Threshold value
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before computing threshold")
        
        # This would require storing training scores, which we don't do currently
        # For now, return a simple estimate based on training stats
        mean = self.training_stats['score_mean']
        std = self.training_stats['score_std']
        
        # Rough approximation assuming normal distribution
        from scipy import stats
        z_score = stats.norm.ppf(percentile / 100.0)
        threshold = mean + z_score * std
        
        return float(threshold)
    
    def evaluate(self, test_updates: List[ModelUpdate], true_labels: List[bool]) -> Dict[str, Any]:
        """
        Evaluate detector performance on labeled test data.
        
        Args:
            test_updates: List of test model updates
            true_labels: True anomaly labels (True = anomaly, False = normal)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before evaluation")
        
        if len(test_updates) != len(true_labels):
            raise ValueError("Number of updates and labels must match")
        
        # Get predictions
        predictions = []
        scores = []
        
        for update in test_updates:
            score = self.predict_anomaly_score(update)
            pred = score < self.anomaly_threshold
            
            scores.append(score)
            predictions.append(pred)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # ROC AUC using scores (lower scores = more anomalous, so negate for AUC)
        try:
            auc = roc_auc_score(true_labels, [-s for s in scores])
        except ValueError:
            auc = 0.0  # Handle case where all labels are the same
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc,
            'n_samples': len(test_updates),
            'n_anomalies': sum(true_labels),
            'n_normal': len(true_labels) - sum(true_labels)
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'training_stats': self.training_stats,
            'anomaly_threshold': self.anomaly_threshold,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_extractor = model_data['feature_extractor']
        self.training_stats = model_data['training_stats']
        self.anomaly_threshold = model_data['anomaly_threshold']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")


class SimpleIsolationForestDetector(IAnomalyDetector):
    """
    Simplified Isolation Forest detector for basic anomaly detection.
    
    Uses minimal features and simpler configuration for lightweight operation.
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """Initialize simple detector."""
        from .feature_extractor import SimpleFeatureExtractor
        
        self.contamination = contamination
        self.random_state = random_state
        self.feature_extractor = SimpleFeatureExtractor()
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=50,
            random_state=random_state
        )
        self.is_fitted = False
    
    def fit(self, normal_updates: List[ModelUpdate]) -> None:
        """Train on normal updates."""
        features_list = [
            self.feature_extractor.extract_features(update) 
            for update in normal_updates
        ]
        X_train = np.array(features_list)
        self.model.fit(X_train)
        self.is_fitted = True
    
    def predict_anomaly_score(self, update: ModelUpdate) -> float:
        """Predict anomaly score."""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        features = self.feature_extractor.extract_features(update)
        score = self.model.decision_function(features.reshape(1, -1))[0]
        return float(score)
    
    def explain_anomaly(self, update: ModelUpdate) -> Dict[str, float]:
        """Generate simple explanation."""
        features = self.feature_extractor.extract_features(update)
        feature_names = self.feature_extractor.get_feature_names()
        
        return {name: float(abs(value)) for name, value in zip(feature_names, features)}
    
    def update_model(self, new_updates: List[ModelUpdate]) -> None:
        """Update with new data."""
        self.fit(new_updates)