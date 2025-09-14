"""
Feature Extraction for Model Updates

Implements feature extraction functions for neural network weights and gradients
with statistical feature computation and normalization pipeline.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

from .interfaces import IFeatureExtractor, ModelUpdate


logger = logging.getLogger(__name__)


class ModelUpdateFeatureExtractor(IFeatureExtractor):
    """
    Extracts statistical and structural features from neural network model updates.
    
    Features extracted include:
    - Weight statistics (mean, variance, skewness, kurtosis)
    - Gradient magnitude statistics
    - Layer-wise feature distributions
    - Structural properties (sparsity, norm ratios)
    """
    
    def __init__(self, normalize: bool = True, scaler_type: str = "standard"):
        """
        Initialize feature extractor.
        
        Args:
            normalize: Whether to normalize extracted features
            scaler_type: Type of scaler ("standard" or "minmax")
        """
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.scaler: Optional[Any] = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        
        if normalize:
            if scaler_type == "standard":
                self.scaler = StandardScaler()
            elif scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def extract_features(self, update: ModelUpdate) -> np.ndarray:
        """
        Extract comprehensive feature vector from model update.
        
        Args:
            update: Model update containing weights and metadata
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Extract weight-based features
        weight_features = self._extract_weight_features(update.weights)
        features.extend(weight_features)
        
        # Extract gradient features (always include, use zeros if not available)
        if 'gradients' in update.metadata:
            gradient_features = self._extract_gradient_features(update.metadata['gradients'])
        else:
            gradient_features = [0.0] * 5  # Default gradient features
        features.extend(gradient_features)
        
        # Extract structural features
        structural_features = self._extract_structural_features(update.weights)
        features.extend(structural_features)
        
        # Extract temporal features
        temporal_features = self._extract_temporal_features(update)
        features.extend(temporal_features)
        
        feature_vector = np.array(features, dtype=np.float32)
        
        # Handle NaN and infinite values
        feature_vector = self._handle_invalid_values(feature_vector)
        
        # Apply normalization if configured and fitted
        if self.normalize and self.scaler is not None and self.is_fitted:
            feature_vector = self.scaler.transform(feature_vector.reshape(1, -1)).flatten()
        
        return feature_vector
    
    def fit_scaler(self, updates: List[ModelUpdate]) -> None:
        """
        Fit the feature scaler on a collection of updates.
        
        Args:
            updates: List of model updates to fit scaler on
        """
        if not self.normalize or self.scaler is None:
            return
        
        logger.info(f"Fitting feature scaler on {len(updates)} updates")
        
        # Extract features from all updates
        all_features = []
        for update in updates:
            features = self._extract_raw_features(update)
            all_features.append(features)
        
        # Fit scaler
        feature_matrix = np.array(all_features)
        self.scaler.fit(feature_matrix)
        self.is_fitted = True
        
        logger.info("Feature scaler fitted successfully")
    
    def _extract_raw_features(self, update: ModelUpdate) -> np.ndarray:
        """Extract features without normalization for scaler fitting."""
        features = []
        
        weight_features = self._extract_weight_features(update.weights)
        features.extend(weight_features)
        
        # Extract gradient features (always include, use zeros if not available)
        if 'gradients' in update.metadata:
            gradient_features = self._extract_gradient_features(update.metadata['gradients'])
        else:
            gradient_features = [0.0] * 5  # Default gradient features
        features.extend(gradient_features)
        
        structural_features = self._extract_structural_features(update.weights)
        features.extend(structural_features)
        
        temporal_features = self._extract_temporal_features(update)
        features.extend(temporal_features)
        
        feature_vector = np.array(features, dtype=np.float32)
        return self._handle_invalid_values(feature_vector)
    
    def _extract_weight_features(self, weights: Dict[str, np.ndarray]) -> List[float]:
        """Extract statistical features from model weights."""
        features = []
        
        # Handle empty weights case
        if not weights:
            return [0.0] * 18  # Return zeros for all weight features
        
        # Flatten all weights
        all_weights = np.concatenate([w.flatten() for w in weights.values()])
        
        # Basic statistics
        features.extend([
            float(np.mean(all_weights)),           # Global mean
            float(np.var(all_weights)),            # Global variance
            float(np.std(all_weights)),            # Global std deviation
            float(np.min(all_weights)),            # Global minimum
            float(np.max(all_weights)),            # Global maximum
            float(np.median(all_weights)),         # Global median
        ])
        
        # Distribution shape statistics
        if len(all_weights) > 1:
            features.extend([
                float(stats.skew(all_weights)),        # Skewness
                float(stats.kurtosis(all_weights)),    # Kurtosis
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Percentile features
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            features.append(float(np.percentile(all_weights, p)))
        
        # Layer-wise statistics
        layer_means = []
        layer_vars = []
        layer_norms = []
        
        for layer_name, layer_weights in weights.items():
            layer_flat = layer_weights.flatten()
            layer_means.append(np.mean(layer_flat))
            layer_vars.append(np.var(layer_flat))
            layer_norms.append(np.linalg.norm(layer_flat))
        
        # Aggregate layer statistics
        if layer_means:
            features.extend([
                float(np.mean(layer_means)),       # Mean of layer means
                float(np.var(layer_means)),        # Variance of layer means
                float(np.mean(layer_vars)),        # Mean of layer variances
                float(np.var(layer_vars)),         # Variance of layer variances
                float(np.mean(layer_norms)),       # Mean of layer norms
                float(np.var(layer_norms)),        # Variance of layer norms
            ])
        else:
            features.extend([0.0] * 6)
        
        return features
    
    def _extract_gradient_features(self, gradients: Dict[str, np.ndarray]) -> List[float]:
        """Extract features from gradients if available."""
        features = []
        
        # Handle empty gradients case
        if not gradients:
            return [0.0] * 5  # Return zeros for all gradient features
        
        # Flatten all gradients
        all_gradients = np.concatenate([g.flatten() for g in gradients.values()])
        
        # Gradient magnitude statistics
        gradient_magnitudes = np.abs(all_gradients)
        features.extend([
            float(np.mean(gradient_magnitudes)),      # Mean gradient magnitude
            float(np.var(gradient_magnitudes)),       # Variance of magnitudes
            float(np.max(gradient_magnitudes)),       # Max gradient magnitude
            float(np.linalg.norm(all_gradients)),     # L2 norm of gradients
        ])
        
        # Gradient direction statistics
        if len(all_gradients) > 0:
            positive_ratio = float(np.sum(all_gradients > 0) / len(all_gradients))
            features.append(positive_ratio)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_structural_features(self, weights: Dict[str, np.ndarray]) -> List[float]:
        """Extract structural properties of the model."""
        features = []
        
        # Handle empty weights case
        if not weights:
            return [0.0] * 6  # Return zeros for all structural features
        
        # Sparsity features
        total_params = sum(w.size for w in weights.values())
        zero_params = sum(np.sum(np.abs(w) < 1e-8) for w in weights.values())
        sparsity_ratio = float(zero_params / total_params) if total_params > 0 else 0.0
        features.append(sparsity_ratio)
        
        # Norm ratios between layers
        layer_norms = [np.linalg.norm(w) for w in weights.values()]
        if len(layer_norms) > 1:
            norm_ratios = [layer_norms[i] / layer_norms[i-1] 
                          for i in range(1, len(layer_norms)) 
                          if layer_norms[i-1] != 0]
            if norm_ratios:
                features.extend([
                    float(np.mean(norm_ratios)),
                    float(np.var(norm_ratios)),
                ])
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        # Weight magnitude distribution
        all_weights = np.concatenate([w.flatten() for w in weights.values()])
        if len(all_weights) > 0:
            # Fraction of weights in different magnitude ranges
            small_weights = np.sum(np.abs(all_weights) < 0.01) / len(all_weights)
            medium_weights = np.sum((np.abs(all_weights) >= 0.01) & 
                                  (np.abs(all_weights) < 0.1)) / len(all_weights)
            large_weights = np.sum(np.abs(all_weights) >= 0.1) / len(all_weights)
            
            features.extend([
                float(small_weights),
                float(medium_weights), 
                float(large_weights)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return features
    
    def _extract_temporal_features(self, update: ModelUpdate) -> List[float]:
        """Extract temporal and metadata features."""
        features = []
        
        # Round information
        try:
            round_num = int(update.round_id.split('_')[-1]) if '_' in update.round_id else 0
            features.append(float(round_num))
        except (ValueError, AttributeError):
            features.append(0.0)
        
        # Update size (total number of parameters)
        total_params = sum(w.size for w in update.weights.values())
        features.append(float(total_params))
        
        # Number of layers
        num_layers = len(update.weights)
        features.append(float(num_layers))
        
        return features
    
    def _handle_invalid_values(self, features: np.ndarray) -> np.ndarray:
        """Handle NaN and infinite values in feature vector."""
        # Replace NaN with 0
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        if not self.feature_names:
            self.feature_names = self._generate_feature_names()
        return self.feature_names
    
    def _generate_feature_names(self) -> List[str]:
        """Generate descriptive names for all features."""
        names = []
        
        # Weight statistics
        names.extend([
            'weight_mean', 'weight_var', 'weight_std', 
            'weight_min', 'weight_max', 'weight_median',
            'weight_skew', 'weight_kurtosis',
            'weight_p10', 'weight_p25', 'weight_p75', 'weight_p90',
            'layer_means_mean', 'layer_means_var',
            'layer_vars_mean', 'layer_vars_var', 
            'layer_norms_mean', 'layer_norms_var'
        ])
        
        # Gradient statistics (if available)
        names.extend([
            'grad_mag_mean', 'grad_mag_var', 'grad_mag_max',
            'grad_l2_norm', 'grad_positive_ratio'
        ])
        
        # Structural features
        names.extend([
            'sparsity_ratio', 'norm_ratio_mean', 'norm_ratio_var',
            'small_weights_frac', 'medium_weights_frac', 'large_weights_frac'
        ])
        
        # Temporal features
        names.extend([
            'round_number', 'total_params', 'num_layers'
        ])
        
        return names


class SimpleFeatureExtractor(IFeatureExtractor):
    """
    Simplified feature extractor for basic anomaly detection.
    
    Extracts only essential statistical features for lightweight operation.
    """
    
    def __init__(self):
        self.feature_names = [
            'mean', 'std', 'min', 'max', 'l2_norm', 'sparsity'
        ]
    
    def extract_features(self, update: ModelUpdate) -> np.ndarray:
        """Extract basic statistical features."""
        # Flatten all weights
        all_weights = np.concatenate([w.flatten() for w in update.weights.values()])
        
        # Basic statistics
        features = [
            float(np.mean(all_weights)),
            float(np.std(all_weights)),
            float(np.min(all_weights)),
            float(np.max(all_weights)),
            float(np.linalg.norm(all_weights)),
            float(np.sum(np.abs(all_weights) < 1e-8) / len(all_weights))
        ]
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names