"""
SHAP Explainability Integration

Implements SHAP-based explanations for anomaly detection results
with visualization utilities and interpretable explanations.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
import warnings

try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap matplotlib")

from .interfaces import IExplainer, ModelUpdate, IAnomalyDetector
from .isolation_forest_detector import IsolationForestDetector


logger = logging.getLogger(__name__)


class SHAPExplainer(IExplainer):
    """
    SHAP-based explainer for anomaly detection results.
    
    Provides interpretable explanations for why specific model updates
    were flagged as anomalous using SHAP (SHapley Additive exPlanations) values.
    """
    
    def __init__(
        self,
        detector: IAnomalyDetector,
        background_data: Optional[List[ModelUpdate]] = None,
        max_background_samples: int = 100,
        explainer_type: str = "auto"
    ):
        """
        Initialize SHAP explainer.
        
        Args:
            detector: Trained anomaly detector to explain
            background_data: Background data for SHAP explainer (optional)
            max_background_samples: Maximum number of background samples to use
            explainer_type: Type of SHAP explainer ("auto", "tree", "kernel", "linear")
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not available. Install with: pip install shap")
        
        self.detector = detector
        self.background_data = background_data
        self.max_background_samples = max_background_samples
        self.explainer_type = explainer_type
        self.explainer = None
        self.background_features = None
        self.feature_names = None
        
        # Initialize explainer if detector is fitted
        if hasattr(detector, 'is_fitted') and detector.is_fitted:
            self._initialize_explainer()
    
    def _initialize_explainer(self) -> None:
        """Initialize the SHAP explainer with background data."""
        try:
            # Get feature names from detector
            if hasattr(self.detector, 'feature_extractor'):
                self.feature_names = self.detector.feature_extractor.get_feature_names()
            else:
                self.feature_names = [f"feature_{i}" for i in range(32)]  # Default
            
            # Prepare background data
            if self.background_data:
                background_features = []
                for update in self.background_data[:self.max_background_samples]:
                    try:
                        features = self.detector.feature_extractor.extract_features(update)
                        background_features.append(features)
                    except Exception as e:
                        logger.warning(f"Failed to extract features for background: {e}")
                        continue
                
                if background_features:
                    self.background_features = np.array(background_features)
                else:
                    logger.warning("No valid background features extracted, using synthetic background")
                    self.background_features = self._create_synthetic_background()
            else:
                # Create synthetic background data
                self.background_features = self._create_synthetic_background()
            
            # Initialize SHAP explainer based on detector type
            self._create_shap_explainer()
            
            logger.info(f"SHAP explainer initialized with {len(self.background_features)} background samples")
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None
    
    def _create_synthetic_background(self) -> np.ndarray:
        """Create synthetic background data when real background is not available."""
        # Create synthetic normal-looking features
        n_features = len(self.feature_names) if self.feature_names else 32
        n_samples = min(self.max_background_samples, 50)
        
        # Generate features that look like normal model updates
        background = np.random.normal(0, 0.1, (n_samples, n_features))
        
        # Add some structure to make it more realistic
        for i in range(n_features):
            if 'weight_mean' in (self.feature_names[i] if self.feature_names else ''):
                background[:, i] = np.random.normal(0, 0.05, n_samples)
            elif 'sparsity' in (self.feature_names[i] if self.feature_names else ''):
                background[:, i] = np.random.uniform(0, 0.1, n_samples)
            elif 'norm' in (self.feature_names[i] if self.feature_names else ''):
                background[:, i] = np.random.lognormal(0, 0.5, n_samples)
        
        return background.astype(np.float32)
    
    def _create_shap_explainer(self) -> None:
        """Create appropriate SHAP explainer based on detector type."""
        try:
            if isinstance(self.detector, IsolationForestDetector):
                # For Isolation Forest, use TreeExplainer if available, otherwise KernelExplainer
                if self.explainer_type == "auto" or self.explainer_type == "tree":
                    try:
                        self.explainer = shap.TreeExplainer(
                            self.detector.model,
                            data=self.background_features,
                            feature_perturbation="interventional"
                        )
                        logger.info("Using SHAP TreeExplainer for Isolation Forest")
                    except Exception as e:
                        logger.warning(f"TreeExplainer failed, falling back to KernelExplainer: {e}")
                        self._create_kernel_explainer()
                else:
                    self._create_kernel_explainer()
            else:
                # For other detectors, use KernelExplainer
                self._create_kernel_explainer()
                
        except Exception as e:
            logger.error(f"Failed to create SHAP explainer: {e}")
            self.explainer = None
    
    def _create_kernel_explainer(self) -> None:
        """Create KernelExplainer as fallback."""
        def predict_fn(X):
            """Prediction function for SHAP KernelExplainer."""
            scores = []
            for row in X:
                # Create a dummy ModelUpdate for prediction
                dummy_update = type('DummyUpdate', (), {
                    'client_id': 'shap_dummy',
                    'round_id': 'shap_round',
                    'weights': {},
                    'signature': b'',
                    'timestamp': None,
                    'metadata': {}
                })()
                
                # Directly use features if detector has a predict method that accepts features
                if hasattr(self.detector.model, 'decision_function'):
                    score = self.detector.model.decision_function(row.reshape(1, -1))[0]
                else:
                    # Fallback: use detector's predict method
                    score = 0.0  # Neutral score
                
                scores.append(score)
            
            return np.array(scores)
        
        # Use a subset of background data for KernelExplainer (it can be slow)
        background_subset = self.background_features[:min(20, len(self.background_features))]
        
        self.explainer = shap.KernelExplainer(
            predict_fn,
            background_subset,
            link="identity"
        )
        logger.info("Using SHAP KernelExplainer")
    
    def explain(self, update: ModelUpdate, anomaly_score: float) -> Dict[str, float]:
        """
        Generate SHAP explanation for anomaly score.
        
        Args:
            update: Model update that was scored
            anomaly_score: Computed anomaly score
            
        Returns:
            Dictionary mapping feature names to SHAP importance scores
        """
        if self.explainer is None:
            logger.warning("SHAP explainer not initialized, falling back to simple explanation")
            return self._fallback_explanation(update)
        
        try:
            # Extract features from update
            features = self.detector.feature_extractor.extract_features(update)
            features = features.reshape(1, -1)
            
            # Compute SHAP values
            shap_values = self.explainer.shap_values(features)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Multi-output case (shouldn't happen for anomaly detection)
                shap_values = shap_values[0]
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]  # Take first sample
            
            # Create explanation dictionary
            explanation = {}
            for i, (name, shap_val) in enumerate(zip(self.feature_names, shap_values)):
                explanation[name] = float(abs(shap_val))  # Use absolute SHAP values
            
            return explanation
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(update)
    
    def _fallback_explanation(self, update: ModelUpdate) -> Dict[str, float]:
        """Fallback explanation when SHAP fails."""
        try:
            # Use detector's built-in explanation if available
            if hasattr(self.detector, 'explain_anomaly'):
                return self.detector.explain_anomaly(update)
            else:
                # Very basic fallback
                features = self.detector.feature_extractor.extract_features(update)
                feature_names = self.detector.feature_extractor.get_feature_names()
                
                return {name: float(abs(val)) for name, val in zip(feature_names, features)}
                
        except Exception as e:
            logger.error(f"Fallback explanation failed: {e}")
            return {}
    
    def explain_batch(self, updates: List[ModelUpdate]) -> List[Dict[str, Any]]:
        """
        Generate explanations for a batch of updates.
        
        Args:
            updates: List of model updates to explain
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for update in updates:
            try:
                # Get anomaly score
                anomaly_score = self.detector.predict_anomaly_score(update)
                
                # Get SHAP explanation
                shap_explanation = self.explain(update, anomaly_score)
                
                # Combine into result
                result = {
                    'client_id': update.client_id,
                    'round_id': update.round_id,
                    'anomaly_score': anomaly_score,
                    'is_anomaly': anomaly_score < getattr(self.detector, 'anomaly_threshold', 0.0),
                    'shap_values': shap_explanation,
                    'top_features': self._get_top_features(shap_explanation, top_k=5)
                }
                
                explanations.append(result)
                
            except Exception as e:
                logger.error(f"Failed to explain update {update.client_id}: {e}")
                explanations.append({
                    'client_id': update.client_id,
                    'round_id': update.round_id,
                    'error': str(e)
                })
        
        return explanations
    
    def _get_top_features(self, shap_values: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top contributing features from SHAP values."""
        if not shap_values:
            return []
        
        # Sort by absolute SHAP value
        sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_features[:top_k]
    
    def create_explanation_summary(self, update: ModelUpdate, anomaly_score: float) -> str:
        """
        Create human-readable explanation summary.
        
        Args:
            update: Model update to explain
            anomaly_score: Anomaly score for the update
            
        Returns:
            Human-readable explanation string
        """
        try:
            shap_values = self.explain(update, anomaly_score)
            top_features = self._get_top_features(shap_values, top_k=3)
            
            is_anomaly = anomaly_score < getattr(self.detector, 'anomaly_threshold', 0.0)
            
            if is_anomaly:
                summary = f"Update from {update.client_id} flagged as ANOMALOUS (score: {anomaly_score:.3f})\n"
                summary += "Top contributing factors:\n"
            else:
                summary = f"Update from {update.client_id} classified as NORMAL (score: {anomaly_score:.3f})\n"
                summary += "Key characteristics:\n"
            
            for i, (feature, importance) in enumerate(top_features, 1):
                # Make feature names more readable
                readable_name = self._make_feature_readable(feature)
                summary += f"  {i}. {readable_name}: {importance:.3f}\n"
            
            return summary
            
        except Exception as e:
            return f"Failed to generate explanation for {update.client_id}: {e}"
    
    def _make_feature_readable(self, feature_name: str) -> str:
        """Convert technical feature names to readable descriptions."""
        readable_names = {
            'weight_mean': 'Average weight value',
            'weight_var': 'Weight variance',
            'weight_std': 'Weight standard deviation',
            'sparsity_ratio': 'Proportion of zero weights',
            'layer_norms_mean': 'Average layer norm',
            'grad_mag_mean': 'Average gradient magnitude',
            'total_params': 'Total parameter count',
            'num_layers': 'Number of layers'
        }
        
        return readable_names.get(feature_name, feature_name.replace('_', ' ').title())
    
    def visualize_explanation(
        self,
        update: ModelUpdate,
        anomaly_score: float,
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Create visualization of SHAP explanation.
        
        Args:
            update: Model update to explain
            anomaly_score: Anomaly score
            save_path: Path to save plot (optional)
            show_plot: Whether to display plot
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP visualization requires matplotlib")
            return
        
        try:
            # Get SHAP values
            shap_values = self.explain(update, anomaly_score)
            
            if not shap_values:
                logger.warning("No SHAP values available for visualization")
                return
            
            # Create bar plot of top features
            top_features = self._get_top_features(shap_values, top_k=10)
            
            if not top_features:
                return
            
            features, values = zip(*top_features)
            readable_features = [self._make_feature_readable(f) for f in features]
            
            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(readable_features)), values)
            
            # Color bars based on contribution (red for high, blue for low)
            max_val = max(values) if values else 1
            for i, (bar, val) in enumerate(zip(bars, values)):
                color_intensity = val / max_val
                bar.set_color(plt.cm.RdYlBu_r(color_intensity))
            
            plt.yticks(range(len(readable_features)), readable_features)
            plt.xlabel('SHAP Value (Feature Importance)')
            plt.title(f'Anomaly Explanation for {update.client_id}\n'
                     f'Anomaly Score: {anomaly_score:.3f}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Explanation plot saved to {save_path}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
    
    def get_global_feature_importance(
        self,
        updates: List[ModelUpdate],
        max_samples: int = 100
    ) -> Dict[str, float]:
        """
        Compute global feature importance across multiple updates.
        
        Args:
            updates: List of updates to analyze
            max_samples: Maximum number of samples to use
            
        Returns:
            Dictionary of average feature importance scores
        """
        if not updates:
            return {}
        
        # Sample updates if too many
        if len(updates) > max_samples:
            import random
            updates = random.sample(updates, max_samples)
        
        # Collect SHAP values for all updates
        all_shap_values = []
        
        for update in updates:
            try:
                anomaly_score = self.detector.predict_anomaly_score(update)
                shap_values = self.explain(update, anomaly_score)
                
                if shap_values:
                    all_shap_values.append(shap_values)
                    
            except Exception as e:
                logger.warning(f"Failed to get SHAP values for {update.client_id}: {e}")
                continue
        
        if not all_shap_values:
            return {}
        
        # Compute average importance for each feature
        feature_importance = {}
        
        for feature_name in self.feature_names:
            values = [shap_vals.get(feature_name, 0.0) for shap_vals in all_shap_values]
            feature_importance[feature_name] = float(np.mean(np.abs(values)))
        
        return feature_importance


class SimpleSHAPExplainer(IExplainer):
    """
    Simplified SHAP explainer for basic use cases.
    
    Provides essential SHAP functionality without advanced features.
    """
    
    def __init__(self, detector: IAnomalyDetector):
        """Initialize simple explainer."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not available")
        
        self.detector = detector
    
    def explain(self, update: ModelUpdate, anomaly_score: float) -> Dict[str, float]:
        """Generate basic SHAP explanation."""
        try:
            # Use detector's built-in explanation as fallback
            if hasattr(self.detector, 'explain_anomaly'):
                return self.detector.explain_anomaly(update)
            else:
                return {}
        except Exception as e:
            logger.error(f"Simple SHAP explanation failed: {e}")
            return {}