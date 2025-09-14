"""
Configuration Settings for QSFL-CAAD
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    
    # Default configuration
    default_config = {
        'security': {
            'anomaly_threshold': 0.6,
            'reputation_decay': 0.95,
            'quarantine_threshold': 0.8
        },
        'federated_learning': {
            'min_clients': 2,
            'max_clients': 100,
            'aggregation_method': 'federated_averaging'
        },
        'monitoring': {
            'log_level': 'INFO',
            'metrics_interval': 60
        }
    }
    
    # Try to load from file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration")
    
    return default_config