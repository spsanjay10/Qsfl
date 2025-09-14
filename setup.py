"""
Setup script for QSFL-CAAD package
"""

from setuptools import setup, find_packages

setup(
    name="qsfl-caad",
    version="1.0.0",
    description="Quantum-Safe Federated Learning with Client Anomaly and Attack Detection",
    author="QSFL-CAAD Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "psutil>=5.8.0",
        "cryptography>=3.4.0",
    ],
    extras_require={
        "full": [
            "shap>=0.41.0",
            "tensorflow>=2.8.0",
            "flask>=2.0.0",
            "flask-socketio>=5.0.0",
        ]
    },
    python_requires=">=3.8",
)