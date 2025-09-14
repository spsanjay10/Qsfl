# QSFL-CAAD Developer Guide

This guide provides comprehensive information for developers who want to contribute to, extend, or integrate with the QSFL-CAAD system.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Coding Standards](#coding-standards)
4. [Testing Guidelines](#testing-guidelines)
5. [Contributing Guidelines](#contributing-guidelines)
6. [Extension Points](#extension-points)
7. [Debugging and Troubleshooting](#debugging-and-troubleshooting)
8. [Performance Optimization](#performance-optimization)

---

## Development Environment Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, or virtualenv)
- Optional: Docker for containerized development

### Installation Steps

1. **Clone the Repository**
```bash
git clone https://github.com/your-org/qsfl-caad.git
cd qsfl-caad
```

2. **Create Virtual Environment**
```bash
# Using venv
python -m venv qsfl_env
source qsfl_env/bin/activate  # On Windows: qsfl_env\Scripts\activate

# Using conda
conda create -n qsfl_env python=3.9
conda activate qsfl_env
```

3. **Install Dependencies**
```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .
```

4. **Install Post-Quantum Cryptography Libraries**
```bash
# Try to install pqcrypto (may require compilation)
pip install pqcrypto

# If pqcrypto fails, the system will use fallback implementations
# based on NIST specifications
```

5. **Setup Pre-commit Hooks**
```bash
pre-commit install
```

6. **Initialize Configuration**
```bash
# Copy example configuration
cp .env.example .env

# Edit configuration as needed
nano .env
```

7. **Run Initial Tests**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest tests/test_pq_security/
pytest tests/test_integration/
```

### Development Dependencies

The `requirements-dev.txt` file includes:

```txt
# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.21.0
pytest-mock>=3.10.0

# Code Quality
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.0.0
pylint>=2.17.0

# Documentation
sphinx>=6.0.0
sphinx-rtd-theme>=1.2.0
myst-parser>=1.0.0

# Development Tools
pre-commit>=3.0.0
jupyter>=1.0.0
ipython>=8.0.0

# Profiling and Debugging
memory-profiler>=0.60.0
line-profiler>=4.0.0
py-spy>=0.3.14
```

### IDE Configuration

#### VS Code Settings

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./qsfl_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ]
}
```

#### PyCharm Configuration

1. Set Python interpreter to virtual environment
2. Enable pytest as test runner
3. Configure code style to use Black formatter
4. Enable type checking with mypy

---

## Project Structure

### Directory Layout

```
qsfl-caad/
├── pq_security/              # Post-quantum cryptography module
│   ├── __init__.py
│   ├── interfaces.py         # Abstract interfaces
│   ├── kyber.py             # CRYSTALS-Kyber implementation
│   ├── dilithium.py         # CRYSTALS-Dilithium implementation
│   └── manager.py           # Unified PQ crypto manager
├── auth/                    # Authentication module
│   ├── __init__.py
│   ├── interfaces.py
│   ├── authentication_service.py
│   ├── credential_manager.py
│   └── revocation_manager.py
├── anomaly_detection/       # Anomaly detection module
│   ├── __init__.py
│   ├── interfaces.py
│   ├── isolation_forest_detector.py
│   ├── shap_explainer.py
│   ├── feature_extractor.py
│   └── reputation_manager.py
├── federated_learning/      # Federated learning core
│   ├── __init__.py
│   ├── interfaces.py
│   ├── server.py
│   ├── model_aggregator.py
│   ├── model_update_handler.py
│   ├── client_simulation.py
│   └── dataset_manager.py
├── monitoring/              # Monitoring and logging
│   ├── __init__.py
│   ├── interfaces.py
│   ├── security_logger.py
│   ├── metrics_collector.py
│   └── alert_manager.py
├── config/                  # Configuration management
│   ├── __init__.py
│   ├── settings.py
│   └── config.yaml
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── conftest.py          # Pytest configuration
│   ├── test_pq_security/
│   ├── test_auth/
│   ├── test_anomaly_detection/
│   ├── test_federated_learning/
│   ├── test_monitoring/
│   └── test_integration/
├── docs/                    # Documentation
│   ├── API_DOCUMENTATION.md
│   ├── USAGE_EXAMPLES.md
│   ├── DEVELOPER_GUIDE.md
│   └── TROUBLESHOOTING.md
├── scripts/                 # Utility scripts
│   ├── setup_dev_env.sh
│   ├── run_tests.sh
│   └── benchmark.py
├── .github/                 # GitHub workflows
│   └── workflows/
│       ├── ci.yml
│       └── security.yml
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── setup.py                # Package setup
├── pyproject.toml          # Modern Python project config
├── .gitignore
├── .pre-commit-config.yaml
└── README.md
```

### Module Architecture

Each module follows a consistent structure:

1. **interfaces.py**: Abstract base classes and data structures
2. **Implementation files**: Concrete implementations of interfaces
3. **__init__.py**: Module exports and public API

### Import Conventions

```python
# Preferred import style
from pq_security.interfaces import IPQCrypto
from pq_security.kyber import KyberKeyExchange
from anomaly_detection.interfaces import ModelUpdate, AnomalyReport

# Avoid wildcard imports
# from pq_security import *  # DON'T DO THIS

# Use relative imports within modules
from .interfaces import IAuthenticationService  # Within auth module
```

---

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications enforced by Black formatter.

#### Code Formatting

```python
# Use Black formatter (line length: 88 characters)
# Install: pip install black
# Format: black .

# Example of properly formatted code
class SecureFederatedServer:
    """Main server for coordinating secure federated learning."""
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        pq_manager: Optional[PQCryptoManager] = None
    ) -> None:
        self.config = config
        self.pq_manager = pq_manager or PQCryptoManager()
        self.clients: Dict[str, ClientInfo] = {}
        
    def process_client_update(
        self, client_id: str, update: ModelUpdate
    ) -> Tuple[bool, str]:
        """Process and validate client model update."""
        try:
            # Authenticate client
            if not self._authenticate_client(client_id, update):
                return False, "Authentication failed"
            
            # Check for anomalies
            anomaly_score = self._detect_anomalies(update)
            if anomaly_score > self.config["anomaly_threshold"]:
                return False, f"Anomaly detected: {anomaly_score:.3f}"
            
            return True, "Update accepted"
            
        except Exception as e:
            logger.error(f"Error processing update from {client_id}: {e}")
            return False, "Processing error"
```

#### Import Organization

Use isort with Black profile:

```python
# Standard library imports
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Third-party imports
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest

# Local imports
from config.settings import load_config
from pq_security.interfaces import IPQCrypto
from anomaly_detection.interfaces import ModelUpdate
```

#### Type Hints

Use comprehensive type hints:

```python
from typing import Dict, List, Optional, Union, Tuple, Any, Protocol

class ModelAggregator:
    def aggregate(
        self, 
        updates: List[ModelUpdate], 
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, np.ndarray]:
        """Aggregate model updates with optional weighting."""
        pass
    
    def _compute_weighted_average(
        self, 
        arrays: List[np.ndarray], 
        weights: List[float]
    ) -> np.ndarray:
        """Compute weighted average of numpy arrays."""
        pass
```

#### Documentation Standards

Use Google-style docstrings:

```python
def detect_anomalies(
    self, 
    update: ModelUpdate, 
    threshold: float = 0.5
) -> Tuple[float, Dict[str, Any]]:
    """Detect anomalies in model update.
    
    Args:
        update: Model update to analyze
        threshold: Anomaly threshold (0.0 to 1.0)
        
    Returns:
        Tuple of (anomaly_score, explanation_dict)
        
    Raises:
        ValueError: If threshold is not in valid range
        AnomalyDetectionError: If detection fails
        
    Example:
        >>> detector = IsolationForestDetector()
        >>> score, explanation = detector.detect_anomalies(update)
        >>> if score > 0.8:
        ...     print("High anomaly detected!")
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be in [0.0, 1.0], got {threshold}")
    
    # Implementation here
    pass
```

#### Error Handling

Use specific exception types and proper logging:

```python
# Define custom exceptions
class QSFLError(Exception):
    """Base exception for QSFL-CAAD system."""
    pass

class AuthenticationError(QSFLError):
    """Authentication-related errors."""
    pass

class AnomalyDetectionError(QSFLError):
    """Anomaly detection errors."""
    pass

# Use in code
def authenticate_client(self, client_id: str, signature: bytes) -> bool:
    """Authenticate client using digital signature."""
    try:
        # Verification logic
        return self.pq_manager.verify_signature(signature, client_id)
    except CryptographicError as e:
        logger.error(f"Cryptographic error during authentication: {e}")
        raise AuthenticationError(f"Failed to authenticate {client_id}") from e
    except Exception as e:
        logger.error(f"Unexpected error during authentication: {e}")
        raise AuthenticationError(f"Authentication system error") from e
```

### Configuration Management

Use structured configuration with validation:

```python
# config/settings.py
from dataclasses import dataclass
from typing import Dict, Any
import yaml

@dataclass
class SecurityConfig:
    """Security-related configuration."""
    anomaly_threshold: float = 0.5
    reputation_decay: float = 0.95
    quarantine_threshold: float = 0.8
    
    def __post_init__(self):
        if not 0.0 <= self.anomaly_threshold <= 1.0:
            raise ValueError("anomaly_threshold must be in [0.0, 1.0]")

@dataclass
class SystemConfig:
    """Main system configuration."""
    security: SecurityConfig
    database_url: str
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            security=SecurityConfig(**data.get('security', {})),
            database_url=data['database_url'],
            log_level=data.get('log_level', 'INFO')
        )
```

---

## Testing Guidelines

### Test Structure

Organize tests to mirror the source structure:

```
tests/
├── conftest.py              # Shared fixtures
├── test_pq_security/
│   ├── test_kyber.py
│   ├── test_dilithium.py
│   └── test_manager.py
├── test_auth/
│   ├── test_authentication_service.py
│   ├── test_credential_manager.py
│   └── test_revocation_manager.py
└── test_integration/
    ├── test_e2e_federated_learning.py
    └── test_security_integration.py
```

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test scalability and performance
5. **Security Tests**: Test security properties

### Writing Good Tests

#### Unit Test Example

```python
# tests/test_pq_security/test_kyber.py
import pytest
import numpy as np
from pq_security.kyber import KyberKeyExchange
from pq_security.interfaces import IKeyExchange

class TestKyberKeyExchange:
    """Test suite for CRYSTALS-Kyber key exchange."""
    
    @pytest.fixture
    def kyber(self) -> KyberKeyExchange:
        """Create KyberKeyExchange instance for testing."""
        return KyberKeyExchange()
    
    def test_implements_interface(self, kyber):
        """Test that KyberKeyExchange implements IKeyExchange."""
        assert isinstance(kyber, IKeyExchange)
    
    def test_generate_keypair(self, kyber):
        """Test keypair generation."""
        public_key, private_key = kyber.generate_keypair()
        
        assert isinstance(public_key, bytes)
        assert isinstance(private_key, bytes)
        assert len(public_key) > 0
        assert len(private_key) > 0
        assert public_key != private_key
    
    def test_key_exchange_success(self, kyber):
        """Test successful key exchange."""
        # Generate keypair
        public_key, private_key = kyber.generate_keypair()
        
        # Encapsulate
        ciphertext, shared_secret1 = kyber.encapsulate(public_key)
        
        # Decapsulate
        shared_secret2 = kyber.decapsulate(ciphertext, private_key)
        
        # Verify shared secrets match
        assert shared_secret1 == shared_secret2
        assert len(shared_secret1) > 0
    
    def test_invalid_public_key(self, kyber):
        """Test encapsulation with invalid public key."""
        invalid_key = b"invalid_key"
        
        with pytest.raises(ValueError):
            kyber.encapsulate(invalid_key)
    
    def test_invalid_private_key(self, kyber):
        """Test decapsulation with invalid private key."""
        public_key, _ = kyber.generate_keypair()
        ciphertext, _ = kyber.encapsulate(public_key)
        invalid_private_key = b"invalid_private_key"
        
        with pytest.raises(ValueError):
            kyber.decapsulate(ciphertext, invalid_private_key)
    
    @pytest.mark.parametrize("num_iterations", [10, 50, 100])
    def test_multiple_key_exchanges(self, kyber, num_iterations):
        """Test multiple key exchanges for consistency."""
        for _ in range(num_iterations):
            public_key, private_key = kyber.generate_keypair()
            ciphertext, secret1 = kyber.encapsulate(public_key)
            secret2 = kyber.decapsulate(ciphertext, private_key)
            assert secret1 == secret2
```

#### Integration Test Example

```python
# tests/test_integration/test_auth_integration.py
import pytest
from datetime import datetime, timedelta
from auth.authentication_service import AuthenticationService
from auth.credential_manager import CredentialManager
from pq_security.manager import PQCryptoManager

class TestAuthenticationIntegration:
    """Integration tests for authentication system."""
    
    @pytest.fixture
    def auth_system(self):
        """Create integrated authentication system."""
        pq_manager = PQCryptoManager()
        cred_manager = CredentialManager()
        auth_service = AuthenticationService(pq_manager, cred_manager)
        
        return {
            'auth_service': auth_service,
            'cred_manager': cred_manager,
            'pq_manager': pq_manager
        }
    
    def test_complete_auth_workflow(self, auth_system):
        """Test complete authentication workflow."""
        auth_service = auth_system['auth_service']
        
        # Register client
        client_id = "test_client_001"
        credentials = auth_service.register_client(client_id)
        
        assert credentials.client_id == client_id
        assert credentials.status.value == "active"
        
        # Create and sign message
        message = b"test_model_update_data"
        signature = auth_system['pq_manager'].sign(
            message, credentials.private_key
        )
        
        # Authenticate
        is_authenticated = auth_service.authenticate_client(
            client_id, signature, message
        )
        
        assert is_authenticated is True
        
        # Test with wrong signature
        wrong_signature = b"wrong_signature"
        is_authenticated = auth_service.authenticate_client(
            client_id, wrong_signature, message
        )
        
        assert is_authenticated is False
```

#### Performance Test Example

```python
# tests/test_performance/test_anomaly_detection_performance.py
import pytest
import time
import numpy as np
from anomaly_detection.isolation_forest_detector import IsolationForestDetector
from anomaly_detection.interfaces import ModelUpdate

class TestAnomalyDetectionPerformance:
    """Performance tests for anomaly detection."""
    
    @pytest.fixture
    def large_dataset(self):
        """Create large dataset for performance testing."""
        updates = []
        for i in range(1000):
            weights = {
                "layer1": np.random.normal(0, 0.1, (100, 50)),
                "layer2": np.random.normal(0, 0.1, (50, 10))
            }
            update = ModelUpdate(
                client_id=f"client_{i}",
                round_id="perf_test",
                weights=weights,
                signature=b"sig",
                timestamp=datetime.now(),
                metadata={}
            )
            updates.append(update)
        return updates
    
    def test_training_performance(self, large_dataset):
        """Test detector training performance."""
        detector = IsolationForestDetector()
        
        start_time = time.time()
        detector.fit(large_dataset[:800])  # Train on 800 samples
        training_time = time.time() - start_time
        
        # Should train within reasonable time
        assert training_time < 30.0  # 30 seconds max
        
    def test_prediction_performance(self, large_dataset):
        """Test prediction performance."""
        detector = IsolationForestDetector()
        detector.fit(large_dataset[:800])
        
        test_updates = large_dataset[800:]
        
        start_time = time.time()
        for update in test_updates:
            score = detector.predict_anomaly_score(update)
            assert 0.0 <= score <= 1.0
        
        prediction_time = time.time() - start_time
        avg_prediction_time = prediction_time / len(test_updates)
        
        # Should predict quickly
        assert avg_prediction_time < 0.1  # 100ms per prediction max
```

### Test Fixtures and Utilities

Create reusable test fixtures in `conftest.py`:

```python
# tests/conftest.py
import pytest
import tempfile
import shutil
from datetime import datetime
import numpy as np

from config.settings import SystemConfig, SecurityConfig
from pq_security.manager import PQCryptoManager
from anomaly_detection.interfaces import ModelUpdate

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_config():
    """Create test configuration."""
    return SystemConfig(
        security=SecurityConfig(
            anomaly_threshold=0.5,
            reputation_decay=0.95,
            quarantine_threshold=0.8
        ),
        database_url="sqlite:///:memory:",
        log_level="DEBUG"
    )

@pytest.fixture
def pq_manager():
    """Create PQCryptoManager for testing."""
    return PQCryptoManager()

@pytest.fixture
def sample_model_update():
    """Create sample model update for testing."""
    weights = {
        "layer1": np.random.normal(0, 0.1, (10, 5)),
        "layer2": np.random.normal(0, 0.1, (5, 1))
    }
    
    return ModelUpdate(
        client_id="test_client",
        round_id="test_round",
        weights=weights,
        signature=b"test_signature",
        timestamp=datetime.now(),
        metadata={"accuracy": 0.85}
    )

@pytest.fixture
def malicious_model_update():
    """Create malicious model update for testing."""
    weights = {
        "layer1": np.random.normal(0.5, 1.0, (10, 5)),  # Anomalous
        "layer2": np.random.normal(-0.3, 0.8, (5, 1))   # Anomalous
    }
    
    return ModelUpdate(
        client_id="malicious_client",
        round_id="test_round",
        weights=weights,
        signature=b"malicious_signature",
        timestamp=datetime.now(),
        metadata={"accuracy": 0.3}
    )
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_pq_security/test_kyber.py

# Run tests matching pattern
pytest -k "test_authentication"

# Run tests with specific markers
pytest -m "integration"

# Run tests in parallel
pytest -n auto

# Run tests with verbose output
pytest -v

# Run tests and stop on first failure
pytest -x
```

### Test Markers

Define custom test markers in `pytest.ini`:

```ini
[tool:pytest]
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    security: Security tests
    slow: Slow tests that take more than 5 seconds
```

Use markers in tests:

```python
@pytest.mark.integration
def test_auth_integration():
    pass

@pytest.mark.slow
@pytest.mark.performance
def test_large_scale_aggregation():
    pass
```

---

## Contributing Guidelines

### Git Workflow

We use a feature branch workflow:

1. **Fork the repository** (for external contributors)
2. **Create feature branch** from `main`
3. **Make changes** with clear, atomic commits
4. **Write tests** for new functionality
5. **Update documentation** as needed
6. **Submit pull request** with detailed description

#### Branch Naming

Use descriptive branch names:

```bash
# Feature branches
feature/add-kyber-implementation
feature/improve-anomaly-detection
feature/dashboard-ui

# Bug fix branches
bugfix/fix-authentication-timeout
bugfix/memory-leak-aggregation

# Documentation branches
docs/api-documentation
docs/developer-guide
```

#### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Examples:

```
feat(pq_security): add CRYSTALS-Kyber key exchange

Implement CRYSTALS-Kyber key exchange mechanism with fallback
to simulated implementation when pqcrypto library is unavailable.

Closes #123

fix(auth): resolve credential expiration check

The credential expiration was not properly handling timezone-aware
datetime objects, causing false positives.

Fixes #456

docs(api): add comprehensive usage examples

Add detailed examples for all major components including
post-quantum security, authentication, and anomaly detection.

test(integration): add end-to-end federated learning test

Add comprehensive test covering complete federated learning
workflow with malicious clients and security validation.
```

### Code Review Process

#### Pull Request Requirements

- [ ] All tests pass
- [ ] Code coverage maintained or improved
- [ ] Documentation updated
- [ ] No security vulnerabilities introduced
- [ ] Performance impact assessed
- [ ] Breaking changes documented

#### Review Checklist

**Functionality**
- [ ] Code solves the intended problem
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] No obvious bugs

**Code Quality**
- [ ] Code follows style guidelines
- [ ] Functions are reasonably sized
- [ ] Variable names are descriptive
- [ ] Comments explain complex logic

**Testing**
- [ ] New code has appropriate tests
- [ ] Tests cover edge cases
- [ ] Tests are reliable and fast
- [ ] Integration tests updated if needed

**Security**
- [ ] No hardcoded secrets
- [ ] Input validation is present
- [ ] Cryptographic operations are correct
- [ ] No information leakage

**Performance**
- [ ] No obvious performance regressions
- [ ] Algorithms are efficient
- [ ] Memory usage is reasonable
- [ ] Database queries are optimized

### Issue Reporting

When reporting issues, include:

1. **Environment information**
   - Python version
   - Operating system
   - Package versions
   - Configuration details

2. **Steps to reproduce**
   - Minimal code example
   - Input data (if applicable)
   - Expected vs actual behavior

3. **Error messages**
   - Full stack traces
   - Log output
   - Screenshots (if UI-related)

#### Issue Template

```markdown
## Bug Report

**Environment:**
- Python version: 3.9.7
- OS: Ubuntu 20.04
- QSFL-CAAD version: 1.2.0
- pqcrypto version: 0.7.2

**Description:**
Brief description of the issue.

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:**
What should happen.

**Actual Behavior:**
What actually happens.

**Error Messages:**
```
Full error message and stack trace
```

**Additional Context:**
Any other relevant information.
```

---

## Extension Points

The QSFL-CAAD system is designed to be extensible. Here are the main extension points:

### Adding New Cryptographic Algorithms

Implement the `IPQCrypto` interface:

```python
from pq_security.interfaces import IPQCrypto
from typing import Tuple

class NewPQAlgorithm(IPQCrypto):
    """Implementation of new post-quantum algorithm."""
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate keypair for new algorithm."""
        # Implementation here
        pass
    
    def encrypt(self, plaintext: bytes, public_key: bytes) -> bytes:
        """Encrypt using new algorithm."""
        # Implementation here
        pass
    
    def decrypt(self, ciphertext: bytes, private_key: bytes) -> bytes:
        """Decrypt using new algorithm."""
        # Implementation here
        pass
    
    def sign(self, message: bytes, private_key: bytes) -> bytes:
        """Sign using new algorithm."""
        # Implementation here
        pass
    
    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify signature using new algorithm."""
        # Implementation here
        pass
```

Register the new algorithm:

```python
# In pq_security/manager.py
from .new_algorithm import NewPQAlgorithm

class PQCryptoManager:
    def __init__(self, algorithm="kyber_dilithium"):
        if algorithm == "new_algorithm":
            self.crypto = NewPQAlgorithm()
        # ... existing algorithms
```

### Adding New Anomaly Detection Methods

Implement the `IAnomalyDetector` interface:

```python
from anomaly_detection.interfaces import IAnomalyDetector, ModelUpdate
from typing import List, Dict
import numpy as np

class NewAnomalyDetector(IAnomalyDetector):
    """New anomaly detection algorithm."""
    
    def __init__(self, **kwargs):
        self.model = None
        self.is_trained = False
    
    def fit(self, normal_updates: List[ModelUpdate]) -> None:
        """Train the new detector."""
        # Extract features
        features = self._extract_features(normal_updates)
        
        # Train your algorithm
        self.model = self._train_model(features)
        self.is_trained = True
    
    def predict_anomaly_score(self, update: ModelUpdate) -> float:
        """Predict anomaly score using new method."""
        if not self.is_trained:
            raise ValueError("Detector must be trained first")
        
        features = self._extract_features([update])
        score = self.model.predict(features[0])
        
        return float(score)
    
    def explain_anomaly(self, update: ModelUpdate) -> Dict[str, float]:
        """Generate explanation for new method."""
        # Implementation specific to your algorithm
        pass
    
    def _extract_features(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Extract features for new algorithm."""
        # Feature extraction logic
        pass
    
    def _train_model(self, features: np.ndarray):
        """Train the underlying model."""
        # Training logic
        pass
```

### Adding New Aggregation Methods

Implement custom aggregation in `ModelAggregator`:

```python
from federated_learning.model_aggregator import ModelAggregator
import numpy as np

class CustomModelAggregator(ModelAggregator):
    """Custom model aggregation with new method."""
    
    def __init__(self):
        super().__init__()
        self.aggregation_methods["custom_method"] = self._custom_aggregation
    
    def _custom_aggregation(
        self, 
        updates: List[ModelUpdate], 
        weights: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """Custom aggregation algorithm."""
        
        # Extract model weights
        client_weights = {}
        for update in updates:
            client_weights[update.client_id] = update.weights
        
        # Implement your custom aggregation logic
        aggregated = {}
        for layer_name in client_weights[updates[0].client_id].keys():
            layer_updates = []
            layer_weights = []
            
            for update in updates:
                layer_updates.append(client_weights[update.client_id][layer_name])
                layer_weights.append(weights.get(update.client_id, 1.0))
            
            # Custom aggregation formula
            aggregated[layer_name] = self._custom_layer_aggregation(
                layer_updates, layer_weights
            )
        
        return aggregated
    
    def _custom_layer_aggregation(
        self, 
        layer_updates: List[np.ndarray], 
        weights: List[float]
    ) -> np.ndarray:
        """Custom layer-wise aggregation."""
        # Implement your custom logic here
        # Example: median aggregation with weights
        
        weighted_updates = []
        for update, weight in zip(layer_updates, weights):
            weighted_updates.append(update * weight)
        
        # Return median (or your custom aggregation)
        return np.median(weighted_updates, axis=0)
```

### Adding New Monitoring Metrics

Extend the metrics collection system:

```python
from monitoring.metrics_collector import MetricsCollector
from monitoring.interfaces import SystemMetrics
from datetime import datetime

class ExtendedMetricsCollector(MetricsCollector):
    """Extended metrics collector with custom metrics."""
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect extended system metrics."""
        base_metrics = super().collect_metrics()
        
        # Add custom metrics
        custom_metrics = self._collect_custom_metrics()
        
        # Extend base metrics (you might need to modify SystemMetrics)
        extended_metrics = SystemMetrics(
            timestamp=base_metrics.timestamp,
            cpu_usage=base_metrics.cpu_usage,
            memory_usage=base_metrics.memory_usage,
            active_clients=base_metrics.active_clients,
            training_rounds_completed=base_metrics.training_rounds_completed,
            anomalies_detected=base_metrics.anomalies_detected,
            authentication_failures=base_metrics.authentication_failures,
            # Add your custom fields
            network_latency=custom_metrics.get('network_latency', 0.0),
            disk_usage=custom_metrics.get('disk_usage', 0.0),
            model_accuracy=custom_metrics.get('model_accuracy', 0.0)
        )
        
        return extended_metrics
    
    def _collect_custom_metrics(self) -> Dict[str, float]:
        """Collect custom metrics."""
        metrics = {}
        
        # Network latency
        metrics['network_latency'] = self._measure_network_latency()
        
        # Disk usage
        metrics['disk_usage'] = self._measure_disk_usage()
        
        # Current model accuracy
        metrics['model_accuracy'] = self._get_current_model_accuracy()
        
        return metrics
    
    def _measure_network_latency(self) -> float:
        """Measure network latency to clients."""
        # Implementation here
        pass
    
    def _measure_disk_usage(self) -> float:
        """Measure disk usage percentage."""
        # Implementation here
        pass
    
    def _get_current_model_accuracy(self) -> float:
        """Get current global model accuracy."""
        # Implementation here
        pass
```

This completes the first part of the Developer Guide. The document continues with debugging, troubleshooting, and performance optimization sections.---


## Debugging and Troubleshooting

### Logging Configuration

Configure comprehensive logging for debugging:

```python
# config/logging_config.py
import logging
import logging.config
from datetime import datetime

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s", "function": "%(funcName)s", "line": %(lineno)d}'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'logs/qsfl_caad.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
        'security_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'WARNING',
            'formatter': 'json',
            'filename': 'logs/security.log',
            'maxBytes': 10485760,
            'backupCount': 10
        }
    },
    'loggers': {
        'qsfl_caad': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'qsfl_caad.security': {
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'security_file'],
            'propagate': False
        },
        'qsfl_caad.performance': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}

def setup_logging():
    """Setup logging configuration."""
    import os
    os.makedirs('logs', exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)
```

Use structured logging in your code:

```python
import logging

# Create module-specific loggers
logger = logging.getLogger('qsfl_caad.anomaly_detection')
security_logger = logging.getLogger('qsfl_caad.security')
perf_logger = logging.getLogger('qsfl_caad.performance')

class IsolationForestDetector:
    def __init__(self):
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
    
    def fit(self, normal_updates):
        self.logger.info(f"Training detector with {len(normal_updates)} normal updates")
        
        try:
            # Training logic
            self.logger.debug("Feature extraction completed")
            self.logger.debug("Model training started")
            # ... training code ...
            self.logger.info("Detector training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            raise
    
    def predict_anomaly_score(self, update):
        client_id = update.client_id
        self.logger.debug(f"Scoring update from {client_id}")
        
        start_time = time.time()
        try:
            score = self._compute_score(update)
            
            # Log performance metrics
            duration = time.time() - start_time
            perf_logger.info(f"Anomaly scoring took {duration:.3f}s for {client_id}")
            
            # Log security events
            if score > 0.8:
                security_logger.warning(
                    f"High anomaly score detected",
                    extra={
                        'client_id': client_id,
                        'anomaly_score': score,
                        'event_type': 'high_anomaly'
                    }
                )
            
            return score
            
        except Exception as e:
            self.logger.error(f"Scoring failed for {client_id}: {e}", exc_info=True)
            raise
```

### Common Issues and Solutions

#### 1. Post-Quantum Cryptography Issues

**Problem**: `pqcrypto` library installation fails

```bash
# Error message
ERROR: Failed building wheel for pqcrypto
```

**Solutions**:

```bash
# Option 1: Install build dependencies
sudo apt-get install build-essential python3-dev

# Option 2: Use conda-forge
conda install -c conda-forge pqcrypto

# Option 3: Use fallback implementation
# The system will automatically use fallback implementations
# if pqcrypto is not available
```

**Problem**: Cryptographic operations are slow

**Debug steps**:

```python
import time
from pq_security.kyber import KyberKeyExchange

def benchmark_crypto_operations():
    kyber = KyberKeyExchange()
    
    # Benchmark key generation
    start = time.time()
    for _ in range(100):
        kyber.generate_keypair()
    keygen_time = time.time() - start
    print(f"Key generation: {keygen_time/100:.4f}s per operation")
    
    # Benchmark encapsulation
    public_key, private_key = kyber.generate_keypair()
    start = time.time()
    for _ in range(100):
        kyber.encapsulate(public_key)
    encap_time = time.time() - start
    print(f"Encapsulation: {encap_time/100:.4f}s per operation")

# Run benchmark
benchmark_crypto_operations()
```

**Solutions**:
- Enable hardware acceleration if available
- Use caching for frequently used keys
- Consider using simulated implementations for development

#### 2. Anomaly Detection Issues

**Problem**: High false positive rate

**Debug approach**:

```python
def debug_anomaly_detection():
    detector = IsolationForestDetector()
    
    # Analyze training data distribution
    normal_updates = load_normal_updates()
    features = detector._extract_features(normal_updates)
    
    print(f"Feature statistics:")
    print(f"Mean: {np.mean(features, axis=0)}")
    print(f"Std: {np.std(features, axis=0)}")
    print(f"Min: {np.min(features, axis=0)}")
    print(f"Max: {np.max(features, axis=0)}")
    
    # Check for outliers in training data
    from scipy import stats
    z_scores = np.abs(stats.zscore(features))
    outliers = np.where(z_scores > 3)
    print(f"Potential outliers in training data: {len(outliers[0])}")
    
    # Analyze threshold sensitivity
    test_updates = load_test_updates()
    scores = [detector.predict_anomaly_score(u) for u in test_updates]
    
    import matplotlib.pyplot as plt
    plt.hist(scores, bins=50)
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Anomaly Scores')
    plt.show()
```

**Solutions**:
- Adjust contamination parameter in Isolation Forest
- Improve feature engineering
- Use more representative training data
- Implement adaptive thresholding

#### 3. Memory Issues

**Problem**: High memory usage during aggregation

**Debug with memory profiling**:

```python
# Install memory_profiler: pip install memory-profiler

from memory_profiler import profile
import psutil
import os

@profile
def debug_memory_usage():
    """Profile memory usage during aggregation."""
    
    # Monitor process memory
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Load large number of updates
    updates = []
    for i in range(1000):
        # Create large model updates
        weights = {
            f"layer_{j}": np.random.normal(0, 0.1, (1000, 1000))
            for j in range(10)
        }
        update = ModelUpdate(
            client_id=f"client_{i}",
            round_id="debug",
            weights=weights,
            signature=b"sig",
            timestamp=datetime.now(),
            metadata={}
        )
        updates.append(update)
        
        if i % 100 == 0:
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f"After {i} updates: {current_memory:.2f} MB")
    
    # Perform aggregation
    aggregator = ModelAggregator()
    result = aggregator.aggregate(updates)
    
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Memory increase: {final_memory - initial_memory:.2f} MB")

# Run with: python -m memory_profiler debug_script.py
```

**Solutions**:
- Implement batch processing for large numbers of updates
- Use memory-efficient data structures
- Clear intermediate variables explicitly
- Consider using memory mapping for large arrays

#### 4. Performance Issues

**Problem**: Slow federated learning rounds

**Performance profiling**:

```python
import cProfile
import pstats
from pstats import SortKey

def profile_federated_round():
    """Profile a complete federated learning round."""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run federated learning round
    server = SecureFederatedServer()
    # ... setup and run round ...
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)  # Top 20 functions
    
    # Save detailed report
    stats.dump_stats('profile_results.prof')

# Analyze with snakeviz: pip install snakeviz
# snakeviz profile_results.prof
```

**Line-by-line profiling**:

```python
# Install line_profiler: pip install line_profiler

@profile
def slow_function():
    """Function to profile line by line."""
    # Your code here
    pass

# Run with: kernprof -l -v script.py
```

### Debugging Tools and Techniques

#### 1. Interactive Debugging

Use `pdb` for interactive debugging:

```python
import pdb

def problematic_function(data):
    # Set breakpoint
    pdb.set_trace()
    
    # Your code here
    result = process_data(data)
    
    return result

# Or use breakpoint() in Python 3.7+
def another_function(data):
    breakpoint()  # Modern way to set breakpoint
    return process_data(data)
```

#### 2. Remote Debugging

For debugging in containers or remote environments:

```python
# Install debugpy: pip install debugpy

import debugpy

# Enable remote debugging
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()

# Your code here
```

#### 3. Assertion-Based Debugging

Use assertions to catch issues early:

```python
def aggregate_updates(self, updates, weights=None):
    """Aggregate model updates with validation."""
    
    # Input validation assertions
    assert isinstance(updates, list), "Updates must be a list"
    assert len(updates) > 0, "Cannot aggregate empty update list"
    assert all(isinstance(u, ModelUpdate) for u in updates), "All items must be ModelUpdate instances"
    
    if weights is not None:
        assert isinstance(weights, dict), "Weights must be a dictionary"
        assert len(weights) == len(updates), "Weights must match number of updates"
        assert all(w >= 0 for w in weights.values()), "All weights must be non-negative"
    
    # Process updates
    result = self._perform_aggregation(updates, weights)
    
    # Output validation
    assert isinstance(result, dict), "Result must be a dictionary"
    assert len(result) > 0, "Result cannot be empty"
    
    return result
```

#### 4. Custom Debug Decorators

Create reusable debugging decorators:

```python
import functools
import time
import logging

def debug_performance(func):
    """Decorator to log function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger = logging.getLogger(f'{func.__module__}.{func.__name__}')
        
        logger.debug(f"Starting {func.__name__} with args={len(args)}, kwargs={len(kwargs)}")
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"Completed {func.__name__} in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {duration:.3f}s: {e}")
            raise
    
    return wrapper

def debug_inputs_outputs(func):
    """Decorator to log function inputs and outputs."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(f'{func.__module__}.{func.__name__}')
        
        # Log inputs (be careful with sensitive data)
        logger.debug(f"Input args: {[type(arg).__name__ for arg in args]}")
        logger.debug(f"Input kwargs: {list(kwargs.keys())}")
        
        result = func(*args, **kwargs)
        
        # Log output type and basic info
        logger.debug(f"Output type: {type(result).__name__}")
        if hasattr(result, '__len__'):
            logger.debug(f"Output length: {len(result)}")
        
        return result
    
    return wrapper

# Usage
@debug_performance
@debug_inputs_outputs
def complex_function(data, threshold=0.5):
    # Your complex logic here
    pass
```

---

## Performance Optimization

### Profiling and Benchmarking

#### System-Wide Performance Monitoring

```python
# scripts/performance_monitor.py
import psutil
import time
import threading
from datetime import datetime
import matplotlib.pyplot as plt

class PerformanceMonitor:
    """Monitor system performance during QSFL-CAAD operations."""
    
    def __init__(self, interval=1.0):
        self.interval = interval
        self.monitoring = False
        self.data = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_percent': [],
            'memory_mb': [],
            'network_sent': [],
            'network_recv': []
        }
    
    def start_monitoring(self):
        """Start performance monitoring in background thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        initial_net = psutil.net_io_counters()
        
        while self.monitoring:
            timestamp = datetime.now()
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            net = psutil.net_io_counters()
            
            self.data['timestamps'].append(timestamp)
            self.data['cpu_percent'].append(cpu_percent)
            self.data['memory_percent'].append(memory.percent)
            self.data['memory_mb'].append(memory.used / 1024 / 1024)
            self.data['network_sent'].append(net.bytes_sent - initial_net.bytes_sent)
            self.data['network_recv'].append(net.bytes_recv - initial_net.bytes_recv)
            
            time.sleep(self.interval)
    
    def generate_report(self, save_path=None):
        """Generate performance report with plots."""
        if not self.data['timestamps']:
            print("No monitoring data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU usage
        axes[0, 0].plot(self.data['timestamps'], self.data['cpu_percent'])
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_ylabel('Percentage')
        
        # Memory usage
        axes[0, 1].plot(self.data['timestamps'], self.data['memory_mb'])
        axes[0, 1].set_title('Memory Usage (MB)')
        axes[0, 1].set_ylabel('MB')
        
        # Network sent
        axes[1, 0].plot(self.data['timestamps'], self.data['network_sent'])
        axes[1, 0].set_title('Network Bytes Sent')
        axes[1, 0].set_ylabel('Bytes')
        
        # Network received
        axes[1, 1].plot(self.data['timestamps'], self.data['network_recv'])
        axes[1, 1].set_title('Network Bytes Received')
        axes[1, 1].set_ylabel('Bytes')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        # Print summary statistics
        print(f"\nPerformance Summary:")
        print(f"Average CPU: {sum(self.data['cpu_percent'])/len(self.data['cpu_percent']):.2f}%")
        print(f"Peak CPU: {max(self.data['cpu_percent']):.2f}%")
        print(f"Average Memory: {sum(self.data['memory_mb'])/len(self.data['memory_mb']):.2f} MB")
        print(f"Peak Memory: {max(self.data['memory_mb']):.2f} MB")

# Usage example
def benchmark_federated_learning():
    """Benchmark federated learning performance."""
    monitor = PerformanceMonitor(interval=0.5)
    
    try:
        monitor.start_monitoring()
        
        # Run your federated learning code here
        server = SecureFederatedServer()
        # ... run multiple rounds ...
        
    finally:
        monitor.stop_monitoring()
        monitor.generate_report('performance_report.png')
```

#### Component-Specific Benchmarks

```python
# scripts/benchmark_components.py
import time
import numpy as np
from contextlib import contextmanager

@contextmanager
def timer(description):
    """Context manager for timing operations."""
    start = time.time()
    yield
    end = time.time()
    print(f"{description}: {end - start:.4f} seconds")

def benchmark_cryptographic_operations():
    """Benchmark post-quantum cryptographic operations."""
    from pq_security.kyber import KyberKeyExchange
    from pq_security.dilithium import DilithiumSigner
    
    kyber = KyberKeyExchange()
    dilithium = DilithiumSigner()
    
    print("=== Cryptographic Operations Benchmark ===")
    
    # Kyber benchmarks
    with timer("Kyber key generation (100 ops)"):
        for _ in range(100):
            kyber.generate_keypair()
    
    public_key, private_key = kyber.generate_keypair()
    
    with timer("Kyber encapsulation (100 ops)"):
        for _ in range(100):
            kyber.encapsulate(public_key)
    
    ciphertext, _ = kyber.encapsulate(public_key)
    
    with timer("Kyber decapsulation (100 ops)"):
        for _ in range(100):
            kyber.decapsulate(ciphertext, private_key)
    
    # Dilithium benchmarks
    with timer("Dilithium key generation (100 ops)"):
        for _ in range(100):
            dilithium.generate_keypair()
    
    public_key, private_key = dilithium.generate_keypair()
    message = b"test message for signing"
    
    with timer("Dilithium signing (100 ops)"):
        for _ in range(100):
            dilithium.sign(message, private_key)
    
    signature = dilithium.sign(message, private_key)
    
    with timer("Dilithium verification (100 ops)"):
        for _ in range(100):
            dilithium.verify(message, signature, public_key)

def benchmark_anomaly_detection():
    """Benchmark anomaly detection performance."""
    from anomaly_detection.isolation_forest_detector import IsolationForestDetector
    from anomaly_detection.interfaces import ModelUpdate
    from datetime import datetime
    
    print("\n=== Anomaly Detection Benchmark ===")
    
    # Generate test data
    def create_test_updates(num_updates, weights_shape=(100, 50)):
        updates = []
        for i in range(num_updates):
            weights = {
                "layer1": np.random.normal(0, 0.1, weights_shape),
                "layer2": np.random.normal(0, 0.1, (weights_shape[1], 10))
            }
            update = ModelUpdate(
                client_id=f"client_{i}",
                round_id="benchmark",
                weights=weights,
                signature=b"sig",
                timestamp=datetime.now(),
                metadata={}
            )
            updates.append(update)
        return updates
    
    detector = IsolationForestDetector()
    
    # Training benchmark
    training_sizes = [100, 500, 1000, 2000]
    for size in training_sizes:
        training_data = create_test_updates(size)
        with timer(f"Training with {size} updates"):
            detector.fit(training_data)
    
    # Prediction benchmark
    test_data = create_test_updates(100)
    with timer("Prediction on 100 updates"):
        for update in test_data:
            detector.predict_anomaly_score(update)

def benchmark_model_aggregation():
    """Benchmark model aggregation performance."""
    from federated_learning.model_aggregator import ModelAggregator
    from anomaly_detection.interfaces import ModelUpdate
    from datetime import datetime
    
    print("\n=== Model Aggregation Benchmark ===")
    
    aggregator = ModelAggregator()
    
    # Test different numbers of clients and model sizes
    client_counts = [10, 50, 100, 200]
    model_sizes = [(100, 50), (500, 200), (1000, 500)]
    
    for num_clients in client_counts:
        for model_size in model_sizes:
            # Create updates
            updates = []
            for i in range(num_clients):
                weights = {
                    "layer1": np.random.normal(0, 0.1, model_size),
                    "layer2": np.random.normal(0, 0.1, (model_size[1], 10))
                }
                update = ModelUpdate(
                    client_id=f"client_{i}",
                    round_id="benchmark",
                    weights=weights,
                    signature=b"sig",
                    timestamp=datetime.now(),
                    metadata={}
                )
                updates.append(update)
            
            with timer(f"Aggregation: {num_clients} clients, {model_size} weights"):
                aggregator.aggregate(updates)

if __name__ == "__main__":
    benchmark_cryptographic_operations()
    benchmark_anomaly_detection()
    benchmark_model_aggregation()
```

### Optimization Strategies

#### 1. Caching and Memoization

```python
from functools import lru_cache
import hashlib
import pickle

class CachedAnomalyDetector:
    """Anomaly detector with feature caching."""
    
    def __init__(self):
        self.detector = IsolationForestDetector()
        self._feature_cache = {}
    
    def _get_update_hash(self, update: ModelUpdate) -> str:
        """Generate hash for model update."""
        # Create deterministic hash of weights
        weights_bytes = pickle.dumps(update.weights, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(weights_bytes).hexdigest()
    
    @lru_cache(maxsize=1000)
    def _extract_features_cached(self, update_hash: str, weights_pickle: bytes) -> np.ndarray:
        """Extract features with caching."""
        weights = pickle.loads(weights_pickle)
        return self._extract_features_impl(weights)
    
    def predict_anomaly_score(self, update: ModelUpdate) -> float:
        """Predict with feature caching."""
        update_hash = self._get_update_hash(update)
        weights_pickle = pickle.dumps(update.weights, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Use cached feature extraction
        features = self._extract_features_cached(update_hash, weights_pickle)
        
        return self.detector.predict_anomaly_score_from_features(features)
```

#### 2. Batch Processing

```python
class BatchModelAggregator:
    """Model aggregator with batch processing optimization."""
    
    def __init__(self, batch_size=50):
        self.batch_size = batch_size
    
    def aggregate_large_scale(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate large number of updates in batches."""
        
        if len(updates) <= self.batch_size:
            return self._aggregate_batch(updates)
        
        # Process in batches
        batch_results = []
        for i in range(0, len(updates), self.batch_size):
            batch = updates[i:i + self.batch_size]
            batch_result = self._aggregate_batch(batch)
            batch_results.append(batch_result)
        
        # Aggregate batch results
        return self._aggregate_batch_results(batch_results)
    
    def _aggregate_batch(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate a single batch of updates."""
        # Standard aggregation logic
        pass
    
    def _aggregate_batch_results(self, batch_results: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Aggregate results from multiple batches."""
        # Combine batch results
        pass
```

#### 3. Parallel Processing

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

class ParallelAnomalyDetector:
    """Anomaly detector with parallel processing."""
    
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or mp.cpu_count()
        self.detector = IsolationForestDetector()
    
    def predict_batch_parallel(self, updates: List[ModelUpdate]) -> List[float]:
        """Predict anomaly scores in parallel."""
        
        # Split updates into chunks
        chunk_size = max(1, len(updates) // self.n_workers)
        chunks = [updates[i:i + chunk_size] for i in range(0, len(updates), chunk_size)]
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self._process_chunk, chunk) for chunk in chunks]
            results = [future.result() for future in futures]
        
        # Flatten results
        return [score for chunk_scores in results for score in chunk_scores]
    
    def _process_chunk(self, chunk: List[ModelUpdate]) -> List[float]:
        """Process a chunk of updates."""
        return [self.detector.predict_anomaly_score(update) for update in chunk]

class ParallelModelAggregator:
    """Model aggregator with parallel layer processing."""
    
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or mp.cpu_count()
    
    def aggregate_parallel(self, updates: List[ModelUpdate]) -> Dict[str, np.ndarray]:
        """Aggregate model updates with parallel layer processing."""
        
        # Group weights by layer
        layer_groups = self._group_weights_by_layer(updates)
        
        # Process layers in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                layer_name: executor.submit(self._aggregate_layer, layer_weights)
                for layer_name, layer_weights in layer_groups.items()
            }
            
            results = {
                layer_name: future.result()
                for layer_name, future in futures.items()
            }
        
        return results
    
    def _group_weights_by_layer(self, updates: List[ModelUpdate]) -> Dict[str, List[np.ndarray]]:
        """Group weights by layer name."""
        layer_groups = {}
        
        for update in updates:
            for layer_name, weights in update.weights.items():
                if layer_name not in layer_groups:
                    layer_groups[layer_name] = []
                layer_groups[layer_name].append(weights)
        
        return layer_groups
    
    def _aggregate_layer(self, layer_weights: List[np.ndarray]) -> np.ndarray:
        """Aggregate weights for a single layer."""
        return np.mean(layer_weights, axis=0)
```

#### 4. Memory Optimization

```python
import gc
from typing import Iterator

class MemoryEfficientProcessor:
    """Memory-efficient processing for large-scale operations."""
    
    def __init__(self):
        self.chunk_size = 100  # Process in chunks to manage memory
    
    def process_updates_streaming(self, updates: Iterator[ModelUpdate]) -> Iterator[float]:
        """Process updates in streaming fashion to minimize memory usage."""
        
        chunk = []
        for update in updates:
            chunk.append(update)
            
            if len(chunk) >= self.chunk_size:
                # Process chunk
                scores = self._process_chunk(chunk)
                
                # Yield results
                for score in scores:
                    yield score
                
                # Clear chunk and force garbage collection
                chunk.clear()
                gc.collect()
        
        # Process remaining updates
        if chunk:
            scores = self._process_chunk(chunk)
            for score in scores:
                yield score
    
    def _process_chunk(self, chunk: List[ModelUpdate]) -> List[float]:
        """Process a chunk of updates."""
        # Your processing logic here
        pass
    
    @staticmethod
    def optimize_numpy_arrays(arrays: List[np.ndarray]) -> List[np.ndarray]:
        """Optimize numpy arrays for memory usage."""
        optimized = []
        
        for arr in arrays:
            # Convert to most efficient dtype
            if arr.dtype == np.float64:
                # Check if we can use float32 without significant precision loss
                if np.allclose(arr, arr.astype(np.float32)):
                    arr = arr.astype(np.float32)
            
            # Use memory mapping for very large arrays
            if arr.nbytes > 100 * 1024 * 1024:  # 100MB
                # Save to temporary file and memory map
                temp_file = f"/tmp/array_{id(arr)}.npy"
                np.save(temp_file, arr)
                arr = np.load(temp_file, mmap_mode='r')
            
            optimized.append(arr)
        
        return optimized
```

### Configuration Tuning

Create performance-optimized configurations:

```yaml
# config/performance_config.yaml
performance:
  # Cryptographic operations
  crypto:
    enable_hardware_acceleration: true
    key_cache_size: 1000
    signature_batch_size: 50
  
  # Anomaly detection
  anomaly_detection:
    feature_cache_size: 5000
    batch_prediction_size: 100
    parallel_workers: 4
    contamination: 0.1
    n_estimators: 100
  
  # Model aggregation
  aggregation:
    batch_size: 50
    parallel_layers: true
    memory_limit_mb: 1000
    compression_enabled: true
  
  # System resources
  system:
    max_memory_usage_percent: 80
    gc_frequency: 100  # Force GC every N operations
    thread_pool_size: 8
    process_pool_size: 4

# Load and apply performance config
def apply_performance_config(config):
    """Apply performance optimizations based on config."""
    
    # Set numpy thread count
    import os
    os.environ['OMP_NUM_THREADS'] = str(config['system']['thread_pool_size'])
    
    # Configure garbage collection
    import gc
    gc.set_threshold(
        config['system']['gc_frequency'],
        config['system']['gc_frequency'] * 10,
        config['system']['gc_frequency'] * 100
    )
    
    # Set memory limits if available
    try:
        import resource
        memory_limit = config['system']['max_memory_usage_percent'] * 1024 * 1024 * 1024 // 100
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    except ImportError:
        pass  # resource module not available on Windows
```

This completes the comprehensive Developer Guide for QSFL-CAAD, covering all aspects from setup to advanced optimization techniques.