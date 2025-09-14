# QSFL-CAAD API Documentation

## Overview

QSFL-CAAD (Quantum-Safe Federated Learning with Client Anomaly and Attack Detection) provides a comprehensive API for building secure federated learning systems with post-quantum cryptography and AI-driven anomaly detection.

## Table of Contents

1. [Post-Quantum Security Module](#post-quantum-security-module)
2. [Authentication Module](#authentication-module)
3. [Anomaly Detection Module](#anomaly-detection-module)
4. [Federated Learning Module](#federated-learning-module)
5. [Monitoring Module](#monitoring-module)
6. [Usage Examples](#usage-examples)
7. [Integration Guide](#integration-guide)

---

## Post-Quantum Security Module

The `pq_security` module provides quantum-resistant cryptographic operations using CRYSTALS-Kyber and CRYSTALS-Dilithium algorithms.

### Core Interfaces

#### IPQCrypto

Base interface for post-quantum cryptographic operations.

```python
from pq_security.interfaces import IPQCrypto

class IPQCrypto(ABC):
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate a public/private key pair."""
        
    def encrypt(self, plaintext: bytes, public_key: bytes) -> bytes:
        """Encrypt plaintext using public key."""
        
    def decrypt(self, ciphertext: bytes, private_key: bytes) -> bytes:
        """Decrypt ciphertext using private key."""
        
    def sign(self, message: bytes, private_key: bytes) -> bytes:
        """Sign a message using private key."""
        
    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify a signature using public key."""
```

### Implementation Classes

#### KyberKeyExchange

Implements CRYSTALS-Kyber key exchange mechanism.

```python
from pq_security.kyber import KyberKeyExchange

# Initialize Kyber key exchange
kyber = KyberKeyExchange()

# Generate keypair
public_key, private_key = kyber.generate_keypair()

# Encapsulate shared secret
ciphertext, shared_secret = kyber.encapsulate(public_key)

# Decapsulate shared secret
recovered_secret = kyber.decapsulate(ciphertext, private_key)
```

**Methods:**
- `generate_keypair()` → `Tuple[bytes, bytes]`: Generate Kyber keypair
- `encapsulate(public_key: bytes)` → `Tuple[bytes, bytes]`: Encapsulate shared secret
- `decapsulate(ciphertext: bytes, private_key: bytes)` → `bytes`: Decapsulate shared secret

#### DilithiumSigner

Implements CRYSTALS-Dilithium digital signatures.

```python
from pq_security.dilithium import DilithiumSigner

# Initialize Dilithium signer
signer = DilithiumSigner()

# Generate keypair
public_key, private_key = signer.generate_keypair()

# Sign message
message = b"Hello, quantum-safe world!"
signature = signer.sign(message, private_key)

# Verify signature
is_valid = signer.verify(message, signature, public_key)
```

**Methods:**
- `generate_keypair()` → `Tuple[bytes, bytes]`: Generate Dilithium keypair
- `sign(message: bytes, private_key: bytes)` → `bytes`: Sign message
- `verify(message: bytes, signature: bytes, public_key: bytes)` → `bool`: Verify signature

#### PQCryptoManager

Unified manager for all post-quantum cryptographic operations.

```python
from pq_security.manager import PQCryptoManager

# Initialize manager
pq_manager = PQCryptoManager()

# Generate client credentials
client_id = "client_001"
credentials = pq_manager.generate_client_credentials(client_id)

# Establish secure session
session_key = pq_manager.establish_session(client_public_key)

# Sign and verify model updates
update_data = b"model_weights_data"
signature = pq_manager.sign_update(update_data, client_private_key)
is_valid = pq_manager.verify_update(update_data, signature, client_public_key)
```

---

## Authentication Module

The `auth` module handles client authentication, credential management, and revocation.

### Core Data Structures

#### ClientCredentials

```python
@dataclass
class ClientCredentials:
    client_id: str
    public_key: bytes
    private_key: bytes
    issued_at: datetime
    expires_at: datetime
    status: CredentialStatus
```

#### CredentialStatus

```python
class CredentialStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"
```

### Implementation Classes

#### AuthenticationService

Main service for client authentication operations.

```python
from auth.authentication_service import AuthenticationService

# Initialize service
auth_service = AuthenticationService()

# Register new client
credentials = auth_service.register_client("client_001")

# Authenticate client
message = b"model_update_data"
signature = sign_message(message, credentials.private_key)
is_authenticated = auth_service.authenticate_client(
    "client_001", signature, message
)

# Check client validity
is_valid = auth_service.is_client_valid("client_001")
```

**Methods:**
- `register_client(client_id: str)` → `ClientCredentials`: Register new client
- `authenticate_client(client_id: str, signature: bytes, message: bytes)` → `bool`: Authenticate client
- `revoke_client(client_id: str)` → `None`: Revoke client credentials
- `is_client_valid(client_id: str)` → `bool`: Check client validity

#### CredentialManager

Manages credential lifecycle and storage.

```python
from auth.credential_manager import CredentialManager

# Initialize manager
cred_manager = CredentialManager()

# Issue new credentials
credentials = cred_manager.issue_credentials("client_001")

# Renew existing credentials
renewed_creds = cred_manager.renew_credentials("client_001")

# Store credentials securely
cred_manager.store_credentials(credentials)

# Load credentials
loaded_creds = cred_manager.load_credentials("client_001")
```

#### RevocationManager

Handles credential revocation and blacklisting.

```python
from auth.revocation_manager import RevocationManager

# Initialize manager
revocation_manager = RevocationManager()

# Revoke client
revocation_manager.revoke_credential("client_001", "suspicious_behavior")

# Check revocation status
is_revoked = revocation_manager.is_revoked("client_001")

# Get revocation list
revoked_clients = revocation_manager.get_revocation_list()
```

---

## Anomaly Detection Module

The `anomaly_detection` module provides AI-driven detection of malicious client behavior.

### Core Data Structures

#### ModelUpdate

```python
@dataclass
class ModelUpdate:
    client_id: str
    round_id: str
    weights: Dict[str, np.ndarray]
    signature: bytes
    timestamp: datetime
    metadata: Dict[str, Any]
```

#### AnomalyReport

```python
@dataclass
class AnomalyReport:
    client_id: str
    anomaly_score: float
    shap_values: Dict[str, float]
    explanation: str
    recommended_action: ResponseAction
    timestamp: datetime
```

### Implementation Classes

#### IsolationForestDetector

Implements anomaly detection using Isolation Forest algorithm.

```python
from anomaly_detection.isolation_forest_detector import IsolationForestDetector

# Initialize detector
detector = IsolationForestDetector()

# Train on normal updates
normal_updates = load_normal_updates()
detector.fit(normal_updates)

# Score new update
new_update = receive_client_update()
anomaly_score = detector.predict_anomaly_score(new_update)

# Generate explanation
explanation = detector.explain_anomaly(new_update)
```

**Methods:**
- `fit(normal_updates: List[ModelUpdate])` → `None`: Train detector
- `predict_anomaly_score(update: ModelUpdate)` → `float`: Score update
- `explain_anomaly(update: ModelUpdate)` → `Dict[str, float]`: Generate explanation

#### SHAPExplainer

Provides interpretable explanations for anomaly scores.

```python
from anomaly_detection.shap_explainer import SHAPExplainer

# Initialize explainer
explainer = SHAPExplainer(detector_model)

# Generate explanation
explanation = explainer.explain(model_update, anomaly_score)

# Get feature importance
feature_importance = explanation["feature_importance"]
explanation_text = explanation["explanation"]
```

#### ClientReputationManager

Manages client reputation scores and influence weights.

```python
from anomaly_detection.reputation_manager import ClientReputationManager

# Initialize manager
reputation_manager = ClientReputationManager()

# Update reputation based on anomaly score
reputation_manager.update_reputation("client_001", anomaly_score=0.8)

# Get current reputation
reputation = reputation_manager.get_reputation("client_001")

# Get influence weight for aggregation
weight = reputation_manager.get_influence_weight("client_001")

# Check quarantine status
is_quarantined = reputation_manager.is_quarantined("client_001")
```

---

## Federated Learning Module

The `federated_learning` module orchestrates secure federated learning rounds.

### Core Data Structures

#### GlobalModel

```python
@dataclass
class GlobalModel:
    model_id: str
    round_id: str
    weights: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    created_at: datetime
```

#### TrainingRound

```python
@dataclass
class TrainingRound:
    round_id: str
    participants: List[str]
    global_model_hash: str
    aggregation_method: str
    security_events: List[str]
    metrics: Dict[str, float]
    started_at: datetime
    completed_at: Optional[datetime] = None
```

### Implementation Classes

#### SecureFederatedServer

Main server coordinating federated learning with security.

```python
from federated_learning.server import SecureFederatedServer

# Initialize server
server = SecureFederatedServer()

# Start training round
round_id = server.start_training_round()

# Receive client update
update_accepted = server.receive_client_update("client_001", model_update)

# Aggregate updates
global_model = server.aggregate_updates(round_id)

# Distribute global model
server.distribute_global_model(global_model)
```

#### ModelAggregator

Aggregates client updates with security considerations.

```python
from federated_learning.model_aggregator import ModelAggregator

# Initialize aggregator
aggregator = ModelAggregator()

# Set aggregation method
aggregator.set_aggregation_method("federated_averaging")

# Aggregate updates with reputation weights
client_weights = {"client_001": 0.8, "client_002": 1.0, "client_003": 0.3}
aggregated_weights = aggregator.aggregate(client_updates, client_weights)
```

---

## Monitoring Module

The `monitoring` module provides comprehensive system monitoring and alerting.

### Core Data Structures

#### SecurityEvent

```python
@dataclass
class SecurityEvent:
    event_id: str
    event_type: EventType
    severity: EventSeverity
    client_id: Optional[str]
    description: str
    metadata: Dict[str, Any]
    timestamp: datetime
```

#### SystemMetrics

```python
@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    active_clients: int
    training_rounds_completed: int
    anomalies_detected: int
    authentication_failures: int
```

### Implementation Classes

#### SecurityEventLogger

Logs all security-related events.

```python
from monitoring.security_logger import SecurityEventLogger

# Initialize logger
logger = SecurityEventLogger()

# Log authentication event
logger.log_authentication_event("client_001", success=True, details="Valid signature")

# Log anomaly event
logger.log_anomaly_event("client_002", anomaly_score=0.9, action="quarantine")

# Retrieve events
events = logger.get_events(start_time, end_time, EventType.ANOMALY_DETECTION)
```

#### MetricsCollector

Collects system performance and security metrics.

```python
from monitoring.metrics_collector import MetricsCollector

# Initialize collector
collector = MetricsCollector()

# Collect current metrics
metrics = collector.collect_metrics()

# Record custom metric
collector.record_metric("model_accuracy", 0.95)

# Get metric history
accuracy_history = collector.get_metric_history("model_accuracy", start_time, end_time)
```

#### AlertManager

Manages alerts and notifications.

```python
from monitoring.alert_manager import AlertManager

# Initialize manager
alert_manager = AlertManager()

# Create alert
alert_id = alert_manager.create_alert(
    "High Anomaly Rate", 
    "Anomaly detection rate exceeded threshold", 
    EventSeverity.HIGH
)

# Configure threshold
alert_manager.configure_threshold("anomaly_rate", 0.1, EventSeverity.MEDIUM)

# Get active alerts
active_alerts = alert_manager.get_active_alerts()
```

---

## Usage Examples

### Basic Federated Learning Setup

```python
from qsfl_caad import QSFLServer, QSFLClient

# Initialize server
server = QSFLServer()

# Register clients
client_ids = ["client_001", "client_002", "client_003"]
for client_id in client_ids:
    credentials = server.register_client(client_id)
    
# Start federated learning
for round_num in range(10):
    round_id = server.start_training_round()
    
    # Clients train and submit updates
    for client_id in client_ids:
        client = QSFLClient(client_id, credentials[client_id])
        update = client.train_local_model()
        server.receive_client_update(client_id, update)
    
    # Aggregate and distribute
    global_model = server.aggregate_updates(round_id)
    server.distribute_global_model(global_model)
```

### Anomaly Detection Integration

```python
from anomaly_detection import IsolationForestDetector, ClientReputationManager

# Initialize components
detector = IsolationForestDetector()
reputation_manager = ClientReputationManager()

# Train detector with normal updates
normal_updates = load_baseline_updates()
detector.fit(normal_updates)

# Process incoming update
def process_client_update(client_id, update):
    # Score for anomalies
    anomaly_score = detector.predict_anomaly_score(update)
    
    # Update reputation
    reputation_manager.update_reputation(client_id, anomaly_score)
    
    # Get influence weight
    weight = reputation_manager.get_influence_weight(client_id)
    
    # Check if quarantined
    if reputation_manager.is_quarantined(client_id):
        return False, "Client quarantined"
    
    return True, weight
```

### Security Event Monitoring

```python
from monitoring import SecurityEventLogger, AlertManager

# Initialize monitoring
logger = SecurityEventLogger()
alert_manager = AlertManager()

# Set up alert thresholds
alert_manager.configure_threshold("authentication_failures", 5, EventSeverity.HIGH)
alert_manager.configure_threshold("anomaly_rate", 0.2, EventSeverity.MEDIUM)

# Monitor authentication
def monitor_authentication(client_id, success, details):
    logger.log_authentication_event(client_id, success, details)
    
    if not success:
        # Check for repeated failures
        recent_failures = count_recent_failures(client_id)
        if recent_failures > 3:
            alert_manager.create_alert(
                f"Repeated auth failures for {client_id}",
                f"Client {client_id} has {recent_failures} recent failures",
                EventSeverity.HIGH
            )
```

---

## Integration Guide

### Setting Up QSFL-CAAD

1. **Installation**
```bash
pip install -r requirements.txt
python setup.py install
```

2. **Configuration**
```python
from config.settings import load_config

# Load configuration
config = load_config("config/config.yaml")

# Initialize system components
from qsfl_caad import initialize_system
system = initialize_system(config)
```

3. **Database Setup**
```python
from qsfl_caad.database import setup_database

# Initialize database
setup_database("sqlite:///qsfl_caad.db")
```

### Integration with Existing FL Frameworks

#### TensorFlow Federated

```python
import tensorflow_federated as tff
from qsfl_caad.integrations.tff import QSFLTFFWrapper

# Wrap TFF computation with QSFL-CAAD security
@tff.federated_computation
def secure_federated_averaging(model_fn, client_data):
    wrapper = QSFLTFFWrapper()
    return wrapper.secure_federated_averaging(model_fn, client_data)
```

#### PySyft

```python
import syft as sy
from qsfl_caad.integrations.syft import QSFLSyftHook

# Add QSFL-CAAD security to PySyft
hook = QSFLSyftHook()
sy.hook = hook

# Use secure federated learning
secure_model = sy.SecureModel(model, security_level="quantum_safe")
```

### Custom Integration

```python
from qsfl_caad.core import SecurityLayer

class CustomFLFramework:
    def __init__(self):
        self.security = SecurityLayer()
    
    def process_client_update(self, client_id, update_data):
        # Authenticate client
        if not self.security.authenticate_client(client_id, update_data):
            return False
        
        # Check for anomalies
        anomaly_score = self.security.detect_anomalies(update_data)
        
        # Apply security policies
        if anomaly_score > 0.8:
            self.security.quarantine_client(client_id)
            return False
        
        return True
```

### Error Handling Best Practices

```python
from qsfl_caad.exceptions import (
    AuthenticationError, 
    CryptographicError, 
    AnomalyDetectionError
)

try:
    # Perform secure operation
    result = server.process_client_update(client_id, update)
except AuthenticationError as e:
    logger.error(f"Authentication failed for {client_id}: {e}")
    # Handle authentication failure
except CryptographicError as e:
    logger.error(f"Cryptographic operation failed: {e}")
    # Handle crypto failure
except AnomalyDetectionError as e:
    logger.warning(f"Anomaly detection issue: {e}")
    # Handle with fallback detection
```

### Performance Optimization

```python
# Configure for high-performance scenarios
config = {
    "anomaly_detection": {
        "batch_size": 100,
        "parallel_processing": True,
        "cache_features": True
    },
    "cryptography": {
        "hardware_acceleration": True,
        "key_caching": True
    },
    "aggregation": {
        "async_processing": True,
        "compression": True
    }
}

system = initialize_system(config)
```

This API documentation provides comprehensive coverage of all public interfaces and implementation classes in the QSFL-CAAD system, along with practical usage examples and integration guidance.