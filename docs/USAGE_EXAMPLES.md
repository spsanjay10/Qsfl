# QSFL-CAAD Usage Examples

This document provides practical examples for using the QSFL-CAAD system in various scenarios.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Post-Quantum Security Examples](#post-quantum-security-examples)
3. [Authentication Examples](#authentication-examples)
4. [Anomaly Detection Examples](#anomaly-detection-examples)
5. [Federated Learning Examples](#federated-learning-examples)
6. [Monitoring Examples](#monitoring-examples)
7. [Complete Integration Examples](#complete-integration-examples)

---

## Quick Start

### Basic Server Setup

```python
#!/usr/bin/env python3
"""
Basic QSFL-CAAD server setup example.
"""

from qsfl_caad import QSFLServer
from config.settings import load_config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load configuration
    config = load_config("config/config.yaml")
    
    # Initialize server
    server = QSFLServer(config)
    
    # Register some clients
    client_ids = ["honest_client_1", "honest_client_2", "malicious_client"]
    credentials = {}
    
    for client_id in client_ids:
        creds = server.register_client(client_id)
        credentials[client_id] = creds
        logger.info(f"Registered client: {client_id}")
    
    # Start federated learning
    logger.info("Starting federated learning...")
    
    for round_num in range(5):
        logger.info(f"Starting round {round_num + 1}")
        
        # Start training round
        round_id = server.start_training_round()
        
        # Simulate client updates (in real scenario, clients would submit these)
        for client_id in client_ids:
            # Create mock update (replace with actual client training)
            update = create_mock_update(client_id, round_id, credentials[client_id])
            
            # Submit update
            accepted = server.receive_client_update(client_id, update)
            logger.info(f"Update from {client_id}: {'accepted' if accepted else 'rejected'}")
        
        # Aggregate updates
        global_model = server.aggregate_updates(round_id)
        logger.info(f"Round {round_num + 1} completed. Model accuracy: {global_model.metadata.get('accuracy', 'N/A')}")
        
        # Distribute global model
        server.distribute_global_model(global_model)

def create_mock_update(client_id, round_id, credentials):
    """Create a mock model update for demonstration."""
    import numpy as np
    from datetime import datetime
    from anomaly_detection.interfaces import ModelUpdate
    
    # Create mock weights
    weights = {
        "layer1": np.random.normal(0, 0.1, (10, 5)),
        "layer2": np.random.normal(0, 0.1, (5, 1))
    }
    
    # Add some malicious behavior for the malicious client
    if "malicious" in client_id:
        # Inject poisoned gradients
        weights["layer1"] *= 10  # Amplify gradients
        weights["layer2"] += np.random.normal(0, 1, weights["layer2"].shape)
    
    # Create update
    update = ModelUpdate(
        client_id=client_id,
        round_id=round_id,
        weights=weights,
        signature=b"mock_signature",  # In real scenario, use actual signature
        timestamp=datetime.now(),
        metadata={"local_epochs": 5, "batch_size": 32}
    )
    
    return update

if __name__ == "__main__":
    main()
```

### Basic Client Setup

```python
#!/usr/bin/env python3
"""
Basic QSFL-CAAD client example.
"""

import numpy as np
import tensorflow as tf
from qsfl_caad import QSFLClient
from datetime import datetime

class SimpleClient:
    def __init__(self, client_id, credentials, dataset):
        self.client_id = client_id
        self.credentials = credentials
        self.dataset = dataset
        self.model = self._create_model()
    
    def _create_model(self):
        """Create a simple neural network model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(784,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_local_model(self, global_weights=None):
        """Train the local model and return update."""
        if global_weights:
            self.model.set_weights(global_weights)
        
        # Train on local data
        history = self.model.fit(
            self.dataset['x'], self.dataset['y'],
            epochs=5, batch_size=32, verbose=0
        )
        
        # Create model update
        from anomaly_detection.interfaces import ModelUpdate
        
        update = ModelUpdate(
            client_id=self.client_id,
            round_id="current_round",
            weights={f"layer_{i}": w for i, w in enumerate(self.model.get_weights())},
            signature=self._sign_update(),
            timestamp=datetime.now(),
            metadata={
                "accuracy": history.history['accuracy'][-1],
                "loss": history.history['loss'][-1],
                "epochs": 5
            }
        )
        
        return update
    
    def _sign_update(self):
        """Sign the model update (simplified for example)."""
        # In real implementation, use proper cryptographic signing
        return f"signature_{self.client_id}".encode()

# Example usage
def main():
    # Load MNIST data (simplified)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    
    # Create client dataset (simulate data partition)
    client_data = {
        'x': x_train[:1000],  # First 1000 samples
        'y': y_train[:1000]
    }
    
    # Mock credentials (in real scenario, received from server)
    credentials = {
        'client_id': 'client_001',
        'public_key': b'mock_public_key',
        'private_key': b'mock_private_key'
    }
    
    # Create client
    client = SimpleClient('client_001', credentials, client_data)
    
    # Train and get update
    update = client.train_local_model()
    print(f"Client {client.client_id} completed training")
    print(f"Local accuracy: {update.metadata['accuracy']:.4f}")

if __name__ == "__main__":
    main()
```

---

## Post-Quantum Security Examples

### Key Exchange Example

```python
#!/usr/bin/env python3
"""
Example of CRYSTALS-Kyber key exchange.
"""

from pq_security.kyber import KyberKeyExchange
import os

def demonstrate_key_exchange():
    """Demonstrate secure key exchange between client and server."""
    
    # Initialize Kyber for both parties
    client_kyber = KyberKeyExchange()
    server_kyber = KyberKeyExchange()
    
    print("=== CRYSTALS-Kyber Key Exchange Demo ===")
    
    # Step 1: Server generates keypair
    server_public_key, server_private_key = server_kyber.generate_keypair()
    print(f"Server generated keypair")
    print(f"Public key length: {len(server_public_key)} bytes")
    
    # Step 2: Client encapsulates shared secret with server's public key
    ciphertext, client_shared_secret = client_kyber.encapsulate(server_public_key)
    print(f"Client encapsulated shared secret")
    print(f"Ciphertext length: {len(ciphertext)} bytes")
    print(f"Client shared secret: {client_shared_secret[:16].hex()}...")
    
    # Step 3: Server decapsulates shared secret
    server_shared_secret = server_kyber.decapsulate(ciphertext, server_private_key)
    print(f"Server decapsulated shared secret")
    print(f"Server shared secret: {server_shared_secret[:16].hex()}...")
    
    # Verify shared secrets match
    if client_shared_secret == server_shared_secret:
        print("✅ Key exchange successful! Shared secrets match.")
    else:
        print("❌ Key exchange failed! Shared secrets don't match.")
    
    return client_shared_secret == server_shared_secret

if __name__ == "__main__":
    demonstrate_key_exchange()
```

### Digital Signature Example

```python
#!/usr/bin/env python3
"""
Example of CRYSTALS-Dilithium digital signatures.
"""

from pq_security.dilithium import DilithiumSigner
import json

def demonstrate_digital_signatures():
    """Demonstrate digital signatures for model updates."""
    
    # Initialize Dilithium signer
    signer = DilithiumSigner()
    
    print("=== CRYSTALS-Dilithium Digital Signatures Demo ===")
    
    # Generate keypair for client
    public_key, private_key = signer.generate_keypair()
    print(f"Generated Dilithium keypair")
    print(f"Public key length: {len(public_key)} bytes")
    print(f"Private key length: {len(private_key)} bytes")
    
    # Create a model update message
    model_update = {
        "client_id": "client_001",
        "round_id": "round_5",
        "weights_hash": "abc123def456",
        "timestamp": "2024-01-15T10:30:00Z"
    }
    
    message = json.dumps(model_update, sort_keys=True).encode()
    print(f"Message to sign: {message.decode()}")
    
    # Sign the message
    signature = signer.sign(message, private_key)
    print(f"Generated signature length: {len(signature)} bytes")
    
    # Verify the signature
    is_valid = signer.verify(message, signature, public_key)
    print(f"Signature verification: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Test with tampered message
    tampered_message = message.replace(b"client_001", b"client_002")
    is_tampered_valid = signer.verify(tampered_message, signature, public_key)
    print(f"Tampered message verification: {'❌ Should be invalid' if is_tampered_valid else '✅ Correctly invalid'}")
    
    return is_valid and not is_tampered_valid

if __name__ == "__main__":
    demonstrate_digital_signatures()
```

---

## Authentication Examples

### Client Registration and Authentication

```python
#!/usr/bin/env python3
"""
Example of client authentication workflow.
"""

from auth.authentication_service import AuthenticationService
from auth.credential_manager import CredentialManager
from auth.revocation_manager import RevocationManager
import json
from datetime import datetime

def demonstrate_authentication():
    """Demonstrate complete authentication workflow."""
    
    print("=== Client Authentication Demo ===")
    
    # Initialize services
    auth_service = AuthenticationService()
    cred_manager = CredentialManager()
    revocation_manager = RevocationManager()
    
    # Register multiple clients
    clients = ["honest_client_1", "honest_client_2", "suspicious_client"]
    credentials = {}
    
    for client_id in clients:
        print(f"\n--- Registering {client_id} ---")
        creds = auth_service.register_client(client_id)
        credentials[client_id] = creds
        print(f"✅ Client {client_id} registered successfully")
        print(f"   Issued at: {creds.issued_at}")
        print(f"   Expires at: {creds.expires_at}")
    
    # Simulate authentication attempts
    print(f"\n--- Authentication Attempts ---")
    
    for client_id in clients:
        # Create a message to authenticate
        message_data = {
            "client_id": client_id,
            "action": "submit_model_update",
            "timestamp": datetime.now().isoformat()
        }
        message = json.dumps(message_data, sort_keys=True).encode()
        
        # Sign message with client's private key
        from pq_security.dilithium import DilithiumSigner
        signer = DilithiumSigner()
        signature = signer.sign(message, credentials[client_id].private_key)
        
        # Authenticate
        is_authenticated = auth_service.authenticate_client(client_id, signature, message)
        print(f"Authentication for {client_id}: {'✅ Success' if is_authenticated else '❌ Failed'}")
    
    # Demonstrate revocation
    print(f"\n--- Revocation Demo ---")
    suspicious_client = "suspicious_client"
    
    # Revoke suspicious client
    revocation_manager.revoke_credential(suspicious_client, "Detected anomalous behavior")
    auth_service.revoke_client(suspicious_client)
    print(f"🚫 Revoked credentials for {suspicious_client}")
    
    # Try to authenticate revoked client
    message_data = {
        "client_id": suspicious_client,
        "action": "submit_model_update",
        "timestamp": datetime.now().isoformat()
    }
    message = json.dumps(message_data, sort_keys=True).encode()
    signature = signer.sign(message, credentials[suspicious_client].private_key)
    
    is_authenticated = auth_service.authenticate_client(suspicious_client, signature, message)
    print(f"Authentication for revoked client: {'❌ Should fail' if is_authenticated else '✅ Correctly failed'}")
    
    # Check revocation status
    is_revoked = revocation_manager.is_revoked(suspicious_client)
    print(f"Revocation status check: {'✅ Correctly revoked' if is_revoked else '❌ Should be revoked'}")

if __name__ == "__main__":
    demonstrate_authentication()
```

---

## Anomaly Detection Examples

### Training Anomaly Detector

```python
#!/usr/bin/env python3
"""
Example of training and using the anomaly detection system.
"""

import numpy as np
from datetime import datetime
from anomaly_detection.isolation_forest_detector import IsolationForestDetector
from anomaly_detection.shap_explainer import SHAPExplainer
from anomaly_detection.reputation_manager import ClientReputationManager
from anomaly_detection.interfaces import ModelUpdate

def generate_normal_updates(num_updates=100):
    """Generate normal model updates for training."""
    updates = []
    
    for i in range(num_updates):
        # Normal weight distributions
        weights = {
            "layer1": np.random.normal(0, 0.1, (10, 5)),
            "layer2": np.random.normal(0, 0.1, (5, 1)),
            "bias1": np.random.normal(0, 0.05, (5,)),
            "bias2": np.random.normal(0, 0.05, (1,))
        }
        
        update = ModelUpdate(
            client_id=f"normal_client_{i % 10}",
            round_id=f"round_{i // 10}",
            weights=weights,
            signature=b"mock_signature",
            timestamp=datetime.now(),
            metadata={"accuracy": np.random.uniform(0.85, 0.95)}
        )
        updates.append(update)
    
    return updates

def generate_malicious_updates(num_updates=20):
    """Generate malicious model updates for testing."""
    updates = []
    
    for i in range(num_updates):
        # Malicious weight distributions (larger variance, different means)
        weights = {
            "layer1": np.random.normal(0.5, 1.0, (10, 5)),  # Shifted mean, high variance
            "layer2": np.random.normal(-0.3, 0.8, (5, 1)),  # Negative shift
            "bias1": np.random.normal(0.2, 0.5, (5,)),      # High bias
            "bias2": np.random.normal(-0.1, 0.3, (1,))      # Negative bias
        }
        
        update = ModelUpdate(
            client_id=f"malicious_client_{i}",
            round_id=f"round_{i}",
            weights=weights,
            signature=b"mock_signature",
            timestamp=datetime.now(),
            metadata={"accuracy": np.random.uniform(0.3, 0.7)}  # Lower accuracy
        )
        updates.append(update)
    
    return updates

def demonstrate_anomaly_detection():
    """Demonstrate anomaly detection workflow."""
    
    print("=== Anomaly Detection Demo ===")
    
    # Initialize components
    detector = IsolationForestDetector()
    reputation_manager = ClientReputationManager()
    
    # Generate training data
    print("Generating training data...")
    normal_updates = generate_normal_updates(100)
    print(f"Generated {len(normal_updates)} normal updates")
    
    # Train detector
    print("Training anomaly detector...")
    detector.fit(normal_updates)
    print("✅ Detector trained successfully")
    
    # Generate test data (mix of normal and malicious)
    print("\nGenerating test data...")
    test_normal = generate_normal_updates(20)
    test_malicious = generate_malicious_updates(10)
    all_test_updates = test_normal + test_malicious
    
    # Shuffle test data
    np.random.shuffle(all_test_updates)
    
    # Test detection
    print(f"\n--- Testing Detection on {len(all_test_updates)} Updates ---")
    
    results = {
        "normal_detected_as_normal": 0,
        "normal_detected_as_anomaly": 0,
        "malicious_detected_as_normal": 0,
        "malicious_detected_as_anomaly": 0
    }
    
    for update in all_test_updates:
        # Predict anomaly score
        anomaly_score = detector.predict_anomaly_score(update)
        
        # Update reputation
        reputation_manager.update_reputation(update.client_id, anomaly_score)
        
        # Determine if anomalous (threshold = 0.5)
        is_anomalous = anomaly_score > 0.5
        is_actually_malicious = "malicious" in update.client_id
        
        # Update results
        if is_actually_malicious:
            if is_anomalous:
                results["malicious_detected_as_anomaly"] += 1
                print(f"🎯 Malicious client {update.client_id}: score={anomaly_score:.3f} (DETECTED)")
            else:
                results["malicious_detected_as_normal"] += 1
                print(f"😞 Malicious client {update.client_id}: score={anomaly_score:.3f} (MISSED)")
        else:
            if is_anomalous:
                results["normal_detected_as_anomaly"] += 1
                print(f"⚠️  Normal client {update.client_id}: score={anomaly_score:.3f} (FALSE POSITIVE)")
            else:
                results["normal_detected_as_normal"] += 1
                print(f"✅ Normal client {update.client_id}: score={anomaly_score:.3f} (CORRECT)")
        
        # Show reputation impact
        reputation = reputation_manager.get_reputation(update.client_id)
        influence = reputation_manager.get_influence_weight(update.client_id)
        is_quarantined = reputation_manager.is_quarantined(update.client_id)
        
        print(f"   Reputation: {reputation:.3f}, Influence: {influence:.3f}, Quarantined: {is_quarantined}")
    
    # Calculate metrics
    print(f"\n--- Detection Results ---")
    total_malicious = results["malicious_detected_as_anomaly"] + results["malicious_detected_as_normal"]
    total_normal = results["normal_detected_as_normal"] + results["normal_detected_as_anomaly"]
    
    if total_malicious > 0:
        detection_rate = results["malicious_detected_as_anomaly"] / total_malicious
        print(f"Detection Rate (Sensitivity): {detection_rate:.2%}")
    
    if total_normal > 0:
        false_positive_rate = results["normal_detected_as_anomaly"] / total_normal
        print(f"False Positive Rate: {false_positive_rate:.2%}")
    
    total_correct = results["malicious_detected_as_anomaly"] + results["normal_detected_as_normal"]
    total_predictions = len(all_test_updates)
    accuracy = total_correct / total_predictions
    print(f"Overall Accuracy: {accuracy:.2%}")
    
    # Show reputation summary
    print(f"\n--- Reputation Summary ---")
    all_clients = set(update.client_id for update in all_test_updates)
    for client_id in sorted(all_clients):
        reputation = reputation_manager.get_reputation(client_id)
        influence = reputation_manager.get_influence_weight(client_id)
        is_quarantined = reputation_manager.is_quarantined(client_id)
        client_type = "Malicious" if "malicious" in client_id else "Normal"
        
        print(f"{client_type:>9} {client_id}: Rep={reputation:.3f}, Inf={influence:.3f}, Quarantined={is_quarantined}")

if __name__ == "__main__":
    demonstrate_anomaly_detection()
```

### SHAP Explanations Example

```python
#!/usr/bin/env python3
"""
Example of generating SHAP explanations for anomaly scores.
"""

import numpy as np
from datetime import datetime
from anomaly_detection.isolation_forest_detector import IsolationForestDetector
from anomaly_detection.shap_explainer import SHAPExplainer
from anomaly_detection.interfaces import ModelUpdate

def demonstrate_shap_explanations():
    """Demonstrate SHAP explanations for anomaly detection."""
    
    print("=== SHAP Explanations Demo ===")
    
    # Initialize detector
    detector = IsolationForestDetector()
    
    # Generate and train on normal data
    normal_updates = []
    for i in range(50):
        weights = {
            "layer1": np.random.normal(0, 0.1, (5, 3)),
            "layer2": np.random.normal(0, 0.1, (3, 1))
        }
        update = ModelUpdate(
            client_id=f"normal_{i}",
            round_id=f"round_{i}",
            weights=weights,
            signature=b"sig",
            timestamp=datetime.now(),
            metadata={}
        )
        normal_updates.append(update)
    
    detector.fit(normal_updates)
    print("✅ Trained detector on normal updates")
    
    # Create a suspicious update
    suspicious_weights = {
        "layer1": np.random.normal(0.8, 0.5, (5, 3)),  # High mean and variance
        "layer2": np.random.normal(-0.5, 0.3, (3, 1))  # Negative mean
    }
    
    suspicious_update = ModelUpdate(
        client_id="suspicious_client",
        round_id="current_round",
        weights=suspicious_weights,
        signature=b"sig",
        timestamp=datetime.now(),
        metadata={}
    )
    
    # Score the suspicious update
    anomaly_score = detector.predict_anomaly_score(suspicious_update)
    print(f"Anomaly score for suspicious update: {anomaly_score:.4f}")
    
    # Generate SHAP explanation
    explainer = SHAPExplainer(detector)
    explanation = explainer.explain(suspicious_update, anomaly_score)
    
    print(f"\n--- SHAP Explanation ---")
    print(f"Overall anomaly score: {anomaly_score:.4f}")
    print(f"Explanation: {explanation.get('explanation', 'No explanation available')}")
    
    # Show feature importance
    feature_importance = explanation.get('feature_importance', {})
    if feature_importance:
        print(f"\nFeature Importance (SHAP values):")
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature, importance in sorted_features[:10]:  # Top 10 features
            direction = "↑" if importance > 0 else "↓"
            print(f"  {feature:30} {direction} {importance:+.4f}")
    
    # Compare with normal update
    normal_update = normal_updates[0]
    normal_score = detector.predict_anomaly_score(normal_update)
    normal_explanation = explainer.explain(normal_update, normal_score)
    
    print(f"\n--- Comparison with Normal Update ---")
    print(f"Normal update anomaly score: {normal_score:.4f}")
    print(f"Suspicious update anomaly score: {anomaly_score:.4f}")
    print(f"Difference: {anomaly_score - normal_score:+.4f}")

if __name__ == "__main__":
    demonstrate_shap_explanations()
```

This completes the first part of the usage examples. The document continues with federated learning examples, monitoring examples, and complete integration examples.
---

## 
Federated Learning Examples

### Complete Federated Learning Round

```python
#!/usr/bin/env python3
"""
Example of a complete federated learning round with security.
"""

import numpy as np
import tensorflow as tf
from datetime import datetime
from federated_learning.server import SecureFederatedServer
from federated_learning.model_aggregator import ModelAggregator
from anomaly_detection.interfaces import ModelUpdate

def create_mnist_model():
    """Create a simple MNIST model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def simulate_client_training(client_id, global_weights, client_data, is_malicious=False):
    """Simulate client training and return model update."""
    
    # Create model and set global weights
    model = create_mnist_model()
    if global_weights:
        model.set_weights(global_weights)
    
    # Train on local data
    x_train, y_train = client_data
    
    if is_malicious:
        # Malicious client: poison some labels
        y_train_poisoned = y_train.copy()
        poison_indices = np.random.choice(len(y_train), size=len(y_train)//10, replace=False)
        y_train_poisoned[poison_indices] = (y_train_poisoned[poison_indices] + 1) % 10
        y_train = y_train_poisoned
        print(f"🦹 {client_id}: Poisoned {len(poison_indices)} labels")
    
    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=3,
        batch_size=32,
        verbose=0
    )
    
    # Create model update
    weights_dict = {}
    for i, weight in enumerate(model.get_weights()):
        weights_dict[f"layer_{i}"] = weight
    
    # Add malicious gradient manipulation
    if is_malicious:
        for key in weights_dict:
            # Amplify gradients to cause model divergence
            weights_dict[key] *= 2.0
            # Add noise
            weights_dict[key] += np.random.normal(0, 0.1, weights_dict[key].shape)
    
    update = ModelUpdate(
        client_id=client_id,
        round_id="current_round",
        weights=weights_dict,
        signature=f"signature_{client_id}".encode(),
        timestamp=datetime.now(),
        metadata={
            "accuracy": history.history['accuracy'][-1],
            "loss": history.history['loss'][-1],
            "samples": len(x_train),
            "is_malicious": is_malicious
        }
    )
    
    return update

def demonstrate_federated_learning():
    """Demonstrate complete federated learning with security."""
    
    print("=== Secure Federated Learning Demo ===")
    
    # Load and prepare MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Initialize server
    server = SecureFederatedServer()
    
    # Create client data partitions
    num_clients = 5
    clients_data = {}
    samples_per_client = len(x_train) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        
        client_id = f"client_{i+1}"
        clients_data[client_id] = (
            x_train[start_idx:end_idx],
            y_train[start_idx:end_idx]
        )
        
        # Register client
        server.register_client(client_id)
        print(f"📝 Registered {client_id} with {end_idx - start_idx} samples")
    
    # Designate one client as malicious
    malicious_client = "client_5"
    print(f"🦹 {malicious_client} will behave maliciously")
    
    # Create initial global model
    global_model = create_mnist_model()
    global_weights = global_model.get_weights()
    
    # Run federated learning rounds
    num_rounds = 5
    
    for round_num in range(num_rounds):
        print(f"\n{'='*50}")
        print(f"ROUND {round_num + 1}")
        print(f"{'='*50}")
        
        # Start training round
        round_id = server.start_training_round()
        print(f"🚀 Started training round: {round_id}")
        
        # Collect client updates
        client_updates = []
        
        for client_id, client_data in clients_data.items():
            is_malicious = (client_id == malicious_client)
            
            # Simulate client training
            update = simulate_client_training(
                client_id, global_weights, client_data, is_malicious
            )
            
            # Submit update to server
            accepted = server.receive_client_update(client_id, update)
            
            status = "✅ Accepted" if accepted else "❌ Rejected"
            malicious_tag = " 🦹" if is_malicious else ""
            print(f"  {client_id}{malicious_tag}: {status} (acc: {update.metadata['accuracy']:.3f})")
            
            if accepted:
                client_updates.append(update)
        
        # Aggregate updates
        if client_updates:
            global_model_data = server.aggregate_updates(round_id)
            global_weights = [global_model_data.weights[f"layer_{i}"] for i in range(len(global_weights))]
            
            # Evaluate global model
            global_model.set_weights(global_weights)
            test_loss, test_accuracy = global_model.evaluate(x_test, y_test, verbose=0)
            
            print(f"📊 Global model accuracy: {test_accuracy:.4f}")
            print(f"📊 Global model loss: {test_loss:.4f}")
            
            # Distribute global model
            server.distribute_global_model(global_model_data)
        else:
            print("⚠️  No updates accepted, skipping aggregation")
        
        # Show security summary
        security_summary = server.get_security_summary()
        print(f"🛡️  Security events this round: {security_summary.get('events_count', 0)}")
        print(f"🎯 Anomalies detected: {security_summary.get('anomalies_detected', 0)}")
        print(f"🚫 Clients quarantined: {security_summary.get('quarantined_clients', 0)}")

if __name__ == "__main__":
    demonstrate_federated_learning()
```

### Model Aggregation with Security

```python
#!/usr/bin/env python3
"""
Example of secure model aggregation with reputation weighting.
"""

import numpy as np
from datetime import datetime
from federated_learning.model_aggregator import ModelAggregator
from anomaly_detection.reputation_manager import ClientReputationManager
from anomaly_detection.interfaces import ModelUpdate

def demonstrate_secure_aggregation():
    """Demonstrate secure model aggregation with reputation weighting."""
    
    print("=== Secure Model Aggregation Demo ===")
    
    # Initialize components
    aggregator = ModelAggregator()
    reputation_manager = ClientReputationManager()
    
    # Create mock client updates
    clients = ["honest_1", "honest_2", "honest_3", "suspicious_1", "malicious_1"]
    updates = []
    
    for client_id in clients:
        # Create different weight patterns for different client types
        if "honest" in client_id:
            # Normal weight distributions
            weights = {
                "layer_0": np.random.normal(0, 0.1, (10, 5)),
                "layer_1": np.random.normal(0, 0.1, (5, 1))
            }
            accuracy = np.random.uniform(0.85, 0.95)
        elif "suspicious" in client_id:
            # Slightly abnormal but not clearly malicious
            weights = {
                "layer_0": np.random.normal(0.1, 0.2, (10, 5)),
                "layer_1": np.random.normal(-0.05, 0.15, (5, 1))
            }
            accuracy = np.random.uniform(0.75, 0.85)
        else:  # malicious
            # Clearly malicious patterns
            weights = {
                "layer_0": np.random.normal(0.5, 1.0, (10, 5)),
                "layer_1": np.random.normal(-0.3, 0.8, (5, 1))
            }
            accuracy = np.random.uniform(0.3, 0.6)
        
        update = ModelUpdate(
            client_id=client_id,
            round_id="demo_round",
            weights=weights,
            signature=f"sig_{client_id}".encode(),
            timestamp=datetime.now(),
            metadata={"accuracy": accuracy}
        )
        updates.append(update)
        
        # Simulate reputation based on client type
        if "honest" in client_id:
            reputation_manager.update_reputation(client_id, 0.1)  # Low anomaly score
        elif "suspicious" in client_id:
            reputation_manager.update_reputation(client_id, 0.6)  # Medium anomaly score
        else:  # malicious
            reputation_manager.update_reputation(client_id, 0.9)  # High anomaly score
    
    print(f"Created {len(updates)} client updates")
    
    # Show client reputations and influence weights
    print(f"\n--- Client Reputations ---")
    client_weights = {}
    
    for client_id in clients:
        reputation = reputation_manager.get_reputation(client_id)
        influence = reputation_manager.get_influence_weight(client_id)
        is_quarantined = reputation_manager.is_quarantined(client_id)
        
        client_weights[client_id] = influence
        
        status = "🚫 QUARANTINED" if is_quarantined else "✅ Active"
        client_type = "Honest" if "honest" in client_id else ("Suspicious" if "suspicious" in client_id else "Malicious")
        
        print(f"  {client_type:>10} {client_id}: Rep={reputation:.3f}, Weight={influence:.3f} {status}")
    
    # Filter out quarantined clients
    active_updates = [u for u in updates if not reputation_manager.is_quarantined(u.client_id)]
    active_weights = {u.client_id: client_weights[u.client_id] for u in active_updates}
    
    print(f"\n--- Aggregation ---")
    print(f"Total updates: {len(updates)}")
    print(f"Active updates (non-quarantined): {len(active_updates)}")
    
    # Perform aggregation
    aggregated_weights = aggregator.aggregate(active_updates, active_weights)
    
    print(f"✅ Aggregation completed")
    print(f"Aggregated model layers: {list(aggregated_weights.keys())}")
    
    # Show weight statistics
    for layer_name, weights in aggregated_weights.items():
        mean_weight = np.mean(weights)
        std_weight = np.std(weights)
        print(f"  {layer_name}: mean={mean_weight:.4f}, std={std_weight:.4f}, shape={weights.shape}")
    
    # Compare with unweighted aggregation
    print(f"\n--- Comparison with Unweighted Aggregation ---")
    
    # Equal weights for all active clients
    equal_weights = {u.client_id: 1.0 for u in active_updates}
    unweighted_aggregated = aggregator.aggregate(active_updates, equal_weights)
    
    # Calculate difference
    for layer_name in aggregated_weights.keys():
        weighted_mean = np.mean(aggregated_weights[layer_name])
        unweighted_mean = np.mean(unweighted_aggregated[layer_name])
        difference = abs(weighted_mean - unweighted_mean)
        
        print(f"  {layer_name}:")
        print(f"    Reputation-weighted mean: {weighted_mean:.6f}")
        print(f"    Equal-weighted mean: {unweighted_mean:.6f}")
        print(f"    Difference: {difference:.6f}")
    
    return aggregated_weights

if __name__ == "__main__":
    demonstrate_secure_aggregation()
```

---

## Monitoring Examples

### Security Event Logging

```python
#!/usr/bin/env python3
"""
Example of comprehensive security event logging.
"""

from datetime import datetime, timedelta
from monitoring.security_logger import SecurityEventLogger
from monitoring.interfaces import SecurityEvent, EventType, EventSeverity
import time

def demonstrate_security_logging():
    """Demonstrate security event logging and retrieval."""
    
    print("=== Security Event Logging Demo ===")
    
    # Initialize logger
    logger = SecurityEventLogger()
    
    # Simulate various security events
    events_to_log = [
        # Authentication events
        ("client_001", "authentication", "low", "Successful client authentication"),
        ("client_002", "authentication", "low", "Successful client authentication"),
        ("unknown_client", "authentication", "high", "Authentication attempt with invalid credentials"),
        ("client_003", "authentication", "medium", "Authentication with expired credentials"),
        
        # Cryptographic events
        ("server", "cryptographic", "low", "Key exchange completed successfully"),
        ("client_001", "cryptographic", "medium", "Signature verification failed - retrying"),
        ("client_002", "cryptographic", "low", "Digital signature verified successfully"),
        
        # Anomaly detection events
        ("client_004", "anomaly_detection", "high", "High anomaly score detected - quarantining client"),
        ("client_005", "anomaly_detection", "medium", "Moderate anomaly score - reducing influence weight"),
        ("client_001", "anomaly_detection", "low", "Normal behavior detected"),
        
        # System events
        ("server", "system", "medium", "High memory usage detected"),
        ("server", "system", "low", "Training round completed successfully"),
    ]
    
    print(f"Logging {len(events_to_log)} security events...")
    
    # Log events with small delays to show temporal ordering
    for client_id, event_type, severity, description in events_to_log:
        logger.log_event(SecurityEvent(
            event_id=f"evt_{int(time.time() * 1000)}",
            event_type=EventType(event_type),
            severity=EventSeverity(severity),
            client_id=client_id if client_id != "server" else None,
            description=description,
            metadata={"source": "demo", "round_id": "demo_round"},
            timestamp=datetime.now()
        ))
        time.sleep(0.1)  # Small delay for temporal separation
    
    print("✅ All events logged")
    
    # Demonstrate event retrieval
    print(f"\n--- Event Retrieval Examples ---")
    
    # Get all events from last hour
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    all_events = logger.get_events(start_time, end_time)
    print(f"Total events in last hour: {len(all_events)}")
    
    # Get events by type
    auth_events = logger.get_events(start_time, end_time, EventType.AUTHENTICATION)
    anomaly_events = logger.get_events(start_time, end_time, EventType.ANOMALY_DETECTION)
    crypto_events = logger.get_events(start_time, end_time, EventType.CRYPTOGRAPHIC)
    
    print(f"Authentication events: {len(auth_events)}")
    print(f"Anomaly detection events: {len(anomaly_events)}")
    print(f"Cryptographic events: {len(crypto_events)}")
    
    # Show high severity events
    high_severity_events = [e for e in all_events if e.severity == EventSeverity.HIGH]
    print(f"\n--- High Severity Events ({len(high_severity_events)}) ---")
    
    for event in high_severity_events:
        print(f"🚨 {event.timestamp.strftime('%H:%M:%S')} - {event.event_type.value}")
        print(f"   Client: {event.client_id or 'N/A'}")
        print(f"   Description: {event.description}")
    
    # Show authentication failures
    auth_failures = [e for e in auth_events if "invalid" in e.description.lower() or "failed" in e.description.lower()]
    print(f"\n--- Authentication Failures ({len(auth_failures)}) ---")
    
    for event in auth_failures:
        print(f"❌ {event.timestamp.strftime('%H:%M:%S')} - {event.client_id}")
        print(f"   {event.description}")
    
    # Demonstrate specific logging methods
    print(f"\n--- Using Specific Logging Methods ---")
    
    # Log authentication event
    logger.log_authentication_event("demo_client", True, "Demo authentication success")
    logger.log_authentication_event("demo_client", False, "Demo authentication failure")
    
    # Log cryptographic event
    logger.log_cryptographic_event("key_exchange", True, "Demo key exchange completed")
    logger.log_cryptographic_event("signature_verify", False, "Demo signature verification failed")
    
    # Log anomaly event
    logger.log_anomaly_event("demo_client", 0.85, "quarantine")
    
    print("✅ Demonstrated specific logging methods")

if __name__ == "__main__":
    demonstrate_security_logging()
```

### Metrics Collection and Analysis

```python
#!/usr/bin/env python3
"""
Example of system metrics collection and analysis.
"""

import time
import random
from datetime import datetime, timedelta
from monitoring.metrics_collector import MetricsCollector
from monitoring.interfaces import SystemMetrics

def simulate_system_activity():
    """Simulate system activity to generate metrics."""
    
    # Simulate varying system load
    base_cpu = 30.0
    base_memory = 45.0
    base_clients = 10
    
    # Add some randomness and trends
    cpu_usage = base_cpu + random.uniform(-10, 20)
    memory_usage = base_memory + random.uniform(-15, 25)
    active_clients = base_clients + random.randint(-3, 5)
    
    # Simulate some correlation (high CPU -> more anomalies)
    anomalies_detected = max(0, int((cpu_usage - 30) / 10) + random.randint(0, 2))
    auth_failures = random.randint(0, 3) if cpu_usage > 50 else random.randint(0, 1)
    
    return SystemMetrics(
        timestamp=datetime.now(),
        cpu_usage=max(0, min(100, cpu_usage)),
        memory_usage=max(0, min(100, memory_usage)),
        active_clients=max(0, active_clients),
        training_rounds_completed=random.randint(0, 2),
        anomalies_detected=anomalies_detected,
        authentication_failures=auth_failures
    )

def demonstrate_metrics_collection():
    """Demonstrate metrics collection and analysis."""
    
    print("=== System Metrics Collection Demo ===")
    
    # Initialize collector
    collector = MetricsCollector()
    
    # Collect metrics over time
    print("Collecting metrics over time...")
    
    metrics_history = []
    for i in range(20):
        # Simulate system activity
        metrics = simulate_system_activity()
        metrics_history.append(metrics)
        
        # Record individual metrics
        collector.record_metric("cpu_usage", metrics.cpu_usage, metrics.timestamp)
        collector.record_metric("memory_usage", metrics.memory_usage, metrics.timestamp)
        collector.record_metric("active_clients", metrics.active_clients, metrics.timestamp)
        collector.record_metric("anomalies_detected", metrics.anomalies_detected, metrics.timestamp)
        collector.record_metric("auth_failures", metrics.authentication_failures, metrics.timestamp)
        
        # Show current metrics
        if i % 5 == 0:
            print(f"  Time {i:2d}: CPU={metrics.cpu_usage:5.1f}%, "
                  f"Mem={metrics.memory_usage:5.1f}%, "
                  f"Clients={metrics.active_clients:2d}, "
                  f"Anomalies={metrics.anomalies_detected:2d}")
        
        time.sleep(0.1)  # Small delay for demonstration
    
    print("✅ Metrics collection completed")
    
    # Analyze collected metrics
    print(f"\n--- Metrics Analysis ---")
    
    # Calculate statistics
    cpu_values = [m.cpu_usage for m in metrics_history]
    memory_values = [m.memory_usage for m in metrics_history]
    anomaly_values = [m.anomalies_detected for m in metrics_history]
    
    print(f"CPU Usage:")
    print(f"  Average: {sum(cpu_values)/len(cpu_values):.1f}%")
    print(f"  Min: {min(cpu_values):.1f}%")
    print(f"  Max: {max(cpu_values):.1f}%")
    
    print(f"Memory Usage:")
    print(f"  Average: {sum(memory_values)/len(memory_values):.1f}%")
    print(f"  Min: {min(memory_values):.1f}%")
    print(f"  Max: {max(memory_values):.1f}%")
    
    print(f"Anomalies Detected:")
    print(f"  Total: {sum(anomaly_values)}")
    print(f"  Average per collection: {sum(anomaly_values)/len(anomaly_values):.1f}")
    print(f"  Max in single collection: {max(anomaly_values)}")
    
    # Retrieve metric history
    print(f"\n--- Metric History Retrieval ---")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=5)
    
    cpu_history = collector.get_metric_history("cpu_usage", start_time, end_time)
    anomaly_history = collector.get_metric_history("anomalies_detected", start_time, end_time)
    
    print(f"CPU usage history points: {len(cpu_history)}")
    print(f"Anomaly detection history points: {len(anomaly_history)}")
    
    # Show recent high CPU periods
    high_cpu_periods = [(timestamp, value) for timestamp, value in cpu_history if value > 60]
    if high_cpu_periods:
        print(f"\nHigh CPU periods (>60%):")
        for timestamp, value in high_cpu_periods:
            print(f"  {timestamp.strftime('%H:%M:%S')}: {value:.1f}%")
    
    # Show correlation between CPU and anomalies
    print(f"\n--- CPU vs Anomaly Correlation ---")
    
    # Simple correlation analysis
    if len(cpu_values) == len(anomaly_values):
        high_cpu_anomalies = sum(anomaly_values[i] for i in range(len(cpu_values)) if cpu_values[i] > 50)
        low_cpu_anomalies = sum(anomaly_values[i] for i in range(len(cpu_values)) if cpu_values[i] <= 50)
        
        high_cpu_count = sum(1 for cpu in cpu_values if cpu > 50)
        low_cpu_count = len(cpu_values) - high_cpu_count
        
        if high_cpu_count > 0 and low_cpu_count > 0:
            high_cpu_avg_anomalies = high_cpu_anomalies / high_cpu_count
            low_cpu_avg_anomalies = low_cpu_anomalies / low_cpu_count
            
            print(f"Average anomalies when CPU > 50%: {high_cpu_avg_anomalies:.2f}")
            print(f"Average anomalies when CPU ≤ 50%: {low_cpu_avg_anomalies:.2f}")
            
            if high_cpu_avg_anomalies > low_cpu_avg_anomalies * 1.5:
                print("📊 Correlation detected: High CPU usage correlates with more anomalies")
            else:
                print("📊 No strong correlation detected between CPU usage and anomalies")

if __name__ == "__main__":
    demonstrate_metrics_collection()
```

### Alert Management

```python
#!/usr/bin/env python3
"""
Example of alert management and threshold monitoring.
"""

import time
import random
from datetime import datetime
from monitoring.alert_manager import AlertManager
from monitoring.interfaces import EventSeverity

def demonstrate_alert_management():
    """Demonstrate alert management and threshold monitoring."""
    
    print("=== Alert Management Demo ===")
    
    # Initialize alert manager
    alert_manager = AlertManager()
    
    # Configure alert thresholds
    print("Configuring alert thresholds...")
    
    thresholds = [
        ("cpu_usage", 80.0, EventSeverity.HIGH),
        ("memory_usage", 85.0, EventSeverity.HIGH),
        ("anomaly_rate", 0.2, EventSeverity.MEDIUM),
        ("authentication_failures", 5, EventSeverity.HIGH),
        ("model_accuracy_drop", 0.1, EventSeverity.MEDIUM),
    ]
    
    for metric_name, threshold, severity in thresholds:
        alert_manager.configure_threshold(metric_name, threshold, severity)
        print(f"  {metric_name}: {threshold} ({severity.value})")
    
    print("✅ Thresholds configured")
    
    # Simulate system conditions that trigger alerts
    print(f"\n--- Simulating System Conditions ---")
    
    scenarios = [
        ("Normal operation", {"cpu_usage": 45, "memory_usage": 60, "anomaly_rate": 0.05}),
        ("High CPU load", {"cpu_usage": 85, "memory_usage": 70, "anomaly_rate": 0.08}),
        ("Memory pressure", {"cpu_usage": 60, "memory_usage": 90, "anomaly_rate": 0.12}),
        ("Security incident", {"cpu_usage": 70, "memory_usage": 75, "anomaly_rate": 0.25, "authentication_failures": 8}),
        ("Model degradation", {"cpu_usage": 50, "memory_usage": 65, "model_accuracy_drop": 0.15}),
        ("Recovery", {"cpu_usage": 40, "memory_usage": 55, "anomaly_rate": 0.03}),
    ]
    
    alert_ids = []
    
    for scenario_name, metrics in scenarios:
        print(f"\n🎭 Scenario: {scenario_name}")
        
        # Check each metric against thresholds
        for metric_name, value in metrics.items():
            # Find threshold for this metric
            threshold_info = next((t for t in thresholds if t[0] == metric_name), None)
            
            if threshold_info:
                _, threshold, severity = threshold_info
                
                if value > threshold:
                    # Create alert
                    alert_id = alert_manager.create_alert(
                        title=f"{metric_name.replace('_', ' ').title()} Threshold Exceeded",
                        description=f"{metric_name} is {value}, exceeding threshold of {threshold}",
                        severity=severity
                    )
                    alert_ids.append(alert_id)
                    
                    severity_emoji = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🚨"}
                    print(f"  {severity_emoji.get(severity.value, '⚠️')} ALERT: {metric_name} = {value} (threshold: {threshold})")
                else:
                    print(f"  ✅ {metric_name} = {value} (within threshold: {threshold})")
        
        time.sleep(1)  # Pause between scenarios
    
    # Show active alerts
    print(f"\n--- Active Alerts ---")
    active_alerts = alert_manager.get_active_alerts()
    
    print(f"Total active alerts: {len(active_alerts)}")
    
    for alert in active_alerts:
        severity_emoji = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🚨"}
        emoji = severity_emoji.get(alert.get('severity', 'medium'), '⚠️')
        
        print(f"{emoji} {alert.get('title', 'Unknown Alert')}")
        print(f"   Severity: {alert.get('severity', 'unknown').upper()}")
        print(f"   Description: {alert.get('description', 'No description')}")
        print(f"   Created: {alert.get('created_at', 'Unknown time')}")
    
    # Demonstrate alert resolution
    print(f"\n--- Alert Resolution ---")
    
    if alert_ids:
        # Resolve some alerts
        alerts_to_resolve = alert_ids[:len(alert_ids)//2]  # Resolve half
        
        for alert_id in alerts_to_resolve:
            alert_manager.resolve_alert(alert_id)
            print(f"✅ Resolved alert: {alert_id}")
        
        # Show remaining active alerts
        remaining_alerts = alert_manager.get_active_alerts()
        print(f"Remaining active alerts: {len(remaining_alerts)}")
    
    # Demonstrate alert escalation simulation
    print(f"\n--- Alert Escalation Simulation ---")
    
    # Create a critical alert
    critical_alert_id = alert_manager.create_alert(
        title="System Under Attack",
        description="Multiple malicious clients detected with coordinated attack patterns",
        severity=EventSeverity.CRITICAL
    )
    
    print(f"🚨 CRITICAL ALERT CREATED: {critical_alert_id}")
    
    # Simulate escalation process
    escalation_steps = [
        "Alert created and logged",
        "Notification sent to security team",
        "Automated response initiated",
        "Malicious clients quarantined",
        "System administrator notified",
        "Incident response team activated"
    ]
    
    for i, step in enumerate(escalation_steps, 1):
        print(f"  Step {i}: {step}")
        time.sleep(0.5)
    
    print("✅ Alert escalation completed")
    
    # Final summary
    final_alerts = alert_manager.get_active_alerts()
    print(f"\n--- Final Alert Summary ---")
    print(f"Total alerts created: {len(alert_ids) + 1}")  # +1 for critical alert
    print(f"Alerts resolved: {len(alert_ids) - len(final_alerts) + (0 if critical_alert_id in [a.get('id') for a in final_alerts] else 1)}")
    print(f"Active alerts remaining: {len(final_alerts)}")
    
    # Show alert distribution by severity
    severity_counts = {}
    for alert in final_alerts:
        severity = alert.get('severity', 'unknown')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    if severity_counts:
        print(f"Alert distribution by severity:")
        for severity, count in sorted(severity_counts.items()):
            emoji = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🚨"}.get(severity, '⚠️')
            print(f"  {emoji} {severity.upper()}: {count}")

if __name__ == "__main__":
    demonstrate_alert_management()
```

---

## Complete Integration Examples

### End-to-End Secure Federated Learning

```python
#!/usr/bin/env python3
"""
Complete end-to-end example of secure federated learning with QSFL-CAAD.
"""

import numpy as np
import tensorflow as tf
from datetime import datetime
import logging
import time

# Import QSFL-CAAD components
from qsfl_caad import QSFLSystem
from config.settings import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecureFederatedLearningDemo:
    """Complete demonstration of secure federated learning."""
    
    def __init__(self, config_path="config/config.yaml"):
        """Initialize the demo with configuration."""
        self.config = load_config(config_path)
        self.system = QSFLSystem(self.config)
        self.clients = {}
        self.global_model = None
        
    def setup_clients(self, num_honest=8, num_malicious=2):
        """Set up honest and malicious clients."""
        logger.info(f"Setting up {num_honest} honest and {num_malicious} malicious clients")
        
        # Load and partition MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        total_clients = num_honest + num_malicious
        samples_per_client = len(x_train) // total_clients
        
        # Create honest clients
        for i in range(num_honest):
            client_id = f"honest_client_{i+1}"
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            
            client_data = {
                'x': x_train[start_idx:end_idx],
                'y': y_train[start_idx:end_idx],
                'type': 'honest'
            }
            
            # Register with system
            credentials = self.system.register_client(client_id)
            self.clients[client_id] = {
                'data': client_data,
                'credentials': credentials,
                'type': 'honest'
            }
            
            logger.info(f"Registered honest client {client_id} with {len(client_data['x'])} samples")
        
        # Create malicious clients
        for i in range(num_malicious):
            client_id = f"malicious_client_{i+1}"
            start_idx = (num_honest + i) * samples_per_client
            end_idx = start_idx + samples_per_client
            
            client_data = {
                'x': x_train[start_idx:end_idx],
                'y': y_train[start_idx:end_idx],
                'type': 'malicious'
            }
            
            # Register with system (they appear legitimate initially)
            credentials = self.system.register_client(client_id)
            self.clients[client_id] = {
                'data': client_data,
                'credentials': credentials,
                'type': 'malicious'
            }
            
            logger.info(f"Registered malicious client {client_id} with {len(client_data['x'])} samples")
        
        # Store test data for evaluation
        self.test_data = (x_test, y_test)
        
    def create_model(self):
        """Create the neural network model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def simulate_client_training(self, client_id, global_weights=None):
        """Simulate local training for a client."""
        client_info = self.clients[client_id]
        client_data = client_info['data']
        client_type = client_info['type']
        
        # Create and initialize model
        model = self.create_model()
        if global_weights:
            model.set_weights(global_weights)
        
        # Prepare training data
        x_train, y_train = client_data['x'], client_data['y']
        
        # Apply malicious behavior
        if client_type == 'malicious':
            # Label flipping attack
            y_train_poisoned = y_train.copy()
            flip_indices = np.random.choice(len(y_train), size=len(y_train)//5, replace=False)
            y_train_poisoned[flip_indices] = (y_train_poisoned[flip_indices] + 1) % 10
            y_train = y_train_poisoned
            
            logger.warning(f"Malicious client {client_id} poisoned {len(flip_indices)} labels")
        
        # Train model
        history = model.fit(
            x_train, y_train,
            epochs=3,
            batch_size=32,
            verbose=0,
            validation_split=0.1
        )
        
        # Create model update
        weights_dict = {}
        for i, weight in enumerate(model.get_weights()):
            weights_dict[f"layer_{i}"] = weight
        
        # Apply gradient manipulation for malicious clients
        if client_type == 'malicious':
            for key in weights_dict:
                # Scale gradients to cause divergence
                weights_dict[key] *= 1.5
                # Add adversarial noise
                noise = np.random.normal(0, 0.05, weights_dict[key].shape)
                weights_dict[key] += noise
        
        # Create update object
        from anomaly_detection.interfaces import ModelUpdate
        
        update = ModelUpdate(
            client_id=client_id,
            round_id="current_round",
            weights=weights_dict,
            signature=self._sign_update(client_id, weights_dict),
            timestamp=datetime.now(),
            metadata={
                'accuracy': history.history['accuracy'][-1],
                'val_accuracy': history.history.get('val_accuracy', [0])[-1],
                'loss': history.history['loss'][-1],
                'samples': len(x_train),
                'epochs': 3,
                'client_type': client_type
            }
        )
        
        return update
    
    def _sign_update(self, client_id, weights_dict):
        """Sign the model update using client credentials."""
        # In a real implementation, this would use proper cryptographic signing
        # For demo purposes, we'll create a mock signature
        credentials = self.clients[client_id]['credentials']
        
        # Create message to sign (simplified)
        message = f"{client_id}_{len(weights_dict)}_{datetime.now().isoformat()}"
        
        # Use the PQ crypto system to sign
        signature = self.system.sign_message(message.encode(), credentials.private_key)
        
        return signature
    
    def run_federated_learning(self, num_rounds=10):
        """Run the complete federated learning process."""
        logger.info(f"Starting federated learning for {num_rounds} rounds")
        
        # Initialize global model
        self.global_model = self.create_model()
        global_weights = self.global_model.get_weights()
        
        # Track metrics
        round_metrics = []
        
        for round_num in range(num_rounds):
            logger.info(f"{'='*60}")
            logger.info(f"ROUND {round_num + 1}/{num_rounds}")
            logger.info(f"{'='*60}")
            
            # Start training round
            round_id = self.system.start_training_round()
            
            # Collect client updates
            client_updates = []
            round_start_time = time.time()
            
            for client_id in self.clients.keys():
                try:
                    # Simulate client training
                    update = self.simulate_client_training(client_id, global_weights)
                    
                    # Submit to system for validation
                    accepted = self.system.receive_client_update(client_id, update)
                    
                    if accepted:
                        client_updates.append(update)
                        status = "✅ ACCEPTED"
                    else:
                        status = "❌ REJECTED"
                    
                    client_type_emoji = "😇" if update.metadata['client_type'] == 'honest' else "😈"
                    logger.info(f"  {client_type_emoji} {client_id}: {status} "
                              f"(acc: {update.metadata['accuracy']:.3f})")
                    
                except Exception as e:
                    logger.error(f"Error processing update from {client_id}: {e}")
            
            # Aggregate updates
            if client_updates:
                try:
                    global_model_data = self.system.aggregate_updates(round_id)
                    
                    # Update global model
                    new_weights = []
                    for i in range(len(global_weights)):
                        layer_key = f"layer_{i}"
                        if layer_key in global_model_data.weights:
                            new_weights.append(global_model_data.weights[layer_key])
                        else:
                            new_weights.append(global_weights[i])
                    
                    global_weights = new_weights
                    self.global_model.set_weights(global_weights)
                    
                    # Evaluate global model
                    test_loss, test_accuracy = self.global_model.evaluate(
                        self.test_data[0], self.test_data[1], verbose=0
                    )
                    
                    round_time = time.time() - round_start_time
                    
                    # Collect round metrics
                    round_metric = {
                        'round': round_num + 1,
                        'test_accuracy': test_accuracy,
                        'test_loss': test_loss,
                        'updates_received': len(client_updates),
                        'updates_total': len(self.clients),
                        'round_time': round_time
                    }
                    round_metrics.append(round_metric)
                    
                    logger.info(f"📊 Global Model Performance:")
                    logger.info(f"   Test Accuracy: {test_accuracy:.4f}")
                    logger.info(f"   Test Loss: {test_loss:.4f}")
                    logger.info(f"   Updates Used: {len(client_updates)}/{len(self.clients)}")
                    logger.info(f"   Round Time: {round_time:.2f}s")
                    
                    # Distribute global model
                    self.system.distribute_global_model(global_model_data)
                    
                except Exception as e:
                    logger.error(f"Error during aggregation: {e}")
                    continue
            else:
                logger.warning("No valid updates received, skipping aggregation")
            
            # Get security summary
            security_summary = self.system.get_security_summary()
            logger.info(f"🛡️  Security Summary:")
            logger.info(f"   Anomalies Detected: {security_summary.get('anomalies_detected', 0)}")
            logger.info(f"   Clients Quarantined: {security_summary.get('quarantined_clients', 0)}")
            logger.info(f"   Authentication Failures: {security_summary.get('auth_failures', 0)}")
            
            # Show client reputations
            logger.info(f"👥 Client Reputations:")
            for client_id in sorted(self.clients.keys()):
                reputation = self.system.get_client_reputation(client_id)
                influence = self.system.get_client_influence(client_id)
                is_quarantined = self.system.is_client_quarantined(client_id)
                
                client_type = self.clients[client_id]['type']
                type_emoji = "😇" if client_type == 'honest' else "😈"
                status = "🚫 QUARANTINED" if is_quarantined else "✅ Active"
                
                logger.info(f"   {type_emoji} {client_id}: Rep={reputation:.3f}, "
                          f"Inf={influence:.3f} {status}")
            
            time.sleep(1)  # Brief pause between rounds
        
        # Final evaluation and summary
        self._generate_final_report(round_metrics)
    
    def _generate_final_report(self, round_metrics):
        """Generate final performance and security report."""
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL REPORT")
        logger.info(f"{'='*60}")
        
        if not round_metrics:
            logger.warning("No round metrics available for report")
            return
        
        # Performance metrics
        final_accuracy = round_metrics[-1]['test_accuracy']
        initial_accuracy = round_metrics[0]['test_accuracy'] if len(round_metrics) > 1 else 0
        accuracy_improvement = final_accuracy - initial_accuracy
        
        avg_accuracy = sum(m['test_accuracy'] for m in round_metrics) / len(round_metrics)
        max_accuracy = max(m['test_accuracy'] for m in round_metrics)
        
        logger.info(f"📈 Performance Metrics:")
        logger.info(f"   Final Accuracy: {final_accuracy:.4f}")
        logger.info(f"   Accuracy Improvement: {accuracy_improvement:+.4f}")
        logger.info(f"   Average Accuracy: {avg_accuracy:.4f}")
        logger.info(f"   Peak Accuracy: {max_accuracy:.4f}")
        
        # Security metrics
        total_clients = len(self.clients)
        honest_clients = sum(1 for c in self.clients.values() if c['type'] == 'honest')
        malicious_clients = total_clients - honest_clients
        
        quarantined_clients = sum(1 for client_id in self.clients.keys() 
                                if self.system.is_client_quarantined(client_id))
        
        # Check which type of clients were quarantined
        quarantined_honest = sum(1 for client_id in self.clients.keys()
                               if self.system.is_client_quarantined(client_id) 
                               and self.clients[client_id]['type'] == 'honest')
        quarantined_malicious = quarantined_clients - quarantined_honest
        
        logger.info(f"🛡️  Security Metrics:")
        logger.info(f"   Total Clients: {total_clients}")
        logger.info(f"   Honest Clients: {honest_clients}")
        logger.info(f"   Malicious Clients: {malicious_clients}")
        logger.info(f"   Clients Quarantined: {quarantined_clients}")
        logger.info(f"   Honest Quarantined: {quarantined_honest}")
        logger.info(f"   Malicious Quarantined: {quarantined_malicious}")
        
        if malicious_clients > 0:
            detection_rate = quarantined_malicious / malicious_clients
            logger.info(f"   Malicious Detection Rate: {detection_rate:.2%}")
        
        if honest_clients > 0:
            false_positive_rate = quarantined_honest / honest_clients
            logger.info(f"   False Positive Rate: {false_positive_rate:.2%}")
        
        # System efficiency
        avg_updates_used = sum(m['updates_received'] for m in round_metrics) / len(round_metrics)
        avg_round_time = sum(m['round_time'] for m in round_metrics) / len(round_metrics)
        
        logger.info(f"⚡ Efficiency Metrics:")
        logger.info(f"   Average Updates Used: {avg_updates_used:.1f}/{total_clients}")
        logger.info(f"   Average Round Time: {avg_round_time:.2f}s")
        logger.info(f"   Update Acceptance Rate: {avg_updates_used/total_clients:.2%}")
        
        # Success assessment
        logger.info(f"\n🎯 Success Assessment:")
        
        success_criteria = [
            (final_accuracy > 0.85, f"Model accuracy > 85%: {final_accuracy:.2%}"),
            (quarantined_malicious >= malicious_clients * 0.8, 
             f"Detected ≥80% of malicious clients: {quarantined_malicious}/{malicious_clients}"),
            (quarantined_honest <= honest_clients * 0.1, 
             f"False positive rate ≤10%: {quarantined_honest}/{honest_clients}"),
            (accuracy_improvement > 0, f"Model improved: {accuracy_improvement:+.4f}")
        ]
        
        passed_criteria = sum(1 for passed, _ in success_criteria if passed)
        
        for passed, description in success_criteria:
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {description}")
        
        overall_success = passed_criteria >= 3
        logger.info(f"\n🏆 Overall Success: {'✅ PASSED' if overall_success else '❌ FAILED'} "
                   f"({passed_criteria}/{len(success_criteria)} criteria met)")

def main():
    """Main demonstration function."""
    print("🚀 Starting QSFL-CAAD Complete Integration Demo")
    
    try:
        # Initialize demo
        demo = SecureFederatedLearningDemo()
        
        # Setup clients
        demo.setup_clients(num_honest=6, num_malicious=2)
        
        # Run federated learning
        demo.run_federated_learning(num_rounds=8)
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
```

This completes the comprehensive usage examples document, covering all major components and integration scenarios of the QSFL-CAAD system.