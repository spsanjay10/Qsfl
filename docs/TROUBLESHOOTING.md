# QSFL-CAAD Troubleshooting Guide

This guide provides solutions to common issues encountered when using, developing, or deploying the QSFL-CAAD system.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Configuration Problems](#configuration-problems)
3. [Runtime Errors](#runtime-errors)
4. [Performance Issues](#performance-issues)
5. [Security Concerns](#security-concerns)
6. [Integration Problems](#integration-problems)
7. [Debugging Tools](#debugging-tools)
8. [Getting Help](#getting-help)

---

## Installation Issues

### Problem: pqcrypto Library Installation Fails

**Symptoms:**
```bash
ERROR: Failed building wheel for pqcrypto
error: Microsoft Visual C++ 14.0 is required
```

**Solutions:**

**For Windows:**
```bash
# Option 1: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Option 2: Use conda-forge
conda install -c conda-forge pqcrypto

# Option 3: Use pre-compiled wheels (if available)
pip install --find-links https://download.pytorch.org/whl/torch_stable.html pqcrypto
```

**For Linux:**
```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install build-essential python3-dev libffi-dev

# For CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel libffi-devel

# Retry installation
pip install pqcrypto
```

**For macOS:**
```bash
# Install Xcode command line tools
xcode-select --install

# Install using Homebrew dependencies
brew install gcc

# Retry installation
pip install pqcrypto
```

**Fallback Solution:**
If pqcrypto installation continues to fail, QSFL-CAAD will automatically use fallback implementations:

```python
# The system will log this message and continue
WARNING: pqcrypto library not available, using fallback implementations
INFO: Fallback implementations provide the same API but may have different performance characteristics
```

### Problem: TensorFlow Installation Issues

**Symptoms:**
```bash
ImportError: No module named 'tensorflow'
# or
Could not find a version that satisfies the requirement tensorflow
```

**Solutions:**

```bash
# For CPU-only version
pip install tensorflow-cpu

# For GPU version (requires CUDA)
pip install tensorflow-gpu

# For Apple Silicon Macs
pip install tensorflow-macos tensorflow-metal

# Alternative: Use conda
conda install tensorflow

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Problem: Dependency Conflicts

**Symptoms:**
```bash
ERROR: pip's dependency resolver does not currently consider all the packages that are installed
```

**Solutions:**

```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # On Windows: fresh_env\Scripts\activate

# Install with specific versions
pip install -r requirements.txt --no-deps
pip install -r requirements.txt

# Use pip-tools for dependency resolution
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt

# Alternative: Use conda environment
conda env create -f environment.yml
```

---

## Configuration Problems

### Problem: Configuration File Not Found

**Symptoms:**
```python
FileNotFoundError: [Errno 2] No such file or directory: 'config/config.yaml'
```

**Solutions:**

```bash
# Copy example configuration
cp .env.example .env
cp config/config.yaml.example config/config.yaml

# Or create minimal configuration
mkdir -p config
cat > config/config.yaml << EOF
security:
  anomaly_threshold: 0.5
  reputation_decay: 0.95
  quarantine_threshold: 0.8

database:
  url: "sqlite:///qsfl_caad.db"

logging:
  level: "INFO"
  file: "logs/qsfl_caad.log"
EOF
```

### Problem: Invalid Configuration Values

**Symptoms:**
```python
ValueError: anomaly_threshold must be in [0.0, 1.0], got 1.5
```

**Solutions:**

```python
# Validate configuration before use
def validate_config(config):
    """Validate configuration values."""
    
    # Security settings
    security = config.get('security', {})
    
    anomaly_threshold = security.get('anomaly_threshold', 0.5)
    if not 0.0 <= anomaly_threshold <= 1.0:
        raise ValueError(f"anomaly_threshold must be in [0.0, 1.0], got {anomaly_threshold}")
    
    reputation_decay = security.get('reputation_decay', 0.95)
    if not 0.0 <= reputation_decay <= 1.0:
        raise ValueError(f"reputation_decay must be in [0.0, 1.0], got {reputation_decay}")
    
    quarantine_threshold = security.get('quarantine_threshold', 0.8)
    if not 0.0 <= quarantine_threshold <= 1.0:
        raise ValueError(f"quarantine_threshold must be in [0.0, 1.0], got {quarantine_threshold}")
    
    print("✅ Configuration validation passed")

# Use validation
try:
    config = load_config("config/config.yaml")
    validate_config(config)
except ValueError as e:
    print(f"❌ Configuration error: {e}")
    # Fix configuration and retry
```

### Problem: Database Connection Issues

**Symptoms:**
```python
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) unable to open database file
```

**Solutions:**

```python
# Check database directory exists
import os
db_path = "logs/qsfl_caad.db"
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Test database connection
from sqlalchemy import create_engine
try:
    engine = create_engine(f"sqlite:///{db_path}")
    connection = engine.connect()
    connection.close()
    print("✅ Database connection successful")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    
    # Alternative: Use in-memory database for testing
    engine = create_engine("sqlite:///:memory:")
```

---

## Runtime Errors

### Problem: Authentication Failures

**Symptoms:**
```python
AuthenticationError: Failed to authenticate client_001
WARNING: Authentication attempt with invalid credentials
```

**Debugging Steps:**

```python
# Enable detailed authentication logging
import logging
logging.getLogger('qsfl_caad.auth').setLevel(logging.DEBUG)

# Check client credentials
def debug_authentication(client_id):
    """Debug authentication issues."""
    
    auth_service = AuthenticationService()
    
    # Check if client is registered
    credentials = auth_service.get_client_credentials(client_id)
    if not credentials:
        print(f"❌ Client {client_id} not registered")
        return
    
    print(f"✅ Client {client_id} found")
    print(f"   Status: {credentials.status}")
    print(f"   Issued: {credentials.issued_at}")
    print(f"   Expires: {credentials.expires_at}")
    
    # Check if credentials are expired
    from datetime import datetime
    if credentials.expires_at < datetime.now():
        print(f"❌ Credentials expired")
        return
    
    # Test signature creation and verification
    message = b"test_message"
    try:
        from pq_security.dilithium import DilithiumSigner
        signer = DilithiumSigner()
        
        signature = signer.sign(message, credentials.private_key)
        is_valid = signer.verify(message, signature, credentials.public_key)
        
        if is_valid:
            print(f"✅ Signature verification successful")
        else:
            print(f"❌ Signature verification failed")
            
    except Exception as e:
        print(f"❌ Signature operation failed: {e}")

# Usage
debug_authentication("client_001")
```

**Common Solutions:**

```python
# Re-register client with fresh credentials
def fix_authentication_issues(client_id):
    """Fix common authentication issues."""
    
    auth_service = AuthenticationService()
    
    # Remove old credentials
    try:
        auth_service.revoke_client(client_id)
    except:
        pass  # Client might not exist
    
    # Register with new credentials
    new_credentials = auth_service.register_client(client_id)
    
    print(f"✅ Client {client_id} re-registered with new credentials")
    return new_credentials
```

### Problem: Anomaly Detection Errors

**Symptoms:**
```python
AnomalyDetectionError: Detector must be trained first
ValueError: Input contains NaN, infinity or a value too large for dtype('float64')
```

**Debugging Steps:**

```python
def debug_anomaly_detection():
    """Debug anomaly detection issues."""
    
    from anomaly_detection.isolation_forest_detector import IsolationForestDetector
    import numpy as np
    
    detector = IsolationForestDetector()
    
    # Check if detector is trained
    if not hasattr(detector, 'model') or detector.model is None:
        print("❌ Detector not trained")
        
        # Load or create training data
        normal_updates = load_normal_training_data()
        if not normal_updates:
            print("❌ No training data available")
            return
        
        print(f"Training detector with {len(normal_updates)} samples")
        detector.fit(normal_updates)
        print("✅ Detector trained successfully")
    
    # Test with sample update
    test_update = create_test_update()
    
    # Check for data quality issues
    for layer_name, weights in test_update.weights.items():
        if np.isnan(weights).any():
            print(f"❌ NaN values found in {layer_name}")
        if np.isinf(weights).any():
            print(f"❌ Infinite values found in {layer_name}")
        if weights.dtype not in [np.float32, np.float64]:
            print(f"⚠️  Unexpected dtype in {layer_name}: {weights.dtype}")
    
    # Test prediction
    try:
        score = detector.predict_anomaly_score(test_update)
        print(f"✅ Anomaly score: {score}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

def clean_model_weights(weights_dict):
    """Clean model weights to handle NaN/inf values."""
    cleaned = {}
    
    for layer_name, weights in weights_dict.items():
        # Replace NaN with zeros
        weights = np.nan_to_num(weights, nan=0.0)
        
        # Clip extreme values
        weights = np.clip(weights, -1e6, 1e6)
        
        # Ensure correct dtype
        weights = weights.astype(np.float32)
        
        cleaned[layer_name] = weights
    
    return cleaned
```

### Problem: Model Aggregation Failures

**Symptoms:**
```python
ValueError: All input arrays must have the same shape
RuntimeError: Aggregation failed: incompatible weight dimensions
```

**Debugging Steps:**

```python
def debug_aggregation_issues(updates):
    """Debug model aggregation issues."""
    
    print(f"Debugging aggregation with {len(updates)} updates")
    
    # Check update structure consistency
    if not updates:
        print("❌ No updates to aggregate")
        return
    
    reference_update = updates[0]
    reference_layers = set(reference_update.weights.keys())
    
    print(f"Reference layers: {reference_layers}")
    
    for i, update in enumerate(updates):
        update_layers = set(update.weights.keys())
        
        if update_layers != reference_layers:
            print(f"❌ Update {i} has different layers: {update_layers}")
            print(f"   Missing: {reference_layers - update_layers}")
            print(f"   Extra: {update_layers - reference_layers}")
            continue
        
        # Check weight shapes
        for layer_name in reference_layers:
            ref_shape = reference_update.weights[layer_name].shape
            update_shape = update.weights[layer_name].shape
            
            if ref_shape != update_shape:
                print(f"❌ Update {i}, layer {layer_name}: shape mismatch")
                print(f"   Expected: {ref_shape}, Got: {update_shape}")
    
    print("✅ Update structure validation completed")

def fix_aggregation_issues(updates):
    """Fix common aggregation issues."""
    
    if not updates:
        return []
    
    # Find common layers across all updates
    common_layers = set(updates[0].weights.keys())
    for update in updates[1:]:
        common_layers &= set(update.weights.keys())
    
    print(f"Common layers: {common_layers}")
    
    # Filter updates to only include common layers
    fixed_updates = []
    for update in updates:
        fixed_weights = {
            layer: weights for layer, weights in update.weights.items()
            if layer in common_layers
        }
        
        # Create new update with fixed weights
        fixed_update = ModelUpdate(
            client_id=update.client_id,
            round_id=update.round_id,
            weights=fixed_weights,
            signature=update.signature,
            timestamp=update.timestamp,
            metadata=update.metadata
        )
        fixed_updates.append(fixed_update)
    
    return fixed_updates
```

---

## Performance Issues

### Problem: Slow Cryptographic Operations

**Symptoms:**
- Key generation takes several seconds
- Signature verification is very slow
- High CPU usage during crypto operations

**Solutions:**

```python
# Enable hardware acceleration
def optimize_crypto_performance():
    """Optimize cryptographic performance."""
    
    # Check for hardware acceleration
    import platform
    system = platform.system()
    
    if system == "Linux":
        # Check for AES-NI support
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'aes' in cpuinfo:
                    print("✅ AES-NI hardware acceleration available")
                else:
                    print("⚠️  AES-NI not available, crypto operations may be slower")
        except:
            pass
    
    # Use caching for frequently used keys
    from functools import lru_cache
    
    @lru_cache(maxsize=1000)
    def cached_key_generation(client_id):
        """Cache key generation results."""
        from pq_security.dilithium import DilithiumSigner
        signer = DilithiumSigner()
        return signer.generate_keypair()
    
    # Batch signature operations
    def batch_verify_signatures(messages_and_sigs):
        """Verify multiple signatures in batch."""
        from pq_security.dilithium import DilithiumSigner
        signer = DilithiumSigner()
        
        results = []
        for message, signature, public_key in messages_and_sigs:
            result = signer.verify(message, signature, public_key)
            results.append(result)
        
        return results

# Monitor crypto performance
def benchmark_crypto_operations():
    """Benchmark cryptographic operations."""
    import time
    from pq_security.kyber import KyberKeyExchange
    from pq_security.dilithium import DilithiumSigner
    
    kyber = KyberKeyExchange()
    dilithium = DilithiumSigner()
    
    # Benchmark key generation
    start = time.time()
    for _ in range(10):
        kyber.generate_keypair()
    keygen_time = (time.time() - start) / 10
    
    if keygen_time > 1.0:
        print(f"⚠️  Slow key generation: {keygen_time:.2f}s per operation")
    else:
        print(f"✅ Key generation performance: {keygen_time:.3f}s per operation")
```

### Problem: High Memory Usage

**Symptoms:**
- Memory usage grows continuously
- Out of memory errors during aggregation
- System becomes unresponsive

**Solutions:**

```python
import gc
import psutil
import os

def monitor_memory_usage():
    """Monitor and optimize memory usage."""
    
    process = psutil.Process(os.getpid())
    
    def get_memory_mb():
        return process.memory_info().rss / 1024 / 1024
    
    initial_memory = get_memory_mb()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Force garbage collection
    gc.collect()
    
    after_gc_memory = get_memory_mb()
    print(f"After GC: {after_gc_memory:.2f} MB")
    
    # Check for memory leaks
    if after_gc_memory > initial_memory * 1.5:
        print("⚠️  Potential memory leak detected")
        
        # Get top memory consumers
        import tracemalloc
        tracemalloc.start()
        
        # Your code here
        
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
        print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
        
        tracemalloc.stop()

def optimize_memory_usage():
    """Implement memory optimization strategies."""
    
    # Use generators instead of lists for large datasets
    def process_updates_generator(updates):
        """Process updates as generator to save memory."""
        for update in updates:
            # Process update
            result = process_single_update(update)
            yield result
            
            # Clear references
            del update
            
            # Periodic garbage collection
            if len(updates) % 100 == 0:
                gc.collect()
    
    # Use memory mapping for large arrays
    import numpy as np
    
    def use_memory_mapping(large_array):
        """Use memory mapping for large arrays."""
        if large_array.nbytes > 100 * 1024 * 1024:  # 100MB
            # Save to temporary file
            temp_file = f"/tmp/large_array_{id(large_array)}.npy"
            np.save(temp_file, large_array)
            
            # Load as memory-mapped array
            return np.load(temp_file, mmap_mode='r+')
        
        return large_array
    
    # Implement object pooling for frequently created objects
    class ObjectPool:
        def __init__(self, factory, max_size=100):
            self.factory = factory
            self.pool = []
            self.max_size = max_size
        
        def get(self):
            if self.pool:
                return self.pool.pop()
            return self.factory()
        
        def put(self, obj):
            if len(self.pool) < self.max_size:
                # Reset object state
                obj.reset()
                self.pool.append(obj)
```

### Problem: Slow Anomaly Detection

**Symptoms:**
- Anomaly detection takes several seconds per update
- High CPU usage during detection
- System becomes unresponsive during batch processing

**Solutions:**

```python
def optimize_anomaly_detection():
    """Optimize anomaly detection performance."""
    
    # Use feature caching
    from functools import lru_cache
    import hashlib
    import pickle
    
    class OptimizedAnomalyDetector:
        def __init__(self):
            self.detector = IsolationForestDetector()
            self._feature_cache = {}
        
        @lru_cache(maxsize=1000)
        def _extract_features_cached(self, weights_hash):
            """Cache feature extraction results."""
            if weights_hash in self._feature_cache:
                return self._feature_cache[weights_hash]
            
            # Extract features (expensive operation)
            features = self._extract_features_impl(weights_hash)
            self._feature_cache[weights_hash] = features
            return features
        
        def predict_batch(self, updates):
            """Batch prediction for better performance."""
            # Extract all features first
            features_list = []
            for update in updates:
                features = self._extract_features_cached(update)
                features_list.append(features)
            
            # Batch prediction
            scores = self.detector.predict_batch(features_list)
            return scores
    
    # Use parallel processing
    from concurrent.futures import ThreadPoolExecutor
    
    def parallel_anomaly_detection(updates, n_workers=4):
        """Detect anomalies in parallel."""
        
        def process_chunk(chunk):
            detector = IsolationForestDetector()
            return [detector.predict_anomaly_score(update) for update in chunk]
        
        # Split into chunks
        chunk_size = len(updates) // n_workers
        chunks = [updates[i:i+chunk_size] for i in range(0, len(updates), chunk_size)]
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            results = [future.result() for future in futures]
        
        # Flatten results
        return [score for chunk_scores in results for score in chunk_scores]
```

---

## Security Concerns

### Problem: Potential Security Vulnerabilities

**Symptoms:**
- Unusual authentication patterns
- Unexpected anomaly scores
- Suspicious client behavior

**Security Audit Checklist:**

```python
def security_audit():
    """Perform comprehensive security audit."""
    
    print("=== QSFL-CAAD Security Audit ===")
    
    # 1. Check cryptographic implementations
    audit_crypto_security()
    
    # 2. Verify authentication mechanisms
    audit_authentication_security()
    
    # 3. Check anomaly detection effectiveness
    audit_anomaly_detection_security()
    
    # 4. Verify logging and monitoring
    audit_logging_security()
    
    print("✅ Security audit completed")

def audit_crypto_security():
    """Audit cryptographic security."""
    
    print("\n--- Cryptographic Security Audit ---")
    
    # Check if using fallback implementations
    try:
        import pqcrypto
        print("✅ Using pqcrypto library (recommended)")
    except ImportError:
        print("⚠️  Using fallback implementations (less secure)")
    
    # Test key generation randomness
    from pq_security.kyber import KyberKeyExchange
    kyber = KyberKeyExchange()
    
    keys = []
    for _ in range(10):
        pub, priv = kyber.generate_keypair()
        keys.append((pub, priv))
    
    # Check for duplicate keys (should never happen)
    public_keys = [k[0] for k in keys]
    if len(set(public_keys)) != len(public_keys):
        print("❌ CRITICAL: Duplicate public keys detected!")
    else:
        print("✅ Key generation randomness verified")

def audit_authentication_security():
    """Audit authentication security."""
    
    print("\n--- Authentication Security Audit ---")
    
    # Check for weak credentials
    auth_service = AuthenticationService()
    
    # Simulate brute force attack
    def simulate_brute_force():
        """Simulate brute force authentication attack."""
        
        failed_attempts = 0
        for i in range(100):
            # Try authentication with random signature
            fake_signature = os.urandom(64)
            fake_message = b"fake_message"
            
            try:
                result = auth_service.authenticate_client("fake_client", fake_signature, fake_message)
                if result:
                    print("❌ CRITICAL: Authentication bypass detected!")
                    return
                else:
                    failed_attempts += 1
            except:
                failed_attempts += 1
        
        print(f"✅ Brute force resistance: {failed_attempts}/100 attempts failed")
    
    simulate_brute_force()

def audit_anomaly_detection_security():
    """Audit anomaly detection effectiveness."""
    
    print("\n--- Anomaly Detection Security Audit ---")
    
    from anomaly_detection.isolation_forest_detector import IsolationForestDetector
    
    detector = IsolationForestDetector()
    
    # Test with known malicious patterns
    malicious_patterns = [
        create_gradient_poisoning_update(),
        create_label_flipping_update(),
        create_backdoor_update()
    ]
    
    detection_count = 0
    for pattern in malicious_patterns:
        score = detector.predict_anomaly_score(pattern)
        if score > 0.5:  # Assuming 0.5 threshold
            detection_count += 1
    
    detection_rate = detection_count / len(malicious_patterns)
    
    if detection_rate < 0.8:
        print(f"⚠️  Low detection rate: {detection_rate:.2%}")
    else:
        print(f"✅ Good detection rate: {detection_rate:.2%}")

def audit_logging_security():
    """Audit logging and monitoring security."""
    
    print("\n--- Logging Security Audit ---")
    
    # Check if sensitive data is being logged
    import re
    
    log_files = ["logs/qsfl_caad.log", "logs/security.log"]
    
    sensitive_patterns = [
        r'private_key',
        r'password',
        r'secret',
        r'token'
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
                
                for pattern in sensitive_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        print(f"⚠️  Potential sensitive data in {log_file}: {pattern}")
    
    print("✅ Log security audit completed")
```

### Problem: Adversarial Attacks

**Symptoms:**
- Coordinated malicious behavior
- Sophisticated evasion attempts
- Gradual model degradation

**Detection and Mitigation:**

```python
def detect_coordinated_attacks():
    """Detect coordinated adversarial attacks."""
    
    # Analyze client behavior patterns
    def analyze_client_patterns(client_histories):
        """Analyze patterns in client behavior."""
        
        suspicious_groups = []
        
        # Look for clients with similar anomaly patterns
        for i, client1 in enumerate(client_histories):
            similar_clients = [client1['id']]
            
            for j, client2 in enumerate(client_histories[i+1:], i+1):
                # Calculate similarity in anomaly scores
                similarity = calculate_pattern_similarity(
                    client1['anomaly_scores'], 
                    client2['anomaly_scores']
                )
                
                if similarity > 0.8:  # High similarity threshold
                    similar_clients.append(client2['id'])
            
            if len(similar_clients) > 1:
                suspicious_groups.append(similar_clients)
        
        return suspicious_groups
    
    # Implement adaptive thresholds
    def adaptive_anomaly_threshold(recent_scores, base_threshold=0.5):
        """Adapt anomaly threshold based on recent activity."""
        
        if not recent_scores:
            return base_threshold
        
        # Increase threshold if many recent anomalies
        recent_anomaly_rate = sum(1 for s in recent_scores if s > base_threshold) / len(recent_scores)
        
        if recent_anomaly_rate > 0.2:  # More than 20% anomalies
            # Increase threshold to reduce false positives
            return min(base_threshold * 1.2, 0.9)
        
        return base_threshold
    
    # Implement reputation-based defense
    def reputation_based_defense(client_id, current_score, history):
        """Apply reputation-based defense mechanisms."""
        
        reputation = calculate_reputation(history)
        
        # Adjust anomaly score based on reputation
        if reputation < 0.3:  # Low reputation
            adjusted_score = current_score * 1.5  # Amplify anomaly score
        elif reputation > 0.8:  # High reputation
            adjusted_score = current_score * 0.8  # Reduce anomaly score
        else:
            adjusted_score = current_score
        
        return min(adjusted_score, 1.0)
```

---

## Integration Problems

### Problem: TensorFlow Integration Issues

**Symptoms:**
```python
AttributeError: 'Sequential' object has no attribute 'get_weights'
ValueError: Incompatible model architectures
```

**Solutions:**

```python
def fix_tensorflow_integration():
    """Fix common TensorFlow integration issues."""
    
    import tensorflow as tf
    
    # Ensure model is built before getting weights
    def safe_get_weights(model):
        """Safely get model weights."""
        
        if not model.built:
            # Build model with dummy input
            dummy_input = tf.zeros((1,) + model.input_shape[1:])
            model(dummy_input)
        
        return model.get_weights()
    
    # Handle different TensorFlow versions
    def version_compatible_operations():
        """Handle TensorFlow version compatibility."""
        
        tf_version = tf.__version__
        major_version = int(tf_version.split('.')[0])
        
        if major_version >= 2:
            # TensorFlow 2.x
            @tf.function
            def train_step(model, x, y):
                with tf.GradientTape() as tape:
                    predictions = model(x, training=True)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(y, predictions)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                return gradients, loss
        else:
            # TensorFlow 1.x (legacy support)
            def train_step(model, x, y):
                # Legacy implementation
                pass
        
        return train_step
    
    # Convert between different model formats
    def convert_model_weights(weights, source_format, target_format):
        """Convert model weights between formats."""
        
        if source_format == target_format:
            return weights
        
        if source_format == "list" and target_format == "dict":
            # Convert list of arrays to dictionary
            return {f"layer_{i}": w for i, w in enumerate(weights)}
        
        elif source_format == "dict" and target_format == "list":
            # Convert dictionary to list
            return [weights[f"layer_{i}"] for i in range(len(weights))]
        
        else:
            raise ValueError(f"Unsupported conversion: {source_format} -> {target_format}")
```

### Problem: PySyft Integration Issues

**Symptoms:**
```python
ImportError: No module named 'syft'
AttributeError: 'VirtualWorker' object has no attribute 'send'
```

**Solutions:**

```python
def setup_syft_integration():
    """Setup PySyft integration with QSFL-CAAD."""
    
    try:
        import syft as sy
        import torch
        
        # Create hook
        hook = sy.TorchHook(torch)
        
        # Create virtual workers
        alice = sy.VirtualWorker(hook, id="alice")
        bob = sy.VirtualWorker(hook, id="bob")
        
        print("✅ PySyft integration setup successful")
        
        return hook, alice, bob
        
    except ImportError:
        print("⚠️  PySyft not available, using alternative implementation")
        return None, None, None
    
    except Exception as e:
        print(f"❌ PySyft setup failed: {e}")
        return None, None, None

def syft_compatible_aggregation(updates):
    """PySyft-compatible model aggregation."""
    
    try:
        import syft as sy
        
        # Convert updates to PySyft tensors
        syft_updates = []
        for update in updates:
            syft_weights = {}
            for layer_name, weights in update.weights.items():
                # Convert numpy to torch tensor
                tensor = torch.from_numpy(weights)
                syft_weights[layer_name] = tensor
            
            syft_updates.append(syft_weights)
        
        # Perform secure aggregation
        aggregated = {}
        for layer_name in syft_updates[0].keys():
            layer_tensors = [update[layer_name] for update in syft_updates]
            aggregated[layer_name] = torch.mean(torch.stack(layer_tensors), dim=0)
        
        # Convert back to numpy
        result = {}
        for layer_name, tensor in aggregated.items():
            result[layer_name] = tensor.detach().numpy()
        
        return result
        
    except Exception as e:
        print(f"PySyft aggregation failed: {e}")
        # Fallback to standard aggregation
        return standard_aggregation(updates)
```

---

## Debugging Tools

### Comprehensive Debugging Script

```python
#!/usr/bin/env python3
"""
Comprehensive debugging script for QSFL-CAAD system.
"""

import sys
import traceback
import logging
from datetime import datetime

def setup_debug_logging():
    """Setup comprehensive debug logging."""
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(f'debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_system_diagnostics():
    """Run comprehensive system diagnostics."""
    
    print("=== QSFL-CAAD System Diagnostics ===")
    
    # 1. Environment check
    check_environment()
    
    # 2. Dependencies check
    check_dependencies()
    
    # 3. Configuration check
    check_configuration()
    
    # 4. Component tests
    test_components()
    
    # 5. Integration tests
    test_integration()
    
    print("\n✅ Diagnostics completed")

def check_environment():
    """Check system environment."""
    
    print("\n--- Environment Check ---")
    
    import platform
    import sys
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Processor: {platform.processor()}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total memory: {memory.total / 1024**3:.2f} GB")
        print(f"Available memory: {memory.available / 1024**3:.2f} GB")
    except ImportError:
        print("⚠️  psutil not available, cannot check memory")

def check_dependencies():
    """Check all dependencies."""
    
    print("\n--- Dependencies Check ---")
    
    required_packages = [
        'numpy', 'tensorflow', 'scikit-learn', 'pqcrypto',
        'cryptography', 'sqlalchemy', 'pyyaml', 'matplotlib'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
        except Exception as e:
            print(f"⚠️  {package} - Import error: {e}")

def check_configuration():
    """Check system configuration."""
    
    print("\n--- Configuration Check ---")
    
    try:
        from config.settings import load_config
        config = load_config("config/config.yaml")
        print("✅ Configuration loaded successfully")
        
        # Validate configuration
        validate_config(config)
        
    except FileNotFoundError:
        print("❌ Configuration file not found")
    except Exception as e:
        print(f"❌ Configuration error: {e}")

def test_components():
    """Test individual components."""
    
    print("\n--- Component Tests ---")
    
    # Test post-quantum crypto
    try:
        from pq_security.kyber import KyberKeyExchange
        kyber = KyberKeyExchange()
        pub, priv = kyber.generate_keypair()
        print("✅ Post-quantum cryptography")
    except Exception as e:
        print(f"❌ Post-quantum cryptography: {e}")
    
    # Test authentication
    try:
        from auth.authentication_service import AuthenticationService
        auth = AuthenticationService()
        print("✅ Authentication service")
    except Exception as e:
        print(f"❌ Authentication service: {e}")
    
    # Test anomaly detection
    try:
        from anomaly_detection.isolation_forest_detector import IsolationForestDetector
        detector = IsolationForestDetector()
        print("✅ Anomaly detection")
    except Exception as e:
        print(f"❌ Anomaly detection: {e}")

def test_integration():
    """Test component integration."""
    
    print("\n--- Integration Tests ---")
    
    try:
        # Test end-to-end workflow
        from qsfl_caad import QSFLSystem
        system = QSFLSystem()
        print("✅ System integration")
    except Exception as e:
        print(f"❌ System integration: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    setup_debug_logging()
    run_system_diagnostics()
```

### Interactive Debugging Session

```python
def start_debug_session():
    """Start interactive debugging session."""
    
    print("Starting QSFL-CAAD debug session...")
    print("Available commands:")
    print("  test_crypto - Test cryptographic operations")
    print("  test_auth - Test authentication")
    print("  test_anomaly - Test anomaly detection")
    print("  test_aggregation - Test model aggregation")
    print("  check_logs - Check recent log entries")
    print("  exit - Exit debug session")
    
    while True:
        try:
            command = input("\nDebug> ").strip().lower()
            
            if command == "exit":
                break
            elif command == "test_crypto":
                test_crypto_interactive()
            elif command == "test_auth":
                test_auth_interactive()
            elif command == "test_anomaly":
                test_anomaly_interactive()
            elif command == "test_aggregation":
                test_aggregation_interactive()
            elif command == "check_logs":
                check_logs_interactive()
            else:
                print(f"Unknown command: {command}")
                
        except KeyboardInterrupt:
            print("\nExiting debug session...")
            break
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

def test_crypto_interactive():
    """Interactive crypto testing."""
    
    print("Testing cryptographic operations...")
    
    try:
        from pq_security.kyber import KyberKeyExchange
        from pq_security.dilithium import DilithiumSigner
        
        # Test Kyber
        kyber = KyberKeyExchange()
        pub, priv = kyber.generate_keypair()
        ciphertext, secret1 = kyber.encapsulate(pub)
        secret2 = kyber.decapsulate(ciphertext, priv)
        
        if secret1 == secret2:
            print("✅ Kyber key exchange working")
        else:
            print("❌ Kyber key exchange failed")
        
        # Test Dilithium
        dilithium = DilithiumSigner()
        pub, priv = dilithium.generate_keypair()
        message = b"test message"
        signature = dilithium.sign(message, priv)
        is_valid = dilithium.verify(message, signature, pub)
        
        if is_valid:
            print("✅ Dilithium signatures working")
        else:
            print("❌ Dilithium signatures failed")
            
    except Exception as e:
        print(f"❌ Crypto test failed: {e}")
        traceback.print_exc()
```

---

## Getting Help

### Community Resources

1. **GitHub Issues**: Report bugs and request features
   - Repository: https://github.com/your-org/qsfl-caad
   - Use issue templates for bug reports and feature requests

2. **Documentation**: Comprehensive guides and API reference
   - API Documentation: `docs/API_DOCUMENTATION.md`
   - Usage Examples: `docs/USAGE_EXAMPLES.md`
   - Developer Guide: `docs/DEVELOPER_GUIDE.md`

3. **Discussion Forums**: Community discussions and Q&A
   - GitHub Discussions for general questions
   - Stack Overflow with tag `qsfl-caad`

### Reporting Issues

When reporting issues, please include:

1. **Environment Information**
   ```bash
   python --version
   pip list | grep -E "(tensorflow|numpy|scikit-learn|pqcrypto)"
   uname -a  # On Linux/macOS
   ```

2. **Minimal Reproducible Example**
   ```python
   # Minimal code that reproduces the issue
   from qsfl_caad import QSFLSystem
   
   system = QSFLSystem()
   # Steps that cause the issue
   ```

3. **Error Messages and Stack Traces**
   ```
   Full error message and stack trace
   ```

4. **Configuration Files** (remove sensitive information)
   ```yaml
   # Relevant parts of config.yaml
   ```

### Professional Support

For enterprise deployments and professional support:

1. **Consulting Services**: Architecture design and implementation guidance
2. **Training**: Team training on QSFL-CAAD usage and development
3. **Custom Development**: Feature development and integration services
4. **Security Audits**: Professional security assessment and recommendations

Contact: support@qsfl-caad.org

### Contributing

Help improve QSFL-CAAD by contributing:

1. **Bug Fixes**: Fix issues and submit pull requests
2. **Feature Development**: Implement new features
3. **Documentation**: Improve documentation and examples
4. **Testing**: Add test cases and improve coverage
5. **Performance**: Optimize algorithms and implementations

See `docs/DEVELOPER_GUIDE.md` for contribution guidelines.

---

This troubleshooting guide covers the most common issues encountered with QSFL-CAAD. If you encounter an issue not covered here, please report it through the appropriate channels so we can help you and improve this guide for future users.