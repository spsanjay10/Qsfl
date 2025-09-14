# QSFL-CAAD System Demonstration Guide

This guide provides comprehensive instructions for demonstrating and evaluating the Quantum-Safe Federated Learning with Client Anomaly and Attack Detection (QSFL-CAAD) system.

## Overview

The QSFL-CAAD system provides:
- **Post-quantum cryptographic security** for future-proof protection
- **Client authentication and authorization** using quantum-safe digital signatures
- **AI-driven anomaly detection** to identify malicious client behavior
- **Secure federated learning** with attack mitigation
- **Real-time monitoring and alerting** for system security

## Quick Start

### Prerequisites

```bash
# Install required dependencies
pip install -r requirements.txt

# Optional: Install visualization dependencies
pip install dash plotly matplotlib seaborn
```

### Basic Demonstration

Run the basic system demonstration:

```bash
python demo_system_capabilities.py --scenario mixed --verbose
```

This will demonstrate all core system capabilities with a mixed scenario of honest and malicious clients.

## Demonstration Scripts

### 1. System Capabilities Demo (`demo_system_capabilities.py`)

Demonstrates the complete system functionality including all security and federated learning features.

#### Usage

```bash
# Basic mixed scenario (default)
python demo_system_capabilities.py

# Specific scenarios
python demo_system_capabilities.py --scenario basic    # Honest clients only
python demo_system_capabilities.py --scenario attack   # Heavy attack scenario
python demo_system_capabilities.py --scenario all      # All scenarios

# With detailed output and result saving
python demo_system_capabilities.py --scenario mixed --verbose --save-results
```

#### Scenarios

- **Basic**: 5 honest clients, 0 malicious clients, 3 rounds
- **Mixed**: 7 honest clients, 3 malicious clients, 5 rounds  
- **Attack**: 3 honest clients, 7 malicious clients, 3 rounds
- **All**: Runs all scenarios sequentially

#### Output

The demonstration will show:
1. **Post-quantum security** - Key generation, signing, verification
2. **Client authentication** - Registration, authentication, revocation
3. **Anomaly detection** - Training, detection, explanation generation
4. **Federated learning** - Complete training rounds with security
5. **Monitoring & alerts** - Event logging, metrics collection, alerting

### 2. Evaluation Metrics (`evaluation_metrics.py`)

Provides comprehensive quantitative evaluation of system performance and security.

#### Usage

```bash
# Basic evaluation with default settings
python evaluation_metrics.py

# Custom output directory and format
python evaluation_metrics.py --output results --format pdf

# Skip plots or data saving
python evaluation_metrics.py --no-plots --no-save

# Use custom configuration
python evaluation_metrics.py --config evaluation_config.json
```

#### Configuration File Format

```json
{
  "evaluation": {
    "num_honest_clients": 20,
    "num_malicious_clients": 5,
    "num_rounds": 10,
    "num_trials": 5,
    "model_size": "medium",
    "attack_types": ["gradient_poisoning", "model_replacement", "byzantine", "backdoor"],
    "attack_intensities": [0.5, 1.0, 2.0, 5.0],
    "detection_threshold": 0.5
  },
  "performance": {
    "max_clients_test": 100,
    "timeout_seconds": 30,
    "memory_limit_mb": 1000,
    "cpu_limit_percent": 80
  },
  "output": {
    "save_plots": true,
    "save_raw_data": true,
    "plot_format": "png",
    "plot_dpi": 300
  }
}
```

#### Metrics Generated

**Security Metrics:**
- Detection accuracy, precision, recall, F1-score
- False positive/negative rates
- AUC-ROC score
- Attack-specific detection rates
- Time to detection
- Quarantine effectiveness

**Performance Metrics:**
- Throughput (updates/second)
- Latency (authentication, detection, aggregation)
- Resource usage (CPU, memory)
- Scalability limits

**Model Quality Metrics:**
- Convergence speed
- Final accuracy
- Robustness under attack
- Byzantine resilience

**Reliability Metrics:**
- System uptime
- Error rates
- Recovery time
- Component failure tolerance

### 3. Visualization Dashboard (`visualization_dashboard.py`)

Creates interactive and static visualizations for system analysis.

#### Usage

```bash
# Generate static visualizations only
python visualization_dashboard.py --static-only

# Run interactive dashboard
python visualization_dashboard.py --port 8050

# Demo mode with simulated real-time data
python visualization_dashboard.py --demo-mode

# Custom data directory
python visualization_dashboard.py --data-dir my_visualizations
```

#### Static Visualizations

Generated plots include:
- **System Overview Dashboard**: Multi-panel view of system metrics
- **Attack Analysis Dashboard**: Attack patterns and detection performance
- **Performance Correlation Heatmap**: Relationships between system metrics

#### Interactive Dashboard

Features:
- Real-time system monitoring
- Configurable time ranges and refresh rates
- Interactive charts and graphs
- Attack event timeline
- Performance metrics tracking

Access at: `http://localhost:8050` (or specified port)

## Integration Tests

### Running Integration Tests

```bash
# Run all integration tests
python -m pytest tests/test_integration_e2e.py -v

# Run specific test categories
python -m pytest tests/test_attack_simulation.py -v
python -m pytest tests/test_performance_benchmarks.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Categories

1. **End-to-End Integration** (`test_integration_e2e.py`)
   - Complete system workflow tests
   - Component integration validation
   - Security and performance integration

2. **Attack Simulation** (`test_attack_simulation.py`)
   - Gradient poisoning attacks
   - Model replacement attacks
   - Byzantine attacks
   - Backdoor attacks
   - Coordinated attacks

3. **Performance Benchmarks** (`test_performance_benchmarks.py`)
   - Cryptographic operation performance
   - Anomaly detection performance
   - Federated learning scalability
   - System resource usage

## Example Demonstration Scenarios

### Scenario 1: Security Showcase

Demonstrate the system's security capabilities:

```bash
# Run security-focused demonstration
python demo_system_capabilities.py --scenario attack --verbose

# Evaluate security metrics
python evaluation_metrics.py --config security_config.json

# Visualize attack patterns
python visualization_dashboard.py --static-only
```

This scenario shows:
- High attack load (70% malicious clients)
- Detection and mitigation in action
- System resilience under attack
- Security metrics and analysis

### Scenario 2: Performance Analysis

Analyze system performance and scalability:

```bash
# Run performance benchmarks
python -m pytest tests/test_performance_benchmarks.py::TestSystemScalability -v

# Generate performance evaluation
python evaluation_metrics.py --output performance_results

# Create performance visualizations
python visualization_dashboard.py --data-dir performance_viz --static-only
```

### Scenario 3: Real-time Monitoring

Demonstrate real-time monitoring capabilities:

```bash
# Start interactive dashboard
python visualization_dashboard.py --demo-mode --port 8050
```

Then open `http://localhost:8050` to see:
- Live system metrics
- Real-time attack detection
- Performance monitoring
- Interactive analysis tools

## Interpreting Results

### Security Metrics

- **Detection Accuracy > 0.85**: Good overall detection performance
- **Precision > 0.80**: Low false positive rate
- **Recall > 0.75**: Catches most attacks
- **F1-Score > 0.80**: Balanced precision and recall

### Performance Metrics

- **Throughput > 50 updates/sec**: Acceptable performance
- **Round Time < 5 seconds**: Good responsiveness
- **CPU Usage < 70%**: Efficient resource usage
- **Memory Usage < 500MB**: Reasonable memory footprint

### Model Quality

- **Convergence < 15 rounds**: Fast convergence
- **Accuracy > 0.85**: Good model quality
- **Degradation < 0.10**: Resilient to attacks
- **Stability > 0.80**: Consistent performance

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   ```

2. **Visualization Issues**
   ```bash
   # Install visualization dependencies
   pip install dash plotly matplotlib seaborn
   ```

3. **Performance Issues**
   - Reduce number of clients in configuration
   - Use smaller model sizes
   - Decrease number of rounds

4. **Memory Issues**
   - Monitor system resources
   - Reduce batch sizes
   - Use smaller datasets

### Debug Mode

Enable verbose output for debugging:

```bash
python demo_system_capabilities.py --verbose
python evaluation_metrics.py --config debug_config.json
```

## Advanced Usage

### Custom Attack Types

Extend the system with custom attack implementations:

```python
from tests.test_utils import ModelUpdateGenerator

# Create custom attack generator
generator = ModelUpdateGenerator()
custom_attack = generator.generate_malicious_update(
    client_id="custom_attacker",
    round_id="test_round",
    model_shape=model_shape,
    attack_type="custom_attack",
    attack_intensity=2.0
)
```

### Custom Evaluation Metrics

Add custom metrics to the evaluation framework:

```python
from evaluation_metrics import MetricsEvaluator

class CustomEvaluator(MetricsEvaluator):
    def evaluate_custom_metrics(self):
        # Implement custom evaluation logic
        pass
```

### Integration with External Systems

The demonstration scripts can be integrated with external monitoring and logging systems:

```python
# Custom logging integration
import logging
logging.basicConfig(level=logging.INFO)

# Custom metrics export
import json
with open('metrics_export.json', 'w') as f:
    json.dump(evaluation_results, f)
```

## Best Practices

1. **Run demonstrations in isolated environments**
2. **Monitor system resources during evaluation**
3. **Save results for comparison and analysis**
4. **Use version control for configuration files**
5. **Document any modifications or customizations**

## Support and Documentation

- **System Architecture**: See `design.md`
- **API Documentation**: Generated from code docstrings
- **Test Documentation**: See individual test files
- **Configuration Reference**: See example configuration files

For additional support or questions about the demonstration system, refer to the main project documentation or contact the development team.