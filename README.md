# QSFL-CAAD: Quantum-Safe Federated Learning with Client Anomaly and Attack Detection

A secure federated learning system designed to protect against both quantum computing threats and malicious client attacks. The system combines post-quantum cryptography, client authentication, and AI-driven anomaly detection to ensure secure and robust distributed machine learning.

## Features

- **Post-Quantum Security**: CRYSTALS-Kyber key exchange and CRYSTALS-Dilithium digital signatures
- **Client Authentication**: Quantum-safe credential management and verification
- **AI-Driven Anomaly Detection**: Isolation Forest with SHAP explanations for malicious update detection
- **Secure Aggregation**: Reputation-based model aggregation with security integration
- **Comprehensive Monitoring**: Security event logging, metrics collection, and alerting

## Architecture

The system is organized into five core modules:

- `pq_security/`: Post-quantum cryptographic operations
- `auth/`: Client authentication and credential management
- `anomaly_detection/`: AI-driven anomaly detection engine
- `federated_learning/`: Federated learning coordination and aggregation
- `monitoring/`: System monitoring, logging, and alerting

## Installation

```bash
# Clone the repository
git clone https://github.com/qsfl-caad/qsfl-caad.git
cd qsfl-caad

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
from config.settings import get_config
from federated_learning.server import SecureFederatedServer

# Load configuration
config = get_config()

# Initialize secure federated learning server
server = SecureFederatedServer(config)

# Start training
server.start_training()
```

## Configuration

Configuration can be managed through:

1. Environment variables (prefixed with `QSFL_`)
2. Configuration file (`config/config.yaml`)
3. Direct configuration object modification

See `config/settings.py` for all available configuration options.

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .

# Type checking
mypy .

# Install pre-commit hooks
pre-commit install
```

## Security Considerations

- Change default secret keys in production
- Use hardware security modules (HSM) for key storage in production
- Regularly rotate cryptographic keys
- Monitor security events and alerts
- Keep cryptographic libraries updated

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{qsfl_caad,
  title={QSFL-CAAD: Quantum-Safe Federated Learning with Client Anomaly and Attack Detection},
  author={QSFL-CAAD Development Team},
  year={2024},
  url={https://github.com/qsfl-caad/qsfl-caad}
}
```