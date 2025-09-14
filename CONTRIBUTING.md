# Contributing to QSFL-CAAD

Thank you for your interest in contributing to the Quantum-Safe Federated Learning with Comprehensive Anomaly and Attack Detection (QSFL-CAAD) project! This document provides guidelines and information for contributors.

## 🚀 Quick Start

### Development Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/qsfl-caad.git
cd qsfl-caad
```

2. **Set up development environment:**
```bash
make install-dev
# or
python scripts/workflow_manager.py setup --env development
```

3. **Verify setup:**
```bash
make test
make run-dashboard
```

## 📋 Development Workflow

### Branch Strategy

We use **Git Flow** branching model:

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `hotfix/*` - Critical bug fixes
- `release/*` - Release preparation

### Workflow Steps

1. **Create feature branch:**
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

2. **Make changes and commit:**
```bash
# Make your changes
git add .
git commit -m "feat: add new anomaly detection algorithm"
```

3. **Run quality checks:**
```bash
make quality-check
make test
```

4. **Push and create PR:**
```bash
git push origin feature/your-feature-name
# Create Pull Request on GitHub
```

## 🔍 Code Quality Standards

### Code Style

We use automated code formatting and linting:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run formatting:
```bash
make format
```

Check code quality:
```bash
make quality-check
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code quality:
```bash
pre-commit install
```

### Type Hints

All new code should include type hints:
```python
def detect_anomaly(data: np.ndarray, threshold: float = 0.5) -> bool:
    """Detect anomalies in the given data."""
    pass
```

### Documentation

- Use Google-style docstrings
- Include type information
- Provide examples for complex functions
- Update README.md for significant changes

Example:
```python
def aggregate_models(models: List[ModelUpdate], weights: Optional[List[float]] = None) -> GlobalModel:
    """Aggregate federated learning models using weighted averaging.
    
    Args:
        models: List of model updates from clients
        weights: Optional weights for each model (defaults to equal weighting)
        
    Returns:
        Aggregated global model
        
    Raises:
        ValueError: If models list is empty or weights don't match models
        
    Example:
        >>> models = [model1, model2, model3]
        >>> weights = [0.5, 0.3, 0.2]
        >>> global_model = aggregate_models(models, weights)
    """
    pass
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── performance/    # Performance benchmarks
├── security/       # Security tests
└── fixtures/       # Test data and fixtures
```

### Writing Tests

1. **Unit Tests:**
```python
import pytest
from qsfl_caad.anomaly_detection import IsolationForestDetector

class TestIsolationForestDetector:
    def test_detect_anomaly_normal_data(self):
        detector = IsolationForestDetector()
        normal_data = np.random.normal(0, 1, (100, 10))
        result = detector.detect(normal_data)
        assert not result.is_anomaly
    
    def test_detect_anomaly_outlier_data(self):
        detector = IsolationForestDetector()
        outlier_data = np.array([[100, 100, 100, 100, 100]])
        result = detector.detect(outlier_data)
        assert result.is_anomaly
```

2. **Integration Tests:**
```python
def test_full_federated_learning_round():
    system = QSFLSystem()
    clients = [create_test_client(i) for i in range(5)]
    
    # Run complete FL round
    result = system.run_training_round(clients)
    
    assert result.success
    assert result.global_model is not None
    assert len(result.client_updates) == 5
```

3. **Performance Tests:**
```python
def test_anomaly_detection_performance(benchmark):
    detector = IsolationForestDetector()
    data = np.random.normal(0, 1, (1000, 50))
    
    result = benchmark(detector.detect, data)
    assert result.execution_time < 1.0  # seconds
```

### Test Coverage

- Maintain minimum 80% code coverage
- Focus on critical paths and edge cases
- Include both positive and negative test cases

Run tests:
```bash
make test              # All tests
make test-unit         # Unit tests only
make test-integration  # Integration tests
make test-performance  # Performance tests
```

## 🔒 Security Guidelines

### Security Best Practices

1. **Never commit secrets:**
   - Use environment variables
   - Add sensitive files to `.gitignore`
   - Use `.env.example` for configuration templates

2. **Input validation:**
   - Validate all external inputs
   - Use type hints and runtime validation
   - Sanitize data before processing

3. **Cryptographic operations:**
   - Use established libraries (cryptography, PyCryptodome)
   - Follow NIST recommendations
   - Implement proper key management

4. **Dependencies:**
   - Regularly update dependencies
   - Use `safety` to check for vulnerabilities
   - Pin dependency versions

### Security Testing

Run security checks:
```bash
make security-check
bandit -r qsfl_caad/
safety check
```

## 📊 Performance Guidelines

### Performance Considerations

1. **Algorithm Efficiency:**
   - Use vectorized operations (NumPy)
   - Avoid nested loops where possible
   - Consider memory usage for large datasets

2. **Profiling:**
   - Use `cProfile` for performance profiling
   - Benchmark critical functions
   - Monitor memory usage

3. **Scalability:**
   - Design for horizontal scaling
   - Use async/await for I/O operations
   - Consider distributed computing

Run performance tests:
```bash
make benchmark
make profile
```

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues
2. Reproduce the bug
3. Test with latest version
4. Gather system information

### Bug Report Template

```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- QSFL-CAAD: [e.g., 1.0.0]

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Any other relevant information
```

## 📝 Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples
```bash
feat(anomaly): add isolation forest detector
fix(auth): resolve token validation issue
docs(readme): update installation instructions
test(integration): add federated learning tests
```

## 🔄 Pull Request Process

### PR Checklist

- [ ] Branch is up to date with `develop`
- [ ] All tests pass
- [ ] Code coverage maintained
- [ ] Documentation updated
- [ ] Security checks pass
- [ ] Performance impact assessed
- [ ] Breaking changes documented

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added for new functionality
```

### Review Process

1. **Automated Checks:** CI/CD pipeline runs automatically
2. **Code Review:** At least one maintainer reviews
3. **Testing:** Reviewer tests functionality
4. **Approval:** Maintainer approves and merges

## 🏗️ Architecture Guidelines

### Project Structure

```
qsfl_caad/
├── __init__.py
├── client.py              # FL client implementation
├── system.py              # Main system orchestrator
├── anomaly_detection/     # Anomaly detection modules
├── auth/                  # Authentication & authorization
├── federated_learning/    # FL algorithms
├── monitoring/            # System monitoring
└── pq_security/          # Post-quantum cryptography
```

### Design Principles

1. **Modularity:** Separate concerns into distinct modules
2. **Extensibility:** Design for easy extension and customization
3. **Testability:** Write testable code with clear interfaces
4. **Performance:** Optimize critical paths
5. **Security:** Security by design, not as an afterthought

### Adding New Components

1. **Create module structure:**
```python
# new_component/__init__.py
from .main import NewComponent
from .interfaces import NewComponentInterface

__all__ = ['NewComponent', 'NewComponentInterface']
```

2. **Define interfaces:**
```python
# new_component/interfaces.py
from abc import ABC, abstractmethod
from typing import Protocol

class NewComponentInterface(Protocol):
    def process(self, data: Any) -> Any:
        """Process data according to component logic."""
        ...
```

3. **Implement component:**
```python
# new_component/main.py
from .interfaces import NewComponentInterface

class NewComponent(NewComponentInterface):
    def process(self, data: Any) -> Any:
        """Implementation of processing logic."""
        pass
```

4. **Add tests:**
```python
# tests/unit/test_new_component.py
import pytest
from qsfl_caad.new_component import NewComponent

class TestNewComponent:
    def test_process(self):
        component = NewComponent()
        result = component.process(test_data)
        assert result is not None
```

## 📚 Documentation

### Types of Documentation

1. **Code Documentation:** Docstrings and comments
2. **API Documentation:** Sphinx-generated docs
3. **User Guide:** Usage examples and tutorials
4. **Developer Guide:** Architecture and contribution guide

### Building Documentation

```bash
make docs
make docs-serve  # Serve locally at http://localhost:8000
```

### Documentation Standards

- Use clear, concise language
- Include code examples
- Keep documentation up to date
- Use proper Sphinx directives

## 🎯 Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- `MAJOR`: Breaking changes
- `MINOR`: New features (backward compatible)
- `PATCH`: Bug fixes (backward compatible)

### Release Steps

1. **Prepare release branch:**
```bash
git checkout develop
git pull origin develop
git checkout -b release/v1.1.0
```

2. **Update version and changelog:**
```bash
# Update version in pyproject.toml and __init__.py
# Update CHANGELOG.md
```

3. **Run full test suite:**
```bash
make ci-local
```

4. **Create release PR:**
```bash
git push origin release/v1.1.0
# Create PR to main branch
```

5. **Tag and deploy:**
```bash
git tag v1.1.0
git push origin v1.1.0
```

## 🤝 Community

### Communication Channels

- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** General questions and discussions
- **Email:** [maintainer@example.com] for private matters

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md).

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Project documentation

## 📞 Getting Help

### Resources

1. **Documentation:** [docs/](docs/)
2. **Examples:** [examples/](examples/)
3. **Tests:** [tests/](tests/) for usage examples
4. **Issues:** Search existing GitHub issues

### Contact

- **General Questions:** GitHub Discussions
- **Bug Reports:** GitHub Issues
- **Security Issues:** Email maintainers privately
- **Feature Requests:** GitHub Issues with feature template

---

Thank you for contributing to QSFL-CAAD! Your contributions help make federated learning more secure and robust. 🚀