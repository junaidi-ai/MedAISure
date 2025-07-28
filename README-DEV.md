# MEDDSAI Benchmark - Developer Documentation

This document provides technical information for developers working on the MEDDSAI benchmark project. It complements the [main README](README.md) by focusing on development workflows, architecture, and technical implementation details.

## Table of Contents
- [Development Environment Setup](#development-environment-setup)
- [Project Architecture](#project-architecture)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Evaluation Framework](#evaluation-framework)
- [Code Style and Standards](#code-style-and-standards)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Development Environment Setup

### Prerequisites
- **Python**: Version 3.8+ (3.10 recommended)
- **Docker**: Latest stable version
- **Git**: Latest stable version
- **GPU** (optional): CUDA-compatible for model training/inference acceleration

### Setting Up a Development Environment

1. **Clone the repository**:
   ```bash
   git clone git@github.com:meddsai/meddsai-benchmark.git
   cd meddsai-benchmark
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Project Architecture

The MEDDSAI benchmark is structured as follows:

```
meddsai-benchmark/
├── meddsai/                 # Main package
│   ├── __init__.py          
│   ├── harness/             # Evaluation harness
│   │   ├── __init__.py
│   │   ├── run_evaluation.py
│   │   └── utils/
│   ├── metrics/             # Evaluation metrics implementation
│   ├── tasks/               # Task definitions and implementations
│   │   ├── diagnostic/      # Diagnostic reasoning tasks
│   │   ├── qa/              # Question-answering tasks
│   │   ├── summarization/   # Clinical summarization tasks
│   │   └── communication/   # Patient communication tasks
│   └── utils/               # Common utilities
├── datasets/                # Dataset loading and processing
├── tests/                   # Unit and integration tests
├── examples/                # Example notebooks and scripts
├── docker/                  # Docker configurations
├── scripts/                 # Utility scripts
├── docs/                    # Documentation
└── evaluation_results/      # (gitignored) Generated evaluation results
```

### Core Components

1. **Evaluation Harness**: The central evaluation framework that orchestrates task execution, model evaluation, and metric calculation.

2. **Tasks**: Modular task implementations that define inputs, expected outputs, and evaluation procedures.

3. **Metrics**: Implementation of various evaluation metrics specific to medical domain tasks.

4. **Docker Environment**: Containerized environment for consistent evaluation across platforms.

## Development Workflow

### Feature Development

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement your changes**:
   - Follow the code style guidelines
   - Add appropriate unit tests
   - Update documentation as needed

3. **Run tests locally**:
   ```bash
   pytest tests/
   ```

4. **Submit a pull request** to the main branch

### Task Development

To add a new task to the benchmark:

1. Create a new task module in the appropriate task directory
2. Implement the required task interfaces (see example tasks)
3. Add task metadata to the task registry
4. Add test cases to validate task functionality
5. Update documentation to include the new task

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_module.py

# Run with coverage report
pytest --cov=meddsai
```

### Test Organization

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test interactions between components
- **End-to-End Tests**: Test the complete evaluation pipeline
- **Task Validation Tests**: Validate that tasks meet the required standards

## Evaluation Framework

The evaluation framework is designed to be extensible and modular:

### Key Components

1. **Task Loader**: Loads task definitions and data from the dataset sources
2. **Model Runner**: Interface for running predictions on models
3. **Metric Calculator**: Computes evaluation metrics for model outputs
4. **Result Aggregator**: Combines results across tasks and metrics

### Adding a New Evaluation Metric

1. Create a new metric module in `meddsai/metrics/`
2. Implement the metric interface
3. Register the metric in the metrics registry
4. Add tests for the new metric

## Code Style and Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) guidelines
- Use type annotations for function parameters and return types
- Maximum line length: 100 characters

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1, param2):
    """Short description of function.
    
    Longer description of function if needed.
    
    Args:
        param1 (type): Description of param1.
        param2 (type): Description of param2.
        
    Returns:
        return_type: Description of return value.
        
    Raises:
        ExceptionType: When and why this exception is raised.
    """
    # Function implementation
```

### Commit Messages

Follow conventional commits format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding or modifying tests
- `chore:` for maintenance tasks

## Pull Request Process

1. **Create a PR from your feature branch** to the main branch
2. **PR Description**:
   - Clearly describe the changes made
   - Reference any related issues
   - Include steps to test the changes
3. **Code Review**:
   - At least one core developer must approve
   - All CI checks must pass
4. **Merging**:
   - Squash and merge is preferred for cleaner history
   - Delete branch after merging

## Release Process

### Versioning

MEDDSAI follows [Semantic Versioning](https://semver.org/):
- **Major**: Breaking changes
- **Minor**: New features without breaking changes
- **Patch**: Bug fixes without breaking changes

### Release Steps

1. Update version number in `setup.py`
2. Update changelog
3. Create a release branch `release/vX.Y.Z`
4. Create a PR from the release branch to main
5. After PR is merged, tag the release:
   ```bash
   git tag -a vX.Y.Z -m "Version X.Y.Z"
   git push origin vX.Y.Z
   ```

## Federated Evaluation (Planned)

For the upcoming federated evaluation feature:

1. **Architecture**:
   - Secure communication layer
   - Local evaluation runners
   - Aggregation protocol
   - Result verification

2. **Security Considerations**:
   - End-to-end encryption
   - Model isolation
   - Data privacy preserving techniques

## Multimodal Tasks (Planned)

For the upcoming multimodal task support:

1. **Image Processing**:
   - Support for radiology images (X-ray, CT, MRI)
   - Image preprocessing pipelines
   - Model inputs for vision-language models

2. **Wearable Data**:
   - Time series processing
   - Signal normalization
   - Feature extraction

## Troubleshooting

Common development issues and their solutions:

1. **Docker permissions issues**:
   ```bash
   sudo groupadd docker
   sudo usermod -aG docker $USER
   # Log out and log back in
   ```

2. **CUDA compatibility issues**:
   - Check CUDA versions with `nvidia-smi` and `python -c "import torch; print(torch.version.cuda)"`
   - Install the correct PyTorch version for your CUDA version

3. **Dependency conflicts**:
   - Use `pip-compile` to generate locked requirements
   - Consider using separate environments for conflicting dependencies

## Community Resources

- **Technical Discussions**: Join our [Discord server](https://discord.gg/meddsai)
- **Development Meetings**: Bi-weekly on Thursdays, see calendar for details
- **Architecture Decisions**: Recorded in ADRs in the `docs/adr/` directory
