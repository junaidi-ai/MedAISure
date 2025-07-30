# Contributing to MEDDSAI Benchmark

We welcome contributions from the community! Whether you're fixing bugs, adding new features, or improving documentation, your help is appreciated.

## 🛠 Development Setup

1. **Fork the repository** and clone it locally
2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   ```
4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## 🧪 Running Tests

Run the test suite with:
```bash
pytest
```

For coverage report:
```bash
pytest --cov=bench tests/
```

## 📝 Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **mypy** for static type checking
- **ruff** for linting

Run all code style checks:
```bash
pre-commit run --all-files
```

## 🚀 Making Changes

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and add tests
3. Run tests and ensure they pass
4. Commit your changes with a descriptive message
5. Push to your fork and open a Pull Request

## 📚 Documentation

- Update any relevant documentation
- Add docstrings for new functions/classes
- Update README.md if needed

## 🤝 Code Review Process

1. Open a Pull Request with a clear description
2. Ensure all CI checks pass
3. A maintainer will review your PR
4. Address any feedback
5. Once approved, your PR will be merged

## 🐛 Reporting Issues

Please open an issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Any relevant logs or screenshots

## 🙏 Thank You!

Your contributions help make MEDDSAI Benchmark better for everyone!
