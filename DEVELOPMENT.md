# MEDDSAI Benchmark - Development Guide

This document provides information for developers working on the MEDDSAI Benchmark framework.

## 🛠️ Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/meddsai/meddsai-benchmark.git
   cd meddsai-benchmark
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install package in development mode
   ```

## 🧪 Running Tests

Run the full test suite with coverage:
```bash
pytest --cov=bench --cov-report=term-missing
```

Run a specific test file:
```bash
pytest tests/test_task_loader.py -v
```

Run tests with detailed logging:
```bash
pytest -v --log-cli-level=INFO
```

## 📚 Code Organization

```
bench/
├── evaluation/           # Core evaluation framework
│   ├── __init__.py      # Package exports
│   ├── task_loader.py   # Task loading and validation
│   ├── model_runner.py  # Model execution
│   ├── metric_calculator.py  # Metrics calculation
│   ├── result_aggregator.py  # Results collection
│   └── harness.py       # Main evaluation harness
├── tasks/               # Task definitions (YAML/JSON)
│   └── medical_nli_task.yaml
└── examples/            # Example scripts
    └── run_example.py
```

## 📝 Adding New Tasks

1. Create a new YAML file in the `bench/tasks/` directory following this structure:
   ```yaml
   name: "Task Name"
   description: "Task description"
   input_schema:
     type: "object"
     properties:
       # Define your input schema here
   output_schema:
     type: "object"
     properties:
       # Define your output schema here
   metrics:
     - name: "metric1"
       # Additional metric configuration
   dataset:
     # Your task dataset
   ```

2. Test your task:
   ```python
   from bench.evaluation import TaskLoader
   
   loader = TaskLoader(tasks_dir="bench/tasks")
   task = loader.load_task("your_task_name")
   print(f"Loaded task: {task.name}")
   ```

## 🧠 Adding New Models

1. Implement your model following the interface expected by `ModelRunner`.
2. Register your model in the `ModelRunner` class or pass it directly to the `evaluate` method.

## 📊 Adding New Metrics

1. Add your metric function to `MetricCalculator`:
   ```python
   def calculate_custom_metric(self, y_true, y_pred, **kwargs):
       # Your metric implementation
       return score
   
   # Register the metric
   metric_calculator.register_metric("custom_metric", calculate_custom_metric)
   ```

2. Use your metric in task definitions:
   ```yaml
   metrics:
     - name: "custom_metric"
       # Additional configuration
   ```

## 🧹 Code Style

We use `black` for code formatting and `isort` for import sorting:

```bash
black bench/
isort bench/
```

## 📦 Building the Package

```bash
python setup.py sdist bdist_wheel
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
