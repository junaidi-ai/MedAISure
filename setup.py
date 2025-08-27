from pathlib import Path

from setuptools import find_packages, setup

# Read the contents of README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="medaisure-benchmark",
    version="0.1.0",
    author="Kresna Sucandra and Junaidi AI Team",
    author_email="kresnasucandra@gmail.com",
    description="A benchmark framework for evaluating medical AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/junaidi-ai/MedAISure",
    # Package discovery
    packages=find_packages(include=["bench", "bench.*"]),
    package_data={
        "bench": ["tasks/*.yaml", "tasks/*.yml", "tasks/*.json"],
    },
    include_package_data=True,
    zip_safe=False,
    # Dependencies
    install_requires=requirements,
    python_requires=">=3.8",
    # Development dependencies
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.910",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    # Metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    # Entry points
    entry_points={
        "console_scripts": [
            "medaisure-benchmark=bench.cli:main",
        ],
    },
    # Other
    keywords=["medical", "ai", "benchmark", "evaluation", "nlp", "healthcare"],
    project_urls={
        "Bug Reports": "https://github.com/junaidi-ai/MedAISure/issues",
        "Source": "https://github.com/junaidi-ai/MedAISure",
    },
)
