from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="meddsai-benchmark",
    version="0.1.0",
    author="Kresna Sucandra and MEDDSAI Team",
    author_email="kresnasucandra@unud.ac.id",
    description="A benchmark framework for evaluating medical AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meddsai/meddsai-benchmark",
    packages=find_packages(include=["bench*", "bench.evaluation*"]),
    package_data={
        "bench": ["tasks/*.yaml", "tasks/*.yml", "tasks/*.json"],
    },
    install_requires=open('requirements.txt').read().splitlines(),
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    entry_points={
        "console_scripts": [
            "meddsai-benchmark=bench.cli:main",
        ],
    },
)
