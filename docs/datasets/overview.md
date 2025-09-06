# Datasets

This page documents the Dataset Registry for MedAISure and how to list and inspect datasets via the CLI and Python API.

The registry currently includes:

- medaisure-core: 200-task composition (Diagnostics 100, Summarization 50, Communication 50)
- medaisure-hard (planned)
- medaisure-specialty (planned)
- medaisure-multimodal (planned)

Registry entries are stored in `bench/data/datasets/registry.json` and validated by the `DatasetMeta` schema.

## CLI Usage (Typer)

List all datasets:

```bash
python -m bench.cli_typer list-datasets
```

JSON output:

```bash
python -m bench.cli_typer list-datasets --json
```

Show composition in the table view:

```bash
python -m bench.cli_typer list-datasets --with-composition
```

Example (truncated):

```text
Datasets
ID               Name               Planned  Size  Categories                                  Composition
medaisure-core   MedAISure Core     no       200   diagnostics, summarization, communication   {"diagnostics": 100, "summarization": 50, "communication": 50}
...
```

Captured CLI output (plain-text fallback):

```text
--8<-- "snippets/list_datasets_with_composition.txt"
```

Show a single dataset:

```bash
python -m bench.cli_typer show-dataset medaisure-core
```

Use a custom registry path if needed:

```bash
python -m bench.cli_typer list-datasets --registry-path bench/data/datasets/registry.json
```

## Python API

Use the high-level API from `bench.data`:

```python
from bench.data import get_default_registry

reg = get_default_registry()
all_datasets = reg.list()
core = reg.get("medaisure-core")
print(core.name, core.size, core.composition)
```

You can also initialize a custom registry path:

```python
from bench.data import DatasetRegistry

reg = DatasetRegistry(registry_path="path/to/registry.json")
print([d.id for d in reg.list()])
```

## Loader Stub (MedAISure-Core)

A simple loader stub is available at `bench/data/datasets/medaisure_core.py`:

```python
from bench.data.datasets.medaisure_core import get_metadata, load_examples

meta = get_metadata()
examples = load_examples(limit=10)  # currently returns an empty list (stub)
```

You can also render example listing rows (for documentation) that match the
structure returned by the task listing command:

```python
from bench.data.datasets import medaisure_core as core

# Validate the core registry entry (raises if inconsistent)
core.validate()

# Composition and categories
print(core.get_composition())
print(core.list_categories())

# Example listing rows (placeholders)
rows = core.example_listing_rows(limit=5)
for r in rows:
    print(r)
```

Doctest (pycon) style quick check:

```pycon
>>> from bench.data.datasets import medaisure_core as core
>>> core.validate()  # does not raise
>>> rows = core.example_listing_rows(limit=2)
>>> isinstance(rows, list)
True
>>> len(rows) > 0
True
>>> sorted(rows[0].keys()) == ['description','difficulty','file','metrics','name','num_examples','task_id']
True

```

## Schema

The registry entries are validated with `DatasetMeta` (Pydantic), defined in `bench/data/dataset_registry.py`.

Key fields:

- `id`: unique identifier (lowercase kebab-case enforced)
- `name`: display name
- `description`: optional text
- `size`: number of tasks/examples in the dataset
- `task_categories`: list of category names
- `source_links`: list of source URLs
- `composition`: category counts (must sum to `size` when both provided)
- `planned`: mark dataset as planned/placeholder

Export the JSON Schema:

```python
from bench.data import DatasetRegistry

schema = DatasetRegistry().json_schema()
```
