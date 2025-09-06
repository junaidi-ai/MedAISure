# Simple automation for docs capture and build

.PHONY: docs docs-serve docs-capture

# Capture deterministic CLI outputs and build the site
.docs-capture:
	@echo "Capturing CLI outputs for docs..."
	@python docs/scripts/capture_cli.py

# Build docs locally
.docs-build:
	@echo "Building MkDocs site..."
	@mkdocs build --strict

# Serve docs locally (blocking)
.docs-serve:
	@echo "Serving MkDocs site at http://127.0.0.1:8000 ..."
	@mkdocs serve

# Public targets

docs: .docs-capture .docs-build

# Run capture then serve
# Usage: make docs-serve

docs-serve: .docs-capture .docs-serve

# Capture only
# Usage: make docs-capture

docs-capture: .docs-capture
