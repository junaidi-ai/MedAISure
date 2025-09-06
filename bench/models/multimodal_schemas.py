"""Draft schemas for future multimodal tasks (scaffold only).

This module defines minimal input/output schema constants for planned
multimodal tasks. These are not imported elsewhere by default and serve
as documentation and scaffolding only.
"""

# Image-Text QA (VQA-style)
IMAGE_TEXT_QA_INPUT_SCHEMA = {"required": ["question", "image_path"]}
IMAGE_TEXT_QA_OUTPUT_SCHEMA = {"required": ["answer"]}

# Clinical Report with Image Context
REPORT_WITH_IMAGE_INPUT_SCHEMA = {"required": ["document", "image_path"]}
REPORT_WITH_IMAGE_OUTPUT_SCHEMA = {"required": ["summary"]}
