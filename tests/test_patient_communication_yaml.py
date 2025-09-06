from bench.evaluation.task_loader import TaskLoader
from bench.models.task_types import TaskType


def test_patient_communication_yaml_loads_and_validates():
    tl = TaskLoader(tasks_dir="bench/tasks")
    # Load both new tasks
    basic = tl.load_task("patient_communication_basic")
    triage = tl.load_task("patient_communication_triage")

    # Types and basic fields
    assert basic.task_type == TaskType.COMMUNICATION
    assert triage.task_type == TaskType.COMMUNICATION

    # Schemas have required fields
    assert "required" in (basic.input_schema or {})
    assert "required" in (basic.output_schema or {})

    # Derived inputs/expected_outputs should not be empty
    assert basic.inputs and isinstance(basic.inputs, list)
    assert triage.inputs and isinstance(triage.inputs, list)

    # Expected outputs should include enriched fields for metrics
    # summary/note are added for communication tasks to support rouge_l/clinical_relevance
    assert isinstance(basic.expected_outputs, list) and len(
        basic.expected_outputs
    ) == len(basic.inputs)
    assert isinstance(triage.expected_outputs, list) and len(
        triage.expected_outputs
    ) == len(triage.inputs)

    # Sample one pair to ensure enrichment exists (may vary by row)
    found_enrichment = any(
        (isinstance(eo, dict) and ("summary" in eo or "note" in eo))
        for eo in basic.expected_outputs
    )
    assert found_enrichment
