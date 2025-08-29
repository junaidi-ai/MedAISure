import json
from pathlib import Path

import pytest

from bench.models.task_types import (
    MedicalQATask,
    DiagnosticReasoningTask,
    ClinicalSummarizationTask,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "bench" / "tasks" / "data"


def test_medical_qa_task_load_and_eval(tmp_path):
    data_path = DATA_DIR / "qa_general.json"
    task = MedicalQATask(task_id="qa_general", description="Test QA")
    task.load_data(data_path)

    # Sanity checks
    assert len(task.dataset) >= 1
    assert task.input_schema["required"] == ["question"]
    assert task.output_schema["required"] == ["answer"]

    # Build perfect predictions
    preds = [{"answer": row["output"]["answer"]} for row in task.dataset]
    metrics = task.evaluate(preds)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["clinical_correctness"] == pytest.approx(1.0)


def test_diagnostic_reasoning_task_load_and_eval(tmp_path):
    # Write a small JSON file for loading test
    rows = [
        {"input": {"case": "fever and cough"}, "output": {"diagnosis": "pneumonia"}},
        {
            "input": {"case": "polyuria polydipsia"},
            "output": {"diagnosis": "diabetes mellitus"},
        },
    ]
    f = tmp_path / "diag.json"
    f.write_text(json.dumps(rows))

    task = DiagnosticReasoningTask(task_id="diag_basic", description="Test DR")
    task.load_data(f)

    assert len(task.dataset) == 2
    assert task.input_schema["required"] == ["case"]
    assert task.output_schema["required"] == ["diagnosis"]

    preds = [
        {"diagnosis": "pneumonia", "explanation": "because of fever and cough"},
        {"diagnosis": "diabetes mellitus", "explanation": "due to symptoms"},
    ]
    metrics = task.evaluate(preds)
    assert metrics["diagnostic_accuracy"] == pytest.approx(1.0)
    assert 0.0 <= metrics["reasoning_quality"] <= 1.0


def test_clinical_summarization_task_load_and_eval(tmp_path):
    rows = [
        {
            "input": {
                "document": "Patient with fever and cough. Treated with antibiotics."
            },
            "output": {"summary": "Fever and cough treated with antibiotics."},
        },
        {
            "input": {"document": "Chest pain with elevated troponin."},
            "output": {"summary": "Chest pain, high troponin."},
        },
    ]
    f = tmp_path / "summ.json"
    f.write_text(json.dumps(rows))

    task = ClinicalSummarizationTask(task_id="sum_basic", description="Test SUM")
    task.load_data(f)

    assert len(task.dataset) == 2
    assert task.input_schema["required"] == ["document"]
    assert task.output_schema["required"] == ["summary"]

    preds = [{"summary": r["output"]["summary"]} for r in rows]
    metrics = task.evaluate(preds)
    # Our lightweight metric returns under 'rouge_l'
    assert "rouge_l" in metrics
    assert 0.0 <= metrics["rouge_l"] <= 1.0
    assert 0.0 <= metrics["clinical_relevance"] <= 1.0
    assert 0.0 <= metrics["factual_consistency"] <= 1.0
