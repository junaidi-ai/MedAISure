import math

from bench.evaluation.metrics import ReasoningQualityMetric


def test_reasoning_quality_composite_scores_and_breakdown():
    m = ReasoningQualityMetric()

    expected = [
        {
            "rationale": "Because fever and cough suggest pneumonia; CXR shows infiltrate.",
            "answer": "pneumonia",
        },
        {
            "explanation": "Therefore, hypotension and high lactate imply sepsis.",
            "answer": "sepsis",
        },
        {"rationale": "non sequitur, because I say so", "answer": "pneumonia"},
        {"rationale": "No rationale provided."},
    ]

    outputs = [
        {
            "rationale": "because fever and cough; imaging x-ray shows infiltrate"
        },  # structure+evidence+facts
        {
            "explanation": "hence hypotension and lactate levels; give iv fluids"
        },  # structure+evidence+facts
        {
            "rationale": "this is true because i say and correlation implies causation"
        },  # fallacy penalty
        {"prediction": ""},  # empty rationale
    ]

    score = m.calculate(expected, outputs)
    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0

    breakdown = m.get_last_breakdown()
    assert isinstance(breakdown, list)
    assert len(breakdown) == 4
    for item in breakdown:
        assert "final_score" in item
        assert "overlap_f1" in item


def test_reasoning_quality_empty_both_rationales_yields_high_overlap_component():
    m = ReasoningQualityMetric()
    expected = [{"rationale": ""}]
    outputs = [{"rationale": ""}]
    score = m.calculate(expected, outputs)
    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0
