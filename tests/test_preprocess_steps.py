from __future__ import annotations

import json
import random
from typing import Dict, Any

from bench.data.preprocess import (
    DataPreprocessor,
    MedicalTextNormalizeStep,
    EntityMaskStep,
    MissingDataHandlerStep,
    AugmentationStep,
)


def test_medical_text_normalize_basic():
    dp = DataPreprocessor()
    dp.add_step(
        MedicalTextNormalizeStep(keys=["text"], replacements={"bp": "blood pressure"})
    )

    item = {"text": "  BP is High  \n", "other": "UNCHANGED"}
    out = dp.process(item)

    assert out["text"] == "blood pressure is high"
    # other key remains unchanged
    assert out["other"] == "UNCHANGED"


def test_entity_mask_step_masks_common_entities():
    dp = DataPreprocessor()
    dp.add_step(EntityMaskStep(keys=["note"]))

    item = {
        "note": "Patient MRN 12345678 on 2024-01-02 called +1 555-123-4567. Email: test@example.com",
    }
    out = dp.process(item)

    assert "<ENT>" in out["note"]
    # ensure multiple patterns replaced
    assert "example.com" not in out["note"]


def test_missing_data_handler_fill_and_drop():
    dp = DataPreprocessor()
    step = MissingDataHandlerStep(
        fill_defaults={"age": 0, "gender": "unknown"},
        drop_if_missing_any=["id"],
        drop_keys=["tmp"],
    )
    dp.add_step(step)

    # fill defaults
    out1 = dp.process({"id": 1, "tmp": "", "note": "ok"})
    assert out1["age"] == 0
    assert out1["gender"] == "unknown"
    assert "tmp" not in out1

    # missing required -> dropped sentinel
    out2 = dp.process({"note": "no id"})
    assert out2.get("__DROPPED__") is True or out2.get("__DROP_ITEM__") is True

    # ensure process_batch drops by default
    items = [{"id": 1}, {"note": "no id"}, {"id": 2}]
    res = dp.process_batch(items)
    assert all("__DROPPED__" not in r for r in res)
    assert len(res) == 2


def test_augmentation_step_dropout_and_synonym():
    dp = DataPreprocessor()
    # deterministic for dropout test
    random.seed(42)
    dp.add_step(
        AugmentationStep(
            keys=["t"], token_dropout_prob=0.5, synonym_map={"fever": "pyrexia"}
        )
    )

    item = {"t": "patient has fever and cough"}
    out = dp.process(item)

    # synonym replacement should occur for 'fever'
    assert "pyrexia" in out["t"] or "fever" not in out["t"]


def test_parallel_processing_and_metrics():
    dp = DataPreprocessor()

    def inc_x(d: Dict[str, Any]) -> Dict[str, Any]:
        dd = dict(d)
        dd["x"] = dd.get("x", 0) + 1
        return dd

    dp.add_step(inc_x)

    items = [{"x": i} for i in range(10)]
    res_serial = dp.process_batch(items, parallel=False)
    res_parallel = dp.process_batch(items, parallel=True, max_workers=4)

    assert res_serial == res_parallel
    assert dp.metrics["runs"] >= 2
    assert dp.metrics["items_processed"] >= 20
    assert any(v["count"] >= 10 for v in dp.metrics["per_step"].values())


def test_pipeline_serialization_roundtrip():
    dp = DataPreprocessor()
    dp.add_steps(
        [
            MedicalTextNormalizeStep(keys=["text"]),
            EntityMaskStep(keys=["text"]),
        ]
    )

    payload = {"text": "Visit 01/01/2020, MRN 999999"}
    out1 = dp.process(payload)

    s = dp.to_json()
    dp2 = DataPreprocessor.from_json(s)
    out2 = dp2.process(payload)

    assert out1 == out2
    assert json.loads(s)["version"] == 1
