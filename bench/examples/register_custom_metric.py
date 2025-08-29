"""
Show how to register and use a custom metric with MetricCalculator directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bench.evaluation.metric_calculator import MetricCalculator


def main() -> None:
    mc = MetricCalculator()

    def exact_match(y_true, y_pred, **kwargs):
        val = sum(int(t == p) for t, p in zip(y_true, y_pred)) / max(1, len(y_true))
        return float(val)

    mc.register_metric("exact_match", exact_match)

    preds = [{"label": "yes"}, {"label": "no"}, {"label": "maybe"}]
    refs = [{"label": "yes"}, {"label": "no"}, {"label": "no"}]

    results = mc.calculate_metrics(
        task_id="demo",
        predictions=preds,
        references=refs,
        metric_names=["accuracy", "exact_match"],
    )

    for name, res in results.items():
        print(name, res.value, res.metadata)


if __name__ == "__main__":
    main()
