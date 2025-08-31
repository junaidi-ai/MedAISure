"""
Demonstrate Hugging Face summarization with generation parameters using ModelRunner.
"""

import logging

from bench.evaluation.model_runner import ModelRunner


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mr = ModelRunner()

    model_name = "facebook/bart-large-cnn"
    mr.load_model(
        model_name,
        model_type="huggingface",
        hf_task="summarization",
        generation_kwargs={
            "max_new_tokens": 96,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
        },
        tokenizer_kwargs={"use_fast": True},
        device=-1,
    )

    inputs = [
        {
            "document": (
                "A 65-year-old male with a history of hypertension presents with shortness of breath. "
                "The chest X-ray shows cardiomegaly and pulmonary edema consistent with heart failure."
            )
        },
        {
            "text": "This is a demonstration of summarization with generation parameters."
        },
    ]

    outputs = mr.run_model(model_name, inputs, batch_size=2)
    for out in outputs:
        print(out["summary"])  # standardized output key for summarization


if __name__ == "__main__":
    main()
