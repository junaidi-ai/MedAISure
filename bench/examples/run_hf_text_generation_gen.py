"""
Demonstrate Hugging Face text-generation with generation parameters using ModelRunner.
"""

import logging

from bench.evaluation.model_runner import ModelRunner


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mr = ModelRunner()

    model_name = "gpt2"
    mr.load_model(
        model_name,
        model_type="huggingface",
        hf_task="text-generation",
        generation_kwargs={
            "max_new_tokens": 64,
            "temperature": 0.8,
            "do_sample": True,
            "top_k": 50,
        },
        tokenizer_kwargs={"use_fast": True},
        device=-1,
    )

    inputs = [
        {
            "text": "Patient presents with chest pain and shortness of breath. The initial assessment"
        },
        {"text": "In summary, the treatment plan should include"},
    ]

    outputs = mr.run_model(model_name, inputs, batch_size=2)
    for out in outputs:
        print(out.get("text") or out.get("prediction") or out)


if __name__ == "__main__":
    main()
