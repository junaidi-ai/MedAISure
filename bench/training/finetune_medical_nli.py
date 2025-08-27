"""Fine-tuning script for medical NLI task."""

import logging
import os
from typing import Any, Dict, List, Tuple

import evaluate
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed for reproducibility
set_seed(42)

# Constants
# Using BioBERT which is pre-trained on biomedical literature
MODEL_NAME = "dmis-lab/biobert-v1.1"
TASK_FILE = "bench/tasks/medical_nli_task.yaml"
OUTPUT_DIR = "models/medical_nli_biobert"
BATCH_SIZE = 8  # Reduced batch size for better stability
NUM_EPOCHS = 15  # Increased epochs for better convergence with domain adaptation
LEARNING_RATE = 2e-5
MAX_SEQ_LENGTH = 128
TRAIN_TEST_SPLIT = 0.2  # 80% train, 20% test
RANDOM_SEED = 42  # For reproducibility

# Number of classes for NLI task
NUM_LABELS = 3

# Label mapping for NLI
LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def load_medical_nli_dataset(task_file: str) -> Dataset:
    """Load the medical NLI dataset from the task file."""
    import yaml

    with open(task_file, "r") as f:
        task_data = yaml.safe_load(f)

    # Extract examples from the dataset
    examples = task_data.get("dataset", [])

    # Convert to format expected by the model
    # Using Any for the list type since it will contain both strings and integers
    data: dict[str, Any] = {
        "premise": [],
        "hypothesis": [],
        "label": [],
    }

    for example in examples:
        data["premise"].append(example["premise"])
        data["hypothesis"].append(example["hypothesis"])
        data["label"].append(LABEL_MAP[example["label"]])

    return Dataset.from_dict(data)


def tokenize_function(
    examples: Dict[str, List[str]], tokenizer: AutoTokenizer
) -> Dict[str, List[List[int]]]:
    """Tokenize the examples for the NLI task."""
    result = tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        max_length=MAX_SEQ_LENGTH,
        truncation=True,
    )
    # Convert any tensors to lists for serialization
    return {k: v.tolist() if hasattr(v, "tolist") else v for k, v in result.items()}


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute metrics for evaluation."""
    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision", average="macro", zero_division=0)
    recall_metric = evaluate.load("recall", average="macro", zero_division=0)
    f1_metric = evaluate.load("f1", average="macro")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate metrics
    metrics = {}
    metrics.update(accuracy_metric.compute(predictions=predictions, references=labels))

    # Handle potential all-zero predictions for precision/recall
    if (
        len(np.unique(predictions)) > 1
    ):  # Only calculate if we have predictions in multiple classes
        metrics.update(
            precision_metric.compute(
                predictions=predictions, references=labels, average="macro"
            )
        )
        metrics.update(
            recall_metric.compute(
                predictions=predictions, references=labels, average="macro"
            )
        )
        metrics.update(
            f1_metric.compute(
                predictions=predictions, references=labels, average="macro"
            )
        )
    else:
        # If all predictions are the same class, set metrics to 0
        metrics.update({"precision": 0.0, "recall": 0.0, "f1": 0.0})

    return metrics


def train() -> None:
    """Fine-tune the NLI model on the medical dataset."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_medical_nli_dataset(TASK_FILE)

    # Split dataset into train and validation sets
    # First shuffle the dataset to ensure random distribution
    shuffled_dataset = dataset.shuffle(seed=RANDOM_SEED)

    # Calculate split indices
    train_size = int((1 - TRAIN_TEST_SPLIT) * len(dataset))

    # Split the dataset
    train_test_split = {
        "train": shuffled_dataset.select(range(train_size)),
        "test": shuffled_dataset.select(range(train_size, len(dataset))),
    }
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    # Load tokenizer and model
    logger.info("Loading BioBERT tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load BioBERT model with sequence classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,  # In case the default head size doesn't match
    )

    # Resize token embeddings in case the tokenizer was modified
    model.resize_token_embeddings(len(tokenizer))

    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["premise", "hypothesis"],
    )
    tokenized_eval = eval_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["premise", "hypothesis"],
    )

    # Calculate training steps
    total_steps = (len(train_test_split["train"]) // BATCH_SIZE) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup

    # Set up training arguments with compatible parameters only
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=10,  # Log every 10 steps
        learning_rate=LEARNING_RATE,
        # Save and evaluate every epoch
        eval_steps=len(train_dataset) // BATCH_SIZE,  # Evaluate once per epoch
        save_steps=len(train_dataset) // BATCH_SIZE,  # Save once per epoch
        seed=RANDOM_SEED,
        # Disable features that might not be supported
        no_cuda=not torch.cuda.is_available(),
        overwrite_output_dir=True,  # Overwrite the output directory
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logger.info("Starting training...")
    train_result = trainer.train()

    # Save the model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Log training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluate the model
    logger.info("Evaluating model...")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(eval_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    logger.info(f"Training complete! Model saved to {OUTPUT_DIR}")
    logger.info(f"Final evaluation metrics: {metrics}")


if __name__ == "__main__":
    train()
