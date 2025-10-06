"""Training pipeline for medical image classification models."""

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from data_loader import MedicalImageLoader
from models import CNNModels


def _resolve_steps(requested: Optional[int]) -> Optional[int]:
    """Return a positive integer for Keras step arguments or ``None``."""
    if requested is None:
        return None
    if requested <= 0:
        raise ValueError("Step counts must be positive integers")
    return int(requested)


def _default_steps(generator) -> int:
    """Compute the default number of steps per epoch for a generator."""
    if generator.samples == 0:
        return 0
    return math.ceil(generator.samples / generator.batch_size)


def _format_class_weights(class_weights) -> str:
    rounded = {cls: round(float(weight), 4) for cls, weight in class_weights.items()}
    return ", ".join(f"{cls}: {weight}" for cls, weight in rounded.items())


def train_model(
    data_dir: str,
    model_type: str = "custom",
    base_model: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 32,
    img_size: Tuple[int, int] = (224, 224),
    models_dir: str = "models",
    results_dir: str = "results",
    train_steps: Optional[int] = None,
    val_steps: Optional[int] = None,
    test_steps: Optional[int] = None,
    validation_split: float = 0.2,
    min_val_images: int = 64,
    patience: int = 7,
    seed: Optional[int] = 42,
):
    """Train a medical image classification model."""
    models_path = Path(models_dir)
    results_path = Path(results_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    results_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MEDICAL IMAGE CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 80)

    # Data pipeline
    print("\n[1/5] Loading and preprocessing data...")
    loader = MedicalImageLoader(data_dir, img_size=img_size, batch_size=batch_size)

    train_gen, val_gen, test_gen = loader.create_data_generators(
        validation_split=validation_split,
        seed=seed,
        min_val_images=min_val_images,
    )

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    if loader.using_separate_val:
        print(
            "Validation strategy: data/val directory "
            f"({loader.val_image_count} images)"
        )
    else:
        approx_val = int(train_gen.samples * validation_split)
        print(
            f"Validation strategy: {validation_split:.0%} split from training data"
            f" (~{approx_val} images)"
        )

    class_weights = loader.get_class_weights(train_gen)
    print(f"Class weights: {_format_class_weights(class_weights)}")

    # Model selection
    print(f"\n[2/5] Building {model_type} model...")
    input_shape = (*img_size, 3)
    if model_type == "custom":
        model_name = "custom_cnn"
        model = CNNModels.custom_cnn(input_shape=input_shape)
    elif model_type == "transfer":
        if base_model is None:
            base_model = "VGG16"
        model_name = f"{base_model.lower()}_transfer"
        model = CNNModels.transfer_learning_model(
            base_model_name=base_model,
            input_shape=input_shape,
        )
    else:
        raise ValueError("model_type must be 'custom' or 'transfer'")

    print(f"\nModel: {model_name}")
    print(f"Total parameters: {model.count_params():,}")
    trainable_params = sum(tf.size(weight).numpy() for weight in model.trainable_weights)
    print(f"Trainable parameters: {trainable_params:,}")

    # Callbacks
    callbacks = CNNModels.get_callbacks(
        model_name,
        patience=patience,
        models_dir=models_path.as_posix(),
        results_dir=results_path.as_posix(),
        monitor="val_accuracy",
    )

    steps_per_epoch = _resolve_steps(train_steps)
    validation_steps = _resolve_steps(val_steps)
    evaluation_steps = _resolve_steps(test_steps)

    if steps_per_epoch:
        print(f"Steps per epoch overridden to {steps_per_epoch}")
    if validation_steps:
        print(f"Validation steps overridden to {validation_steps}")
    if evaluation_steps:
        print(f"Test steps overridden to {evaluation_steps}")

    # Training
    print(f"\n[3/5] Training model for {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluation
    print("\n[4/5] Evaluating on test set...")
    eval_kwargs = {"verbose": 1, "return_dict": True}
    if evaluation_steps is not None:
        eval_kwargs["steps"] = evaluation_steps
    metrics = model.evaluate(test_gen, **eval_kwargs)
    metric_names = list(metrics.keys())

    metrics_dict = {key: float(value) for key, value in metrics.items()}
    metrics_lower = {key.lower(): float(value) for key, value in metrics.items()}

    test_loss = metrics_lower.get("loss", 0.0)
    test_accuracy = metrics_lower.get(
        "accuracy", metrics_lower.get("binary_accuracy", 0.0)
    )
    test_precision = metrics_lower.get("precision", 0.0)
    test_recall = metrics_lower.get("recall", 0.0)
    f1_denominator = test_precision + test_recall
    test_f1 = float(0.0)
    if f1_denominator:
        test_f1 = float(2 * (test_precision * test_recall) / f1_denominator)

    print("\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    if "precision" in metrics_lower:
        print(f"  Precision: {test_precision:.4f}")
    if "recall" in metrics_lower:
        print(f"  Recall: {test_recall:.4f}")
    if "precision" in metrics_lower and "recall" in metrics_lower:
        print(f"  F1-Score: {test_f1:.4f}")

    # Persist artifacts
    print("\n[5/5] Saving model and results...")
    final_model_path = models_path / f"{model_name}_final.keras"
    model.save(final_model_path.as_posix())
    print(f"Model saved to {final_model_path}")

    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    results_payload = {
        "model_name": model_name,
        "model_type": model_type,
        "base_model": base_model if model_type == "transfer" else None,
        "epochs_requested": epochs,
        "epochs_trained": len(history_data.get("loss", [])),
        "batch_size": batch_size,
        "img_size": list(img_size),
        "seed": seed,
        "train_samples": int(train_gen.samples),
        "val_samples": int(val_gen.samples),
        "test_samples": int(test_gen.samples),
        "val_image_count": int(loader.val_image_count),
        "steps_per_epoch": steps_per_epoch or _default_steps(train_gen),
        "validation_steps": validation_steps or _default_steps(val_gen),
        "test_steps": evaluation_steps or _default_steps(test_gen),
        "class_weights": {str(k): float(v) for k, v in class_weights.items()},
        "using_separate_validation": loader.using_separate_val,
        "metrics": {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": test_f1,
        },
        "raw_metrics": metrics_dict,
        "history": history_data,
        "metric_names": metric_names,
    }

    results_path.mkdir(parents=True, exist_ok=True)
    results_file = results_path / f"{model_name}_results.json"
    with results_file.open("w", encoding="utf-8") as fh:
        json.dump(results_payload, fh, indent=4)

    print(f"Results saved to {results_file}")
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    return model, history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train medical image classification models"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="custom",
        choices=["custom", "transfer"],
        help="Model type: custom CNN or transfer learning",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="VGG16",
        choices=["VGG16", "ResNet50", "InceptionV3", "DenseNet121", "EfficientNetB0"],
        help="Base model to use for transfer learning",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size (square)",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.2,
        help="Validation split to use when a dedicated val/ directory is absent",
    )
    parser.add_argument(
        "--min_val_images",
        type=int,
        default=64,
        help=(
            "Minimum number of images required in data/val before using it "
            "as the validation set"
        ),
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Directory to store trained models",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to store metrics and logs",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=None,
        help="Optional override for steps per epoch",
    )
    parser.add_argument(
        "--val_steps",
        type=int,
        default=None,
        help="Optional override for validation steps",
    )
    parser.add_argument(
        "--test_steps",
        type=int,
        default=None,
        help="Optional override for evaluation steps",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (set to -1 to disable)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    if seed is not None:
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)

    img_size = (args.img_size, args.img_size)

    train_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        base_model=args.base_model if args.model_type == "transfer" else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=img_size,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        train_steps=args.train_steps,
        val_steps=args.val_steps,
        test_steps=args.test_steps,
        validation_split=args.validation_split,
        min_val_images=args.min_val_images,
        patience=args.patience,
        seed=seed,
    )


if __name__ == "__main__":
    main()
