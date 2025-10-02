"""
Training pipeline for medical image classification models.
"""

import os
import argparse
import json
import numpy as np
import tensorflow as tf
from data_loader import MedicalImageLoader
from models import CNNModels


def train_model(data_dir, model_type='custom', base_model=None, epochs=50,
                batch_size=32, img_size=(224, 224)):
    """
    Train a medical image classification model.

    Args:
        data_dir: Path to dataset directory
        model_type: 'custom' or 'transfer'
        base_model: Base model for transfer learning (e.g., 'VGG16')
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size

    Returns:
        Trained model and training history
    """
    print("=" * 80)
    print("MEDICAL IMAGE CLASSIFICATION - TRAINING PIPELINE")
    print("=" * 80)

    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Initialize data loader
    print("\n[1/5] Loading and preprocessing data...")
    loader = MedicalImageLoader(data_dir, img_size=img_size, batch_size=batch_size)

    # Create data generators
    train_gen, val_gen, test_gen = loader.create_data_generators(validation_split=0.2)

    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")

    # Calculate class weights for imbalanced data
    class_weights = loader.get_class_weights(train_gen)
    print(f"Class weights: {class_weights}")

    # Build model
    print(f"\n[2/5] Building {model_type} model...")
    if model_type == 'custom':
        model_name = 'custom_cnn'
        model = CNNModels.custom_cnn(input_shape=(*img_size, 3))
    elif model_type == 'transfer':
        if base_model is None:
            base_model = 'VGG16'
        model_name = f'{base_model.lower()}_transfer'
        model = CNNModels.transfer_learning_model(
            base_model_name=base_model,
            input_shape=(*img_size, 3)
        )
    else:
        raise ValueError("model_type must be 'custom' or 'transfer'")

    print(f"\nModel: {model_name}")
    print(f"Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_params:,}")

    # Get callbacks
    callbacks = CNNModels.get_callbacks(model_name, patience=7)

    # Train model
    print(f"\n[3/5] Training model for {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    print("\n[4/5] Evaluating on test set...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(test_gen, verbose=1)

    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
    print(f"  F1-Score: {f1_score:.4f}")

    # Save final model
    print("\n[5/5] Saving model...")
    model.save(f'models/{model_name}_final.h5')
    print(f"Model saved to models/{model_name}_final.h5")

    # Save training history and results
    results = {
        'model_name': model_name,
        'model_type': model_type,
        'base_model': base_model if model_type == 'transfer' else None,
        'epochs_trained': len(history.history['loss']),
        'batch_size': batch_size,
        'img_size': img_size,
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1_score': float(f1_score),
        'test_loss': float(test_loss),
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
    }

    with open(f'results/{model_name}_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to results/{model_name}_results.json")
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

    return model, history


def main():
    """Main training function with command-line arguments."""
    parser = argparse.ArgumentParser(description='Train medical image classification model')

    parser.add_argument('--data_dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--model_type', type=str, default='custom',
                       choices=['custom', 'transfer'],
                       help='Model type: custom or transfer learning')
    parser.add_argument('--base_model', type=str, default='VGG16',
                       choices=['VGG16', 'ResNet50', 'InceptionV3', 'DenseNet121', 'EfficientNetB0'],
                       help='Base model for transfer learning')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size (square)')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Train model
    model, history = train_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        base_model=args.base_model if args.model_type == 'transfer' else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size)
    )


if __name__ == "__main__":
    main()
