"""
Model evaluation and visualization utilities.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
)
import tensorflow as tf


def plot_training_history(history_file, save_path='results/training_history.png'):
    """
    Plot training history from saved results.

    Args:
        history_file: Path to JSON file with training history
        save_path: Where to save the plot
    """
    with open(history_file, 'r') as f:
        results = json.load(f)

    history = results['history']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Loss
    axes[0, 1].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision
    axes[1, 0].plot(history['precision'], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history['val_precision'], label='Val Precision', linewidth=2)
    axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Recall
    axes[1, 1].plot(history['recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history['val_recall'], label='Val Recall', linewidth=2)
    axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def evaluate_model(model_path, test_generator, save_dir='results'):
    """
    Comprehensive model evaluation with visualizations.

    Args:
        model_path: Path to saved model
        test_generator: Test data generator
        save_dir: Directory to save evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    # Get predictions
    print("Generating predictions...")
    y_pred_prob = model.predict(test_generator, verbose=1)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    # Classification report
    print("\nClassification Report:")
    print("=" * 60)
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # Save report
    with open(f'{save_dir}/classification_report.txt', 'w') as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {save_dir}/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_dir}/roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curve saved to {save_dir}/precision_recall_curve.png")
    plt.close()

    # Save evaluation metrics
    metrics = {
        'accuracy': float(np.mean(y_pred == y_true)),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist()
    }

    with open(f'{save_dir}/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"\nEvaluation metrics saved to {save_dir}/evaluation_metrics.json")
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    return metrics


def visualize_predictions(model_path, test_generator, num_samples=16, save_path='results/predictions.png'):
    """
    Visualize model predictions on sample images.

    Args:
        model_path: Path to saved model
        test_generator: Test data generator
        num_samples: Number of samples to visualize
        save_path: Where to save the plot
    """
    model = tf.keras.models.load_model(model_path)

    # Get a batch of images
    images, labels = next(test_generator)
    predictions = model.predict(images[:num_samples])

    # Create plot
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))

    class_names = list(test_generator.class_indices.keys())

    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(images[i])

            true_label = class_names[int(labels[i])]
            pred_label = class_names[int(predictions[i] > 0.5)]
            confidence = predictions[i][0] if predictions[i] > 0.5 else 1 - predictions[i][0]

            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2%})',
                        color=color, fontsize=10, fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction visualization saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Example usage
    print("Evaluation utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - plot_training_history()")
    print("  - evaluate_model()")
    print("  - visualize_predictions()")
