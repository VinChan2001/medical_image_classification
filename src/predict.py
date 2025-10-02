"""
Inference script for making predictions on new medical images.
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


class MedicalImagePredictor:
    """Predict pneumonia from chest X-ray images."""

    def __init__(self, model_path, img_size=(224, 224)):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model
            img_size: Input image size expected by model
        """
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = img_size
        self.class_names = ['NORMAL', 'PNEUMONIA']

    def preprocess_image(self, img_path):
        """
        Preprocess a single image for prediction.

        Args:
            img_path: Path to image file

        Returns:
            Preprocessed image array
        """
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def predict(self, img_path, return_confidence=True):
        """
        Predict class for a single image.

        Args:
            img_path: Path to image file
            return_confidence: Whether to return confidence score

        Returns:
            Prediction (and confidence if requested)
        """
        img_array = self.preprocess_image(img_path)
        prediction_prob = self.model.predict(img_array, verbose=0)[0][0]

        predicted_class = self.class_names[int(prediction_prob > 0.5)]
        confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob

        if return_confidence:
            return predicted_class, confidence
        return predicted_class

    def predict_batch(self, img_paths):
        """
        Predict classes for multiple images.

        Args:
            img_paths: List of image file paths

        Returns:
            List of (prediction, confidence) tuples
        """
        results = []
        for img_path in img_paths:
            pred, conf = self.predict(img_path)
            results.append((pred, conf))
        return results

    def visualize_prediction(self, img_path, save_path=None):
        """
        Visualize prediction on an image.

        Args:
            img_path: Path to image file
            save_path: Optional path to save visualization
        """
        # Get prediction
        predicted_class, confidence = self.predict(img_path)

        # Load original image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')

        # Add prediction text
        color = 'green' if predicted_class == 'NORMAL' else 'red'
        title = f'Prediction: {predicted_class}\nConfidence: {confidence:.2%}'
        plt.title(title, fontsize=16, fontweight='bold', color=color, pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_report(self, img_path):
        """
        Generate a detailed prediction report.

        Args:
            img_path: Path to image file

        Returns:
            Dictionary with prediction details
        """
        img_array = self.preprocess_image(img_path)
        prediction_prob = self.model.predict(img_array, verbose=0)[0][0]

        predicted_class = self.class_names[int(prediction_prob > 0.5)]
        confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob

        report = {
            'image_path': img_path,
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'normal_probability': float(1 - prediction_prob),
            'pneumonia_probability': float(prediction_prob),
            'recommendation': self._generate_recommendation(predicted_class, confidence)
        }

        return report

    def _generate_recommendation(self, predicted_class, confidence):
        """Generate medical recommendation based on prediction."""
        if predicted_class == 'PNEUMONIA':
            if confidence > 0.9:
                return "High confidence pneumonia detection. Immediate medical consultation recommended."
            elif confidence > 0.7:
                return "Pneumonia detected. Medical consultation recommended."
            else:
                return "Possible pneumonia detected. Further examination recommended."
        else:
            if confidence > 0.9:
                return "Chest X-ray appears normal. No immediate concerns."
            elif confidence > 0.7:
                return "Likely normal. Routine follow-up recommended if symptoms persist."
            else:
                return "Uncertain diagnosis. Additional imaging or medical evaluation recommended."


def main():
    """Main prediction function with command-line interface."""
    parser = argparse.ArgumentParser(description='Predict pneumonia from chest X-ray images')

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to X-ray image for prediction')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize the prediction')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed prediction report')

    args = parser.parse_args()

    # Initialize predictor
    predictor = MedicalImagePredictor(args.model)

    # Make prediction
    print("=" * 80)
    print("MEDICAL IMAGE CLASSIFICATION - PREDICTION")
    print("=" * 80)
    print(f"\nImage: {args.image}")
    print(f"Model: {args.model}")

    if args.report:
        report = predictor.generate_report(args.image)
        print("\n" + "-" * 80)
        print("PREDICTION REPORT")
        print("-" * 80)
        print(f"Predicted Class: {report['predicted_class']}")
        print(f"Confidence: {report['confidence']:.2%}")
        print(f"\nProbabilities:")
        print(f"  Normal: {report['normal_probability']:.2%}")
        print(f"  Pneumonia: {report['pneumonia_probability']:.2%}")
        print(f"\nRecommendation:")
        print(f"  {report['recommendation']}")
    else:
        predicted_class, confidence = predictor.predict(args.image)
        print(f"\nPrediction: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")

    if args.visualize:
        predictor.visualize_prediction(args.image, save_path=args.output)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
