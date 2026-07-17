import cv2
import numpy as np

from src.antispoof.model import AntiSpoofModel

# Load the model
model = AntiSpoofModel("models/antispoof/best_model_quantized.onnx")

# Load a test image
image = cv2.imread("test.jpeg")

if image is None:
    raise FileNotFoundError("Could not find test.jpg")

# Get raw scores
scores = model.predict(image)

# Convert logits to probabilities
exp_scores = np.exp(scores - np.max(scores))
probabilities = exp_scores / np.sum(exp_scores)

# Get final prediction
is_live, confidence = model.check(image)

print("=" * 50)
print("Raw Scores      :", scores)
print("Probabilities   :", probabilities)
print("Predicted Class :", np.argmax(scores))
print("Live            :", is_live)
print("Confidence      :", confidence)
print("=" * 50)