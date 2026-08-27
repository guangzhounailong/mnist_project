import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import MLP


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "mlp_state_dict.pth"


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)

    image_tensor = transforms.ToTensor()(image)

    was_inverted = image_tensor.mean().item() > 0.5
    if was_inverted:
        image_tensor = 1.0 - image_tensor

    image_batch = image_tensor.unsqueeze(0)
    return image_batch, was_inverted


def predict(image_path):
    model = MLP()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    image_batch, was_inverted = preprocess_image(image_path)

    with torch.no_grad():
        logits = model(image_batch)
        probabilities = logits.softmax(dim=1)
        confidence, predicted_class = probabilities.max(dim=1)

    print(f"Input tensor shape: {image_batch.shape}")
    print(f"Colors inverted: {was_inverted}")
    print(f"Prediction: {predicted_class.item()}")
    print(f"Confidence: {confidence.item() * 100:.2f}%")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict.py path/to/image.png")

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    predict(image_path)


if __name__ == "__main__":
    main()
