from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = PROJECT_DIR / "models" / "mlp_state_dict.pth"
BATCH_SIZE = 64


def evaluate(model):
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transforms.ToTensor(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()

    accuracy = 100 * correct_predictions / len(test_dataset)
    return accuracy


def main():
    model = MLP()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)

    accuracy = evaluate(model)
    print(f"Loaded model from: {MODEL_PATH}")
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
