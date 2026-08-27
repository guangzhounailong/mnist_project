from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn import CNN
from mlp import MLP


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 5
RANDOM_SEED = 42


def create_datasets():
    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    return train_dataset, test_dataset


def train_and_evaluate(model_name, model_class, train_dataset, test_dataset):
    torch.manual_seed(RANDOM_SEED)
    model = model_class()
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    start_time = perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        average_loss = total_loss / len(train_dataset)
        print(
            f"{model_name} | Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f}"
        )

    training_time = perf_counter() - start_time
    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images).argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()

    return {
        "name": model_name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "accuracy": 100 * correct_predictions / len(test_dataset),
        "training_time": training_time,
    }


def main():
    train_dataset, test_dataset = create_datasets()
    results = [
        train_and_evaluate("MLP", MLP, train_dataset, test_dataset),
        train_and_evaluate("CNN", CNN, train_dataset, test_dataset),
    ]

    print()
    print(f"{'Model':<8} {'Parameters':>12} {'Test Accuracy':>16} {'Training Time':>16}")
    print("-" * 56)
    for result in results:
        print(
            f"{result['name']:<8} "
            f"{result['parameters']:>12,} "
            f"{result['accuracy']:>15.2f}% "
            f"{result['training_time']:>14.2f}s"
        )


if __name__ == "__main__":
    main()
