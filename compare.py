import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cnn import CNN
from experiment_utils import calculate_metrics, write_csv
from mlp import MLP


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 5
RANDOM_SEED = 42
RANDOM_SEEDS = [42, 123, 2026]
RESULTS_PATH = PROJECT_DIR / "results" / "experiment_results.csv"


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


def train_and_evaluate(
    model_name,
    model_class,
    train_dataset,
    test_dataset,
    seed=RANDOM_SEED,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
):
    torch.manual_seed(seed)
    model = model_class()
    generator = torch.Generator().manual_seed(seed)
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
    optimizer = SGD(model.parameters(), lr=learning_rate)
    start_time = perf_counter()

    for epoch in range(epochs):
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
            f"{model_name} | seed {seed} | Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.4f}"
        )

    training_time = perf_counter() - start_time
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            all_labels.append(labels)
            all_predictions.append(model(images).argmax(dim=1))

    metrics = calculate_metrics(
        torch.cat(all_labels),
        torch.cat(all_predictions),
    )

    return {
        "model": model_name,
        "random_seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "training_time": training_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Reproducible MLP/CNN comparison")
    parser.add_argument("--seeds", nargs="+", type=int, default=RANDOM_SEEDS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    train_dataset, test_dataset = create_datasets()
    results = []
    for seed in args.seeds:
        results.extend(
            [
                train_and_evaluate(
                    "MLP", MLP, train_dataset, test_dataset,
                    seed, args.epochs, args.learning_rate,
                ),
                train_and_evaluate(
                    "CNN", CNN, train_dataset, test_dataset,
                    seed, args.epochs, args.learning_rate,
                ),
            ]
        )

    csv_rows = [
        {
            key: f"{value:.6f}" if isinstance(value, float) else value
            for key, value in result.items()
        }
        for result in results
    ]
    write_csv(
        RESULTS_PATH,
        [
            "model", "random_seed", "epochs", "learning_rate", "parameters",
            "training_time", "accuracy", "macro_f1",
        ],
        csv_rows,
    )

    print()
    print(
        f"{'Model':<8} {'Seed':>6} {'Parameters':>12} "
        f"{'Accuracy':>12} {'Macro F1':>12} {'Training Time':>16}"
    )
    print("-" * 74)
    for result in results:
        print(
            f"{result['model']:<8} "
            f"{result['random_seed']:>6} "
            f"{result['parameters']:>12,} "
            f"{result['accuracy']:>11.2%} "
            f"{result['macro_f1']:>11.2%} "
            f"{result['training_time']:>14.2f}s"
        )

    print()
    for model_name in ("MLP", "CNN"):
        model_results = [row for row in results if row["model"] == model_name]
        mean_accuracy = sum(row["accuracy"] for row in model_results) / len(model_results)
        mean_f1 = sum(row["macro_f1"] for row in model_results) / len(model_results)
        print(
            f"{model_name} mean over {len(model_results)} seeds: "
            f"accuracy={mean_accuracy:.2%}, macro F1={mean_f1:.2%}"
        )
    print(f"Experiment rows saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
