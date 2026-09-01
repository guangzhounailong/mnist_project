import csv
import random
from pathlib import Path

import torch


CLASS_NAMES = list(range(10))


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def collect_predictions(model, data_loader):
    model.eval()
    labels = []
    predictions = []
    confidences = []
    images = []

    with torch.no_grad():
        for batch_images, batch_labels in data_loader:
            probabilities = model(batch_images).softmax(dim=1)
            batch_confidences, batch_predictions = probabilities.max(dim=1)
            images.append(batch_images.cpu())
            labels.append(batch_labels.cpu())
            predictions.append(batch_predictions.cpu())
            confidences.append(batch_confidences.cpu())

    return {
        "images": torch.cat(images),
        "labels": torch.cat(labels),
        "predictions": torch.cat(predictions),
        "confidences": torch.cat(confidences),
    }


def confusion_matrix(labels, predictions, number_of_classes=10):
    indices = labels.to(torch.int64) * number_of_classes + predictions.to(torch.int64)
    return torch.bincount(
        indices,
        minlength=number_of_classes * number_of_classes,
    ).reshape(number_of_classes, number_of_classes)


def calculate_metrics(labels, predictions, number_of_classes=10):
    matrix = confusion_matrix(labels, predictions, number_of_classes)
    true_positives = matrix.diag().to(torch.float64)
    predicted_totals = matrix.sum(dim=0).to(torch.float64)
    actual_totals = matrix.sum(dim=1).to(torch.float64)

    precision = torch.where(
        predicted_totals > 0,
        true_positives / predicted_totals,
        torch.zeros_like(true_positives),
    )
    recall = torch.where(
        actual_totals > 0,
        true_positives / actual_totals,
        torch.zeros_like(true_positives),
    )
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    accuracy = true_positives.sum() / matrix.sum().clamp_min(1)

    return {
        "accuracy": accuracy.item(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual_totals.to(torch.int64),
        "confusion_matrix": matrix,
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
    }


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
