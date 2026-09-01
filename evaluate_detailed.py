import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import cnn
import mlp
from experiment_utils import calculate_metrics, collect_predictions, write_csv


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_DIR / "results" / "classification_results.csv"
CLASS_RESULTS_PATH = PROJECT_DIR / "results" / "per_class_results.csv"
CONFUSION_MATRIX_PATH = PROJECT_DIR / "images" / "confusion_matrix.png"
ERROR_SAMPLES_PATH = PROJECT_DIR / "images" / "error_samples.png"


def evaluate_model(model_name, model, data_loader):
    outputs = collect_predictions(model, data_loader)
    metrics = calculate_metrics(outputs["labels"], outputs["predictions"])
    return {"model": model_name, "outputs": outputs, "metrics": metrics}


def most_common_confusions(matrix, limit=3):
    mistakes = matrix.clone()
    mistakes.fill_diagonal_(0)
    flat_indices = mistakes.flatten().argsort(descending=True)
    pairs = []
    for index in flat_indices:
        count = mistakes.flatten()[index].item()
        if count == 0 or len(pairs) == limit:
            break
        pairs.append((index.item() // 10, index.item() % 10, count))
    return pairs


def save_confusion_matrices(results, output_path):
    figure, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    axes = [axes] if len(results) == 1 else axes

    for axis, result in zip(axes, results):
        matrix = result["metrics"]["confusion_matrix"]
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{result['model']} Confusion Matrix")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def select_confidence_examples(outputs, samples_per_group=5):
    correct = outputs["predictions"] == outputs["labels"]
    wrong_indices = torch.where(~correct)[0]
    correct_indices = torch.where(correct)[0]
    high_confidence_wrong = wrong_indices[
        outputs["confidences"][wrong_indices].argsort(descending=True)
    ][:samples_per_group]
    low_confidence_correct = correct_indices[
        outputs["confidences"][correct_indices].argsort()
    ][:samples_per_group]
    return high_confidence_wrong, low_confidence_correct


def save_error_samples(results, output_path, samples_per_group=5):
    figure, axes = plt.subplots(
        len(results) * 2,
        samples_per_group,
        figsize=(2.4 * samples_per_group, 4.2 * len(results)),
        squeeze=False,
    )

    for model_index, result in enumerate(results):
        outputs = result["outputs"]
        high_wrong, low_correct = select_confidence_examples(
            outputs,
            samples_per_group,
        )
        groups = (
            (high_wrong, "High-confidence wrong", "red"),
            (low_correct, "Low-confidence correct", "green"),
        )
        for group_index, (indices, group_name, color) in enumerate(groups):
            row = model_index * 2 + group_index
            for column, index in enumerate(indices):
                axis = axes[row, column]
                axis.imshow(outputs["images"][index].squeeze(0), cmap="gray")
                axis.set_title(
                    f"True {outputs['labels'][index].item()} / "
                    f"Pred {outputs['predictions'][index].item()}\n"
                    f"Confidence {outputs['confidences'][index].item():.1%}",
                    color=color,
                    fontsize=9,
                )
                axis.axis("off")
            axes[row, 0].set_ylabel(
                f"{result['model']}\n{group_name}",
                color=color,
                fontsize=10,
            )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_results(results):
    summary_rows = []
    class_rows = []
    for result in results:
        metrics = result["metrics"]
        summary_rows.append(
            {
                "model": result["model"],
                "accuracy": f"{metrics['accuracy']:.6f}",
                "macro_precision": f"{metrics['macro_precision']:.6f}",
                "macro_recall": f"{metrics['macro_recall']:.6f}",
                "macro_f1": f"{metrics['macro_f1']:.6f}",
            }
        )
        for digit in range(10):
            class_rows.append(
                {
                    "model": result["model"],
                    "digit": digit,
                    "precision": f"{metrics['precision'][digit].item():.6f}",
                    "recall": f"{metrics['recall'][digit].item():.6f}",
                    "f1": f"{metrics['f1'][digit].item():.6f}",
                    "class_accuracy": f"{metrics['recall'][digit].item():.6f}",
                    "support": metrics["support"][digit].item(),
                }
            )

    write_csv(
        RESULTS_PATH,
        ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1"],
        summary_rows,
    )
    write_csv(
        CLASS_RESULTS_PATH,
        ["model", "digit", "precision", "recall", "f1", "class_accuracy", "support"],
        class_rows,
    )


def run(model_names):
    _, test_loader = mlp.create_data_loader(train=False, shuffle=False)
    model_factories = {"MLP": mlp.load_model, "CNN": cnn.load_model}
    results = [
        evaluate_model(name, model_factories[name](), test_loader)
        for name in model_names
    ]
    save_results(results)
    save_confusion_matrices(results, CONFUSION_MATRIX_PATH)
    save_error_samples(results, ERROR_SAMPLES_PATH)

    for result in results:
        metrics = result["metrics"]
        confusions = most_common_confusions(metrics["confusion_matrix"])
        confusion_text = ", ".join(
            f"{actual}->{predicted} ({count})"
            for actual, predicted, count in confusions
        )
        print(
            f"{result['model']}: accuracy={metrics['accuracy']:.2%}, "
            f"macro F1={metrics['macro_f1']:.2%}, "
            f"top confusions: {confusion_text}"
        )
    print(f"Detailed metrics saved to: {RESULTS_PATH}")
    print(f"Per-class metrics saved to: {CLASS_RESULTS_PATH}")
    print(f"Confusion matrices saved to: {CONFUSION_MATRIX_PATH}")
    print(f"Confidence examples saved to: {ERROR_SAMPLES_PATH}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Detailed MNIST evaluation")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["MLP", "CNN"],
        default=["MLP", "CNN"],
    )
    args = parser.parse_args()
    run(args.models)


if __name__ == "__main__":
    main()
