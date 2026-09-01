import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

import cnn
import mlp
from experiment_utils import write_csv


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_DIR / "results" / "robustness_results.csv"
PLOT_PATH = PROJECT_DIR / "images" / "robustness_comparison.png"
RANDOM_SEED = 42

CONDITIONS = {
    "original": "No change",
    "rotation": "10 degrees clockwise",
    "translation": "2 pixels right and down",
    "gaussian_noise": "standard deviation 0.20",
    "blur": "3x3 Gaussian blur, sigma 1.0",
    "brightness": "60% brightness",
}


def apply_condition(images, condition, generator=None):
    if condition == "original":
        return images
    if condition == "rotation":
        return transform_functional.rotate(
            images,
            angle=10,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
    if condition == "translation":
        return transform_functional.affine(
            images,
            angle=0,
            translate=[2, 2],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
    if condition == "gaussian_noise":
        noise = torch.randn(
            images.shape,
            dtype=images.dtype,
            device=images.device,
            generator=generator,
        )
        return (images + 0.20 * noise).clamp(0, 1)
    if condition == "blur":
        return transform_functional.gaussian_blur(images, kernel_size=[3, 3], sigma=[1.0, 1.0])
    if condition == "brightness":
        return transform_functional.adjust_brightness(images, brightness_factor=0.6)
    raise ValueError(f"Unknown robustness condition: {condition}")


def evaluate_condition(model, data_loader, condition, seed=RANDOM_SEED):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            changed_images = apply_condition(images, condition, generator)
            predictions = model(changed_images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return correct / total


def save_plot(rows, output_path):
    conditions = list(CONDITIONS)
    model_names = list(dict.fromkeys(row["model"] for row in rows))
    x_positions = torch.arange(len(conditions), dtype=torch.float64)
    width = 0.8 / len(model_names)
    figure, axis = plt.subplots(figsize=(11, 5))

    for model_index, model_name in enumerate(model_names):
        accuracies = [
            100 * float(next(
                row["accuracy"]
                for row in rows
                if row["model"] == model_name and row["condition"] == condition
            ))
            for condition in conditions
        ]
        offset = (model_index - (len(model_names) - 1) / 2) * width
        axis.bar(x_positions + offset, accuracies, width=width, label=model_name)

    axis.set_title("MNIST Robustness Comparison")
    axis.set_ylabel("Accuracy (%)")
    axis.set_xticks(x_positions, conditions, rotation=20, ha="right")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run(model_names, seed=RANDOM_SEED):
    _, test_loader = mlp.create_data_loader(train=False, shuffle=False)
    model_factories = {"MLP": mlp.load_model, "CNN": cnn.load_model}
    rows = []
    for model_name in model_names:
        model = model_factories[model_name]()
        for condition, description in CONDITIONS.items():
            accuracy = evaluate_condition(model, test_loader, condition, seed)
            rows.append(
                {
                    "model": model_name,
                    "condition": condition,
                    "parameters": description,
                    "seed": seed,
                    "accuracy": f"{accuracy:.6f}",
                }
            )
            print(f"{model_name} | {condition}: {accuracy:.2%}")

    write_csv(
        RESULTS_PATH,
        ["model", "condition", "parameters", "seed", "accuracy"],
        rows,
    )
    save_plot(rows, PLOT_PATH)
    print(f"Robustness results saved to: {RESULTS_PATH}")
    print(f"Robustness chart saved to: {PLOT_PATH}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="MNIST robustness benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["MLP", "CNN"],
        default=["MLP", "CNN"],
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    run(args.models, args.seed)


if __name__ == "__main__":
    main()
