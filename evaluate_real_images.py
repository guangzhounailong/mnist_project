import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import cnn
import mlp
from experiment_utils import write_csv


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = PROJECT_DIR / "real_images"
DETAIL_PATH = PROJECT_DIR / "results" / "real_image_predictions.csv"
SUMMARY_PATH = PROJECT_DIR / "results" / "real_image_results.csv"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_label(image_path, images_dir):
    relative_path = image_path.relative_to(images_dir)
    if relative_path.parent != Path(".") and relative_path.parts[0].isdigit():
        label = int(relative_path.parts[0])
    elif image_path.stem and image_path.stem[0].isdigit():
        label = int(image_path.stem[0])
    else:
        raise ValueError(
            f"Cannot find a label for {relative_path}. Use 7_example.jpg or 7/example.jpg."
        )
    if label not in range(10):
        raise ValueError(f"Label must be between 0 and 9: {relative_path}")
    return label


def simple_preprocess(image_source):
    with Image.open(image_source) as image:
        grayscale = image.convert("L").resize((28, 28), Image.Resampling.LANCZOS)
    return transforms.ToTensor()(grayscale).unsqueeze(0)


def predict_tensor(model, image_batch):
    with torch.no_grad():
        probabilities = model(image_batch).softmax(dim=1)
        confidence, prediction = probabilities.max(dim=1)
    return prediction.item(), confidence.item()


def most_error_prone_digits(rows):
    totals = {digit: 0 for digit in range(10)}
    errors = {digit: 0 for digit in range(10)}
    for row in rows:
        label = row["true_label"]
        totals[label] += 1
        errors[label] += not row["correct"]
    error_rates = [
        (errors[digit] / totals[digit], digit)
        for digit in range(10)
        if totals[digit] > 0 and errors[digit] > 0
    ]
    if not error_rates:
        return "none"
    highest_rate = max(rate for rate, _ in error_rates)
    return ";".join(str(digit) for rate, digit in error_rates if rate == highest_rate)


def run(images_dir=DEFAULT_IMAGES_DIR):
    images_dir = Path(images_dir)
    image_paths = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(
            f"No real images found in {images_dir}. "
            "Add files such as 7_01.jpg or 7/01.jpg before running this experiment."
        )

    models = {"MLP": mlp.load_model(), "CNN": cnn.load_model()}
    preprocessors = {
        "simple_resize_grayscale": simple_preprocess,
        "enhanced_detection_crop_center": lambda path: cnn.preprocess_image(path)[0],
    }
    detail_rows = []
    for image_path in image_paths:
        label = parse_label(image_path, images_dir)
        for preprocessing_name, preprocessor in preprocessors.items():
            image_batch = preprocessor(image_path)
            for model_name, model in models.items():
                prediction, confidence = predict_tensor(model, image_batch)
                detail_rows.append(
                    {
                        "image": str(image_path.relative_to(images_dir)),
                        "true_label": label,
                        "model": model_name,
                        "preprocessing": preprocessing_name,
                        "prediction": prediction,
                        "confidence": f"{confidence:.6f}",
                        "correct": prediction == label,
                    }
                )

    write_csv(
        DETAIL_PATH,
        [
            "image", "true_label", "model", "preprocessing", "prediction",
            "confidence", "correct",
        ],
        detail_rows,
    )

    summary_rows = []
    for model_name in models:
        for preprocessing_name in preprocessors:
            selected = [
                row for row in detail_rows
                if row["model"] == model_name
                and row["preprocessing"] == preprocessing_name
            ]
            correct = sum(row["correct"] for row in selected)
            total = len(selected)
            summary_rows.append(
                {
                    "model": model_name,
                    "preprocessing": preprocessing_name,
                    "total_images": total,
                    "correct": correct,
                    "incorrect": total - correct,
                    "accuracy": f"{correct / total:.6f}",
                    "most_error_prone_digits": most_error_prone_digits(selected),
                }
            )

    write_csv(
        SUMMARY_PATH,
        [
            "model", "preprocessing", "total_images", "correct", "incorrect",
            "accuracy", "most_error_prone_digits",
        ],
        summary_rows,
    )
    for row in summary_rows:
        print(
            f"{row['model']} | {row['preprocessing']}: "
            f"{row['correct']}/{row['total_images']} ({float(row['accuracy']):.2%})"
        )
    print(f"Per-image predictions saved to: {DETAIL_PATH}")
    print(f"Real-image summary saved to: {SUMMARY_PATH}")
    return detail_rows, summary_rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate labeled real digit images")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    args = parser.parse_args()
    run(args.images_dir)


if __name__ == "__main__":
    main()
