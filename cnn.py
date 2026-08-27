import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = PROJECT_DIR / "models" / "cnn_state_dict.pth"
HISTORY_PLOT_PATH = PROJECT_DIR / "images" / "cnn_training_history.png"
ANALYSIS_IMAGE_PATH = PROJECT_DIR / "images" / "cnn_prediction_analysis.png"

BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 10
RANDOM_SEED = 42
SAMPLES_PER_GROUP = 4
CANVAS_SIZE = 28
DIGIT_SIZE = 21


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, images):
        features = self.features(images)
        return self.classifier(features)


def create_transform(train):
    if train:
        return transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
            ]
        )
    return transforms.ToTensor()


def create_data_loader(train, shuffle):
    dataset = datasets.MNIST(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=create_transform(train),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(RANDOM_SEED) if shuffle else None,
    )
    return dataset, data_loader


def load_model():
    model = CNN()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def train():
    import matplotlib.pyplot as plt

    torch.manual_seed(RANDOM_SEED)
    train_dataset, train_loader = create_data_loader(train=True, shuffle=True)
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    loss_history = []
    accuracy_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct_predictions = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()

        average_loss = total_loss / len(train_dataset)
        accuracy = 100 * correct_predictions / len(train_dataset)
        loss_history.append(average_loss)
        accuracy_history.append(accuracy)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Accuracy: {accuracy:.2f}% | "
            f"Learning rate: {LEARNING_RATE}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"CNN state_dict saved to: {MODEL_PATH}")

    epoch_numbers = range(1, EPOCHS + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(10, 4))
    loss_axis.plot(epoch_numbers, loss_history, marker="o")
    loss_axis.set_title("CNN Training Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(True)

    accuracy_axis.plot(epoch_numbers, accuracy_history, marker="o")
    accuracy_axis.set_title("CNN Training Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.grid(True)

    figure.tight_layout()
    HISTORY_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(HISTORY_PLOT_PATH)
    plt.close(figure)
    print(f"CNN training history saved to: {HISTORY_PLOT_PATH}")


def evaluate():
    test_dataset, test_loader = create_data_loader(train=False, shuffle=False)
    model = load_model()
    correct_predictions = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()

    accuracy = 100 * correct_predictions / len(test_dataset)
    print(f"Loaded CNN from: {MODEL_PATH}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def preprocess_image(image_source):
    if isinstance(image_source, Image.Image):
        image = image_source.copy()
    else:
        image = Image.open(image_source)

    image_tensor = transforms.ToTensor()(image.convert("L")).squeeze(0)
    border_pixels = torch.cat(
        (
            image_tensor[0, :],
            image_tensor[-1, :],
            image_tensor[:, 0],
            image_tensor[:, -1],
        )
    )
    background = border_pixels.median()
    dark_strokes = (background - image_tensor).clamp(min=0)
    light_strokes = (image_tensor - background).clamp(min=0)

    dark_contrast = torch.quantile(dark_strokes, 0.99)
    light_contrast = torch.quantile(light_strokes, 0.99)
    was_inverted = bool((dark_contrast > light_contrast).item())
    foreground = dark_strokes if was_inverted else light_strokes

    foreground_peak = torch.quantile(foreground, 0.99).item()
    border_noise = torch.quantile((border_pixels - background).abs(), 0.99).item()
    threshold = max(0.05, foreground_peak * 0.2, border_noise * 1.5)
    foreground = (foreground - threshold) / max(
        foreground_peak - threshold,
        1e-6,
    )
    foreground = foreground.clamp(0, 1)

    foreground_pixels = torch.nonzero(foreground > 0.1)
    if foreground_pixels.numel() == 0:
        raise ValueError("No handwritten digit was found in the image.")

    top, left = foreground_pixels.min(dim=0).values
    bottom, right = foreground_pixels.max(dim=0).values
    cropped = foreground[top : bottom + 1, left : right + 1]

    cropped_image = transforms.ToPILImage()(cropped.unsqueeze(0))
    width, height = cropped_image.size
    resize_scale = DIGIT_SIZE / max(width, height)
    resized_size = (
        max(1, round(width * resize_scale)),
        max(1, round(height * resize_scale)),
    )
    resized_image = cropped_image.resize(resized_size, Image.Resampling.LANCZOS)
    resized_tensor = transforms.ToTensor()(resized_image).squeeze(0)
    resized_tensor = resized_tensor / resized_tensor.max().clamp_min(1e-6)

    rows, columns = torch.meshgrid(
        torch.arange(resized_size[1]),
        torch.arange(resized_size[0]),
        indexing="ij",
    )
    total_ink = resized_tensor.sum()
    center_x = (columns * resized_tensor).sum() / total_ink
    center_y = (rows * resized_tensor).sum() / total_ink
    target_center = CANVAS_SIZE / 2
    paste_position = (
        math.ceil(target_center - center_x.item()),
        math.ceil(target_center - center_y.item()),
    )

    centered_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
    centered_image.paste(
        transforms.ToPILImage()(resized_tensor.unsqueeze(0)),
        paste_position,
    )
    return transforms.ToTensor()(centered_image).unsqueeze(0), was_inverted


def predict(image_source):
    model = load_model()
    image_batch, was_inverted = preprocess_image(image_source)

    with torch.no_grad():
        probabilities = model(image_batch).softmax(dim=1)
        confidence, predicted_class = probabilities.max(dim=1)

    return (
        predicted_class.item(),
        confidence.item(),
        image_batch,
        was_inverted,
    )


def analyze():
    import matplotlib.pyplot as plt

    _, test_loader = create_data_loader(train=False, shuffle=True)
    model = load_model()
    correct_examples = []
    incorrect_examples = []

    with torch.no_grad():
        for images, labels in test_loader:
            probabilities = model(images).softmax(dim=1)
            confidences, predictions = probabilities.max(dim=1)

            for image, label, prediction, confidence in zip(
                images, labels, predictions, confidences
            ):
                example = (
                    image,
                    label.item(),
                    prediction.item(),
                    confidence.item(),
                )
                if prediction.item() == label.item() and len(correct_examples) < 4:
                    correct_examples.append(example)
                elif prediction.item() != label.item() and len(incorrect_examples) < 4:
                    incorrect_examples.append(example)

            if len(correct_examples) == 4 and len(incorrect_examples) == 4:
                break

    figure, axes = plt.subplots(2, SAMPLES_PER_GROUP, figsize=(10, 5))
    for column, (image, label, prediction, confidence) in enumerate(correct_examples):
        axis = axes[0, column]
        axis.imshow(image.squeeze(0), cmap="gray")
        axis.set_title(
            f"Correct\nTrue: {label} | Pred: {prediction}\nConf: {confidence:.1%}"
        )
        axis.axis("off")

    for column, (image, label, prediction, confidence) in enumerate(incorrect_examples):
        axis = axes[1, column]
        axis.imshow(image.squeeze(0), cmap="gray")
        axis.set_title(
            f"Wrong\nTrue: {label} | Pred: {prediction}\nConf: {confidence:.1%}",
            color="red",
        )
        axis.axis("off")

    figure.tight_layout()
    ANALYSIS_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(ANALYSIS_IMAGE_PATH)
    plt.close(figure)
    print(f"CNN prediction analysis saved to: {ANALYSIS_IMAGE_PATH}")


def main():
    parser = argparse.ArgumentParser(description="MNIST CNN workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Train and save the CNN")
    subparsers.add_parser("evaluate", help="Evaluate the saved CNN")
    subparsers.add_parser("analyze", help="Visualize correct and wrong predictions")
    predict_parser = subparsers.add_parser("predict", help="Predict one image")
    predict_parser.add_argument("image", type=Path)
    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "analyze":
        analyze()
    else:
        prediction, confidence, image_batch, was_inverted = predict(args.image)
        print(f"Input tensor shape: {image_batch.shape}")
        print(f"Colors inverted: {was_inverted}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()
