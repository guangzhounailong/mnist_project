import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = PROJECT_DIR / "models" / "mlp_state_dict.pth"
HISTORY_PLOT_PATH = PROJECT_DIR / "images" / "training_history.png"
ANALYSIS_IMAGE_PATH = PROJECT_DIR / "images" / "prediction_analysis.png"

BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 5
SAMPLES_PER_GROUP = 4


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, images):
        return self.network(images)


def create_data_loader(train, shuffle):
    dataset = datasets.MNIST(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )
    return dataset, data_loader


def load_model():
    model = MLP()
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def train():
    import matplotlib.pyplot as plt

    train_dataset, train_loader = create_data_loader(train=True, shuffle=True)
    model = MLP()
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
    print(f"Model state_dict saved to: {MODEL_PATH}")

    epoch_numbers = range(1, EPOCHS + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(10, 4))
    loss_axis.plot(epoch_numbers, loss_history, marker="o")
    loss_axis.set_title("MLP Training Loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(True)

    accuracy_axis.plot(epoch_numbers, accuracy_history, marker="o")
    accuracy_axis.set_title("MLP Training Accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Accuracy (%)")
    accuracy_axis.grid(True)

    figure.tight_layout()
    HISTORY_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(HISTORY_PLOT_PATH)
    plt.close(figure)
    print(f"Training history saved to: {HISTORY_PLOT_PATH}")


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
    print(f"Loaded MLP from: {MODEL_PATH}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def preprocess_image(image_source):
    if isinstance(image_source, Image.Image):
        image = image_source.copy()
    else:
        image = Image.open(image_source)

    image = image.convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image_tensor = transforms.ToTensor()(image)

    was_inverted = image_tensor.mean().item() > 0.5
    if was_inverted:
        image_tensor = 1.0 - image_tensor

    return image_tensor.unsqueeze(0), was_inverted


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
    print(f"Prediction analysis saved to: {ANALYSIS_IMAGE_PATH}")


def main():
    parser = argparse.ArgumentParser(description="MNIST MLP workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Train and save the MLP")
    subparsers.add_parser("evaluate", help="Evaluate the saved MLP")
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
