from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_PATH = PROJECT_DIR / "models" / "mlp_state_dict.pth"
ANALYSIS_IMAGE_PATH = PROJECT_DIR / "images" / "prediction_analysis.png"

SAMPLES_PER_GROUP = 4


test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=transforms.ToTensor(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True,
)

model = MLP()
state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()

correct_examples = []
incorrect_examples = []

with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images)
        probabilities = logits.softmax(dim=1)
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

            if prediction == label and len(correct_examples) < SAMPLES_PER_GROUP:
                correct_examples.append(example)
            elif prediction != label and len(incorrect_examples) < SAMPLES_PER_GROUP:
                incorrect_examples.append(example)

        if (
            len(correct_examples) == SAMPLES_PER_GROUP
            and len(incorrect_examples) == SAMPLES_PER_GROUP
        ):
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
figure.savefig(ANALYSIS_IMAGE_PATH)

for index, (_, label, prediction, confidence) in enumerate(incorrect_examples, start=1):
    print(
        f"Mistake {index}: True: {label} | "
        f"Predicted: {prediction} | Confidence: {confidence:.2%}"
    )

print(f"Prediction analysis saved to: {ANALYSIS_IMAGE_PATH}")
plt.show()
