from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
HISTORY_PLOT_PATH = PROJECT_DIR / "images" / "training_history.png"
MODEL_PATH = PROJECT_DIR / "models" / "mlp_state_dict.pth"

BATCH_SIZE = 64
LEARNING_RATE = 0.01
EPOCHS = 5


train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=transforms.ToTensor(),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

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

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model state_dict saved to: {MODEL_PATH}")

epoch_numbers = range(1, EPOCHS + 1)
figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(10, 4))

loss_axis.plot(epoch_numbers, loss_history, marker="o")
loss_axis.set_title("Training Loss")
loss_axis.set_xlabel("Epoch")
loss_axis.set_ylabel("Loss")
loss_axis.grid(True)

accuracy_axis.plot(epoch_numbers, accuracy_history, marker="o")
accuracy_axis.set_title("Training Accuracy")
accuracy_axis.set_xlabel("Epoch")
accuracy_axis.set_ylabel("Accuracy (%)")
accuracy_axis.grid(True)

figure.tight_layout()
figure.savefig(HISTORY_PLOT_PATH)

print(f"Training history saved to: {HISTORY_PLOT_PATH}")
plt.show()
