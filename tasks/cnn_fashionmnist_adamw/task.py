import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


def get_task_metadata():
    return {
        "task_name": "cnn_fashionmnist_adamw",
        "dataset": "FashionMNIST",
        "model": "CNN",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "epochs": 5,
        "batch_size": 64,
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_dataloaders(batch_size=64):
    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=ToTensor()
    )
    val_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=ToTensor()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def build_model(device=None):
    if device is None:
        device = get_device()
    model = CNN().to(device)
    return model


def train(model, train_loader, val_loader, epochs=5, lr=1e-3):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / num_batches
        loss_history.append(avg_train_loss)

        model.eval()
        val_running_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_running_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_running_loss / val_batches
        val_loss_history.append(avg_val_loss)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

    return {
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
    }


def evaluate(model, data_loader):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            num_batches += 1
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    return {
        "accuracy": correct / total,
        "loss": total_loss / num_batches,
    }


def predict(model, X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        outputs = model(X)
        preds = outputs.argmax(dim=1)
    return preds


def save_artifacts(model, metrics, loss_history, val_loss_history):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pt'))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Train')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FashionMNIST Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve.png'))
    plt.close()


if __name__ == '__main__':
    try:
        set_seed(42)
        device = get_device()
        print(f"Using device: {device}")

        train_loader, val_loader = make_dataloaders()
        model = build_model(device)
        result = train(model, train_loader, val_loader, epochs=5)

        train_metrics = evaluate(model, train_loader)
        val_metrics = evaluate(model, val_loader)

        print(f"Train Accuracy: {train_metrics['accuracy']:.4f}, Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val Accuracy:   {val_metrics['accuracy']:.4f}, Val Loss:   {val_metrics['loss']:.4f}")

        save_artifacts(model, val_metrics, result['loss_history'], result['val_loss_history'])

        assert val_metrics['accuracy'] > 0.85, f"Val accuracy {val_metrics['accuracy']:.4f} <= 0.85"
        print("OK")
        sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)
