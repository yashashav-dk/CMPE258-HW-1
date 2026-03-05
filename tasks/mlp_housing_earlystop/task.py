import sys
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


def get_task_metadata():
    return {
        'task_name': 'mlp_housing_earlystop',
        'description': 'MLP Regression on California Housing with Early Stopping',
        'dataset': 'California Housing',
        'model': 'HousingMLP',
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_dataloaders(batch_size=64, test_size=0.2):
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class HousingMLP(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)


def build_model(input_dim=8, device=None):
    if device is None:
        device = get_device()
    model = HousingMLP(input_dim=input_dim).to(device)
    return model


def train(model, train_loader, val_loader, epochs=200, lr=1e-3, patience=5):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    stopped_epoch = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        avg_train_loss = running_loss / n_batches
        loss_history.append(avg_train_loss)

        model.eval()
        val_running = 0.0
        val_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_loss = criterion(preds, y_batch)
                val_running += val_loss.item()
                val_batches += 1
        avg_val_loss = val_running / val_batches
        val_loss_history.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            stopped_epoch = epoch
            model.load_state_dict(best_model_state)
            print(f"Early stopping at epoch {epoch}")
            break

    return {
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'stopped_epoch': stopped_epoch,
    }


def evaluate(model, data_loader):
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_preds.append(preds.cpu())
            all_targets.append(y_batch)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mse = torch.mean((all_preds - all_targets) ** 2).item()
    ss_res = torch.sum((all_targets - all_preds) ** 2).item()
    ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2).item()
    r2 = 1.0 - (ss_res / ss_tot)

    return {'mse': float(mse), 'r2': float(r2)}


def predict(model, X):
    device = next(model.parameters()).device
    model.eval()
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    X = X.to(device)
    with torch.no_grad():
        preds = model(X)
    return preds.cpu()


def save_artifacts(model, metrics, loss_history, val_loss_history, stopped_epoch):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pth'))

    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    if stopped_epoch is not None:
        plt.axvline(x=stopped_epoch, color='r', linestyle='--', label=f'Early Stop (epoch {stopped_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Housing Regression Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_housing_loss.png'))
    plt.close()


if __name__ == '__main__':
    try:
        set_seed(42)
        device = get_device()
        print(f"Device: {device}")

        train_loader, val_loader = make_dataloaders()
        model = build_model(device=device)

        result = train(model, train_loader, val_loader, epochs=200, patience=5)
        loss_history = result['loss_history']
        val_loss_history = result['val_loss_history']
        stopped_epoch = result['stopped_epoch']

        train_metrics = evaluate(model, train_loader)
        val_metrics = evaluate(model, val_loader)

        print(f"Stopped epoch: {stopped_epoch}")
        print(f"Train MSE: {train_metrics['mse']:.4f}, Train R2: {train_metrics['r2']:.4f}")
        print(f"Val MSE: {val_metrics['mse']:.4f}, Val R2: {val_metrics['r2']:.4f}")

        save_artifacts(model, val_metrics, loss_history, val_loss_history, stopped_epoch)

        assert val_metrics['r2'] > 0.60, f"Val R2 {val_metrics['r2']:.4f} <= 0.60"
        print("OK")
        sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)
