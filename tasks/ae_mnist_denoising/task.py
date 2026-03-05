import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')


def get_task_metadata():
    return {
        'task_name': 'ae_mnist_denoising',
        'description': 'Denoising Autoencoder on MNIST',
        'dataset': 'MNIST',
        'model': 'DenoisingAutoencoder',
        'protocol': 'pytorch_task_v1',
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


def add_noise(images, noise_factor=0.3):
    noisy = images + noise_factor * torch.randn_like(images)
    return torch.clamp(noisy, 0.0, 1.0)


class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def make_dataloaders(batch_size=128):
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def build_model(device=None):
    if device is None:
        device = get_device()
    model = DenoisingAutoencoder().to(device)
    return model


def train(model, train_loader, val_loader, epochs=10, lr=1e-3, noise_factor=0.3):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for images, _ in train_loader:
            images = images.to(device)
            noisy = add_noise(images, noise_factor).to(device)
            output = model(noisy)
            loss = criterion(output, images.view(-1, 784))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        epoch_loss = running_loss / count
        loss_history.append(epoch_loss)

        model.eval()
        val_running = 0.0
        val_count = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                noisy = add_noise(images, noise_factor).to(device)
                output = model(noisy)
                loss = criterion(output, images.view(-1, 784))
                val_running += loss.item() * images.size(0)
                val_count += images.size(0)

        val_loss = val_running / val_count
        val_loss_history.append(val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_loss:.6f} Val Loss: {val_loss:.6f}")

    return {
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
    }


def evaluate(model, data_loader, noise_factor=0.3):
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    model.eval()
    running_loss = 0.0
    count = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            noisy = add_noise(images, noise_factor).to(device)
            output = model(noisy)
            loss = criterion(output, images.view(-1, 784))
            running_loss += loss.item() * images.size(0)
            count += images.size(0)
    mse = running_loss / count
    return {'mse': mse}


def predict(model, X):
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        output = model(X)
    return output


def save_artifacts(model, metrics, loss_history, val_loss_history, data_loader, noise_factor=0.3):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model.pth'))

    # grab some samples for visualization
    device = next(model.parameters()).device
    images_list = []
    for images, _ in data_loader:
        images_list.append(images)
        if sum(b.size(0) for b in images_list) >= 8:
            break
    sample_images = torch.cat(images_list, dim=0)[:8]

    noisy_images = add_noise(sample_images, noise_factor)
    model.eval()
    with torch.no_grad():
        reconstructed = model(noisy_images.to(device)).cpu().view(-1, 1, 28, 28)

    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    for i in range(8):
        axes[0, i].imshow(sample_images[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(noisy_images[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[2, i].axis('off')
    axes[0, 0].set_title('Original')
    axes[1, 0].set_title('Noisy')
    axes[2, 0].set_title('Reconstructed')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ae_mnist_samples.png'))
    plt.close()


if __name__ == '__main__':
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = make_dataloaders()
    model = build_model(device)
    result = train(model, train_loader, val_loader, epochs=10)

    loss_history = result['loss_history']
    val_loss_history = result['val_loss_history']

    train_metrics = evaluate(model, train_loader, noise_factor=0.3)
    val_metrics = evaluate(model, val_loader, noise_factor=0.3)

    print(f"Train MSE: {train_metrics['mse']:.6f}")
    print(f"Val MSE: {val_metrics['mse']:.6f}")

    save_artifacts(model, val_metrics, loss_history, val_loss_history, val_loader, noise_factor=0.3)

    assert val_metrics['mse'] < 0.05, f"val mse too high: {val_metrics['mse']:.4f}"
    print("OK")
    sys.exit(0)
